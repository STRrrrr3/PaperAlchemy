import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

try:
    import gradio as gr
except ImportError as exc:
    raise RuntimeError("Gradio is required to launch the PaperAlchemy web UI.") from exc

try:
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
    from playwright.sync_api import sync_playwright
except ImportError:
    PlaywrightTimeoutError = RuntimeError
    sync_playwright = None

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from src.agent_patch import (
    patch_agent_node,
    patch_executor_node,
)
from src.agent_coder import run_coder_agent
from src.agent_planner import run_planner_agent
from src.agent_reader import run_reader_agent
from src.agent_translator import translator_node
from src.deterministic_template_selector import score_and_select_templates
from src.human_feedback import (
    build_human_feedback_payload,
    empty_human_feedback,
    extract_human_feedback_text,
)
from src.page_manifest import (
    enrich_page_plan_shell_contracts,
    missing_shell_contract_block_ids,
)
from src.parser import parse_pdf
from src.schemas import CoderArtifact, PagePlan, StructuredPaper
from src.state import WorkflowState
from src.template_resources import SyncedTemplateAssets, ensure_autopage_template_assets

PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_ROOT / "data" / "input"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
PREVIEW_CACHE_DIR = OUTPUT_DIR / "_preview_cache"
TEMPLATE_TOP_K = 5

APP_CSS = """
#paperalchemy-preview {
  min-height: 78vh;
  border: 1px solid #d9dde7;
  border-radius: 16px;
  overflow: auto;
  background: #ffffff;
  box-shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
}

#paperalchemy-preview img {
  width: 100%;
  height: auto;
  display: block;
}

#paperalchemy-logs textarea {
  font-family: Consolas, "SFMono-Regular", monospace;
}
"""


def load_cached_structured_data(path: Path) -> StructuredPaper | None:
    if not path.exists():
        return None

    print("[PaperAlchemy] Found cached structured paper, loading...")
    try:
        with open(path, "r", encoding="utf-8") as file:
            data_dict = json.load(file)
        structured_data = StructuredPaper(**data_dict)
        print(f"[PaperAlchemy] Structured paper loaded: {structured_data.paper_title}")
        return structured_data
    except Exception as exc:
        print(f"[PaperAlchemy] Structured cache is invalid, rerunning Reader: {exc}")
        return None


def save_structured_data(path: Path, structured_data: StructuredPaper) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(structured_data.model_dump(), file, indent=2, ensure_ascii=False)


def save_page_plan(path: Path, page_plan: PagePlan) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(page_plan.model_dump(), file, indent=2, ensure_ascii=False)


def save_coder_artifact(path: Path, artifact: CoderArtifact) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(artifact.model_dump(), file, indent=2, ensure_ascii=False)


def _read_text_with_fallback(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def _review_accordion_updates(stage: str) -> tuple[dict[str, Any], dict[str, Any]]:
    normalized_stage = str(stage or "").strip().lower()
    if normalized_stage == "overview":
        return gr.update(open=True), gr.update(open=False)
    if normalized_stage == "outline":
        return gr.update(open=False), gr.update(open=True)
    return gr.update(open=False), gr.update(open=False)


def _enrich_page_plan_shell_contracts(page_plan: PagePlan) -> PagePlan:
    template_entry_path = (
        PROJECT_ROOT
        / page_plan.template_selection.selected_root_dir
        / str(page_plan.template_selection.selected_entry_html or "").strip()
    )
    if not template_entry_path.exists():
        raise FileNotFoundError(f"Template entry html not found for shell extraction: {template_entry_path}")

    enriched_plan = enrich_page_plan_shell_contracts(
        page_plan,
        _read_text_with_fallback(template_entry_path),
    )
    missing_blocks = missing_shell_contract_block_ids(enriched_plan)
    if missing_blocks:
        raise ValueError(
            "Template shell extraction failed for block(s): " + ", ".join(missing_blocks)
        )
    return enriched_plan


def normalize_coder_artifact(artifact: Any) -> CoderArtifact | None:
    if artifact is None:
        return None
    if isinstance(artifact, CoderArtifact):
        return artifact
    try:
        return CoderArtifact.model_validate(artifact)
    except Exception:
        return None


def ensure_parsed_output(pdf_filename: str, output_dir: Path) -> bool:
    output_md_path = output_dir / "full_paper.md"
    parsed_json_path = output_dir / "parsed_data.json"
    if output_md_path.exists() and parsed_json_path.exists():
        print("[PaperAlchemy] Parsed PDF assets already exist, skipping parser.")
        return True

    print("[PaperAlchemy] Parsing PDF...")
    parse_pdf(pdf_filename)

    if output_md_path.exists() and parsed_json_path.exists():
        return True

    print("[PaperAlchemy] Parser output is incomplete: missing full_paper.md or parsed_data.json.")
    return False


def list_available_pdfs() -> list[str]:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(path.name for path in INPUT_DIR.glob("*.pdf"))


def get_default_pdf() -> str:
    available_pdfs = list_available_pdfs()
    if not available_pdfs:
        raise FileNotFoundError(
            f"No PDFs found in {INPUT_DIR}. Add a paper to data/input before starting generation."
        )
    return available_pdfs[0]


def build_user_constraints(
    background_color: str,
    density: str,
    navigation: str,
    layout: str,
) -> dict[str, str]:
    return {
        "background_color": background_color,
        "page_density": density,
        "has_navigation": navigation,
        "image_layout": layout,
    }


def build_planner_constraints(tags_json_path: Path, top_k: int = TEMPLATE_TOP_K) -> dict[str, Any]:
    return {
        "target_framework": "static-html",
        "max_templates_for_planner": 120,
        "max_entry_candidates": 3,
        "template_candidate_top_k": max(1, int(top_k)),
        "max_blocks": 10,
        "min_blocks": 6,
        "template_tags_json_path": str(tags_json_path),
    }


def build_generation_constraints(
    synced_assets: SyncedTemplateAssets,
    selected_candidate: dict[str, Any],
    ranked_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    planner_candidates = [
        {
            "template_id": item.get("template_id"),
            "template_name": item.get("template_name"),
            "template_path": item.get("template_path"),
            "entry_html": item.get("entry_html"),
            "entry_html_path": item.get("entry_html_path"),
            "score": item.get("score", 0.0),
            "max_possible_score": item.get("max_possible_score", 0.0),
            "matched_features": list(item.get("matched_features") or []),
            "mismatched_features": list(item.get("mismatched_features") or []),
            "template_tags": dict(item.get("template_tags") or {}),
            "reasons": list(item.get("reasons") or []),
        }
        for item in ranked_candidates
    ]

    return {
        **build_planner_constraints(
            synced_assets.tags_json_path,
            top_k=max(TEMPLATE_TOP_K, len(planner_candidates)),
        ),
        "designated_template_id": str(selected_candidate.get("template_id") or "").strip(),
        "designated_template_path": str(selected_candidate.get("template_path") or "").strip(),
        "designated_entry_html": str(selected_candidate.get("entry_html") or "").strip(),
        "designated_template_score": float(selected_candidate.get("score") or 0.0),
        "ui_ranked_candidates": planner_candidates,
    }


def get_output_paths(paper_folder_name: str) -> tuple[Path, Path, Path, Path]:
    output_dir = OUTPUT_DIR / paper_folder_name
    structured_json_path = output_dir / "structured_paper.json"
    planner_json_path = output_dir / "page_plan.json"
    coder_json_path = output_dir / "coder_artifact.json"
    return output_dir, structured_json_path, planner_json_path, coder_json_path


def append_log_lines(existing_text: str, new_lines: list[str]) -> str:
    merged = [line for line in str(existing_text or "").splitlines() if line.strip()]
    merged.extend(line for line in new_lines if str(line).strip())
    return "\n".join(merged)


def _sanitize_preview_name(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(value or "").strip())
    return normalized.strip("._") or "preview"


def build_template_preview_path(candidate: dict[str, Any]) -> Path:
    template_id = _sanitize_preview_name(str(candidate.get("template_id") or "template"))
    entry_name = _sanitize_preview_name(Path(str(candidate.get("entry_html") or "index.html")).stem)
    PREVIEW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return PREVIEW_CACHE_DIR / f"{template_id}_{entry_name}.png"


def build_final_preview_path(entry_html_path: Path) -> Path:
    site_dir = entry_html_path.parent
    site_dir.mkdir(parents=True, exist_ok=True)
    return site_dir / "final_render.png"


def take_local_screenshot(html_absolute_path: str, output_image_path: str) -> str:
    html_path = Path(html_absolute_path).absolute()
    image_path = Path(output_image_path).absolute()

    if not html_path.exists():
        print(f"[Preview] Screenshot skipped: HTML file not found at {html_path}")
        return ""

    if sync_playwright is None:
        print("[Preview] Screenshot skipped: playwright is not installed.")
        return ""

    image_path.parent.mkdir(parents=True, exist_ok=True)
    target_uri = html_path.as_uri()

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            context = browser.new_context(viewport={"width": 1920, "height": 1080})
            page = context.new_page()

            try:
                page.goto(target_uri, wait_until="networkidle", timeout=45000)
            except PlaywrightTimeoutError as exc:
                print(f"[Preview] networkidle timeout for {target_uri}: {exc}. Capturing best-effort screenshot.")
                try:
                    page.wait_for_timeout(1500)
                except Exception:
                    pass

            page.screenshot(path=str(image_path), full_page=True)
            context.close()
            browser.close()
            return str(image_path)
    except Exception as exc:
        print(
            "[Preview] Playwright screenshot failed for "
            f"{html_path}: {exc}. Ensure `playwright` is installed and Chromium is available."
        )
        return ""


def _message_content_to_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str) and item.strip():
                parts.append(item.strip())
                continue
            if isinstance(item, dict):
                text = str(item.get("text") or "").strip()
                if text:
                    parts.append(text)
        return "\n".join(parts).strip()

    return str(content or "").strip()


def _coerce_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        cleaned = value.strip()
        return [cleaned] if cleaned else []

    if isinstance(value, list):
        results: list[str] = []
        for item in value:
            if isinstance(item, str):
                cleaned = item.strip()
                if cleaned:
                    results.append(cleaned)
        return results

    return []


def _extract_front_matter_candidates(structured_paper_dict: dict[str, Any]) -> tuple[list[str], list[str]]:
    authors = _coerce_string_list(
        structured_paper_dict.get("authors")
        or structured_paper_dict.get("author_names")
        or structured_paper_dict.get("paper_authors")
    )
    affiliations = _coerce_string_list(
        structured_paper_dict.get("affiliations")
        or structured_paper_dict.get("institutions")
        or structured_paper_dict.get("paper_affiliations")
    )

    if authors and affiliations:
        return authors, affiliations

    candidate_texts: list[str] = []
    overall_summary = str(structured_paper_dict.get("overall_summary") or "").strip()
    if overall_summary:
        candidate_texts.append(overall_summary)

    for section in list(structured_paper_dict.get("sections") or [])[:2]:
        rich_web_content = str(section.get("rich_web_content") or "").strip()
        if rich_web_content:
            candidate_texts.append(rich_web_content)

    author_keywords = ("author", "authors", "corresponding author", "affiliated")
    affiliation_keywords = (
        "university",
        "institute",
        "college",
        "school",
        "department",
        "lab",
        "laboratory",
        "company",
        "research",
        "affiliation",
        "affiliations",
    )

    for raw_text in candidate_texts:
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", raw_text):
            normalized = " ".join(sentence.split()).strip(" -")
            if not normalized:
                continue

            lowered = normalized.lower()
            if any(keyword in lowered for keyword in author_keywords) and normalized not in authors:
                authors.append(normalized)
            if any(keyword in lowered for keyword in affiliation_keywords) and normalized not in affiliations:
                affiliations.append(normalized)

    return authors[:3], affiliations[:4]


def _trim_review_text(text: Any, max_chars: int = 700) -> str:
    normalized = " ".join(str(text or "").split()).strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."


def format_paper_to_markdown(structured_paper_dict: dict[str, Any]) -> str:
    paper_title = str(structured_paper_dict.get("paper_title") or "Untitled Paper").strip()
    overall_summary = str(structured_paper_dict.get("overall_summary") or "").strip()
    sections = list(structured_paper_dict.get("sections") or [])
    authors, affiliations = _extract_front_matter_candidates(structured_paper_dict)

    lines = [
        f"# {paper_title}",
        "",
        "This is the Reader extraction source pack that the Planner will use as input.",
        "It is not the final webpage outline.",
        "Use Revise Extraction to add missing source material, remove incorrect content, or fix metadata and asset grounding before webpage planning.",
        "",
        "## Authors",
    ]

    if authors:
        lines.extend(f"- {author}" for author in authors)
    else:
        lines.append("- Not explicitly extracted in the current StructuredPaper output.")

    lines.extend(["", "## Affiliations"])
    if affiliations:
        lines.extend(f"- {affiliation}" for affiliation in affiliations)
    else:
        lines.append("- Not explicitly extracted in the current StructuredPaper output.")

    lines.extend(["", "## Overall Summary"])
    lines.append(overall_summary or "No overall summary was extracted.")

    lines.extend(["", "## Source Sections"])
    if not sections:
        lines.append("- No sections were extracted.")
        return "\n".join(lines)

    for index, section in enumerate(sections, start=1):
        section_title = str(section.get("section_title") or f"Section {index}").strip()
        rich_web_content = str(section.get("rich_web_content") or "").strip()
        related_figures = list(section.get("related_figures") or [])

        lines.extend(["", f"### {index}. {section_title}"])
        lines.append(_trim_review_text(rich_web_content) or "No rich web content was extracted.")

        if related_figures:
            lines.extend(["", "Linked assets:"])
            for figure in related_figures[:3]:
                image_path = str(figure.get("image_path") or "").strip()
                caption = str(figure.get("caption") or "").strip()
                figure_label = f"`{image_path}`" if image_path else "`(missing path)`"
                if caption:
                    lines.append(f"- {figure_label}: {caption}")
                else:
                    lines.append(f"- {figure_label}")

    return "\n".join(lines)


def format_page_plan_to_markdown(
    page_plan_dict: dict[str, Any],
    structured_paper_dict: dict[str, Any] | None = None,
) -> str:
    page_plan = PagePlan.model_validate(page_plan_dict)
    outline_items = sorted(page_plan.page_outline, key=lambda item: item.order)
    structured_sections = list((structured_paper_dict or {}).get("sections") or [])
    source_section_titles = [
        str(section.get("section_title") or "").strip()
        for section in structured_sections
        if str(section.get("section_title") or "").strip()
    ]

    usage_by_section: dict[str, list[str]] = defaultdict(list)
    merged_blocks: list[tuple[str, list[str]]] = []
    for item in outline_items:
        for source_section in item.source_sections:
            usage_by_section[source_section].append(item.block_id)
        if len(item.source_sections) > 1:
            merged_blocks.append((item.block_id, list(item.source_sections)))

    unused_sections = [
        section_title
        for section_title in source_section_titles
        if section_title not in usage_by_section
    ]
    split_sections = [
        (section_title, block_ids)
        for section_title, block_ids in usage_by_section.items()
        if len(block_ids) > 1
    ]

    lines = [
        "# Planned Webpage Outline",
        "",
        "This is the planner-stage webpage outline that the first draft will follow.",
        "Use Revise Outline to add, remove, merge, split, rename, or reorder webpage sections before generating the draft.",
        "",
        "## Ordered Webpage Sections",
    ]

    if not outline_items:
        lines.append("- No webpage sections are currently planned.")
        return "\n".join(lines)

    for item in outline_items:
        lines.extend(
            [
                "",
                f"### {item.order}. {item.title}",
                f"- `block_id`: `{item.block_id}`",
                f"- Objective: {item.objective}",
                "- Source sections: "
                + (", ".join(item.source_sections) if item.source_sections else "(none)"),
                f"- Estimated height: {item.estimated_height}",
            ]
        )

    lines.extend(["", "## Mapping Diagnostics"])
    lines.append(
        "- Unused source sections: "
        + (", ".join(unused_sections) if unused_sections else "None")
    )
    if split_sections:
        lines.append(
            "- Split source sections: "
            + "; ".join(
                f"`{section_title}` -> {', '.join(block_ids)}"
                for section_title, block_ids in split_sections
            )
        )
    else:
        lines.append("- Split source sections: None")

    if merged_blocks:
        lines.append(
            "- Merged webpage blocks: "
            + "; ".join(
                f"`{block_id}` <- {', '.join(source_sections)}"
                for block_id, source_sections in merged_blocks
            )
        )
    else:
        lines.append("- Merged webpage blocks: None")

    return "\n".join(lines)


def build_candidate_label(candidate: dict[str, Any]) -> str:
    template_name = str(candidate.get("display_name") or candidate.get("template_id") or "unknown-template")
    rank = int(candidate.get("rank") or 0)
    score = float(candidate.get("score") or 0.0)
    max_score = float(candidate.get("max_possible_score") or 0.0)
    return f"{rank}. {template_name} (score {score:.2f}/{max_score:.2f})"


def attach_candidate_labels(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labeled: list[dict[str, Any]] = []
    for candidate in candidates:
        enriched = dict(candidate)
        enriched["ui_label"] = build_candidate_label(enriched)
        labeled.append(enriched)
    return labeled


def resolve_selected_candidate(
    selected_label: str | None,
    search_state: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not selected_label:
        return None

    for candidate in (search_state or {}).get("candidates", []):
        if str(candidate.get("ui_label") or "") == str(selected_label):
            return candidate
    return None


def ensure_template_assets() -> SyncedTemplateAssets:
    return ensure_autopage_template_assets(PROJECT_ROOT)


def reader_phase_node(state: WorkflowState) -> dict[str, Any]:
    paper_folder_name = str(state.get("paper_folder_name") or "").strip()
    if not paper_folder_name:
        raise ValueError("paper_folder_name is missing for reader phase.")

    human_directives = extract_human_feedback_text(state.get("human_directives"))
    previous_structured_paper = state.get("structured_paper")
    previous_structured_data: StructuredPaper | None = None
    if previous_structured_paper:
        try:
            previous_structured_data = StructuredPaper.model_validate(previous_structured_paper)
        except Exception:
            previous_structured_data = None

    _, structured_json_path, _, _ = get_output_paths(paper_folder_name)
    structured_data: StructuredPaper | None = None
    if not human_directives:
        structured_data = load_cached_structured_data(structured_json_path)

    if human_directives:
        print("[Reader] Revising structured extraction from human directives...")
    if not structured_data:
        print("[Reader] Running reader agent...")
        structured_data = run_reader_agent(
            paper_folder_name,
            human_directives=human_directives,
            previous_structured_paper=previous_structured_data,
        )
        if not structured_data:
            raise RuntimeError("Reader agent failed to produce structured paper data.")
        save_structured_data(structured_json_path, structured_data)
        print(f"[Reader] Saved structured paper to {structured_json_path}")
    else:
        print(f"[Reader] Reused cached structured paper from {structured_json_path}")

    return {"structured_paper": structured_data}


def overview_node(state: WorkflowState) -> dict[str, Any]:
    structured_data = StructuredPaper.model_validate(state.get("structured_paper"))
    print("[Overview] Building deterministic reader extraction review...")
    return {
        "paper_overview": format_paper_to_markdown(structured_data.model_dump()),
        "outline_overview": "",
        "review_stage": "overview",
    }


def planner_phase_node(state: WorkflowState) -> dict[str, Any]:
    paper_folder_name = str(state.get("paper_folder_name") or "").strip()
    if not paper_folder_name:
        raise ValueError("paper_folder_name is missing for planner phase.")

    structured_data = StructuredPaper.model_validate(state.get("structured_paper"))
    generation_constraints = dict(state.get("generation_constraints") or {})
    user_constraints = dict(state.get("user_constraints") or {})
    previous_page_plan: PagePlan | None = None
    for candidate in (state.get("page_plan"), state.get("approved_page_plan")):
        if not candidate:
            continue
        try:
            previous_page_plan = PagePlan.model_validate(candidate)
            break
        except Exception:
            continue

    print("[Planner] Running template-first planner graph with designated template...")
    page_plan = run_planner_agent(
        paper_folder_name=paper_folder_name,
        structured_data=structured_data,
        generation_constraints=generation_constraints,
        user_constraints=user_constraints,
        human_directives=state.get("human_directives"),
        previous_page_plan=previous_page_plan,
        max_retry=2,
    )
    if not page_plan:
        raise RuntimeError("Planner agent failed to produce a page plan.")
    page_plan = _enrich_page_plan_shell_contracts(page_plan)

    _, _, planner_json_path, _ = get_output_paths(paper_folder_name)
    save_page_plan(planner_json_path, page_plan)
    print(f"[Planner] Saved page plan to {planner_json_path}")
    return {"page_plan": page_plan, "approved_page_plan": None}


def outline_review_node(state: WorkflowState) -> dict[str, Any]:
    page_plan = PagePlan.model_validate(state.get("page_plan"))
    structured_data = StructuredPaper.model_validate(state.get("structured_paper"))
    print("[Outline] Building deterministic webpage outline review...")
    return {
        "outline_overview": format_page_plan_to_markdown(
            page_plan.model_dump(),
            structured_data.model_dump(),
        ),
        "review_stage": "outline",
    }


def coder_phase_node(state: WorkflowState) -> dict[str, Any]:
    paper_folder_name = str(state.get("paper_folder_name") or "").strip()
    if not paper_folder_name:
        raise ValueError("paper_folder_name is missing for coder phase.")

    structured_data = StructuredPaper.model_validate(state.get("structured_paper"))
    approved_page_plan = state.get("approved_page_plan") or state.get("page_plan")
    page_plan = _enrich_page_plan_shell_contracts(PagePlan.model_validate(approved_page_plan))
    previous_coder_artifact = normalize_coder_artifact(state.get("coder_artifact"))

    print("[Coder] Running coder agent...")
    coder_artifact = run_coder_agent(
        paper_folder_name=paper_folder_name,
        structured_data=structured_data,
        page_plan=page_plan,
        human_directives=state.get("human_directives"),
        coder_instructions=str(state.get("coder_instructions") or ""),
        previous_coder_artifact=previous_coder_artifact,
        max_retry=2,
    )
    if not coder_artifact:
        raise RuntimeError("Coder agent failed to build the final webpage.")

    _, _, _, coder_json_path = get_output_paths(paper_folder_name)
    save_coder_artifact(coder_json_path, coder_artifact)
    print(f"[Coder] Generated entry html at {coder_artifact.entry_html}")
    return {
        "page_plan": page_plan,
        "approved_page_plan": page_plan,
        "coder_artifact": coder_artifact,
        "patch_error": "",
        "revision_plan": None,
        "targeted_replacement_plan": None,
        "patch_agent_output": "",
    }


def webpage_review_node(_: WorkflowState) -> dict[str, Any]:
    return {"review_stage": "webpage"}


def human_review_router(state: WorkflowState) -> str:
    if bool(state.get("is_approved")):
        return "planner"
    return "reader"


def outline_review_router(state: WorkflowState) -> str:
    if bool(state.get("is_outline_approved")):
        return "coder"
    return "planner"


def webpage_review_router(state: WorkflowState) -> str:
    if bool(state.get("is_webpage_approved")):
        return "end"
    return "translator"


def build_hitl_workflow():
    workflow = StateGraph(WorkflowState)
    workflow.add_node("reader", reader_phase_node)
    workflow.add_node("overview", overview_node)
    workflow.add_node("planner", planner_phase_node)
    workflow.add_node("outline_review", outline_review_node)
    workflow.add_node("webpage_review", webpage_review_node)
    workflow.add_node("translator", translator_node)
    workflow.add_node("patch_agent", patch_agent_node)
    workflow.add_node("patch_executor", patch_executor_node)
    workflow.add_node("coder", coder_phase_node)

    workflow.set_entry_point("reader")
    workflow.add_edge("reader", "overview")
    workflow.add_conditional_edges(
        "overview",
        human_review_router,
        {
            "planner": "planner",
            "reader": "reader",
        },
    )
    workflow.add_edge("planner", "outline_review")
    workflow.add_conditional_edges(
        "outline_review",
        outline_review_router,
        {
            "planner": "planner",
            "coder": "coder",
        },
    )
    workflow.add_edge("coder", "webpage_review")
    workflow.add_conditional_edges(
        "webpage_review",
        webpage_review_router,
        {
            "translator": "translator",
            "end": END,
        },
    )
    workflow.add_edge("translator", "patch_agent")
    workflow.add_edge("patch_agent", "patch_executor")
    workflow.add_edge("patch_executor", "webpage_review")

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory, interrupt_after=["overview", "outline_review", "webpage_review"])


HITL_WORKFLOW = build_hitl_workflow()


def run_langgraph_batch(
    pdf_filename: str,
    user_constraints: dict[str, str],
    selected_candidate: dict[str, Any],
    ranked_candidates: list[dict[str, Any]],
    log: Callable[[str], None],
) -> CoderArtifact:
    safe_pdf_filename = str(pdf_filename or "").strip()
    if not safe_pdf_filename:
        raise ValueError("A valid PDF filename is required.")

    input_path = INPUT_DIR / safe_pdf_filename
    if not input_path.exists():
        raise FileNotFoundError(f"Input PDF not found: {input_path}")

    template_id = str(selected_candidate.get("template_id") or "").strip()
    template_path = str(selected_candidate.get("template_path") or "").strip()
    entry_html = str(selected_candidate.get("entry_html") or "").strip()
    if not (template_id and template_path and entry_html):
        raise ValueError("The selected template is missing template_id, template_path, or entry_html.")

    log(f"[UI] Using input PDF: {safe_pdf_filename}")
    log(f"[UI] Confirmed template: {template_id}")
    log(f"[UI] Template root: {template_path}")
    log(f"[UI] Entry HTML: {entry_html}")
    log(f"[UI] User constraints: {json.dumps(user_constraints, ensure_ascii=False)}")

    synced_assets = ensure_template_assets()
    log(f"[Assets] Template resources ready at {synced_assets.templates_dir}")

    paper_folder_name = Path(safe_pdf_filename).stem
    output_dir = OUTPUT_DIR / paper_folder_name
    structured_json_path = output_dir / "structured_paper.json"
    planner_json_path = output_dir / "page_plan.json"
    coder_json_path = output_dir / "coder_artifact.json"

    if not ensure_parsed_output(safe_pdf_filename, output_dir):
        raise RuntimeError("PDF parsing failed. Please inspect the parser output logs.")

    structured_data = load_cached_structured_data(structured_json_path)
    if not structured_data:
        log("[Reader] Running reader agent...")
        structured_data = run_reader_agent(paper_folder_name)
        if not structured_data:
            raise RuntimeError("Reader agent failed to produce structured paper data.")
        save_structured_data(structured_json_path, structured_data)
        log(f"[Reader] Saved structured paper to {structured_json_path}")
    else:
        log(f"[Reader] Reused cached structured paper from {structured_json_path}")

    generation_constraints = build_generation_constraints(
        synced_assets=synced_assets,
        selected_candidate=selected_candidate,
        ranked_candidates=ranked_candidates,
    )

    log("[Planner] Running template-first planner graph with designated template...")
    page_plan = run_planner_agent(
        paper_folder_name=paper_folder_name,
        structured_data=structured_data,
        generation_constraints=generation_constraints,
        user_constraints=user_constraints,
        max_retry=2,
    )
    if not page_plan:
        raise RuntimeError("Planner agent failed to produce a page plan.")

    page_plan = _enrich_page_plan_shell_contracts(page_plan)

    save_page_plan(planner_json_path, page_plan)
    log(f"[Planner] Saved page plan to {planner_json_path}")

    log("[Coder] Running coder agent...")
    coder_artifact = run_coder_agent(
        paper_folder_name=paper_folder_name,
        structured_data=structured_data,
        page_plan=page_plan,
        human_directives=empty_human_feedback(),
        coder_instructions="",
        max_retry=2,
    )
    if not coder_artifact:
        raise RuntimeError("Coder agent failed to build the final webpage.")

    save_coder_artifact(coder_json_path, coder_artifact)
    log(f"[Coder] Generated entry html at {coder_artifact.entry_html}")
    log("[Done] Generation completed successfully.")
    return coder_artifact


def find_templates(
    background_color: str,
    density: str,
    navigation: str,
    layout: str,
) -> tuple[Any, ...]:
    log_lines: list[str] = []
    preview_image_path: str | None = None

    try:
        user_constraints = build_user_constraints(
            background_color=background_color,
            density=density,
            navigation=navigation,
            layout=layout,
        )
        synced_assets = ensure_template_assets()
        candidates = score_and_select_templates(
            user_constraints=user_constraints,
            tags_json_path=synced_assets.tags_json_path,
            top_k=TEMPLATE_TOP_K,
        )
        candidates = attach_candidate_labels(candidates)

        if not candidates:
            raise ValueError("No templates matched the provided constraints.")

        search_state = {
            "user_constraints": user_constraints,
            "tags_json_path": str(synced_assets.tags_json_path),
            "candidates": candidates,
        }

        log_lines.append(f"[Search] user_constraints={json.dumps(user_constraints, ensure_ascii=False)}")
        for candidate in candidates:
            log_lines.append(f"[Candidate] {candidate['ui_label']} -> {candidate['template_path']}")

        return (
            gr.update(
                choices=[candidate["ui_label"] for candidate in candidates],
                value=None,
                interactive=True,
            ),
            search_state,
            None,
            "\n".join(log_lines),
            preview_image_path,
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            "",
            "",
            *_review_accordion_updates("webpage"),
            "",
            None,
            "",
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            "",
        )
    except Exception as exc:
        log_lines.append(f"[Error] {exc}")
        return (
            gr.update(choices=[], value=None, interactive=False),
            {"user_constraints": {}, "tags_json_path": "", "candidates": []},
            None,
            "\n".join(log_lines),
            None,
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            "",
            "",
            *_review_accordion_updates("webpage"),
            "",
            None,
            "",
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            "",
        )


def preview_selected_template(
    selected_label: str | None,
    search_state: dict[str, Any] | None,
    current_logs: str,
    current_preview_path: str | None,
) -> tuple[Any, ...]:
    candidate = resolve_selected_candidate(selected_label, search_state)
    if not candidate:
        return (
            None,
            None,
            current_logs,
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            "",
            "",
            *_review_accordion_updates("webpage"),
            "",
            None,
            "",
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            "",
        )

    try:
        entry_html_path = Path(str(candidate.get("entry_html_path") or "")).resolve()
        preview_image_path = take_local_screenshot(
            str(entry_html_path),
            str(build_template_preview_path(candidate)),
        )
        if not preview_image_path:
            raise RuntimeError(f"Failed to render screenshot for {entry_html_path}")
        updated_logs = append_log_lines(
            current_logs,
            [
                f"[Preview] Rendered template screenshot for {candidate['template_id']} from {entry_html_path}",
                f"[Preview] Screenshot path: {preview_image_path}",
            ],
        )
        return (
            preview_image_path,
            candidate,
            updated_logs,
            gr.update(interactive=True),
            gr.update(interactive=False),
            gr.update(interactive=False),
            "",
            "",
            *_review_accordion_updates("webpage"),
            "",
            None,
            "",
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            "",
        )
    except Exception as exc:
        updated_logs = append_log_lines(current_logs, [f"[Error] Failed to render template preview: {exc}"])
        return (
            current_preview_path,
            None,
            updated_logs,
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            "",
            "",
            *_review_accordion_updates("webpage"),
            "",
            None,
            "",
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            "",
        )


def run_extraction(
    pdf_filename: str | None,
    selected_label: str | None,
    search_state: dict[str, Any] | None,
    selected_candidate_state: dict[str, Any] | None,
    current_logs: str,
) -> tuple[Any, ...]:
    selected_candidate = selected_candidate_state or resolve_selected_candidate(selected_label, search_state)
    if not selected_candidate:
        raise gr.Error("Select and preview one of the Top 5 candidate templates before running Step 1.")

    chosen_pdf = str(pdf_filename or "").strip() or get_default_pdf()
    safe_pdf_filename = str(chosen_pdf).strip()
    if not safe_pdf_filename:
        raise gr.Error("A valid PDF filename is required.")

    input_path = INPUT_DIR / safe_pdf_filename
    if not input_path.exists():
        raise gr.Error(f"Input PDF not found: {input_path}")

    run_log_lines = [line for line in str(current_logs or "").splitlines() if line.strip()]

    def log(message: str) -> None:
        print(message)
        run_log_lines.append(message)

    try:
        template_id = str(selected_candidate.get("template_id") or "").strip()
        template_path = str(selected_candidate.get("template_path") or "").strip()
        entry_html = str(selected_candidate.get("entry_html") or "").strip()
        if not (template_id and template_path and entry_html):
            raise ValueError("The selected template is missing template_id, template_path, or entry_html.")

        user_constraints = dict((search_state or {}).get("user_constraints") or {})
        ranked_candidates = list((search_state or {}).get("candidates") or [])
        synced_assets = ensure_template_assets()
        paper_folder_name = Path(safe_pdf_filename).stem
        output_dir, _, _, _ = get_output_paths(paper_folder_name)

        log(f"[UI] Using input PDF: {safe_pdf_filename}")
        log(f"[UI] Confirmed template: {template_id}")
        log(f"[UI] Template root: {template_path}")
        log(f"[UI] Entry HTML: {entry_html}")
        log(f"[UI] User constraints: {json.dumps(user_constraints, ensure_ascii=False)}")
        log(f"[Assets] Template resources ready at {synced_assets.templates_dir}")

        if not ensure_parsed_output(safe_pdf_filename, output_dir):
            raise RuntimeError("PDF parsing failed. Please inspect the parser output logs.")

        generation_constraints = build_generation_constraints(
            synced_assets=synced_assets,
            selected_candidate=selected_candidate,
            ranked_candidates=ranked_candidates,
        )
        thread_id = f"hitl::{paper_folder_name}::{uuid4().hex}"
        config = {"configurable": {"thread_id": thread_id}}
        initial_state: WorkflowState = {
            "paper_folder_name": paper_folder_name,
            "user_constraints": user_constraints,
            "generation_constraints": generation_constraints,
            "human_directives": empty_human_feedback(),
            "coder_instructions": "",
            "patch_agent_output": "",
            "revision_plan": None,
            "targeted_replacement_plan": None,
            "patch_error": "",
            "paper_overview": "",
            "outline_overview": "",
            "is_approved": False,
            "is_outline_approved": False,
            "is_webpage_approved": False,
            "review_stage": "overview",
            "structured_paper": None,
            "page_plan": None,
            "approved_page_plan": None,
            "coder_artifact": None,
        }

        log("[Reader] Running workflow until the extraction review checkpoint...")
        HITL_WORKFLOW.invoke(initial_state, config=config)
        paused_state = HITL_WORKFLOW.get_state(config)
        paused_values = dict(paused_state.values or {})
        paper_overview = str(paused_values.get("paper_overview") or "").strip()
        if not paper_overview:
            structured_data = StructuredPaper.model_validate(paused_values.get("structured_paper"))
            paper_overview = format_paper_to_markdown(structured_data.model_dump())
        log("[Overview] Reader extraction complete. Review the source pack, revise if needed, or approve it to plan the webpage outline.")
        return (
            "\n".join(run_log_lines),
            paper_overview,
            "",
            *_review_accordion_updates("overview"),
            "",
            None,
            gr.update(interactive=True),
            gr.update(interactive=True),
            thread_id,
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            "",
        )
    except Exception as exc:
        log(f"[Error] {exc}")
        return (
            "\n".join(run_log_lines),
            "",
            "",
            *_review_accordion_updates("webpage"),
            "",
            None,
            gr.update(interactive=False),
            gr.update(interactive=False),
            "",
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            "",
        )


def revise_extraction(
    feedback_text: str,
    feedback_images: Any,
    workflow_thread_id: str,
    current_logs: str,
    current_overview: str,
    current_outline_overview: str,
) -> tuple[Any, ...]:
    run_log_lines = [line for line in str(current_logs or "").splitlines() if line.strip()]

    def log(message: str) -> None:
        print(message)
        run_log_lines.append(message)

    try:
        thread_id = str(workflow_thread_id or "").strip()
        if not thread_id:
            raise ValueError("No paused workflow was found. Run Step 1 before requesting a revision.")

        config = {"configurable": {"thread_id": thread_id}}
        snapshot = HITL_WORKFLOW.get_state(config)
        snapshot_values = dict(snapshot.values or {})
        if str(snapshot_values.get("review_stage") or "overview") != "overview":
            raise ValueError("Extraction revisions are only available during the overview review stage.")

        paper_folder_name = str(snapshot_values.get("paper_folder_name") or "").strip()
        if not paper_folder_name:
            raise ValueError("The paused workflow state is missing paper metadata. Run Step 1 again.")

        feedback_payload = build_human_feedback_payload(feedback_text, feedback_images)
        human_directives = extract_human_feedback_text(feedback_payload)
        if not human_directives:
            raise ValueError("Enter feedback text before requesting an extraction revision.")

        log(f"[HITL] Reader revision feedback captured: {human_directives}")

        HITL_WORKFLOW.update_state(
            config,
            {
                "human_directives": feedback_payload,
                "coder_instructions": "",
                "patch_agent_output": "",
                "revision_plan": None,
                "targeted_replacement_plan": None,
                "patch_error": "",
                "is_approved": False,
            },
            as_node="overview",
        )

        log("[Reader] Resuming workflow to revise extraction...")
        HITL_WORKFLOW.invoke(None, config=config)
        paused_state = HITL_WORKFLOW.get_state(config)
        paused_values = dict(paused_state.values or {})
        paper_overview = str(paused_values.get("paper_overview") or "").strip()
        if not paper_overview:
            structured_data = StructuredPaper.model_validate(paused_values.get("structured_paper"))
            paper_overview = format_paper_to_markdown(structured_data.model_dump())
        log("[Overview] Revised extraction complete. Review the updated source pack or approve it to plan the webpage outline.")
        return (
            "\n".join(run_log_lines),
            paper_overview,
            "",
            *_review_accordion_updates("overview"),
            "",
            None,
        )
    except Exception as exc:
        log(f"[Error] {exc}")
        return (
            "\n".join(run_log_lines),
            current_overview,
            current_outline_overview,
            *_review_accordion_updates("overview"),
            feedback_text,
            feedback_images,
        )


def approve_extraction_and_plan_outline(
    workflow_thread_id: str,
    current_logs: str,
    current_overview: str,
    current_outline_overview: str,
) -> tuple[Any, ...]:
    run_log_lines = [line for line in str(current_logs or "").splitlines() if line.strip()]

    def log(message: str) -> None:
        print(message)
        run_log_lines.append(message)

    try:
        thread_id = str(workflow_thread_id or "").strip()
        if not thread_id:
            raise ValueError("No paused workflow was found. Run Step 1 before approving.")

        config = {"configurable": {"thread_id": thread_id}}
        snapshot = HITL_WORKFLOW.get_state(config)
        snapshot_values = dict(snapshot.values or {})
        if str(snapshot_values.get("review_stage") or "overview") != "overview":
            raise ValueError("The extraction review stage has already been completed.")

        paper_folder_name = str(snapshot_values.get("paper_folder_name") or "").strip()
        if not paper_folder_name:
            raise ValueError("The paused workflow state is missing paper metadata. Run Step 1 again.")

        HITL_WORKFLOW.update_state(
            config,
            {
                "human_directives": empty_human_feedback(),
                "coder_instructions": "",
                "patch_agent_output": "",
                "revision_plan": None,
                "targeted_replacement_plan": None,
                "patch_error": "",
                "is_approved": True,
                "is_outline_approved": False,
                "is_webpage_approved": False,
            },
            as_node="overview",
        )

        log("[Planner] Extraction approved. Planning the webpage outline...")
        HITL_WORKFLOW.invoke(None, config=config)
        paused_state = HITL_WORKFLOW.get_state(config)
        paused_values = dict(paused_state.values or {})
        if str(paused_values.get("review_stage") or "") != "outline":
            raise RuntimeError("Workflow did not pause at the outline review stage.")
        outline_overview = str(paused_values.get("outline_overview") or "").strip()
        if not outline_overview:
            page_plan = PagePlan.model_validate(paused_values.get("page_plan"))
            structured_data = StructuredPaper.model_validate(paused_values.get("structured_paper"))
            outline_overview = format_page_plan_to_markdown(
                page_plan.model_dump(),
                structured_data.model_dump(),
            )
        paper_overview = str(paused_values.get("paper_overview") or "").strip() or current_overview
        log("[Outline] Planner outline ready. Review the planned webpage sections, revise them if needed, or approve the outline to generate the first draft.")
        return (
            "\n".join(run_log_lines),
            paper_overview,
            outline_overview,
            *_review_accordion_updates("outline"),
            "",
            None,
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
            "",
        )
    except Exception as exc:
        log(f"[Error] {exc}")
        return (
            "\n".join(run_log_lines),
            current_overview,
            current_outline_overview,
            *_review_accordion_updates("overview"),
            "",
            None,
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            "",
        )


def revise_outline(
    feedback_text: str,
    feedback_images: Any,
    workflow_thread_id: str,
    current_logs: str,
    current_outline_overview: str,
) -> tuple[Any, ...]:
    run_log_lines = [line for line in str(current_logs or "").splitlines() if line.strip()]

    def log(message: str) -> None:
        print(message)
        run_log_lines.append(message)

    try:
        thread_id = str(workflow_thread_id or "").strip()
        if not thread_id:
            raise ValueError("No paused workflow was found. Approve extraction before revising the outline.")

        config = {"configurable": {"thread_id": thread_id}}
        snapshot = HITL_WORKFLOW.get_state(config)
        snapshot_values = dict(snapshot.values or {})
        if str(snapshot_values.get("review_stage") or "") != "outline":
            raise ValueError("Outline revisions are only available during the planner outline review stage.")

        feedback_payload = build_human_feedback_payload(feedback_text, feedback_images)
        human_directives = extract_human_feedback_text(feedback_payload)
        if not human_directives:
            raise ValueError("Enter feedback text before requesting an outline revision.")

        log(f"[HITL] Planner outline revision feedback captured: {human_directives}")

        HITL_WORKFLOW.update_state(
            config,
            {
                "human_directives": feedback_payload,
                "is_outline_approved": False,
                "approved_page_plan": None,
            },
            as_node="outline_review",
        )

        log("[Planner] Resuming workflow to revise the webpage outline...")
        HITL_WORKFLOW.invoke(None, config=config)
        paused_state = HITL_WORKFLOW.get_state(config)
        paused_values = dict(paused_state.values or {})
        outline_overview = str(paused_values.get("outline_overview") or "").strip()
        if not outline_overview:
            page_plan = PagePlan.model_validate(paused_values.get("page_plan"))
            structured_data = StructuredPaper.model_validate(paused_values.get("structured_paper"))
            outline_overview = format_page_plan_to_markdown(
                page_plan.model_dump(),
                structured_data.model_dump(),
            )
        log("[Outline] Revised outline ready. Review the updated webpage structure or approve it to generate the first draft.")
        return (
            "\n".join(run_log_lines),
            outline_overview,
            *_review_accordion_updates("outline"),
            "",
            None,
        )
    except Exception as exc:
        log(f"[Error] {exc}")
        return (
            "\n".join(run_log_lines),
            current_outline_overview,
            *_review_accordion_updates("outline"),
            feedback_text,
            feedback_images,
        )


def approve_outline_and_generate_draft(
    workflow_thread_id: str,
    current_logs: str,
    current_outline_overview: str,
    current_preview: str | None,
    current_html_path: str,
) -> tuple[Any, ...]:
    run_log_lines = [line for line in str(current_logs or "").splitlines() if line.strip()]

    def log(message: str) -> None:
        print(message)
        run_log_lines.append(message)

    try:
        thread_id = str(workflow_thread_id or "").strip()
        if not thread_id:
            raise ValueError("No paused workflow was found. Approve extraction before generating the first draft.")

        config = {"configurable": {"thread_id": thread_id}}
        snapshot = HITL_WORKFLOW.get_state(config)
        snapshot_values = dict(snapshot.values or {})
        if str(snapshot_values.get("review_stage") or "") != "outline":
            raise ValueError("The first draft can only be generated from the outline review stage.")

        approved_page_plan = PagePlan.model_validate(snapshot_values.get("page_plan"))
        HITL_WORKFLOW.update_state(
            config,
            {
                "human_directives": empty_human_feedback(),
                "coder_instructions": "",
                "patch_agent_output": "",
                "revision_plan": None,
                "targeted_replacement_plan": None,
                "patch_error": "",
                "approved_page_plan": approved_page_plan,
                "is_outline_approved": True,
                "is_webpage_approved": False,
            },
            as_node="outline_review",
        )

        log("[Coder] Outline approved. Generating the first webpage draft...")
        HITL_WORKFLOW.invoke(None, config=config)
        paused_state = HITL_WORKFLOW.get_state(config)
        paused_values = dict(paused_state.values or {})
        preview_image_path, entry_html_path = render_current_workflow_preview(paused_values)
        log(f"[Preview] Rendered webpage draft screenshot from {entry_html_path}")
        log("[Webpage] First draft ready. Review the preview, then approve it or request a revision with text and screenshots.")
        return (
            "\n".join(run_log_lines),
            current_outline_overview,
            *_review_accordion_updates("webpage"),
            preview_image_path,
            entry_html_path,
            "",
            None,
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=True),
            gr.update(interactive=True),
        )
    except Exception as exc:
        log(f"[Error] {exc}")
        return (
            "\n".join(run_log_lines),
            current_outline_overview,
            *_review_accordion_updates("outline"),
            current_preview,
            str(current_html_path or ""),
            "",
            None,
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=False),
            gr.update(interactive=False),
        )


def request_webpage_revision(
    feedback_text: str,
    feedback_images: Any,
    workflow_thread_id: str,
    current_logs: str,
    current_preview: str | None,
    current_html_path: str,
) -> tuple[Any, ...]:
    run_log_lines = [line for line in str(current_logs or "").splitlines() if line.strip()]

    def log(message: str) -> None:
        print(message)
        run_log_lines.append(message)

    try:
        thread_id = str(workflow_thread_id or "").strip()
        if not thread_id:
            raise ValueError("No paused workflow was found. Generate the first draft before requesting a revision.")

        config = {"configurable": {"thread_id": thread_id}}
        snapshot = HITL_WORKFLOW.get_state(config)
        snapshot_values = dict(snapshot.values or {})
        if str(snapshot_values.get("review_stage") or "") != "webpage":
            raise ValueError("Webpage revisions are only available after the first draft is generated.")

        feedback_payload = build_human_feedback_payload(feedback_text, feedback_images)
        if not (extract_human_feedback_text(feedback_payload) or feedback_payload["images"]):
            raise ValueError("Enter feedback text or upload at least one screenshot before requesting a webpage revision.")

        # Gradio uploads arrive as temporary file paths, so we convert them once into
        # data URLs here and resume LangGraph with a standard multimodal Gemini payload.
        log(
            "[HITL] Webpage revision feedback captured: "
            f"text_length={len(extract_human_feedback_text(feedback_payload))}, screenshots={len(feedback_payload['images'])}"
        )
        HITL_WORKFLOW.update_state(
            config,
            {
                "human_directives": feedback_payload,
                "coder_instructions": "",
                "patch_agent_output": "",
                "revision_plan": None,
                "targeted_replacement_plan": None,
                "patch_error": "",
                "is_webpage_approved": False,
            },
            as_node="webpage_review",
        )

        log("[Translator] Resuming workflow through Translator -> Patch Agent -> Patch Executor...")
        HITL_WORKFLOW.invoke(None, config=config)
        paused_state = HITL_WORKFLOW.get_state(config)
        paused_values = dict(paused_state.values or {})
        patch_error = str(paused_values.get("patch_error") or "").strip()
        patch_agent_output = str(paused_values.get("patch_agent_output") or "").strip()
        targeted_replacement_plan = paused_values.get("targeted_replacement_plan") or {}
        if hasattr(targeted_replacement_plan, "model_dump"):
            targeted_replacement_plan = targeted_replacement_plan.model_dump()
        if isinstance(targeted_replacement_plan, dict):
            replacements_count = len(targeted_replacement_plan.get("replacements") or [])
            style_change_count = len(targeted_replacement_plan.get("style_changes") or [])
            attribute_change_count = len(targeted_replacement_plan.get("attribute_changes") or [])
            override_rule_count = len(targeted_replacement_plan.get("override_css_rules") or [])
            fallback_count = len(targeted_replacement_plan.get("fallback_blocks") or [])
        else:
            replacements_count = 0
            style_change_count = 0
            attribute_change_count = 0
            override_rule_count = 0
            fallback_count = 0
        if patch_error:
            log(f"[Patch] Safe fail: {patch_error}")
        elif replacements_count or style_change_count or attribute_change_count or override_rule_count or fallback_count:
            log(
                "[Patch] Anchored DOM revision applied: "
                f"{replacements_count} targeted replacement(s), "
                f"{style_change_count} style change(s), "
                f"{attribute_change_count} attribute change(s), "
                f"{override_rule_count} override css rule(s), "
                f"{fallback_count} regenerated block(s)."
            )
        elif patch_agent_output:
            log(f"[Patch] Anchored DOM revision applied: {patch_agent_output}")
        preview_image_path, entry_html_path = render_current_workflow_preview(paused_values)
        log(f"[Preview] Rendered revised webpage screenshot from {entry_html_path}")
        log("[Webpage] Revised draft ready. Review it, then approve it or request another revision.")
        return (
            "\n".join(run_log_lines),
            *_review_accordion_updates("webpage"),
            preview_image_path,
            "",
            None,
            entry_html_path,
            gr.update(interactive=True),
            gr.update(interactive=True),
        )
    except Exception as exc:
        log(f"[Error] {exc}")
        return (
            "\n".join(run_log_lines),
            *_review_accordion_updates("webpage"),
            current_preview,
            feedback_text,
            feedback_images,
            str(current_html_path or ""),
            gr.update(interactive=True),
            gr.update(interactive=True),
        )


def approve_webpage(
    workflow_thread_id: str,
    current_logs: str,
    current_preview: str | None,
    current_html_path: str,
) -> tuple[Any, ...]:
    run_log_lines = [line for line in str(current_logs or "").splitlines() if line.strip()]

    def log(message: str) -> None:
        print(message)
        run_log_lines.append(message)

    try:
        thread_id = str(workflow_thread_id or "").strip()
        if not thread_id:
            raise ValueError("No paused workflow was found. Generate the first draft before approving the webpage.")

        config = {"configurable": {"thread_id": thread_id}}
        snapshot = HITL_WORKFLOW.get_state(config)
        snapshot_values = dict(snapshot.values or {})
        if str(snapshot_values.get("review_stage") or "") != "webpage":
            raise ValueError("The webpage approval step is only available while reviewing a generated draft.")

        HITL_WORKFLOW.update_state(
            config,
            {
                "human_directives": empty_human_feedback(),
                "coder_instructions": "",
                "patch_agent_output": "",
                "revision_plan": None,
                "targeted_replacement_plan": None,
                "patch_error": "",
                "is_webpage_approved": True,
            },
            as_node="webpage_review",
        )

        log("[Webpage] Final approval received. Completing the workflow...")
        HITL_WORKFLOW.invoke(None, config=config)
        log("[Done] Webpage approved and workflow completed successfully.")
        return (
            "\n".join(run_log_lines),
            *_review_accordion_updates("webpage"),
            current_preview,
            str(current_html_path or ""),
            gr.update(interactive=False),
            gr.update(interactive=False),
            "",
            None,
        )
    except Exception as exc:
        log(f"[Error] {exc}")
        return (
            "\n".join(run_log_lines),
            *_review_accordion_updates("webpage"),
            current_preview,
            str(current_html_path or ""),
            gr.update(interactive=True),
            gr.update(interactive=True),
            "",
            None,
        )


def confirm_and_start_generation(
    pdf_filename: str | None,
    selected_label: str | None,
    search_state: dict[str, Any] | None,
    selected_candidate_state: dict[str, Any] | None,
    current_logs: str,
) -> tuple[str, str | None]:
    selected_candidate = selected_candidate_state or resolve_selected_candidate(selected_label, search_state)
    if not selected_candidate:
        raise gr.Error("Select and preview one of the Top 5 candidate templates before starting generation.")

    chosen_pdf = str(pdf_filename or "").strip() or get_default_pdf()
    user_constraints = dict((search_state or {}).get("user_constraints") or {})
    ranked_candidates = list((search_state or {}).get("candidates") or [])

    run_log_lines = [line for line in str(current_logs or "").splitlines() if line.strip()]

    def log(message: str) -> None:
        print(message)
        run_log_lines.append(message)

    preview_image_path: str | None = None

    try:
        coder_artifact = run_langgraph_batch(
            pdf_filename=chosen_pdf,
            user_constraints=user_constraints,
            selected_candidate=selected_candidate,
            ranked_candidates=ranked_candidates,
            log=log,
        )
        entry_html_path = Path(coder_artifact.entry_html).resolve()
        preview_image_path = take_local_screenshot(
            str(entry_html_path),
            str(build_final_preview_path(entry_html_path)),
        )
        if not preview_image_path:
            raise RuntimeError(f"Failed to render final screenshot for {entry_html_path}")
        log(f"[Preview] Rendered generated webpage screenshot from {entry_html_path}")
    except Exception as exc:
        log(f"[Error] {exc}")
        preview_image_path = None

    return "\n".join(run_log_lines), preview_image_path


def render_current_workflow_preview(state_values: dict[str, Any]) -> tuple[str, str]:
    coder_artifact = normalize_coder_artifact(state_values.get("coder_artifact"))
    if not coder_artifact:
        raise RuntimeError("Workflow preview is unavailable because coder_artifact is missing.")
    entry_html_path = Path(coder_artifact.entry_html).resolve()
    preview_image_path = take_local_screenshot(
        str(entry_html_path),
        str(build_final_preview_path(entry_html_path)),
    )
    if not preview_image_path:
        raise RuntimeError(f"Failed to render workflow preview for {entry_html_path}")
    return preview_image_path, str(entry_html_path)


def build_app() -> gr.Blocks:
    available_pdfs = list_available_pdfs()
    default_pdf = available_pdfs[0] if available_pdfs else None

    with gr.Blocks(title="PaperAlchemy", css=APP_CSS) as demo:
        search_state = gr.State({"user_constraints": {}, "tags_json_path": "", "candidates": []})
        selected_candidate_state = gr.State(None)
        workflow_thread_state = gr.State("")
        current_render_html_state = gr.State("")

        gr.Markdown("# PaperAlchemy")

        with gr.Row():
            with gr.Column(scale=1, min_width=320):
                gr.Markdown("### Control Panel")

                pdf_dropdown = gr.Dropdown(
                    choices=available_pdfs,
                    value=default_pdf,
                    label="Paper PDF",
                    allow_custom_value=True,
                )
                background_color = gr.Radio(
                    choices=["light", "dark"],
                    value="light",
                    label="Background Color",
                )
                density = gr.Radio(
                    choices=["spacious", "compact"],
                    value="spacious",
                    label="Page Density",
                )
                navigation = gr.Radio(
                    choices=["yes", "no"],
                    value="no",
                    label="Navigation",
                )
                layout = gr.Radio(
                    choices=["parallelism", "rotation"],
                    value="parallelism",
                    label="Layout Style",
                )

                find_templates_button = gr.Button("Find Templates")
                candidates_radio = gr.Radio(
                    choices=[],
                    label="Top 5 Candidates",
                    interactive=False,
                )
                step1_button = gr.Button(
                    "Step 1: Extract Source Pack",
                    variant="primary",
                    interactive=False,
                )
                with gr.Accordion("Reader Extraction Review", open=False) as paper_review_accordion:
                    paper_markdown = gr.Markdown(
                        value="Run Step 1 to extract the paper into a reviewable source pack."
                    )
                with gr.Accordion("Planned Webpage Outline", open=False) as outline_review_accordion:
                    outline_markdown = gr.Markdown(
                        value="Approve the Reader extraction to generate a reviewable webpage outline."
                    )
                feedback_text = gr.Textbox(
                    label="Human Feedback",
                    placeholder=(
                        "During extraction review: fix missing or incorrect source material. "
                        "During outline review: add, remove, merge, split, rename, or reorder sections. "
                        "During webpage review: describe the visual issue and attach screenshots below."
                    ),
                    lines=4,
                    value="",
                )
                feedback_images = gr.File(
                    label="Reference Screenshots (Optional)",
                    file_count="multiple",
                    file_types=["image"],
                    type="filepath",
                    value=None,
                )
                revise_button = gr.Button(
                    "Revise Extraction",
                    variant="secondary",
                    interactive=False,
                )
                approve_button = gr.Button(
                    "Approve Extraction & Plan Outline",
                    variant="primary",
                    interactive=False,
                )
                revise_outline_button = gr.Button(
                    "Revise Outline",
                    variant="secondary",
                    interactive=False,
                )
                approve_outline_button = gr.Button(
                    "Approve Outline & Generate First Draft",
                    variant="primary",
                    interactive=False,
                )
                request_revision_button = gr.Button(
                    "Request Webpage Revision",
                    variant="secondary",
                    interactive=False,
                )
                approve_webpage_button = gr.Button(
                    "Approve Final Webpage",
                    variant="primary",
                    interactive=False,
                )
                system_logs = gr.Textbox(
                    label="System Logs",
                    value=(
                        "Choose constraints, click Find Templates, preview a candidate, then run Step 1 to extract "
                        "a reviewable source pack. Use Human Feedback plus Revise Extraction until satisfied, then "
                        "approve the extraction to plan the webpage outline. Revise the outline until it matches the "
                        "sections you want on the final page, then generate the first draft. After that, attach "
                        "screenshots and request webpage revisions through the Translator loop until the draft is ready to approve."
                    ),
                    lines=24,
                    interactive=False,
                    elem_id="paperalchemy-logs",
                )

            with gr.Column(scale=2):
                preview_image = gr.Image(
                    value=None,
                    type="filepath",
                    interactive=False,
                    label="Live Webpage Preview",
                    elem_id="paperalchemy-preview",
                )

        find_templates_button.click(
            fn=find_templates,
            inputs=[background_color, density, navigation, layout],
            outputs=[
                candidates_radio,
                search_state,
                selected_candidate_state,
                system_logs,
                preview_image,
                step1_button,
                revise_button,
                approve_button,
                paper_markdown,
                outline_markdown,
                paper_review_accordion,
                outline_review_accordion,
                feedback_text,
                feedback_images,
                workflow_thread_state,
                revise_outline_button,
                approve_outline_button,
                request_revision_button,
                approve_webpage_button,
                current_render_html_state,
            ],
            api_name="find_templates",
        )

        candidates_radio.change(
            fn=preview_selected_template,
            inputs=[candidates_radio, search_state, system_logs, preview_image],
            outputs=[
                preview_image,
                selected_candidate_state,
                system_logs,
                step1_button,
                revise_button,
                approve_button,
                paper_markdown,
                outline_markdown,
                paper_review_accordion,
                outline_review_accordion,
                feedback_text,
                feedback_images,
                workflow_thread_state,
                revise_outline_button,
                approve_outline_button,
                request_revision_button,
                approve_webpage_button,
                current_render_html_state,
            ],
            api_name="preview_template",
        )

        step1_button.click(
            fn=run_extraction,
            inputs=[pdf_dropdown, candidates_radio, search_state, selected_candidate_state, system_logs],
            outputs=[
                system_logs,
                paper_markdown,
                outline_markdown,
                paper_review_accordion,
                outline_review_accordion,
                feedback_text,
                feedback_images,
                revise_button,
                approve_button,
                workflow_thread_state,
                revise_outline_button,
                approve_outline_button,
                request_revision_button,
                approve_webpage_button,
                current_render_html_state,
            ],
            api_name="extract_and_review",
        )

        revise_button.click(
            fn=revise_extraction,
            inputs=[feedback_text, feedback_images, workflow_thread_state, system_logs, paper_markdown, outline_markdown],
            outputs=[
                system_logs,
                paper_markdown,
                outline_markdown,
                paper_review_accordion,
                outline_review_accordion,
                feedback_text,
                feedback_images,
            ],
            api_name="revise_extraction",
        )

        approve_button.click(
            fn=approve_extraction_and_plan_outline,
            inputs=[workflow_thread_state, system_logs, paper_markdown, outline_markdown],
            outputs=[
                system_logs,
                paper_markdown,
                outline_markdown,
                paper_review_accordion,
                outline_review_accordion,
                feedback_text,
                feedback_images,
                revise_button,
                approve_button,
                revise_outline_button,
                approve_outline_button,
                request_revision_button,
                approve_webpage_button,
                current_render_html_state,
            ],
            api_name="approve_extraction_and_plan_outline",
        )

        revise_outline_button.click(
            fn=revise_outline,
            inputs=[feedback_text, feedback_images, workflow_thread_state, system_logs, outline_markdown],
            outputs=[
                system_logs,
                outline_markdown,
                paper_review_accordion,
                outline_review_accordion,
                feedback_text,
                feedback_images,
            ],
            api_name="revise_outline",
        )

        approve_outline_button.click(
            fn=approve_outline_and_generate_draft,
            inputs=[workflow_thread_state, system_logs, outline_markdown, preview_image, current_render_html_state],
            outputs=[
                system_logs,
                outline_markdown,
                paper_review_accordion,
                outline_review_accordion,
                preview_image,
                current_render_html_state,
                feedback_text,
                feedback_images,
                revise_outline_button,
                approve_outline_button,
                request_revision_button,
                approve_webpage_button,
            ],
            api_name="approve_outline_and_generate_draft",
        )

        request_revision_button.click(
            fn=request_webpage_revision,
            inputs=[feedback_text, feedback_images, workflow_thread_state, system_logs, preview_image, current_render_html_state],
            outputs=[
                system_logs,
                paper_review_accordion,
                outline_review_accordion,
                preview_image,
                feedback_text,
                feedback_images,
                current_render_html_state,
                request_revision_button,
                approve_webpage_button,
            ],
            api_name="request_webpage_revision",
        )

        approve_webpage_button.click(
            fn=approve_webpage,
            inputs=[workflow_thread_state, system_logs, preview_image, current_render_html_state],
            outputs=[
                system_logs,
                paper_review_accordion,
                outline_review_accordion,
                preview_image,
                current_render_html_state,
                request_revision_button,
                approve_webpage_button,
                feedback_text,
                feedback_images,
            ],
            api_name="approve_webpage",
        )

    return demo


def main() -> None:
    allowed_paths = [str(OUTPUT_DIR.resolve())]
    try:
        synced_assets = ensure_template_assets()
        allowed_paths.append(str(synced_assets.templates_dir.resolve()))
    except Exception:
        pass

    app = build_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        allowed_paths=allowed_paths,
    )


if __name__ == "__main__":
    main()
