import json
import re
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

try:
    import gradio as gr
except ImportError as exc:
    raise RuntimeError("Gradio is required to launch the PaperAlchemy web UI.") from exc

from bs4 import BeautifulSoup, Tag
try:
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
    from playwright.sync_api import sync_playwright
except ImportError:
    PlaywrightTimeoutError = RuntimeError
    sync_playwright = None

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from src.agent_coder import run_coder_agent
from src.agent_planner import run_planner_agent
from src.agent_reader import run_reader_agent
from src.deterministic_template_selector import score_and_select_templates
from src.llm import get_llm
from src.parser import parse_pdf
from src.prompts import OVERVIEW_SYSTEM_PROMPT, OVERVIEW_USER_PROMPT_TEMPLATE
from src.schemas import CoderArtifact, PagePlan, StructuredPaper, VisualTweakPlan
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


def _read_text_with_fallback(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


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


def _selector_component(tag: Tag) -> str:
    tag_name = tag.name or "div"
    element_id = tag.get("id")
    if element_id:
        return f"{tag_name}#{element_id}"

    classes = [str(item).strip() for item in tag.get("class", []) if str(item).strip()]
    if classes:
        return tag_name + "".join(f".{class_name}" for class_name in classes[:3])

    return tag_name


def _selector_hint(tag: Tag, max_depth: int = 3) -> str:
    parts: list[str] = []
    current: Tag | None = tag

    while current is not None and len(parts) < max_depth:
        parts.append(_selector_component(current))
        if current.get("id"):
            break

        parent = current.parent
        current = parent if isinstance(parent, Tag) else None

    return " > ".join(reversed(parts))


def _build_html_outline(html_text: str, max_lines: int = 120) -> str:
    soup = BeautifulSoup(html_text, "html.parser")
    root = soup.body or soup
    lines: list[str] = []
    seen: set[str] = set()

    for tag in root.find_all(True):
        if tag.name in {"script", "style", "noscript"}:
            continue

        selector = _selector_hint(tag)
        if selector in seen:
            continue
        seen.add(selector)

        attrs: list[str] = []
        for attr_name in ("id", "class", "src", "href", "alt", "aria-label", "style"):
            value = tag.get(attr_name)
            if not value:
                continue
            if isinstance(value, list):
                value_text = " ".join(str(item).strip() for item in value if str(item).strip())
            else:
                value_text = str(value).strip()
            if value_text:
                attrs.append(f'{attr_name}="{value_text[:100]}"')

        text_preview = " ".join(tag.get_text(" ", strip=True).split())
        attrs_suffix = f" ({', '.join(attrs)})" if attrs else ""
        preview_suffix = f' text="{text_preview[:120]}"' if text_preview else ""
        lines.append(f"- {selector}{attrs_suffix}{preview_suffix}")

        if len(lines) >= max_lines:
            break

    return "\n".join(lines) if lines else "- <empty body>"


def _normalize_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        clean = value.strip()
        return [clean] if clean else []

    if not isinstance(value, list):
        return []

    normalized: list[str] = []
    seen: set[str] = set()
    for item in value:
        clean = str(item or "").strip()
        if clean and clean not in seen:
            seen.add(clean)
            normalized.append(clean)
    return normalized


def apply_visual_tweak(current_html_path: str, human_command: str) -> str:
    html_path = Path(str(current_html_path or "").strip()).resolve()
    tweak_command = str(human_command or "").strip()

    if not html_path.exists():
        raise FileNotFoundError(f"Current HTML file not found: {html_path}")
    if not tweak_command:
        raise ValueError("Visual tweak command is empty.")

    html_text = _read_text_with_fallback(html_path)
    soup = BeautifulSoup(html_text, "html.parser")
    body_html = str(soup.body or soup)[:12000]
    dom_outline = _build_html_outline(html_text)

    system_prompt = (
        "You are a precise CSS and DOM manipulator.\n"
        f"The human wants to tweak the webpage: '{tweak_command}'.\n"
        "Do NOT rewrite the whole HTML.\n"
        "You must return a response that matches the VisualTweakPlan schema exactly.\n"
        "Rules:\n"
        "- Use only selectors that plausibly exist in the provided DOM outline/body snippet.\n"
        "- Prefer targeted selectors over broad selectors.\n"
        "- If the human asks to hide/remove something, use elements_to_remove when safe.\n"
        "- If the human asks for a purely stylistic change, use css_patches.\n"
        "- css_patches values must be inline CSS declaration strings.\n"
        "- Leave fields empty instead of inventing unsafe selectors."
    )
    user_prompt = (
        f"### HUMAN_COMMAND\n{tweak_command}\n\n"
        f"### DOM_OUTLINE\n{dom_outline}\n\n"
        f"### BODY_HTML_SNIPPET\n{body_html}\n"
    )

    llm = get_llm(temperature=0, use_smart_model=True)
    structured_llm = llm.with_structured_output(VisualTweakPlan)
    response = structured_llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    try:
        tweak_plan = response if isinstance(response, VisualTweakPlan) else VisualTweakPlan.model_validate(response)
    except Exception as exc:
        raise ValueError(f"Visual Editor Agent returned invalid structured output: {exc}") from exc

    css_patches: dict[str, str] = {}
    for selector, style_text in (tweak_plan.css_patches or {}).items():
        selector_text = str(selector or "").strip()
        style_value = str(style_text or "").strip()
        if selector_text and style_value:
            css_patches[selector_text] = style_value

    elements_to_remove = _normalize_string_list(tweak_plan.elements_to_remove)

    for selector in elements_to_remove:
        try:
            targets = soup.select(selector)
        except Exception as exc:
            print(f"[VisualTweak] warning: invalid removal selector '{selector}': {exc}")
            continue

        if not targets:
            print(f"[VisualTweak] warning: removal selector '{selector}' matched no elements.")
            continue

        for target in targets:
            try:
                target.decompose()
            except Exception as exc:
                print(f"[VisualTweak] warning: failed removing selector '{selector}': {exc}")

    for selector, new_style in css_patches.items():
        try:
            targets = soup.select(selector)
        except Exception as exc:
            print(f"[VisualTweak] warning: invalid css patch selector '{selector}': {exc}")
            continue

        if not targets:
            print(f"[VisualTweak] warning: css patch selector '{selector}' matched no elements.")
            continue

        for target in targets:
            try:
                existing_style = str(target.get("style") or "").strip().rstrip(";")
                patch_style = new_style.strip().lstrip(";")
                if existing_style and patch_style:
                    target["style"] = f"{existing_style}; {patch_style}"
                elif patch_style:
                    target["style"] = patch_style
            except Exception as exc:
                print(f"[VisualTweak] warning: failed applying css patch to '{selector}': {exc}")

    html_path.write_text(str(soup), encoding="utf-8")
    return str(html_path)


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


def format_paper_to_markdown(structured_paper_dict: dict[str, Any]) -> str:
    paper_title = str(structured_paper_dict.get("paper_title") or "Untitled Paper").strip()
    overall_summary = str(structured_paper_dict.get("overall_summary") or "").strip()
    sections = list(structured_paper_dict.get("sections") or [])
    authors, affiliations = _extract_front_matter_candidates(structured_paper_dict)

    lines = [
        f"# {paper_title}",
        "",
        "This is the reader-stage extraction that the Planner will use as source material.",
        "Use the Human Directives box to tell the Planner what to skip, compress, or emphasize on the webpage.",
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

    lines.extend(["", "## Candidate Web Content"])
    if not sections:
        lines.append("- No sections were extracted.")
        return "\n".join(lines)

    for index, section in enumerate(sections, start=1):
        section_title = str(section.get("section_title") or f"Section {index}").strip()
        rich_web_content = str(section.get("rich_web_content") or "").strip()
        related_figures = list(section.get("related_figures") or [])

        lines.extend(["", f"### {index}. {section_title}"])
        lines.append(rich_web_content or "No rich web content was extracted.")

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

    human_directives = str(state.get("human_directives") or "").strip()
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
    structured_json = json.dumps(structured_data.model_dump(), indent=2, ensure_ascii=False)
    llm = get_llm(temperature=0.2, use_smart_model=False)

    print("[Overview] Generating human-readable paper overview...")
    try:
        response = llm.invoke(
            [
                SystemMessage(content=OVERVIEW_SYSTEM_PROMPT),
                HumanMessage(
                    content=OVERVIEW_USER_PROMPT_TEMPLATE.format(
                        structured_paper_json=structured_json,
                    )
                ),
            ]
        )
        paper_overview = _message_content_to_text(response)
    except Exception as exc:
        print(f"[Overview] Warning: overview generation failed, falling back to deterministic formatter: {exc}")
        paper_overview = format_paper_to_markdown(structured_data.model_dump())

    if not paper_overview:
        paper_overview = format_paper_to_markdown(structured_data.model_dump())

    return {"paper_overview": paper_overview}


def planner_phase_node(state: WorkflowState) -> dict[str, Any]:
    paper_folder_name = str(state.get("paper_folder_name") or "").strip()
    if not paper_folder_name:
        raise ValueError("paper_folder_name is missing for planner phase.")

    structured_data = StructuredPaper.model_validate(state.get("structured_paper"))
    generation_constraints = dict(state.get("generation_constraints") or {})
    user_constraints = dict(state.get("user_constraints") or {})

    print("[Planner] Running template-first planner graph with designated template...")
    page_plan = run_planner_agent(
        paper_folder_name=paper_folder_name,
        structured_data=structured_data,
        generation_constraints=generation_constraints,
        user_constraints=user_constraints,
        human_directives=str(state.get("human_directives") or ""),
        max_retry=2,
    )
    if not page_plan:
        raise RuntimeError("Planner agent failed to produce a page plan.")

    _, _, planner_json_path, _ = get_output_paths(paper_folder_name)
    save_page_plan(planner_json_path, page_plan)
    print(f"[Planner] Saved page plan to {planner_json_path}")
    return {"page_plan": page_plan}


def coder_phase_node(state: WorkflowState) -> dict[str, Any]:
    paper_folder_name = str(state.get("paper_folder_name") or "").strip()
    if not paper_folder_name:
        raise ValueError("paper_folder_name is missing for coder phase.")

    structured_data = StructuredPaper.model_validate(state.get("structured_paper"))
    page_plan = PagePlan.model_validate(state.get("page_plan"))

    print("[Coder] Running coder agent...")
    coder_artifact = run_coder_agent(
        paper_folder_name=paper_folder_name,
        structured_data=structured_data,
        page_plan=page_plan,
        human_directives=str(state.get("human_directives") or ""),
        max_retry=1,
    )
    if not coder_artifact:
        raise RuntimeError("Coder agent failed to build the final webpage.")

    _, _, _, coder_json_path = get_output_paths(paper_folder_name)
    save_coder_artifact(coder_json_path, coder_artifact)
    print(f"[Coder] Generated entry html at {coder_artifact.entry_html}")
    return {"coder_artifact": coder_artifact}


def human_review_router(state: WorkflowState) -> str:
    if bool(state.get("is_approved")):
        return "planner"
    return "reader"


def build_hitl_workflow():
    workflow = StateGraph(WorkflowState)
    workflow.add_node("reader", reader_phase_node)
    workflow.add_node("overview", overview_node)
    workflow.add_node("planner", planner_phase_node)
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
    workflow.add_edge("planner", "coder")
    workflow.add_edge("coder", END)

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory, interrupt_after=["overview"])


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

    save_page_plan(planner_json_path, page_plan)
    log(f"[Planner] Saved page plan to {planner_json_path}")

    log("[Coder] Running coder agent...")
    coder_artifact = run_coder_agent(
        paper_folder_name=paper_folder_name,
        structured_data=structured_data,
        page_plan=page_plan,
        human_directives="",
        max_retry=1,
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
) -> tuple[Any, dict[str, Any], Any, str, str | None, Any, Any, Any, str, str, str, str, Any, str]:
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
            log_lines.append(
                f"[Candidate] {candidate['ui_label']} -> {candidate['template_path']}"
            )

        radio_update = gr.update(
            choices=[candidate["ui_label"] for candidate in candidates],
            value=None,
            interactive=True,
        )
        step1_update = gr.update(interactive=False)
        revise_update = gr.update(interactive=False)
        approve_update = gr.update(interactive=False)
        return (
            radio_update,
            search_state,
            None,
            "\n".join(log_lines),
            preview_image_path,
            step1_update,
            revise_update,
            approve_update,
            "",
            "",
            "",
            "",
            gr.update(interactive=False),
            "",
        )
    except Exception as exc:
        log_lines.append(f"[Error] {exc}")
        empty_state = {"user_constraints": {}, "tags_json_path": "", "candidates": []}
        radio_update = gr.update(choices=[], value=None, interactive=False)
        step1_update = gr.update(interactive=False)
        revise_update = gr.update(interactive=False)
        approve_update = gr.update(interactive=False)
        return (
            radio_update,
            empty_state,
            None,
            "\n".join(log_lines),
            None,
            step1_update,
            revise_update,
            approve_update,
            "",
            "",
            "",
            "",
            gr.update(interactive=False),
            "",
        )


def preview_selected_template(
    selected_label: str | None,
    search_state: dict[str, Any] | None,
    current_logs: str,
    current_preview_path: str | None,
) -> tuple[str | None, dict[str, Any] | None, str, Any, Any, Any, str, str, str, str, Any, str]:
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
            "",
            "",
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
            "",
            "",
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
            "",
            "",
            gr.update(interactive=False),
            "",
        )


def run_extraction(
    pdf_filename: str | None,
    selected_label: str | None,
    search_state: dict[str, Any] | None,
    selected_candidate_state: dict[str, Any] | None,
    current_logs: str,
) -> tuple[str, str, str, Any, Any, str, str, Any, str]:
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
            "human_directives": "",
            "paper_overview": "",
            "is_approved": False,
            "structured_paper": None,
            "page_plan": None,
            "coder_artifact": None,
        }

        log("[Reader] Running workflow until HITL breakpoint after overview...")
        HITL_WORKFLOW.invoke(initial_state, config=config)
        paused_state = HITL_WORKFLOW.get_state(config)
        paused_values = dict(paused_state.values or {})
        paper_overview = str(paused_values.get("paper_overview") or "").strip()
        if not paper_overview:
            structured_data = StructuredPaper.model_validate(paused_values.get("structured_paper"))
            paper_overview = format_paper_to_markdown(structured_data.model_dump())
        log("[Overview] Extraction complete. Review the overview, revise if needed, or approve to continue.")
        return (
            "\n".join(run_log_lines),
            paper_overview,
            "",
            gr.update(interactive=True),
            gr.update(interactive=True),
            thread_id,
            "",
            gr.update(interactive=False),
            "",
        )
    except Exception as exc:
        log(f"[Error] {exc}")
        return (
            "\n".join(run_log_lines),
            "",
            "",
            gr.update(interactive=False),
            gr.update(interactive=False),
            "",
            "",
            gr.update(interactive=False),
            "",
        )


def revise_extraction(
    human_directives_text: str,
    workflow_thread_id: str,
    current_logs: str,
    current_overview: str,
) -> tuple[str, str]:
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
        paper_folder_name = str(snapshot_values.get("paper_folder_name") or "").strip()
        if not paper_folder_name:
            raise ValueError("The paused workflow state is missing paper metadata. Run Step 1 again.")

        human_directives = str(human_directives_text or "").strip()
        if not human_directives:
            raise ValueError("Enter Human Directives before requesting an extraction revision.")

        log(f"[HITL] Human directives captured for Reader revision: {human_directives}")

        HITL_WORKFLOW.update_state(
            config,
            {"human_directives": human_directives, "is_approved": False},
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
        log("[Overview] Revised extraction complete. Review the updated overview or approve to continue.")
        return "\n".join(run_log_lines), paper_overview
    except Exception as exc:
        log(f"[Error] {exc}")
        return "\n".join(run_log_lines), current_overview


def approve_and_generate(
    workflow_thread_id: str,
    current_logs: str,
    current_preview: str | None,
    current_html_path: str,
) -> tuple[str, str | None, str, Any]:
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
        paper_folder_name = str(snapshot_values.get("paper_folder_name") or "").strip()
        if not paper_folder_name:
            raise ValueError("The paused workflow state is missing paper metadata. Run Step 1 again.")

        HITL_WORKFLOW.update_state(
            config,
            {"human_directives": "", "is_approved": True},
            as_node="overview",
        )

        log("[Planner] Approval received. Resuming workflow toward webpage generation...")
        HITL_WORKFLOW.invoke(None, config=config)
        final_state = HITL_WORKFLOW.get_state(config)
        final_values = dict(final_state.values or {})
        coder_artifact = CoderArtifact.model_validate(final_values.get("coder_artifact"))
        entry_html_path = Path(coder_artifact.entry_html).resolve()
        preview_image_path = take_local_screenshot(
            str(entry_html_path),
            str(build_final_preview_path(entry_html_path)),
        )
        if not preview_image_path:
            raise RuntimeError(f"Failed to render final screenshot for {entry_html_path}")
        log(f"[Preview] Rendered generated webpage screenshot from {entry_html_path}")
        log("[Done] Generation completed successfully.")
        return "\n".join(run_log_lines), preview_image_path, str(entry_html_path), gr.update(interactive=True)
    except Exception as exc:
        log(f"[Error] {exc}")
        return "\n".join(run_log_lines), current_preview, str(current_html_path or ""), gr.update(interactive=False)


def apply_visual_tweak_from_ui(
    human_command: str,
    current_html_path: str,
    current_logs: str,
    current_preview: str | None,
) -> tuple[str, str | None, str, str]:
    run_log_lines = [line for line in str(current_logs or "").splitlines() if line.strip()]

    def log(message: str) -> None:
        print(message)
        run_log_lines.append(message)

    html_path = str(current_html_path or "").strip()
    tweak_command = str(human_command or "").strip()
    if not html_path:
        log("[Error] No generated webpage is available for visual tweaking yet.")
        return "\n".join(run_log_lines), current_preview, human_command, current_html_path
    if not tweak_command:
        log("[Error] Visual tweak command is empty.")
        return "\n".join(run_log_lines), current_preview, human_command, current_html_path

    try:
        updated_html_path = apply_visual_tweak(html_path, tweak_command)
        preview_image_path = take_local_screenshot(
            updated_html_path,
            str(build_final_preview_path(Path(updated_html_path))),
        )
        if not preview_image_path:
            raise RuntimeError(f"Failed to render screenshot after visual tweak for {updated_html_path}")

        log(f"[VisualTweak] Applied tweak: {tweak_command}")
        log(f"[Preview] Re-rendered tweaked webpage screenshot from {updated_html_path}")
        return "\n".join(run_log_lines), preview_image_path, "", updated_html_path
    except Exception as exc:
        log(f"[Error] Visual tweak failed: {exc}")
        return "\n".join(run_log_lines), current_preview, human_command, current_html_path


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
                    "Step 1: Extract Paper Content",
                    variant="primary",
                    interactive=False,
                )
                gr.Markdown("### Extracted Paper Overview")
                paper_markdown = gr.Markdown(
                    value="Run Step 1 to extract the paper into a readable review overview."
                )
                human_directives = gr.Textbox(
                    label="Human Directives (e.g., 'Extract more details about the dataset')",
                    placeholder="Example: Extract more dataset details, shorten related work, and preserve all author affiliations.",
                    lines=3,
                    value="",
                )
                revise_button = gr.Button(
                    "Revise Extraction",
                    variant="secondary",
                    interactive=False,
                )
                approve_button = gr.Button(
                    "Approve & Generate Webpage",
                    variant="primary",
                    interactive=False,
                )
                visual_tweak_command = gr.Textbox(
                    label="Visual Tweak Command (e.g., 'Hide the second image')",
                    placeholder="Example: Make the title red, hide the bottom table, or reduce hero spacing.",
                    lines=2,
                    value="",
                )
                apply_tweak_button = gr.Button(
                    "Apply Tweak",
                    variant="secondary",
                    interactive=False,
                )
                system_logs = gr.Textbox(
                    label="System Logs",
                    value=(
                        "Choose constraints, click Find Templates, preview a candidate, then run Step 1 to extract "
                        "a readable overview. Use Human Directives plus Revise Extraction until satisfied, then "
                        "approve to generate the webpage. After generation, you can issue natural-language visual tweaks."
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
                human_directives,
                workflow_thread_state,
                visual_tweak_command,
                apply_tweak_button,
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
                human_directives,
                workflow_thread_state,
                visual_tweak_command,
                apply_tweak_button,
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
                human_directives,
                revise_button,
                approve_button,
                workflow_thread_state,
                visual_tweak_command,
                apply_tweak_button,
                current_render_html_state,
            ],
            api_name="extract_and_review",
        )

        revise_button.click(
            fn=revise_extraction,
            inputs=[human_directives, workflow_thread_state, system_logs, paper_markdown],
            outputs=[system_logs, paper_markdown],
            api_name="revise_extraction",
        )

        approve_button.click(
            fn=approve_and_generate,
            inputs=[workflow_thread_state, system_logs, preview_image, current_render_html_state],
            outputs=[system_logs, preview_image, current_render_html_state, apply_tweak_button],
            api_name="approve_and_generate",
        )

        apply_tweak_button.click(
            fn=apply_visual_tweak_from_ui,
            inputs=[visual_tweak_command, current_render_html_state, system_logs, preview_image],
            outputs=[system_logs, preview_image, visual_tweak_command, current_render_html_state],
            api_name="apply_visual_tweak",
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
