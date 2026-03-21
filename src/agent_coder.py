import os
import re
import shutil
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from src.agent_coder_critic import (
    build_coder_critic_router,
    build_vision_qa_router,
    coder_critic_node,
    take_screenshot_action,
    vision_critic_node,
)
from src.human_feedback import extract_human_feedback_text, normalize_human_feedback
from src.json_utils import to_pretty_json
from src.llm import get_llm
from src.page_manifest import build_page_manifest_path, extract_page_manifest, save_page_manifest
from src.prompts import CODER_SYSTEM_PROMPT, CODER_USER_PROMPT_TEMPLATE
from src.schemas import CoderArtifact, PagePlan, StructuredPaper
from src.state import CoderState


def _normalize_page_plan(plan: Any) -> PagePlan | None:
    if isinstance(plan, PagePlan):
        return plan
    if plan is None:
        return None
    try:
        return PagePlan.model_validate(plan)
    except Exception:
        return None


def _normalize_structured_paper(paper: Any) -> StructuredPaper | None:
    if isinstance(paper, StructuredPaper):
        return paper
    if paper is None:
        return None
    try:
        return StructuredPaper.model_validate(paper)
    except Exception:
        return None


def _normalize_coder_artifact(artifact: Any) -> CoderArtifact | None:
    if isinstance(artifact, CoderArtifact):
        return artifact
    if artifact is None:
        return None
    try:
        return CoderArtifact.model_validate(artifact)
    except Exception:
        return None


def _safe_slug(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", text.strip())
    slug = slug.strip("-").lower()
    return slug or "asset"


def _to_html_relative_path(target_path: Path, base_dir: Path) -> str:
    rel_path = os.path.relpath(target_path, start=base_dir)
    web_path = str(rel_path).replace("\\", "/")
    if not web_path.startswith((".", "/")):
        web_path = f"./{web_path}"
    return web_path


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


def _extract_html_document(text: str) -> str:
    raw_text = str(text or "").strip()
    if not raw_text:
        return ""

    fenced_match = re.search(r"```(?:html)?\s*(.*?)```", raw_text, flags=re.IGNORECASE | re.DOTALL)
    if fenced_match:
        raw_text = fenced_match.group(1).strip()

    doctype_match = re.search(r"<!DOCTYPE\s+html[^>]*>", raw_text, flags=re.IGNORECASE)
    html_match = re.search(r"<html\b.*?</html>", raw_text, flags=re.IGNORECASE | re.DOTALL)

    if html_match:
        start = html_match.start()
        end = html_match.end()
        if doctype_match and doctype_match.start() <= start:
            raw_text = raw_text[doctype_match.start() : end]
        else:
            raw_text = raw_text[start:end]
    elif raw_text.lower().startswith("<!doctype") or raw_text.lower().startswith("<html"):
        pass
    else:
        return ""

    if "<html" not in raw_text.lower():
        return ""

    if "<!doctype" not in raw_text.lower():
        raw_text = "<!DOCTYPE html>\n" + raw_text

    return raw_text.strip()


def _ensure_body_markers(html_text: str) -> str:
    result = str(html_text or "")
    if "PaperAlchemy Generated Body Start" not in result:
        result = re.sub(
            r"(<body\b[^>]*>)",
            r"\1\n<!-- PaperAlchemy Generated Body Start -->",
            result,
            count=1,
            flags=re.IGNORECASE,
        )
    if "PaperAlchemy Generated Body End" not in result:
        result = re.sub(
            r"(</body>)",
            r"<!-- PaperAlchemy Generated Body End -->\n\1",
            result,
            count=1,
            flags=re.IGNORECASE,
        )
    return result


def _normalize_html_whitespace(html_text: str) -> str:
    return str(html_text or "").replace("\r\n", "\n").strip() + "\n"


def _normalize_asset_key(value: str) -> str:
    return str(value or "").strip().replace("\\", "/")


def _collect_figure_paths(page_plan: PagePlan, structured_paper: StructuredPaper) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []

    for block in page_plan.blocks:
        for path in block.asset_binding.figure_paths:
            clean = _normalize_asset_key(path)
            if clean and clean not in seen:
                seen.add(clean)
                ordered.append(clean)

    if ordered:
        return ordered

    for section in structured_paper.sections:
        for fig in section.related_figures:
            clean = _normalize_asset_key(fig.image_path)
            if clean and clean not in seen:
                seen.add(clean)
                ordered.append(clean)

    return ordered


def _build_asset_lookup(structured_paper: StructuredPaper) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    for section in structured_paper.sections:
        for fig in section.related_figures:
            key = _normalize_asset_key(fig.image_path)
            if not key or key in lookup:
                continue
            lookup[key] = {
                "caption": str(fig.caption or "").strip(),
                "type": str(fig.type or "").strip(),
                "section_title": str(section.section_title or "").strip(),
            }
    return lookup


def _copy_paper_assets(
    project_root: Path,
    paper_folder_name: str,
    site_dir: Path,
    entry_html_path: Path,
    structured_paper: StructuredPaper,
    figure_paths: list[str],
) -> tuple[list[dict[str, str]], list[str]]:
    asset_manifest: list[dict[str, str]] = []
    copied_assets: list[str] = []
    if not figure_paths:
        return asset_manifest, copied_assets

    source_root = project_root / "data" / "output" / paper_folder_name
    target_dir = site_dir / "assets" / "paper"
    target_dir.mkdir(parents=True, exist_ok=True)

    asset_lookup = _build_asset_lookup(structured_paper)
    used_names: set[str] = set()

    for idx, rel_path in enumerate(figure_paths, start=1):
        clean_rel_path = _normalize_asset_key(rel_path)
        source_path = source_root / clean_rel_path
        if not source_path.exists() or not source_path.is_file():
            continue

        base_name = _safe_slug(source_path.stem)[:60]
        suffix = source_path.suffix or ".png"
        target_name = f"{base_name}{suffix}"
        disambiguation = 2
        while target_name in used_names:
            target_name = f"{base_name}-{disambiguation}{suffix}"
            disambiguation += 1
        used_names.add(target_name)

        target_path = target_dir / target_name
        shutil.copy2(source_path, target_path)

        copied_rel_path = str(target_path.relative_to(site_dir)).replace("\\", "/")
        web_path = _to_html_relative_path(target_path, entry_html_path.parent)
        metadata = asset_lookup.get(clean_rel_path, {})
        asset_manifest.append(
            {
                "source_path": clean_rel_path,
                "web_path": web_path,
                "filename": target_name,
                "caption": str(metadata.get("caption") or ""),
                "type": str(metadata.get("type") or ""),
                "section_title": str(metadata.get("section_title") or ""),
            }
        )
        copied_assets.append(copied_rel_path)

    return asset_manifest, copied_assets


def _format_feedback_block(value: Any) -> str:
    if not isinstance(value, list):
        return "(none)"

    lines: list[str] = []
    for index, item in enumerate(value, start=1):
        clean = str(item or "").strip()
        if clean:
            lines.append(f"{index}. {clean}")
    return "\n".join(lines) if lines else "(none)"


def _read_previous_generated_html(state: CoderState) -> str:
    artifact = _normalize_coder_artifact(state.get("coder_artifact"))
    if not artifact:
        return "(none)"

    entry_path = Path(artifact.entry_html)
    if not entry_path.exists():
        return "(none)"

    try:
        previous_html = _read_text_with_fallback(entry_path).strip()
    except Exception:
        return "(none)"

    return previous_html or "(none)"


def coder_node(state: CoderState) -> dict[str, Any]:
    print(
        f"[PaperAlchemy-Coder] building site "
        f"(attempt {state.get('coder_retry_count', 0) + 1})..."
    )

    page_plan = _normalize_page_plan(state.get("page_plan"))
    structured_paper = _normalize_structured_paper(state.get("structured_paper"))
    paper_folder_name = str(state.get("paper_folder_name") or "").strip()
    human_directives = extract_human_feedback_text(state.get("human_directives"))
    coder_instructions = str(state.get("coder_instructions") or "").strip()
    if not page_plan or not structured_paper or not paper_folder_name:
        print("[PaperAlchemy-Coder] missing page_plan/structured_paper/paper_folder_name.")
        return {}

    previous_generated_html = _read_previous_generated_html(state)

    project_root = Path(__file__).resolve().parent.parent
    template_root = project_root / page_plan.template_selection.selected_root_dir
    template_entry_rel = str(page_plan.template_selection.selected_entry_html or "").strip()
    template_entry_path = template_root / template_entry_rel

    output_dir = project_root / "data" / "output" / paper_folder_name
    site_dir = output_dir / "site"
    generated_entry_html_path = site_dir / "index.html"

    if site_dir.exists():
        shutil.rmtree(site_dir)

    if not template_root.exists():
        print(f"[PaperAlchemy-Coder] template root not found: {template_root}")
        return {}

    if not template_entry_path.exists():
        print(f"[PaperAlchemy-Coder] template entry html not found: {template_entry_path}")
        return {}

    try:
        template_reference_html = _read_text_with_fallback(template_entry_path)
    except Exception as exc:
        print(f"[PaperAlchemy-Coder] failed reading template reference html: {exc}")
        return {}

    shutil.copytree(template_root, site_dir)

    figure_paths = _collect_figure_paths(page_plan, structured_paper)
    asset_manifest, copied_assets = _copy_paper_assets(
        project_root=project_root,
        paper_folder_name=paper_folder_name,
        site_dir=site_dir,
        entry_html_path=generated_entry_html_path,
        structured_paper=structured_paper,
        figure_paths=figure_paths,
    )

    try:
        llm = get_llm(temperature=0.2, use_smart_model=True)
        response = llm.invoke(
            [
                SystemMessage(content=CODER_SYSTEM_PROMPT),
                HumanMessage(
                    content=CODER_USER_PROMPT_TEMPLATE.format(
                        structured_paper_json=to_pretty_json(structured_paper),
                        page_plan_json=to_pretty_json(page_plan),
                        template_reference_html=template_reference_html,
                        coder_instructions=coder_instructions or "(none)",
                        human_directives=human_directives or "(none)",
                        available_paper_assets_json=to_pretty_json(asset_manifest),
                        prior_coder_feedback=_format_feedback_block(state.get("coder_feedback_history")),
                        prior_visual_feedback=_format_feedback_block(state.get("visual_feedback")),
                        previous_generated_html=previous_generated_html,
                    )
                ),
            ]
        )
    except Exception as exc:
        print(f"[PaperAlchemy-Coder] llm generation failed: {exc}")
        return {}

    generated_html = _extract_html_document(_message_content_to_text(response))
    if not generated_html:
        print("[PaperAlchemy-Coder] model did not return a valid HTML document.")
        return {}

    generated_html = _ensure_body_markers(generated_html)
    generated_html = _normalize_html_whitespace(generated_html)
    try:
        page_manifest = extract_page_manifest(
            html_text=generated_html,
            entry_html=generated_entry_html_path,
            selected_template_id=page_plan.template_selection.selected_template_id,
            page_plan=page_plan,
        )
    except Exception as exc:
        print(f"[PaperAlchemy-Coder] generated HTML is missing stable revision anchors: {exc}")
        return {}

    generated_entry_html_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        generated_entry_html_path.write_text(generated_html, encoding="utf-8")
        save_page_manifest(build_page_manifest_path(generated_entry_html_path), page_manifest)
    except Exception as exc:
        print(f"[PaperAlchemy-Coder] failed writing generated html or page manifest: {exc}")
        return {}

    edited_files = ["index.html"]
    copied_entry_html_path = site_dir / template_entry_rel
    if copied_entry_html_path.resolve() != generated_entry_html_path.resolve():
        copied_entry_html_path.parent.mkdir(parents=True, exist_ok=True)
        copied_entry_html_path.write_text(generated_html, encoding="utf-8")
        edited_files.append(str(copied_entry_html_path.relative_to(site_dir)).replace("\\", "/"))

    artifact = CoderArtifact(
        site_dir=str(site_dir),
        entry_html=str(generated_entry_html_path),
        selected_template_id=page_plan.template_selection.selected_template_id,
        copied_assets=copied_assets,
        edited_files=edited_files,
        notes=(
            "v5-anchored-llm-render: generated a complete HTML document with stable data-pa-block and "
            "data-pa-slot anchors, plus page_manifest.json for targeted revisions."
        ),
    )
    return {"coder_artifact": artifact}


def build_coder_graph(max_retry: int = 1):
    workflow = StateGraph(CoderState)
    workflow.add_node("coder", coder_node)
    workflow.add_node("coder_critic", coder_critic_node)
    workflow.add_node("take_screenshot", take_screenshot_action)
    workflow.add_node("vision_critic", vision_critic_node)

    workflow.set_entry_point("coder")
    workflow.add_edge("coder", "coder_critic")
    workflow.add_conditional_edges(
        "coder_critic",
        build_coder_critic_router(max_retry=max_retry),
        {"retry": "coder", "visual_qa": "take_screenshot", "end": END},
    )
    workflow.add_edge("take_screenshot", "vision_critic")
    workflow.add_conditional_edges(
        "vision_critic",
        build_vision_qa_router(),
        {"retry": "coder", "end": END},
    )

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


def run_coder_agent(
    paper_folder_name: str,
    structured_data: StructuredPaper,
    page_plan: PagePlan,
    human_directives: str | dict = "",
    coder_instructions: str = "",
    previous_coder_artifact: CoderArtifact | None = None,
    max_retry: int = 2,
) -> CoderArtifact | None:
    app = build_coder_graph(max_retry=max_retry)
    thread = {"configurable": {"thread_id": f"coder_{paper_folder_name}"}}

    initial_state: CoderState = {
        "paper_folder_name": paper_folder_name,
        "human_directives": normalize_human_feedback(human_directives),
        "coder_instructions": str(coder_instructions or "").strip(),
        "structured_paper": structured_data,
        "page_plan": page_plan,
        "coder_feedback_history": [],
        "visual_feedback": [],
        "visual_screenshot_path": "",
        "visual_iterations": 0,
        "is_visually_approved": False,
        "coder_artifact": previous_coder_artifact,
        "coder_critic_passed": False,
        "coder_retry_count": 0,
    }

    print("[PaperAlchemy-Coder] running Coder + CoderCritic graph...")
    for _ in app.stream(initial_state, thread):
        pass

    final_state = app.get_state(thread)
    artifact_result = final_state.values.get("coder_artifact")
    normalized_artifact: CoderArtifact | None = None
    if artifact_result is not None:
        try:
            normalized_artifact = CoderArtifact.model_validate(artifact_result)
        except Exception:
            normalized_artifact = None

    if not normalized_artifact or not final_state.values.get("coder_critic_passed"):
        print("[PaperAlchemy-Coder] coder completed but critic did not fully pass.")
        return None

    if not final_state.values.get("is_visually_approved") and int(final_state.values.get("visual_iterations", 0)) > 0:
        print("[PaperAlchemy-Coder] visual QA ended without full approval, returning the latest artifact after retry cap.")

    print(f"[PaperAlchemy-Coder] build completed: {normalized_artifact.entry_html}")
    return normalized_artifact
