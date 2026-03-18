import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup, Tag
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from src.agent_coder_critic import (
    build_coder_critic_router,
    build_vision_qa_router,
    coder_critic_node,
    take_screenshot_action,
    vision_critic_node,
)
from src.schemas import CoderArtifact, PagePlan, StructuredPaper
from src.state import CoderState

VOID_HTML_TAGS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}


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


def _safe_slug(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", text.strip())
    slug = slug.strip("-").lower()
    return slug or "section"


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


def _collect_figure_paths(page_plan: PagePlan) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for block in page_plan.blocks:
        for path in block.asset_binding.figure_paths:
            clean = str(path).strip().replace("\\", "/")
            if not clean or clean in seen:
                continue
            seen.add(clean)
            ordered.append(clean)
    return ordered


def _copy_paper_assets(
    project_root: Path,
    paper_folder_name: str,
    site_dir: Path,
    entry_html_path: Path,
    figure_paths: list[str],
) -> tuple[dict[str, str], list[str]]:
    html_src_map: dict[str, str] = {}
    copied_assets: list[str] = []
    if not figure_paths:
        return html_src_map, copied_assets

    source_root = project_root / "data" / "output" / paper_folder_name
    target_dir = site_dir / "assets" / "paper"
    target_dir.mkdir(parents=True, exist_ok=True)

    for idx, rel_path in enumerate(figure_paths, start=1):
        source_path = source_root / rel_path
        if not source_path.exists() or not source_path.is_file():
            continue

        stem = _safe_slug(source_path.stem)[:60]
        suffix = source_path.suffix or ".png"
        target_name = f"{idx:02d}-{stem}{suffix}"
        target_path = target_dir / target_name
        shutil.copy2(source_path, target_path)

        site_relative_path = str(target_path.relative_to(site_dir)).replace("\\", "/")
        html_relative_path = _to_html_relative_path(target_path, entry_html_path.parent)
        html_src_map[rel_path] = html_relative_path
        copied_assets.append(site_relative_path)

    return html_src_map, copied_assets


def _build_valid_asset_set(structured_paper: StructuredPaper) -> set[str]:
    asset_paths: set[str] = set()
    for section in structured_paper.sections:
        for fig in section.related_figures:
            key = str(fig.image_path or "").strip().replace("\\", "/")
            if key:
                asset_paths.add(key)
    return asset_paths


def _candidate_asset_keys(value: str) -> list[str]:
    clean = str(value or "").strip().replace("\\", "/")
    if not clean:
        return []

    keys = [clean]
    if clean.startswith("./"):
        keys.append(clean[2:])
    elif not clean.startswith("/"):
        keys.append(f"./{clean}")
    if clean.startswith("/"):
        keys.append(clean.lstrip("/"))

    ordered: list[str] = []
    seen: set[str] = set()
    for item in keys:
        if item and item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def _resolve_asset_path(value: str, path_map: dict[str, str]) -> str | None:
    for key in _candidate_asset_keys(value):
        mapped = path_map.get(key)
        if mapped:
            return mapped
    return None


def _collect_dom_mapping_figure_paths(
    dom_mapping: dict[str, str],
    valid_asset_paths: set[str],
) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()

    for content in dom_mapping.values():
        content_text = str(content or "").strip().replace("\\", "/")
        for key in _candidate_asset_keys(content_text):
            if key in valid_asset_paths and key not in seen:
                seen.add(key)
                ordered.append(key)

        fragment = BeautifulSoup(str(content or ""), "html.parser")
        for element in fragment.find_all(True):
            for attr_name in ("src", "href", "data-src", "poster"):
                attr_value = element.get(attr_name)
                if not isinstance(attr_value, str):
                    continue
                for key in _candidate_asset_keys(attr_value):
                    if key in valid_asset_paths and key not in seen:
                        seen.add(key)
                        ordered.append(key)
                        break

    return ordered


def _rewrite_fragment_asset_paths(fragment: BeautifulSoup, copied_asset_map: dict[str, str]) -> None:
    if not copied_asset_map:
        return

    for element in fragment.find_all(True):
        for attr_name in ("src", "href", "data-src", "poster"):
            attr_value = element.get(attr_name)
            if not isinstance(attr_value, str):
                continue
            mapped = _resolve_asset_path(attr_value, copied_asset_map)
            if mapped:
                element[attr_name] = mapped


def _fragment_nodes(fragment: BeautifulSoup) -> list[Any]:
    if fragment.body is not None:
        return list(fragment.body.contents)
    return list(fragment.contents)


def _inject_content_into_target(target: Tag, content: str, copied_asset_map: dict[str, str]) -> None:
    fragment = BeautifulSoup(str(content or ""), "html.parser")
    _rewrite_fragment_asset_paths(fragment, copied_asset_map)

    if target.name in VOID_HTML_TAGS:
        replacement_tag = next(
            (node for node in fragment.find_all(True) if node.name == target.name),
            None,
        )
        if replacement_tag is None and target.name == "img":
            replacement_tag = fragment.find("img")

        if replacement_tag is not None:
            target.attrs.clear()
            for attr_name, attr_value in replacement_tag.attrs.items():
                target[attr_name] = attr_value
            return

        text_value = fragment.get_text(" ", strip=True)
        mapped_text = _resolve_asset_path(text_value, copied_asset_map) if text_value else None
        if text_value and target.name in {"img", "source"}:
            target["src"] = mapped_text or text_value
        return

    target.clear()
    for node in _fragment_nodes(fragment):
        target.append(node)


def _tag_contains(container: Tag, other: Tag) -> bool:
    current = other
    while isinstance(current, Tag):
        if current is container:
            return True
        parent = current.parent
        current = parent if isinstance(parent, Tag) else None
    return False


def _overlaps_injected_content(target: Tag, injected_targets: list[Tag]) -> bool:
    for injected_target in injected_targets:
        if target is injected_target:
            return True
        if _tag_contains(target, injected_target):
            return True
        if _tag_contains(injected_target, target):
            return True
    return False


def _extract_json_object(text: str) -> dict[str, Any] | None:
    raw_text = str(text or "").strip()
    if not raw_text:
        return None

    try:
        payload = json.loads(raw_text)
        return payload if isinstance(payload, dict) else None
    except Exception:
        pass

    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        payload = json.loads(raw_text[start : end + 1])
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


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


def _collect_visual_feedback_actions(visual_feedback: Any) -> tuple[list[str], list[str]]:
    selectors_to_remove: list[str] = []
    css_rules_to_inject: list[str] = []
    seen_selectors: set[str] = set()
    seen_css_rules: set[str] = set()

    if not isinstance(visual_feedback, list):
        return selectors_to_remove, css_rules_to_inject

    for feedback_entry in visual_feedback:
        payload = _extract_json_object(str(feedback_entry or ""))
        if not payload:
            continue

        for selector in _normalize_string_list(payload.get("selectors_to_remove")):
            if selector not in seen_selectors:
                seen_selectors.add(selector)
                selectors_to_remove.append(selector)

        css_values = payload.get("css_rules_to_inject") or payload.get("css_rules")
        for css_rule in _normalize_string_list(css_values):
            if css_rule not in seen_css_rules:
                seen_css_rules.add(css_rule)
                css_rules_to_inject.append(css_rule)

    return selectors_to_remove, css_rules_to_inject


def _ensure_head_tag(soup: BeautifulSoup) -> Tag:
    if soup.head is not None:
        return soup.head

    html_tag = soup.html
    if html_tag is None:
        html_tag = soup.new_tag("html")
        for child in list(soup.contents):
            html_tag.append(child.extract())
        soup.append(html_tag)

    head_tag = soup.new_tag("head")
    html_tag.insert(0, head_tag)
    return head_tag


def _apply_visual_feedback(
    soup: BeautifulSoup,
    visual_feedback: Any,
    injected_targets: list[Tag],
) -> None:
    selectors_to_remove, css_rules_to_inject = _collect_visual_feedback_actions(visual_feedback)

    for css_selector in selectors_to_remove:
        selector_text = str(css_selector or "").strip()
        if not selector_text:
            continue

        try:
            targets = soup.select(selector_text)
        except Exception as exc:
            print(f"[PaperAlchemy-Coder] warning: invalid visual removal selector '{selector_text}': {exc}")
            continue

        if not targets:
            print(f"[PaperAlchemy-Coder] warning: visual removal selector '{selector_text}' matched no elements.")
            continue

        for index, target in enumerate(targets, start=1):
            if _overlaps_injected_content(target, injected_targets):
                print(
                    "[PaperAlchemy-Coder] warning: skipped visual removal selector "
                    f"'{selector_text}' target #{index} because it overlaps injected content."
                )
                continue
            try:
                target.decompose()
            except Exception as exc:
                print(
                    "[PaperAlchemy-Coder] warning: failed visual removal "
                    f"selector '{selector_text}' target #{index}: {exc}"
                )

    if css_rules_to_inject:
        head_tag = _ensure_head_tag(soup)
        existing_style = soup.find("style", attrs={"id": "paperalchemy-visual-qa-overrides"})
        if existing_style is not None:
            existing_style.decompose()

        style_tag = soup.new_tag("style", id="paperalchemy-visual-qa-overrides")
        style_tag.string = "\n\n".join(css_rules_to_inject)
        head_tag.append(style_tag)


def coder_node(state: CoderState) -> dict[str, Any]:
    print(
        f"[PaperAlchemy-Coder] building site "
        f"(attempt {state.get('coder_retry_count', 0) + 1})..."
    )

    page_plan = _normalize_page_plan(state.get("page_plan"))
    structured_paper = _normalize_structured_paper(state.get("structured_paper"))
    paper_folder_name = str(state.get("paper_folder_name") or "").strip()
    if not page_plan or not structured_paper or not paper_folder_name:
        print("[PaperAlchemy-Coder] missing page_plan/structured_paper/paper_folder_name.")
        return {}

    project_root = Path(__file__).resolve().parent.parent
    template_root_rel = page_plan.template_selection.selected_root_dir
    template_root = project_root / template_root_rel
    template_entry_rel = page_plan.template_selection.selected_entry_html

    output_dir = project_root / "data" / "output" / paper_folder_name
    site_dir = output_dir / "site"

    if site_dir.exists():
        shutil.rmtree(site_dir)

    if not template_root.exists():
        print(f"[PaperAlchemy-Coder] template root not found: {template_root}")
        return {}

    shutil.copytree(template_root, site_dir)

    entry_html_path = site_dir / template_entry_rel
    if not entry_html_path.exists():
        print(f"[PaperAlchemy-Coder] template entry html not found: {entry_html_path}")
        return {}

    try:
        template_html = _read_text_with_fallback(entry_html_path)
    except Exception as exc:
        print(f"[PaperAlchemy-Coder] failed reading template entry html: {exc}")
        return {}

    soup = BeautifulSoup(template_html, "html.parser")
    valid_asset_paths = _build_valid_asset_set(structured_paper)
    dom_mapping = page_plan.dom_mapping or {}
    figure_paths = _collect_dom_mapping_figure_paths(dom_mapping, valid_asset_paths)
    if not figure_paths:
        figure_paths = _collect_figure_paths(page_plan)

    copied_asset_map, copied_assets = _copy_paper_assets(
        project_root=project_root,
        paper_folder_name=paper_folder_name,
        site_dir=site_dir,
        entry_html_path=entry_html_path,
        figure_paths=figure_paths,
    )
    visual_feedback = state.get("visual_feedback") or []
    if visual_feedback:
        print(
            "[PaperAlchemy-Coder] applying visual feedback iteration "
            f"{int(state.get('visual_iterations', 0)) + 1} with {len(visual_feedback)} feedback item(s)."
        )

    if not dom_mapping:
        print("[PaperAlchemy-Coder] warning: page_plan.dom_mapping is empty; template HTML will be preserved as-is.")

    injected_targets: list[Tag] = []
    for css_selector, content in dom_mapping.items():
        selector_text = str(css_selector or "").strip()
        if not selector_text:
            continue

        try:
            targets = soup.select(selector_text)
        except Exception as exc:
            print(f"[PaperAlchemy-Coder] warning: invalid selector '{selector_text}': {exc}")
            continue

        if not targets:
            print(f"[PaperAlchemy-Coder] warning: selector '{selector_text}' matched no elements.")
            continue

        for index, target in enumerate(targets, start=1):
            try:
                _inject_content_into_target(target, str(content or ""), copied_asset_map)
                injected_targets.append(target)
            except Exception as exc:
                print(
                    "[PaperAlchemy-Coder] warning: failed to inject "
                    f"selector '{selector_text}' target #{index}: {exc}"
                )

    selectors_to_remove = page_plan.selectors_to_remove or []
    for css_selector in selectors_to_remove:
        selector_text = str(css_selector or "").strip()
        if not selector_text:
            continue

        try:
            targets = soup.select(selector_text)
        except Exception as exc:
            print(f"[PaperAlchemy-Coder] warning: invalid removal selector '{selector_text}': {exc}")
            continue

        if not targets:
            print(f"[PaperAlchemy-Coder] warning: removal selector '{selector_text}' matched no elements.")
            continue

        for index, target in enumerate(targets, start=1):
            if _overlaps_injected_content(target, injected_targets):
                print(
                    "[PaperAlchemy-Coder] warning: skipped removal selector "
                    f"'{selector_text}' target #{index} because it overlaps injected content."
                )
                continue
            try:
                target.decompose()
            except Exception as exc:
                print(
                    "[PaperAlchemy-Coder] warning: failed to remove "
                    f"selector '{selector_text}' target #{index}: {exc}"
                )

    if visual_feedback:
        _apply_visual_feedback(
            soup=soup,
            visual_feedback=visual_feedback,
            injected_targets=injected_targets,
        )

    entry_html_path.write_text(str(soup), encoding="utf-8")

    artifact = CoderArtifact(
        site_dir=str(site_dir),
        entry_html=str(entry_html_path),
        selected_template_id=page_plan.template_selection.selected_template_id,
        copied_assets=copied_assets,
        edited_files=[str(entry_html_path.relative_to(site_dir)).replace("\\", "/")],
        notes=(
            "v2-dom-injection-wash: preserved the original template DOM, injected page_plan.dom_mapping "
            "content with BeautifulSoup, decompose()-removed selectors_to_remove, and applied visual QA fixes."
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
    max_retry: int = 1,
) -> CoderArtifact | None:
    app = build_coder_graph(max_retry=max_retry)
    thread = {"configurable": {"thread_id": f"coder_{paper_folder_name}"}}

    initial_state: CoderState = {
        "paper_folder_name": paper_folder_name,
        "structured_paper": structured_data,
        "page_plan": page_plan,
        "coder_feedback_history": [],
        "visual_feedback": [],
        "visual_screenshot_path": "",
        "visual_iterations": 0,
        "is_visually_approved": False,
        "coder_artifact": None,
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
