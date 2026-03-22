import json
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup, Tag
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from src.agent_planner_critic import build_planner_critic_router, planner_critic_node
from src.deterministic_template_selector import score_and_select_template
from src.human_feedback import extract_human_feedback_text, normalize_human_feedback
from src.json_utils import to_pretty_json
from src.llm import get_llm
from src.planner_template_catalog import build_template_catalog, load_module_index, load_template_link_map
from src.prompts import (
    SEMANTIC_PLANNER_SYSTEM_PROMPT,
    SEMANTIC_PLANNER_USER_PROMPT_TEMPLATE,
    TEMPLATE_BINDER_SYSTEM_PROMPT,
    TEMPLATE_BINDER_USER_PROMPT_TEMPLATE,
)
from src.schemas import PagePlan, SemanticPlan, StructuredPaper, TemplateCandidate
from src.state import PlannerState
from src.template_resources import ensure_autopage_template_assets


def _normalize_structured_paper(paper: Any) -> StructuredPaper | None:
    if isinstance(paper, StructuredPaper):
        return paper
    if paper is None:
        return None
    try:
        return StructuredPaper.model_validate(paper)
    except Exception:
        return None


def _normalize_semantic_plan(plan: Any) -> SemanticPlan | None:
    if isinstance(plan, SemanticPlan):
        return plan
    if plan is None:
        return None


def _normalize_page_plan(plan: Any) -> PagePlan | None:
    if isinstance(plan, PagePlan):
        return plan
    if plan is None:
        return None
    try:
        return PagePlan.model_validate(plan)
    except Exception:
        return None
    try:
        return SemanticPlan.model_validate(plan)
    except Exception:
        return None


def _normalize_template_candidate(candidate: Any) -> TemplateCandidate | None:
    if isinstance(candidate, TemplateCandidate):
        return candidate
    if candidate is None:
        return None
    try:
        return TemplateCandidate.model_validate(candidate)
    except Exception:
        return None


def _to_project_relative_path(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def _read_text_with_fallback(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


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


def _build_dom_outline(template_html: str, max_lines: int = 120) -> str:
    soup = BeautifulSoup(template_html, "html.parser")
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
        for attr_name in ("id", "class", "src", "href", "alt", "aria-label"):
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

    return "\n".join(lines) if lines else "- <empty template body>"


def _candidate_from_ranked_match(
    ranked_match: dict[str, Any],
    project_root: Path,
) -> TemplateCandidate | None:
    template_id = str(ranked_match.get("template_id") or "").strip()
    template_path = str(ranked_match.get("template_path") or "").strip()
    entry_html = str(ranked_match.get("entry_html") or "").strip()
    if not template_id or not template_path or not entry_html:
        return None

    reasons = [
        str(item).strip()
        for item in (ranked_match.get("reasons") or [])
        if str(item).strip()
    ]

    return TemplateCandidate(
        template_id=template_id,
        root_dir=_to_project_relative_path(Path(template_path), project_root),
        chosen_entry_html=entry_html,
        score=float(ranked_match.get("score") or 0.0),
        reasons=reasons[:8],
    )


def _select_designated_template(
    constraints: dict[str, Any],
    project_root: Path,
) -> tuple[list[TemplateCandidate], TemplateCandidate | None, str | None]:
    raw_candidates = constraints.get("ui_ranked_candidates") or []
    if not isinstance(raw_candidates, list):
        raw_candidates = []

    top_k = max(1, int(constraints.get("template_candidate_top_k", 3)))
    candidates = [
        candidate
        for candidate in (
            _candidate_from_ranked_match(item, project_root)
            for item in raw_candidates[:top_k]
            if isinstance(item, dict)
        )
        if candidate is not None
    ]

    designated_template_id = str(constraints.get("designated_template_id") or "").strip()
    designated_template_path = str(constraints.get("designated_template_path") or "").strip()
    designated_entry_html = str(constraints.get("designated_entry_html") or "").strip()

    if not (designated_template_id and designated_template_path and designated_entry_html):
        return candidates, None, None

    selected_match = next(
        (
            item
            for item in raw_candidates
            if isinstance(item, dict)
            and str(item.get("template_id") or "").strip() == designated_template_id
            and str(item.get("entry_html") or "").strip() == designated_entry_html
        ),
        None,
    )

    if selected_match is None:
        selected_match = {
            "template_id": designated_template_id,
            "template_name": designated_template_id,
            "template_path": designated_template_path,
            "entry_html": designated_entry_html,
            "entry_html_path": str((Path(designated_template_path) / designated_entry_html).resolve()),
            "score": float(constraints.get("designated_template_score") or 0.0),
            "reasons": ["manual_ui_selection"],
        }

    selected = _candidate_from_ranked_match(selected_match, project_root)
    if selected is None:
        return candidates, None, None

    existing_ids = {candidate.template_id for candidate in candidates}
    if selected.template_id not in existing_ids:
        candidates = [selected, *candidates]
    else:
        candidates = [
            selected,
            *[candidate for candidate in candidates if candidate.template_id != selected.template_id],
        ]

    return candidates, selected, designated_template_path


def semantic_planner_node(state: PlannerState) -> dict[str, Any]:
    print(
        f"[PaperAlchemy-SemanticPlanner] generating semantic plan "
        f"(attempt {state.get('planner_retry_count', 0) + 1})..."
    )

    constraints = state.get("generation_constraints") or {}
    llm = get_llm(temperature=0.3, use_smart_model=True)
    structured_llm = llm.with_structured_output(SemanticPlan)

    structured_paper = _normalize_structured_paper(state.get("structured_paper"))
    if not structured_paper:
        print("[PaperAlchemy-SemanticPlanner] missing structured_paper, cannot proceed.")
        return {"semantic_plan": None, "page_plan": None}

    previous_page_plan = _normalize_page_plan(state.get("previous_page_plan"))
    feedback_history = state.get("planner_feedback_history") or []
    human_directives = extract_human_feedback_text(state.get("human_directives"))

    user_msg = SEMANTIC_PLANNER_USER_PROMPT_TEMPLATE.format(
        structured_paper_json=to_pretty_json(structured_paper),
        previous_page_plan_json=to_pretty_json(previous_page_plan) if previous_page_plan else "null",
        generation_constraints_json=json.dumps(constraints, indent=2, ensure_ascii=False),
        human_directives=human_directives,
        prior_feedback=json.dumps(feedback_history, indent=2, ensure_ascii=False),
    )

    try:
        result = structured_llm.invoke(
            [
                SystemMessage(content=SEMANTIC_PLANNER_SYSTEM_PROMPT),
                HumanMessage(content=user_msg),
            ]
        )
        if not result:
            raise ValueError("Semantic planner returned empty result")

        return {
            "semantic_plan": result,
            "page_plan": None,
        }
    except Exception as exc:
        print(f"[PaperAlchemy-SemanticPlanner] generation error: {exc}")
        return {"semantic_plan": None, "page_plan": None}


def template_selector_node(state: PlannerState) -> dict[str, Any]:
    print("[PaperAlchemy-TemplateSelector] selecting template from user constraints...")

    project_root = Path(__file__).resolve().parent.parent
    constraints = state.get("generation_constraints") or {}
    user_constraints = state.get("user_constraints") or {}
    tags_json_path = str(constraints.get("template_tags_json_path") or "").strip()

    designated_candidates, designated_selected, designated_path = _select_designated_template(
        constraints=constraints,
        project_root=project_root,
    )
    if designated_selected is not None:
        print(
            "[PaperAlchemy-TemplateSelector] using designated template from UI selection: "
            f"{designated_selected.template_id} entry={designated_selected.chosen_entry_html}"
        )
        return {
            "template_candidates": designated_candidates or [designated_selected],
            "selected_template": designated_selected,
            "selected_template_path": designated_path,
            "generation_constraints": {
                **constraints,
                "selected_template_id": designated_selected.template_id,
            },
        }

    if not tags_json_path:
        print("[PaperAlchemy-TemplateSelector] template_tags_json_path is missing.")
        return {
            "template_candidates": [],
            "selected_template": None,
            "selected_template_path": None,
        }

    try:
        selection_result = score_and_select_template(
            user_constraints=user_constraints,
            tags_json_path=tags_json_path,
        )
    except Exception as exc:
        print(f"[PaperAlchemy-TemplateSelector] deterministic selection failed: {exc}")
        return {
            "template_candidates": [],
            "selected_template": None,
            "selected_template_path": None,
        }

    top_k = max(1, int(constraints.get("template_candidate_top_k", 3)))
    ranked_matches = list(selection_result.get("ranking") or [])
    candidates = [
        candidate
        for candidate in (
            _candidate_from_ranked_match(item, project_root)
            for item in ranked_matches[:top_k]
        )
        if candidate is not None
    ]

    selected = _candidate_from_ranked_match(selection_result, project_root)
    if selected is None:
        print("[PaperAlchemy-TemplateSelector] selected template candidate is invalid.")
        return {
            "template_candidates": [],
            "selected_template": None,
            "selected_template_path": None,
        }

    if not candidates:
        candidates = [selected]

    print(
        "[PaperAlchemy-TemplateSelector] selected="
        f"{selected.template_id} entry={selected.chosen_entry_html} score={selected.score}"
    )

    return {
        "template_candidates": candidates,
        "selected_template": selected,
        "selected_template_path": str(selection_result.get("selected_template_path") or ""),
        "generation_constraints": {
            **constraints,
            "selected_template_id": selected.template_id,
        },
    }


def template_binder_node(state: PlannerState) -> dict[str, Any]:
    print("[PaperAlchemy-TemplateBinder] binding semantic plan to selected template...")

    structured_paper = _normalize_structured_paper(state.get("structured_paper"))
    semantic_plan = _normalize_semantic_plan(state.get("semantic_plan"))
    if not structured_paper or not semantic_plan:
        print("[PaperAlchemy-TemplateBinder] missing structured_paper or semantic_plan.")
        return {}

    raw_candidates = state.get("template_candidates") or []
    candidates = [
        candidate
        for candidate in (_normalize_template_candidate(item) for item in raw_candidates)
        if candidate is not None
    ]

    selected = _normalize_template_candidate(state.get("selected_template"))
    if not selected and candidates:
        selected = candidates[0]

    if not selected:
        print("[PaperAlchemy-TemplateBinder] selected template is missing.")
        return {}

    project_root = Path(__file__).resolve().parent.parent
    template_entry_path = project_root / selected.root_dir / selected.chosen_entry_html
    if not template_entry_path.exists():
        print(f"[PaperAlchemy-TemplateBinder] template entry html not found: {template_entry_path}")
        return {}

    try:
        template_html = _read_text_with_fallback(template_entry_path)
    except Exception as exc:
        print(f"[PaperAlchemy-TemplateBinder] failed reading template html: {exc}")
        return {}

    template_dom_outline = _build_dom_outline(template_html)
    constraints = state.get("generation_constraints") or {}
    llm = get_llm(temperature=0.2, use_smart_model=True)
    structured_llm = llm.with_structured_output(PagePlan)

    template_link_map = state.get("template_link_map") or {}
    module_index = state.get("module_index") or {}
    planner_feedback_history = state.get("planner_feedback_history") or []
    human_directives = extract_human_feedback_text(state.get("human_directives"))
    previous_page_plan = _normalize_page_plan(state.get("previous_page_plan"))

    user_msg = TEMPLATE_BINDER_USER_PROMPT_TEMPLATE.format(
        structured_paper_json=to_pretty_json(structured_paper),
        previous_page_plan_json=to_pretty_json(previous_page_plan) if previous_page_plan else "null",
        semantic_plan_json=to_pretty_json(semantic_plan),
        template_candidates_json=json.dumps(
            [candidate.model_dump() for candidate in candidates],
            indent=2,
            ensure_ascii=False,
        ),
        selected_template_json=json.dumps(selected.model_dump(), indent=2, ensure_ascii=False),
        template_entry_html_path=_to_project_relative_path(template_entry_path, project_root),
        template_dom_outline=template_dom_outline,
        template_link_map_json=json.dumps(template_link_map, indent=2, ensure_ascii=False),
        module_index_json=json.dumps(module_index, indent=2, ensure_ascii=False),
        generation_constraints_json=json.dumps(constraints, indent=2, ensure_ascii=False),
        human_directives=human_directives,
        prior_feedback=json.dumps(planner_feedback_history, indent=2, ensure_ascii=False),
    )

    try:
        result = structured_llm.invoke(
            [
                SystemMessage(content=TEMPLATE_BINDER_SYSTEM_PROMPT),
                HumanMessage(content=user_msg),
            ]
        )
        if not result:
            raise ValueError("Template binder returned empty result")

        result.plan_meta.planning_mode = "hybrid_template_bind"
        result.template_selection.selected_template_id = selected.template_id
        result.template_selection.selected_root_dir = selected.root_dir
        result.template_selection.selected_entry_html = selected.chosen_entry_html

        if not result.template_selection.fallback_template_id and len(candidates) > 1:
            result.template_selection.fallback_template_id = candidates[1].template_id

        normalized_dom_mapping: dict[str, str] = {}
        for selector, content in (result.dom_mapping or {}).items():
            selector_text = str(selector or "").strip()
            if not selector_text:
                continue
            normalized_dom_mapping[selector_text] = str(content or "")

        if not normalized_dom_mapping:
            raise ValueError("Template binder returned empty dom_mapping")

        result.dom_mapping = normalized_dom_mapping
        normalized_selectors_to_remove: list[str] = []
        seen_remove_selectors: set[str] = set()
        for selector in (result.selectors_to_remove or []):
            selector_text = str(selector or "").strip()
            if not selector_text or selector_text in seen_remove_selectors:
                continue
            seen_remove_selectors.add(selector_text)
            normalized_selectors_to_remove.append(selector_text)

        result.selectors_to_remove = normalized_selectors_to_remove
        return {"page_plan": result}
    except Exception as exc:
        print(f"[PaperAlchemy-TemplateBinder] generation error: {exc}")
        return {}


def build_planner_graph(max_retry: int = 2):
    workflow = StateGraph(PlannerState)
    workflow.add_node("template_selector", template_selector_node)
    workflow.add_node("semantic_planner", semantic_planner_node)
    workflow.add_node("template_binder", template_binder_node)
    workflow.add_node("planner_critic", planner_critic_node)

    workflow.set_entry_point("template_selector")
    workflow.add_edge("template_selector", "semantic_planner")
    workflow.add_edge("semantic_planner", "template_binder")
    workflow.add_edge("template_binder", "planner_critic")

    workflow.add_conditional_edges(
        "planner_critic",
        build_planner_critic_router(max_retry=max_retry),
        {"retry": "semantic_planner", "end": END},
    )

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


def run_planner_agent(
    paper_folder_name: str,
    structured_data: StructuredPaper,
    generation_constraints: dict[str, Any] | None = None,
    user_constraints: dict[str, Any] | None = None,
    human_directives: str | dict = "",
    previous_page_plan: PagePlan | None = None,
    max_retry: int = 2,
):
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent

    constraints = generation_constraints or {}
    synced_assets = ensure_autopage_template_assets(
        project_root=project_root,
        force=bool(constraints.get("force_template_sync")),
    )

    templates_dir = synced_assets.templates_dir
    template_links_path = synced_assets.template_link_json_path
    module_index_path = project_root / "data" / "collectors" / "modules" / "module_index.json"

    if synced_assets.missing_template_ids:
        print(
            "[PaperAlchemy-Planner] warning: missing AutoPage templates for tagged ids: "
            f"{synced_assets.missing_template_ids[:8]}"
        )

    constraints = {
        **constraints,
        "template_tags_json_path": str(synced_assets.tags_json_path),
    }
    max_templates = int(constraints.get("max_templates_for_planner", 120))
    max_entry_candidates = int(constraints.get("max_entry_candidates", 3))

    print(
        "[PaperAlchemy-Planner] loading template inventory "
        f"(max_templates={max_templates}, max_entry_candidates={max_entry_candidates})..."
    )
    template_catalog = build_template_catalog(
        templates_dir=templates_dir,
        project_root=project_root,
        max_templates=max_templates,
        max_entry_candidates=max_entry_candidates,
    )
    template_link_map = load_template_link_map(template_links_path)
    module_index = load_module_index(module_index_path)

    if not template_catalog:
        print("[PaperAlchemy-Planner] no valid templates discovered, planner cannot proceed.")
        return None

    app = build_planner_graph(max_retry=max_retry)
    thread = {"configurable": {"thread_id": f"planner_{paper_folder_name}"}}

    initial_state: PlannerState = {
        "structured_paper": structured_data,
        "previous_page_plan": previous_page_plan,
        "template_catalog": template_catalog,
        "template_link_map": template_link_map,
        "module_index": module_index,
        "generation_constraints": constraints,
        "user_constraints": user_constraints or {},
        "human_directives": normalize_human_feedback(human_directives),
        "semantic_plan": None,
        "template_candidates": [],
        "selected_template": None,
        "selected_template_path": None,
        "planner_feedback_history": [],
        "page_plan": None,
        "planner_critic_passed": False,
        "planner_retry_count": 0,
    }

    print("[PaperAlchemy-Planner] running Selector + SemanticPlanner + Binder + Critic graph...")
    for _ in app.stream(initial_state, thread):
        pass

    final_state = app.get_state(thread)
    plan_result = final_state.values.get("page_plan")
    normalized_plan: PagePlan | None = None
    if plan_result is not None:
        try:
            normalized_plan = PagePlan.model_validate(plan_result)
        except Exception:
            normalized_plan = None

    if not normalized_plan or not final_state.values.get("planner_critic_passed"):
        print("[PaperAlchemy-Planner] planner completed but critic did not fully pass.")
    else:
        print("[PaperAlchemy-Planner] planner phase completed successfully.")

    return normalized_plan
