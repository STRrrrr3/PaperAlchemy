import json
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from src.agent_planner_critic import build_planner_critic_router, planner_critic_node
from src.llm import get_llm
from src.prompts import (
    SEMANTIC_PLANNER_SYSTEM_PROMPT,
    SEMANTIC_PLANNER_USER_PROMPT_TEMPLATE,
    TEMPLATE_BINDER_SYSTEM_PROMPT,
    TEMPLATE_BINDER_USER_PROMPT_TEMPLATE,
)
from src.schemas import PagePlan, SemanticPlan, StructuredPaper, TemplateCandidate
from src.state import PlannerState
from src.planner_template_catalog import build_template_catalog, load_module_index, load_template_link_map
from src.planner_template_selector import select_template_candidates


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


def _pick_candidate_by_id(
    candidates: list[TemplateCandidate],
    template_id: str | None,
) -> TemplateCandidate | None:
    if not template_id:
        return None
    target = str(template_id).strip()
    if not target:
        return None
    for candidate in candidates:
        if candidate.template_id == target:
            return candidate
    return None


def _choose_template_with_human(
    candidates: list[TemplateCandidate],
    template_link_map: dict[str, str],
) -> TemplateCandidate:
    print("[PaperAlchemy-TemplateSelector] template recommendations:")
    for idx, candidate in enumerate(candidates, start=1):
        source_url = template_link_map.get(candidate.template_id, "")
        reason_text = ", ".join(candidate.reasons[:3]) if candidate.reasons else "no reason"
        print(
            f"  [{idx}] {candidate.template_id} "
            f"(score={candidate.score}, entry={candidate.chosen_entry_html})"
        )
        if source_url:
            print(f"      source: {source_url}")
        print(f"      reasons: {reason_text}")

    prompt = (
        "Select template by index or template_id "
        "(press Enter to use #1): "
    )
    try:
        raw = input(prompt).strip()
    except EOFError:
        print("[PaperAlchemy-TemplateSelector] non-interactive shell detected, fallback to #1.")
        return candidates[0]

    if not raw:
        return candidates[0]

    if raw.isdigit():
        index = int(raw)
        if 1 <= index <= len(candidates):
            return candidates[index - 1]
        print("[PaperAlchemy-TemplateSelector] invalid index, fallback to #1.")
        return candidates[0]

    by_id = _pick_candidate_by_id(candidates, raw)
    if by_id:
        return by_id

    print("[PaperAlchemy-TemplateSelector] unknown template_id, fallback to #1.")
    return candidates[0]


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
        return {}

    feedback_history = state.get("planner_feedback_history") or []

    user_msg = SEMANTIC_PLANNER_USER_PROMPT_TEMPLATE.format(
        structured_paper_json=structured_paper.model_dump_json(indent=2, ensure_ascii=False),
        generation_constraints_json=json.dumps(constraints, indent=2, ensure_ascii=False),
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
            "template_candidates": [],
            "selected_template": None,
            "page_plan": None,
        }
    except Exception as exc:
        print(f"[PaperAlchemy-SemanticPlanner] generation error: {exc}")
        return {}


def template_selector_node(state: PlannerState) -> dict[str, Any]:
    print("[PaperAlchemy-TemplateSelector] ranking templates against semantic plan...")

    semantic_plan = _normalize_semantic_plan(state.get("semantic_plan"))
    template_catalog = state.get("template_catalog") or []
    constraints = state.get("generation_constraints") or {}

    if not semantic_plan:
        print("[PaperAlchemy-TemplateSelector] semantic_plan is missing.")
        return {}

    top_k = int(constraints.get("template_candidate_top_k", 3))
    candidates = select_template_candidates(
        template_catalog=template_catalog,
        semantic_plan=semantic_plan,
        top_k=top_k,
    )

    if not candidates:
        print("[PaperAlchemy-TemplateSelector] no template candidates were selected.")
        return {}

    template_link_map = state.get("template_link_map") or {}
    selection_mode = str(constraints.get("template_selection_mode", "human")).strip().lower()
    configured_id = str(
        constraints.get("selected_template_id")
        or constraints.get("template_selection_id")
        or ""
    ).strip()

    selected = _pick_candidate_by_id(candidates, configured_id)
    if selected is None:
        if selection_mode in {"human", "human_in_the_loop", "manual"}:
            selected = _choose_template_with_human(candidates, template_link_map)
        else:
            selected = candidates[0]

    print(
        "[PaperAlchemy-TemplateSelector] selected="
        f"{selected.template_id} entry={selected.chosen_entry_html} score={selected.score}"
    )

    return {
        "template_candidates": candidates,
        "selected_template": selected,
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

    constraints = state.get("generation_constraints") or {}
    llm = get_llm(temperature=0.2, use_smart_model=True)
    structured_llm = llm.with_structured_output(PagePlan)

    template_link_map = state.get("template_link_map") or {}
    module_index = state.get("module_index") or {}
    planner_feedback_history = state.get("planner_feedback_history") or []

    user_msg = TEMPLATE_BINDER_USER_PROMPT_TEMPLATE.format(
        structured_paper_json=structured_paper.model_dump_json(indent=2, ensure_ascii=False),
        semantic_plan_json=semantic_plan.model_dump_json(indent=2, ensure_ascii=False),
        template_candidates_json=json.dumps(
            [candidate.model_dump() for candidate in candidates],
            indent=2,
            ensure_ascii=False,
        ),
        selected_template_json=json.dumps(selected.model_dump(), indent=2, ensure_ascii=False),
        template_link_map_json=json.dumps(template_link_map, indent=2, ensure_ascii=False),
        module_index_json=json.dumps(module_index, indent=2, ensure_ascii=False),
        generation_constraints_json=json.dumps(constraints, indent=2, ensure_ascii=False),
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

        # Enforce consistency with selected candidate to reduce downstream ambiguity.
        result.plan_meta.planning_mode = "hybrid_template_bind"
        result.template_selection.selected_template_id = selected.template_id
        result.template_selection.selected_root_dir = selected.root_dir
        result.template_selection.selected_entry_html = selected.chosen_entry_html

        if not result.template_selection.fallback_template_id and len(candidates) > 1:
            result.template_selection.fallback_template_id = candidates[1].template_id

        return {"page_plan": result}
    except Exception as exc:
        print(f"[PaperAlchemy-TemplateBinder] generation error: {exc}")
        return {}


def build_planner_graph(max_retry: int = 2):
    workflow = StateGraph(PlannerState)
    workflow.add_node("semantic_planner", semantic_planner_node)
    workflow.add_node("template_selector", template_selector_node)
    workflow.add_node("template_binder", template_binder_node)
    workflow.add_node("planner_critic", planner_critic_node)

    workflow.set_entry_point("semantic_planner")
    workflow.add_edge("semantic_planner", "template_selector")
    workflow.add_edge("template_selector", "template_binder")
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
    max_retry: int = 2,
):
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent

    templates_dir = project_root / "templates"
    template_links_path = templates_dir / "template_link.json"
    module_index_path = project_root / "data" / "collectors" / "modules" / "module_index.json"

    constraints = generation_constraints or {}
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
        "template_catalog": template_catalog,
        "template_link_map": template_link_map,
        "module_index": module_index,
        "generation_constraints": constraints,
        "semantic_plan": None,
        "template_candidates": [],
        "selected_template": None,
        "planner_feedback_history": [],
        "page_plan": None,
        "planner_critic_passed": False,
        "planner_retry_count": 0,
    }

    print("[PaperAlchemy-Planner] running SemanticPlanner + Selector + Binder + Critic graph...")
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
