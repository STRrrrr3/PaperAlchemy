import json
import re
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from src.agents.planner_critic import build_planner_critic_router, planner_critic_node
from src.services.human_feedback import extract_human_feedback_text, normalize_human_feedback
from src.utils.json_utils import to_pretty_json
from src.services.llm import get_llm
from src.template.catalog import build_template_catalog, load_module_index, load_template_link_map
from src.prompts import PLANNER_SYSTEM_PROMPT, PLANNER_USER_PROMPT_TEMPLATE
from src.contracts.schemas import BlockShellContract, PagePlan, StructuredPaper, TemplateCandidate, TemplateProfile
from src.contracts.state import PlannerState
from src.template.compile import prepare_template_compile_bundle
from src.template.resources import ensure_autopage_template_assets


def _normalize_structured_paper(paper: Any) -> StructuredPaper | None:
    if isinstance(paper, StructuredPaper):
        return paper
    if paper is None:
        return None
    try:
        return StructuredPaper.model_validate(paper)
    except Exception:
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


def _normalize_template_candidate(candidate: Any) -> TemplateCandidate | None:
    if isinstance(candidate, TemplateCandidate):
        return candidate
    if candidate is None:
        return None
    try:
        return TemplateCandidate.model_validate(candidate)
    except Exception:
        return None


def _normalize_template_profile(profile: Any) -> TemplateProfile | None:
    if isinstance(profile, TemplateProfile):
        return profile
    if profile is None:
        return None
    try:
        return TemplateProfile.model_validate(profile)
    except Exception:
        return None


def _dedupe_strings(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        clean = str(value or "").strip()
        if clean and clean not in seen:
            seen.add(clean)
            deduped.append(clean)
    return deduped


def _selector_tokens(selector: str) -> set[str]:
    return {
        token
        for token in re.split(r"[^a-z0-9]+", str(selector or "").lower())
        if token
    }


def _build_shell_contract_from_candidate(candidate: Any) -> BlockShellContract:
    return BlockShellContract(
        root_tag=str(candidate.root_tag or "div"),
        required_classes=list(candidate.required_classes or []),
        preserve_ids=list(candidate.preserve_ids or []),
        wrapper_chain=list(candidate.wrapper_chain or []),
        actionable_root_selector=str(candidate.selector or "").strip(),
    )


def _ordered_shell_candidates(profile: TemplateProfile) -> list[Any]:
    return sorted(
        profile.shell_candidates,
        key=lambda item: (item.dom_index, -item.confidence, item.selector),
    )


def _choose_shell_candidate(
    profile: TemplateProfile,
    *,
    desired_role: str,
    preferred_selector: str,
    used_selectors: set[str],
) -> Any | None:
    selector_tokens = _selector_tokens(preferred_selector)
    best_candidate = None
    best_score = float("-inf")

    for candidate in _ordered_shell_candidates(profile):
        if candidate.selector in used_selectors:
            continue
        score = float(candidate.confidence or 0.0) * 10.0
        if str(candidate.role or "") == desired_role:
            score += 5.0
        elif desired_role in {"hero", "section"} and str(candidate.role or "") in {"hero", "section"}:
            score += 2.5
        elif desired_role == "gallery" and str(candidate.role or "") == "section":
            score += 1.5
        elif desired_role == "table" and str(candidate.role or "") == "section":
            score += 1.0
        else:
            score -= 1.5

        if preferred_selector and candidate.selector == preferred_selector:
            score += 6.0
        elif selector_tokens:
            score += min(3.0, float(len(selector_tokens & _selector_tokens(candidate.selector))))

        if candidate.preserve_ids:
            score += 0.6
        if candidate.required_classes:
            score += 0.4
        if score > best_score:
            best_candidate = candidate
            best_score = score

    return best_candidate


def _normalize_blocks_with_template_profile(page_plan: PagePlan, template_profile: TemplateProfile) -> PagePlan:
    outline_lookup = {item.block_id: item for item in page_plan.page_outline}
    updated_blocks = []
    used_selectors: set[str] = set()

    for block in page_plan.blocks:
        preferred_selector = str(block.target_template_region.selector_hint or "").strip()
        desired_role = str(block.target_template_region.region_role or "section").strip() or "section"
        chosen_candidate = _choose_shell_candidate(
            template_profile,
            desired_role=desired_role,
            preferred_selector=preferred_selector,
            used_selectors=used_selectors,
        )
        if chosen_candidate is None:
            updated_blocks.append(block)
            continue

        used_selectors.add(chosen_candidate.selector)
        updated_region = block.target_template_region.model_copy(
            update={
                "selector_hint": chosen_candidate.selector,
                "region_role": str(chosen_candidate.role or desired_role),
            },
            deep=True,
        )
        updated_outline = outline_lookup.get(block.block_id)
        updated_responsive_rules = block.responsive_rules.model_copy(
            update={
                "mobile_order": int(updated_outline.order if updated_outline is not None else block.responsive_rules.mobile_order)
            },
            deep=True,
        )
        updated_blocks.append(
            block.model_copy(
                update={
                    "target_template_region": updated_region,
                    "shell_contract": _build_shell_contract_from_candidate(chosen_candidate),
                    "responsive_rules": updated_responsive_rules,
                },
                deep=True,
            )
        )

    return page_plan.model_copy(update={"blocks": updated_blocks}, deep=True)


def _compat_dom_mapping(template_profile: TemplateProfile) -> dict[str, str]:
    return {
        selector: "preserve_global_anchor"
        for selector in template_profile.global_preserve_selectors
        if str(selector or "").strip()
    }


def _normalize_selectors_to_remove(page_plan: PagePlan, template_profile: TemplateProfile) -> list[str]:
    protected_selectors = {
        selector
        for selector in template_profile.global_preserve_selectors
        if str(selector or "").strip()
    }
    protected_selectors.update(
        str(candidate.selector or "").strip()
        for candidate in template_profile.shell_candidates
        if str(candidate.selector or "").strip()
    )
    merged = list(page_plan.selectors_to_remove or []) + list(template_profile.removable_demo_selectors or [])
    normalized = []
    for selector in _dedupe_strings(merged):
        if selector in protected_selectors:
            continue
        normalized.append(selector)
    return normalized


def _render_strategy_risks(template_profile: TemplateProfile) -> list[str]:
    risks: list[str] = []
    if float(template_profile.compile_confidence or 0.0) < 0.7:
        risks.append(
            f"template_compile_confidence={template_profile.compile_confidence:.2f} is below compiled block threshold 0.70"
        )
    high_risk_widget_types = {
        widget.widget_type
        for widget in template_profile.optional_widgets
        if {"fetch_runtime_dependency", "chart_runtime_dependency", "math_runtime_dependency"} & set(widget.risk_flags)
    }
    if high_risk_widget_types:
        risks.append(
            "template widgets require risky runtime handling: " + ", ".join(sorted(high_risk_widget_types))
        )
    for risk_flag in template_profile.risk_flags:
        if risk_flag not in risks:
            if risk_flag in {"fetch_runtime_dependency", "chart_runtime_dependency", "math_runtime_dependency"}:
                risks.append(risk_flag)
    return _dedupe_strings(risks)


def unified_planner_node(state: PlannerState) -> dict[str, Any]:
    print(
        "[PaperAlchemy-UnifiedPlanner] generating PagePlan from compiled template profile "
        f"(attempt {state.get('planner_retry_count', 0) + 1})..."
    )

    structured_paper = _normalize_structured_paper(state.get("structured_paper"))
    if not structured_paper:
        print("[PaperAlchemy-UnifiedPlanner] missing structured_paper, cannot proceed.")
        return {"page_plan": None}

    raw_candidates = state.get("template_candidates") or []
    candidates = [
        candidate
        for candidate in (_normalize_template_candidate(item) for item in raw_candidates)
        if candidate is not None
    ]
    selected = _normalize_template_candidate(state.get("selected_template"))
    template_profile = _normalize_template_profile(state.get("template_profile"))
    if selected is None and candidates:
        selected = candidates[0]
    if selected is None or template_profile is None:
        print("[PaperAlchemy-UnifiedPlanner] selected template or template_profile is missing.")
        return {"page_plan": None}

    constraints = state.get("generation_constraints") or {}
    template_catalog = state.get("template_catalog") or []
    template_link_map = state.get("template_link_map") or {}
    module_index = state.get("module_index") or {}
    planner_feedback_history = state.get("planner_feedback_history") or []
    human_directives = extract_human_feedback_text(state.get("human_directives"))
    previous_page_plan = _normalize_page_plan(state.get("previous_page_plan"))

    llm = get_llm(temperature=0.2, use_smart_model=True)
    structured_llm = llm.with_structured_output(PagePlan)
    user_msg = PLANNER_USER_PROMPT_TEMPLATE.format(
        structured_paper_json=to_pretty_json(structured_paper),
        previous_page_plan_json=to_pretty_json(previous_page_plan) if previous_page_plan else "null",
        template_catalog_json=json.dumps(template_catalog, indent=2, ensure_ascii=False),
        template_candidates_json=json.dumps(
            [candidate.model_dump() for candidate in candidates],
            indent=2,
            ensure_ascii=False,
        ),
        selected_template_json=json.dumps(selected.model_dump(), indent=2, ensure_ascii=False),
        template_entry_html_path=str(selected.chosen_entry_html),
        template_profile_json=json.dumps(template_profile.model_dump(), indent=2, ensure_ascii=False),
        template_link_map_json=json.dumps(template_link_map, indent=2, ensure_ascii=False),
        module_index_json=json.dumps(module_index, indent=2, ensure_ascii=False),
        generation_constraints_json=json.dumps(constraints, indent=2, ensure_ascii=False),
        human_directives=human_directives,
        prior_feedback=json.dumps(planner_feedback_history, indent=2, ensure_ascii=False),
    )

    try:
        result = structured_llm.invoke(
            [
                SystemMessage(content=PLANNER_SYSTEM_PROMPT),
                HumanMessage(content=user_msg),
            ]
        )
        if not result:
            raise ValueError("Unified planner returned empty result")

        result.plan_meta.planning_mode = "hybrid_template_bind"
        result.template_selection.selected_template_id = selected.template_id
        result.template_selection.selected_root_dir = selected.root_dir
        result.template_selection.selected_entry_html = selected.chosen_entry_html
        if not result.template_selection.fallback_template_id and len(candidates) > 1:
            result.template_selection.fallback_template_id = candidates[1].template_id

        result = _normalize_blocks_with_template_profile(result, template_profile)
        result.dom_mapping = _compat_dom_mapping(template_profile)
        result.selectors_to_remove = _normalize_selectors_to_remove(result, template_profile)

        render_strategy_risks = _render_strategy_risks(template_profile)
        result.plan_meta.render_strategy = (
            "legacy_fullpage" if render_strategy_risks else "compiled_block_assembly"
        )
        result.coder_handoff = result.coder_handoff.model_copy(
            update={
                "known_risks": _dedupe_strings(
                    list(result.coder_handoff.known_risks or []) + render_strategy_risks
                )
            },
            deep=True,
        )
        return {"page_plan": result}
    except Exception as exc:
        print(f"[PaperAlchemy-UnifiedPlanner] generation error: {exc}")
        return {"page_plan": None}


def build_planner_graph(max_retry: int = 2):
    workflow = StateGraph(PlannerState)
    workflow.add_node("unified_planner", unified_planner_node)
    workflow.add_node("planner_critic", planner_critic_node)

    workflow.set_entry_point("unified_planner")
    workflow.add_edge("unified_planner", "planner_critic")
    workflow.add_conditional_edges(
        "planner_critic",
        build_planner_critic_router(max_retry=max_retry),
        {"retry": "unified_planner", "end": END},
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
    template_candidates: list[TemplateCandidate] | None = None,
    selected_template: TemplateCandidate | None = None,
    template_profile: TemplateProfile | None = None,
):
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[2]
    constraints = dict(generation_constraints or {})

    synced_assets = ensure_autopage_template_assets(
        project_root=project_root,
        force=bool(constraints.get("force_template_sync")),
    )
    constraints.setdefault("template_tags_json_path", str(synced_assets.tags_json_path))

    if selected_template is None or template_profile is None:
        compiled_candidates, compiled_selected, compiled_profile, _, _, _ = prepare_template_compile_bundle(
            project_root=project_root,
            generation_constraints=constraints,
            user_constraints=user_constraints or {},
            synced_assets=synced_assets,
            allow_llm=bool(constraints.get("template_compile_use_llm", True)),
            force_recompile=bool(constraints.get("force_template_recompile")),
        )
        if template_candidates is None:
            template_candidates = compiled_candidates
        selected_template = selected_template or compiled_selected
        template_profile = template_profile or compiled_profile

    templates_dir = synced_assets.templates_dir
    template_links_path = synced_assets.template_link_json_path
    module_index_path = project_root / "data" / "collectors" / "modules" / "module_index.json"

    max_templates = int(constraints.get("max_templates_for_planner", 120))
    max_entry_candidates = int(constraints.get("max_entry_candidates", 3))
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
        "template_candidates": list(template_candidates or []),
        "selected_template": selected_template,
        "template_profile": template_profile,
        "planner_feedback_history": [],
        "page_plan": None,
        "planner_critic_passed": False,
        "planner_retry_count": 0,
    }

    print("[PaperAlchemy-Planner] running UnifiedPlanner + Critic graph...")
    for _ in app.stream(initial_state, thread):
        pass

    final_state = app.get_state(thread)
    plan_result = final_state.values.get("page_plan")
    normalized_plan = _normalize_page_plan(plan_result)
    if not normalized_plan or not final_state.values.get("planner_critic_passed"):
        print("[PaperAlchemy-Planner] planner completed but critic did not fully pass.")
    else:
        print("[PaperAlchemy-Planner] planner phase completed successfully.")
    return normalized_plan

