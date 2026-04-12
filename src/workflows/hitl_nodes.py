from __future__ import annotations

import operator
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, StateGraph

from src.agents.coder import coder_node, run_coder_agent_with_diagnostics
from src.agents.coder_critic import (
    build_coder_critic_router,
    build_vision_qa_router,
    coder_critic_node,
    take_screenshot_action,
    vision_critic_node,
)
from src.agents.planner import finalize_planner_output, run_planner_agent, unified_planner_node
from src.agents.planner_critic import build_planner_critic_router, planner_critic_node
from src.agents.reader import _load_reader_inputs, reader_node, run_reader_agent
from src.agents.reader_critic import build_critic_router, critic_node
from src.contracts.schemas import (
    CoderArtifact,
    CssRevisionPlan,
    LayoutComposeSession,
    LayoutComposeUpdate,
    PagePlan,
    ShellBindingReview,
    ShellManualSelection,
    StructuredPaper,
    TemplateCandidate,
    TemplateProfile,
    VisualSmokeReport,
)
from src.contracts.state import CoderState, PlannerState, ReaderState, WorkflowState
from src.services.artifact_store import (
    get_output_paths,
    get_template_profile_output_path,
    load_cached_structured_data,
    load_coder_artifact,
    load_page_plan,
    load_template_profile,
    save_coder_artifact,
    save_page_plan,
    save_structured_data,
    save_template_profile,
)
from src.services.human_feedback import extract_human_feedback_text, normalize_human_feedback
from src.template.catalog import build_template_catalog, load_module_index, load_template_link_map
from src.template.compile import prepare_template_compile_bundle
from src.template.resources import ensure_autopage_template_assets
from src.template.shell_resolver import build_layout_compose_session, resolve_page_plan_shells
from src.ui.formatters import (
    _planner_recovery_feedback_from_visual_smoke,
    format_page_plan_to_markdown,
    format_paper_to_markdown,
)
from src.utils.html_utils import read_text_with_fallback, resolve_template_entry_html_path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ReaderPhaseState(ReaderState, total=False):
    paper_folder_name: str
    paper_overview: str
    outline_overview: str
    review_stage: str
    reader_cache_hit: bool


class PlannerPhaseState(PlannerState, total=False):
    paper_folder_name: str
    outline_overview: str
    review_stage: str
    approved_page_plan: PagePlan | None
    shell_binding_review: ShellBindingReview | None
    shell_manual_selection: ShellManualSelection | None
    layout_compose_session: LayoutComposeSession | None
    layout_compose_update: LayoutComposeUpdate | None
    visual_smoke_report: VisualSmokeReport | None
    template_profile_path: str
    template_compile_cache_hit: bool


class CoderPhaseState(CoderState, total=False):
    paper_folder_name: str
    coder_instructions: str
    patch_error: str
    revision_plan: Any
    targeted_replacement_plan: Any
    css_revision_plan: CssRevisionPlan | None
    css_revision_summary: str
    patch_agent_output: str
    shell_binding_review: ShellBindingReview | None
    shell_manual_selection: ShellManualSelection | None
    layout_compose_session: LayoutComposeSession | None
    layout_compose_update: LayoutComposeUpdate | None


def _workflow_paper_folder_name(state: Any) -> str:
    paper_folder_name = str(state.get("paper_folder_name") or "").strip()
    if not paper_folder_name:
        raise ValueError("paper_folder_name is missing for workflow state.")
    return paper_folder_name


def _load_previous_structured_paper(state: Any) -> StructuredPaper | None:
    current = state.get("structured_paper")
    if current:
        try:
            return StructuredPaper.model_validate(current)
        except Exception:
            pass
    paper_folder_name = str(state.get("paper_folder_name") or "").strip()
    if not paper_folder_name:
        return None
    _, structured_json_path, _, _ = get_output_paths(paper_folder_name)
    return load_cached_structured_data(structured_json_path)


def _load_structured_paper_for_state(state: Any) -> StructuredPaper:
    current = state.get("structured_paper")
    if current:
        try:
            return StructuredPaper.model_validate(current)
        except Exception:
            pass
    paper_folder_name = _workflow_paper_folder_name(state)
    _, structured_json_path, _, _ = get_output_paths(paper_folder_name)
    structured_data = load_cached_structured_data(structured_json_path)
    if structured_data is None:
        raise ValueError("structured_paper is missing from workflow state and disk cache.")
    return structured_data


def _load_page_plan_for_state(state: Any) -> PagePlan:
    for value in (state.get("approved_page_plan"), state.get("page_plan")):
        if not value:
            continue
        try:
            return PagePlan.model_validate(value)
        except Exception:
            continue
    paper_folder_name = _workflow_paper_folder_name(state)
    _, _, planner_json_path, _ = get_output_paths(paper_folder_name)
    page_plan = load_page_plan(planner_json_path)
    if page_plan is None:
        raise ValueError("page_plan is missing from workflow state and disk cache.")
    return page_plan


def _load_previous_page_plan_for_state(state: Any) -> PagePlan | None:
    for value in (state.get("page_plan"), state.get("approved_page_plan")):
        if not value:
            continue
        try:
            return PagePlan.model_validate(value)
        except Exception:
            continue
    paper_folder_name = str(state.get("paper_folder_name") or "").strip()
    if not paper_folder_name:
        return None
    _, _, planner_json_path, _ = get_output_paths(paper_folder_name)
    return load_page_plan(planner_json_path)


def _load_template_profile_for_state(state: Any) -> TemplateProfile | None:
    current = state.get("template_profile")
    if current:
        try:
            return TemplateProfile.model_validate(current)
        except Exception:
            pass
    template_profile_path = str(state.get("template_profile_path") or "").strip()
    if template_profile_path:
        profile = load_template_profile(Path(template_profile_path))
        if profile is not None:
            return profile
    paper_folder_name = str(state.get("paper_folder_name") or "").strip()
    if not paper_folder_name:
        return None
    return load_template_profile(get_template_profile_output_path(paper_folder_name))


def _load_coder_artifact_for_state(state: Any) -> CoderArtifact | None:
    artifact = normalize_coder_artifact(state.get("coder_artifact"))
    if artifact is not None:
        return artifact
    paper_folder_name = str(state.get("paper_folder_name") or "").strip()
    if not paper_folder_name:
        return None
    _, _, _, coder_json_path = get_output_paths(paper_folder_name)
    return load_coder_artifact(coder_json_path)


def _normalize_template_candidates(values: Any) -> list[TemplateCandidate]:
    candidates: list[TemplateCandidate] = []
    for candidate in values or []:
        try:
            candidates.append(TemplateCandidate.model_validate(candidate))
        except Exception:
            continue
    return candidates


def _normalize_selected_template(value: Any) -> TemplateCandidate | None:
    if not value:
        return None
    try:
        return TemplateCandidate.model_validate(value)
    except Exception:
        return None


def _resolve_page_plan_shell_contracts(
    page_plan: PagePlan,
    template_profile: TemplateProfile | None = None,
) -> tuple[PagePlan, ShellBindingReview | None]:
    template_entry_path = _get_template_entry_path(page_plan)
    resolved_plan, binding_review = resolve_page_plan_shells(
        page_plan=page_plan,
        template_reference_html=read_text_with_fallback(template_entry_path),
        template_entry_html_path=template_entry_path,
        template_profile=template_profile,
    )
    return resolved_plan, binding_review


def _get_template_entry_path(page_plan: PagePlan) -> Path:
    template_entry_path = resolve_template_entry_html_path(
        page_plan,
        project_root=PROJECT_ROOT,
    )
    if template_entry_path is None or not template_entry_path.exists():
        raise FileNotFoundError(f"Template entry html not found for shell resolution: {template_entry_path}")
    return template_entry_path


def _build_layout_compose_session_for_plan(
    page_plan: PagePlan,
    structured_data: StructuredPaper,
    paper_folder_name: str,
    template_profile: TemplateProfile | None = None,
) -> LayoutComposeSession:
    template_entry_path = _get_template_entry_path(page_plan)
    return build_layout_compose_session(
        page_plan=page_plan,
        structured_paper=structured_data,
        template_reference_html=read_text_with_fallback(template_entry_path),
        template_entry_html_path=template_entry_path,
        template_profile=template_profile,
        paper_folder_name=paper_folder_name,
    )


def normalize_coder_artifact(artifact: Any) -> CoderArtifact | None:
    if artifact is None:
        return None
    if isinstance(artifact, CoderArtifact):
        return artifact
    try:
        return CoderArtifact.model_validate(artifact)
    except Exception:
        return None


def normalize_shell_binding_review(review: Any) -> ShellBindingReview | None:
    if review is None:
        return None
    if isinstance(review, ShellBindingReview):
        return review
    try:
        return ShellBindingReview.model_validate(review)
    except Exception:
        return None


def normalize_shell_manual_selection(selection: Any) -> ShellManualSelection | None:
    if selection is None:
        return None
    if isinstance(selection, ShellManualSelection):
        return selection
    try:
        return ShellManualSelection.model_validate(selection)
    except Exception:
        return None


def normalize_layout_compose_session(session: Any) -> LayoutComposeSession | None:
    if session is None:
        return None
    if isinstance(session, LayoutComposeSession):
        return session
    try:
        return LayoutComposeSession.model_validate(session)
    except Exception:
        return None


def normalize_layout_compose_update(update: Any) -> LayoutComposeUpdate | None:
    if update is None:
        return None
    if isinstance(update, LayoutComposeUpdate):
        return update
    try:
        return LayoutComposeUpdate.model_validate(update)
    except Exception:
        return None


def normalize_visual_smoke_report(report: Any) -> VisualSmokeReport | None:
    if report is None:
        return None
    if isinstance(report, VisualSmokeReport):
        return report
    try:
        return VisualSmokeReport.model_validate(report)
    except Exception:
        return None


def _apply_shell_manual_selection_to_plan(
    page_plan: PagePlan,
    manual_selection: ShellManualSelection,
) -> PagePlan:
    updated_blocks = []
    found = False
    for block in page_plan.blocks:
        if block.block_id != manual_selection.block_id:
            updated_blocks.append(block)
            continue
        updated_region = block.target_template_region.model_copy(
            update={
                "shell_id": str(manual_selection.shell_id or "").strip(),
                "selector_hint": str(manual_selection.selector_hint or "").strip(),
            },
            deep=True,
        )
        updated_blocks.append(
            block.model_copy(
                update={
                    "target_template_region": updated_region,
                    "shell_contract": None,
                },
                deep=True,
            )
        )
        found = True
    if not found:
        raise ValueError(f"Manual shell binding referenced unknown block '{manual_selection.block_id}'.")
    return page_plan.model_copy(update={"blocks": updated_blocks}, deep=True)


def reader_phase_node(state: WorkflowState) -> dict[str, Any]:
    paper_folder_name = str(state.get("paper_folder_name") or "").strip()
    if not paper_folder_name:
        raise ValueError("paper_folder_name is missing for reader phase.")

    human_directives = extract_human_feedback_text(state.get("human_directives"))
    previous_structured_data = _load_previous_structured_paper(state)

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
    structured_data = _load_structured_paper_for_state(state)
    print("[Overview] Building deterministic reader extraction review...")
    return {
        "paper_overview": format_paper_to_markdown(structured_data.model_dump()),
        "outline_overview": "",
        "review_stage": "overview",
    }


def template_compile_phase_node(state: WorkflowState) -> dict[str, Any]:
    paper_folder_name = str(state.get("paper_folder_name") or "").strip()
    if not paper_folder_name:
        raise ValueError("paper_folder_name is missing for template compile phase.")

    generation_constraints = dict(state.get("generation_constraints") or {})
    user_constraints = dict(state.get("user_constraints") or {})
    print("[TemplateCompile] Selecting template and compiling TemplateProfile...")
    template_candidates, selected_template, template_profile, _, cache_hit, _ = prepare_template_compile_bundle(
        project_root=PROJECT_ROOT,
        generation_constraints=generation_constraints,
        user_constraints=user_constraints,
        allow_llm=bool(generation_constraints.get("template_compile_use_llm", True)),
        force_recompile=bool(generation_constraints.get("force_template_recompile")),
    )
    template_profile_path = get_template_profile_output_path(paper_folder_name)
    save_template_profile(template_profile_path, template_profile)
    return {
        "template_candidates": template_candidates,
        "selected_template": selected_template,
        "template_profile_path": str(template_profile_path),
        "template_compile_cache_hit": cache_hit,
    }


def planner_phase_node(state: WorkflowState) -> dict[str, Any]:
    paper_folder_name = str(state.get("paper_folder_name") or "").strip()
    if not paper_folder_name:
        raise ValueError("paper_folder_name is missing for planner phase.")

    structured_data = _load_structured_paper_for_state(state)
    generation_constraints = dict(state.get("generation_constraints") or {})
    user_constraints = dict(state.get("user_constraints") or {})
    template_candidates = _normalize_template_candidates(state.get("template_candidates"))
    selected_template = _normalize_selected_template(state.get("selected_template"))
    template_profile = _load_template_profile_for_state(state)
    previous_page_plan = _load_previous_page_plan_for_state(state)

    print("[Planner] Running template-first planner graph with designated template...")
    page_plan = run_planner_agent(
        paper_folder_name=paper_folder_name,
        structured_data=structured_data,
        generation_constraints=generation_constraints,
        user_constraints=user_constraints,
        human_directives=state.get("human_directives"),
        previous_page_plan=previous_page_plan,
        max_retry=2,
        template_candidates=template_candidates,
        selected_template=selected_template,
        template_profile=template_profile,
    )
    if not page_plan:
        raise RuntimeError("Planner agent failed to produce a page plan.")

    _, _, planner_json_path, _ = get_output_paths(paper_folder_name)
    save_page_plan(planner_json_path, page_plan)
    print(f"[Planner] Saved page plan to {planner_json_path}")
    return {
        "page_plan": page_plan,
        "approved_page_plan": None,
        "shell_binding_review": None,
        "shell_manual_selection": None,
        "layout_compose_session": None,
        "layout_compose_update": None,
    }


def outline_review_node(state: WorkflowState) -> dict[str, Any]:
    page_plan = _load_page_plan_for_state(state)
    structured_data = _load_structured_paper_for_state(state)
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

    structured_data = _load_structured_paper_for_state(state)
    page_plan = _load_page_plan_for_state(state)
    previous_coder_artifact = _load_coder_artifact_for_state(state)
    template_profile = _load_template_profile_for_state(state)

    print("[Coder] Running coder agent...")
    coder_artifact, visual_smoke_report, resolved_page_plan = run_coder_agent_with_diagnostics(
        paper_folder_name=paper_folder_name,
        structured_data=structured_data,
        page_plan=page_plan,
        human_directives=state.get("human_directives"),
        coder_instructions=str(state.get("coder_instructions") or ""),
        previous_coder_artifact=previous_coder_artifact,
        max_retry=2,
        template_profile=template_profile,
    )
    if not coder_artifact:
        raise RuntimeError("Coder agent failed to build the final webpage.")

    effective_page_plan = resolved_page_plan or page_plan
    _, _, planner_json_path, coder_json_path = get_output_paths(paper_folder_name)
    save_page_plan(planner_json_path, effective_page_plan)
    save_coder_artifact(coder_json_path, coder_artifact)
    print(f"[Coder] Generated entry html at {coder_artifact.entry_html}")
    updated_human_directives = _planner_recovery_feedback_from_visual_smoke(
        state.get("human_directives"),
        visual_smoke_report,
    )
    return {
        "page_plan": effective_page_plan,
        "approved_page_plan": effective_page_plan,
        "coder_artifact": coder_artifact,
        "human_directives": updated_human_directives,
        "patch_error": "",
        "revision_plan": None,
        "targeted_replacement_plan": None,
        "css_revision_plan": None,
        "css_revision_summary": "",
        "patch_agent_output": "",
        "shell_binding_review": None,
        "shell_manual_selection": None,
        "layout_compose_session": None,
        "layout_compose_update": None,
        "visual_smoke_report": visual_smoke_report,
    }


def layout_compose_prepare_node(state: WorkflowState) -> dict[str, Any]:
    paper_folder_name = str(state.get("paper_folder_name") or "").strip()
    if not paper_folder_name:
        raise ValueError("paper_folder_name is missing for layout compose preparation.")

    page_plan = _load_page_plan_for_state(state)
    structured_data = _load_structured_paper_for_state(state)
    template_profile = _load_template_profile_for_state(state)
    compose_session = _build_layout_compose_session_for_plan(
        page_plan=page_plan,
        structured_data=structured_data,
        paper_folder_name=paper_folder_name,
        template_profile=template_profile,
    )
    print("[LayoutCompose] Prepared layout compose session for manual review.")
    return {
        "layout_compose_session": compose_session,
        "layout_compose_update": None,
        "shell_binding_review": None,
        "shell_manual_selection": None,
        "visual_smoke_report": None,
    }


def layout_compose_review_node(state: WorkflowState) -> dict[str, Any]:
    session = normalize_layout_compose_session(state.get("layout_compose_session"))
    if session is None:
        raise ValueError("layout_compose_session is missing for manual review.")
    return {
        "layout_compose_session": session,
        "review_stage": "layout_compose",
    }


def shell_resolver_phase_node(state: WorkflowState) -> dict[str, Any]:
    paper_folder_name = str(state.get("paper_folder_name") or "").strip()
    if not paper_folder_name:
        raise ValueError("paper_folder_name is missing for shell resolver phase.")

    page_plan = _load_page_plan_for_state(state)
    manual_selection = normalize_shell_manual_selection(state.get("shell_manual_selection"))
    template_profile = _load_template_profile_for_state(state)
    if manual_selection is not None:
        page_plan = _apply_shell_manual_selection_to_plan(page_plan, manual_selection)

    resolved_page_plan, binding_review = _resolve_page_plan_shell_contracts(page_plan, template_profile)
    _, _, planner_json_path, _ = get_output_paths(paper_folder_name)
    save_page_plan(planner_json_path, resolved_page_plan)
    if binding_review is not None:
        print(
            "[ShellResolver] Human review required for "
            f"block '{binding_review.block_id}' in template {binding_review.template_entry_html}"
        )
        return {
            "shell_binding_review": binding_review,
            "shell_manual_selection": None,
            "visual_smoke_report": None,
        }

    print("[ShellResolver] All blocks resolved to template shells.")
    return {
        "shell_binding_review": None,
        "shell_manual_selection": None,
        "visual_smoke_report": None,
    }


def binding_review_node(_: WorkflowState) -> dict[str, Any]:
    return {"review_stage": "binding"}


def webpage_review_node(_: WorkflowState) -> dict[str, Any]:
    return {"review_stage": "webpage"}


def non_patch_feedback_node(state: WorkflowState) -> dict[str, Any]:
    reason = str(state.get("edit_intent_reason") or "").strip()
    message = "EditIntentRouter routed this request to non_patch; current patch path only supports anchored local edits."
    if reason:
        message = f"{message} Reason: {reason}."
    print(f"[EditIntentRouter] {message}")
    return {
        "patch_error": message,
        "patch_agent_output": "",
        "targeted_replacement_plan": None,
    }


def _reader_phase_prepare_node(state: ReaderPhaseState) -> dict[str, Any]:
    paper_folder_name = _workflow_paper_folder_name(state)
    output_dir, structured_json_path, _, _ = get_output_paths(paper_folder_name)
    human_directives = extract_human_feedback_text(state.get("human_directives"))

    structured_data: StructuredPaper | None = None
    if not human_directives:
        structured_data = load_cached_structured_data(structured_json_path)

    if human_directives:
        print("[Reader] Revising structured extraction from human directives...")

    if structured_data is not None:
        print(f"[Reader] Reused cached structured paper from {structured_json_path}")
        return {
            "structured_paper": structured_data,
            "critic_passed": True,
            "reader_cache_hit": True,
        }

    raw_markdown, assets_list = _load_reader_inputs(output_dir)
    return {
        "raw_markdown": raw_markdown,
        "assets_list": assets_list,
        "previous_structured_paper": load_cached_structured_data(structured_json_path),
        "human_directives": normalize_human_feedback(state.get("human_directives")),
        "feedback_history": [],
        "structured_paper": None,
        "critic_passed": False,
        "retry_count": 0,
        "reader_cache_hit": False,
    }


def _reader_phase_prepare_router(state: ReaderPhaseState) -> str:
    if bool(state.get("reader_cache_hit")):
        return "finalize"
    return "reader"


def _reader_phase_finalize_node(state: ReaderPhaseState) -> dict[str, Any]:
    paper_folder_name = _workflow_paper_folder_name(state)
    structured_data = StructuredPaper.model_validate(state.get("structured_paper"))
    _, structured_json_path, _, _ = get_output_paths(paper_folder_name)
    if not bool(state.get("reader_cache_hit")):
        save_structured_data(structured_json_path, structured_data)
        print(f"[Reader] Saved structured paper to {structured_json_path}")
    return {
        "structured_paper": None,
        "previous_structured_paper": None,
        "paper_overview": format_paper_to_markdown(structured_data.model_dump()),
        "outline_overview": "",
        "review_stage": "overview",
    }


def build_reader_phase_graph(max_retry: int = 3):
    workflow = StateGraph(ReaderPhaseState)
    workflow.add_node("reader_prepare", _reader_phase_prepare_node)
    workflow.add_node("reader", reader_node)
    workflow.add_node("reader_critic", critic_node)
    workflow.add_node("reader_finalize", _reader_phase_finalize_node)

    workflow.set_entry_point("reader_prepare")
    workflow.add_conditional_edges(
        "reader_prepare",
        _reader_phase_prepare_router,
        {"finalize": "reader_finalize", "reader": "reader"},
    )
    workflow.add_edge("reader", "reader_critic")
    workflow.add_conditional_edges(
        "reader_critic",
        build_critic_router(max_retry=max_retry),
        {"retry": "reader", "end": "reader_finalize"},
    )
    workflow.add_edge("reader_finalize", END)
    return workflow.compile(name="reader_phase")


def _planner_phase_prepare_node(state: PlannerPhaseState) -> dict[str, Any]:
    paper_folder_name = _workflow_paper_folder_name(state)
    structured_data = _load_structured_paper_for_state(state)
    constraints = dict(state.get("generation_constraints") or {})
    user_constraints = dict(state.get("user_constraints") or {})

    synced_assets = ensure_autopage_template_assets(
        project_root=PROJECT_ROOT,
        force=bool(constraints.get("force_template_sync")),
    )
    constraints.setdefault("template_tags_json_path", str(synced_assets.tags_json_path))

    template_candidates = _normalize_template_candidates(state.get("template_candidates"))
    selected_template = _normalize_selected_template(state.get("selected_template"))
    template_profile = _load_template_profile_for_state(state)
    if selected_template is None or template_profile is None:
        compiled_candidates, compiled_selected, compiled_profile, _, _, _ = prepare_template_compile_bundle(
            project_root=PROJECT_ROOT,
            generation_constraints=constraints,
            user_constraints=user_constraints,
            synced_assets=synced_assets,
            allow_llm=bool(constraints.get("template_compile_use_llm", True)),
            force_recompile=bool(constraints.get("force_template_recompile")),
        )
        if not template_candidates:
            template_candidates = compiled_candidates
        selected_template = selected_template or compiled_selected
        template_profile = template_profile or compiled_profile
        save_template_profile(get_template_profile_output_path(paper_folder_name), template_profile)

    templates_dir = synced_assets.templates_dir
    template_links_path = synced_assets.template_link_json_path
    module_index_path = PROJECT_ROOT / "data" / "collectors" / "modules" / "module_index.json"

    max_templates = int(constraints.get("max_templates_for_planner", 120))
    max_entry_candidates = int(constraints.get("max_entry_candidates", 3))
    template_catalog = build_template_catalog(
        templates_dir=templates_dir,
        project_root=PROJECT_ROOT,
        max_templates=max_templates,
        max_entry_candidates=max_entry_candidates,
    )
    template_link_map = load_template_link_map(template_links_path)
    module_index = load_module_index(module_index_path)

    return {
        "structured_paper": structured_data,
        "previous_page_plan": _load_previous_page_plan_for_state(state),
        "template_catalog": template_catalog,
        "template_link_map": template_link_map,
        "module_index": module_index,
        "generation_constraints": constraints,
        "user_constraints": user_constraints,
        "human_directives": normalize_human_feedback(state.get("human_directives")),
        "template_candidates": template_candidates,
        "selected_template": selected_template,
        "template_profile": template_profile,
        "planner_feedback_history": [],
        "semantic_page_plan": None,
        "page_plan": None,
        "planner_critic_passed": False,
        "planner_retry_count": 0,
    }


def _planner_phase_finalize_node(state: PlannerPhaseState) -> dict[str, Any]:
    paper_folder_name = _workflow_paper_folder_name(state)
    page_plan = finalize_planner_output(
        semantic_page_plan=state.get("semantic_page_plan"),
        selected_template=_normalize_selected_template(state.get("selected_template")),
        template_profile=_load_template_profile_for_state(state),
        generation_constraints=dict(state.get("generation_constraints") or {}),
        template_candidates=_normalize_template_candidates(state.get("template_candidates")),
        planner_critic_passed=bool(state.get("planner_critic_passed")),
        planner_feedback_history=list(state.get("planner_feedback_history") or []),
    )
    if page_plan is None:
        raise RuntimeError("Planner agent failed to produce a valid bound page plan.")
    _, _, planner_json_path, _ = get_output_paths(paper_folder_name)
    if not bool(state.get("planner_critic_passed")):
        print("[PaperAlchemy-Planner] planner completed but semantic critic did not fully pass.")
    else:
        print("[PaperAlchemy-Planner] planner phase completed successfully.")
    save_page_plan(planner_json_path, page_plan)
    print(f"[Planner] Saved page plan to {planner_json_path}")
    return {
        "structured_paper": None,
        "template_profile": None,
        "semantic_page_plan": None,
        "page_plan": None,
        "previous_page_plan": None,
        "approved_page_plan": None,
        "shell_binding_review": None,
        "shell_manual_selection": None,
        "layout_compose_session": None,
        "layout_compose_update": None,
        "visual_smoke_report": None,
    }


def build_planner_phase_graph(max_retry: int = 2):
    workflow = StateGraph(PlannerPhaseState)
    workflow.add_node("planner_prepare", _planner_phase_prepare_node)
    workflow.add_node("unified_planner", unified_planner_node)
    workflow.add_node("planner_critic", planner_critic_node)
    workflow.add_node("planner_finalize", _planner_phase_finalize_node)

    workflow.set_entry_point("planner_prepare")
    workflow.add_edge("planner_prepare", "unified_planner")
    workflow.add_edge("unified_planner", "planner_critic")
    workflow.add_conditional_edges(
        "planner_critic",
        build_planner_critic_router(max_retry=max_retry),
        {"retry": "unified_planner", "end": "planner_finalize"},
    )
    workflow.add_edge("planner_finalize", END)
    return workflow.compile(name="planner_phase")


def _coder_phase_prepare_node(state: CoderPhaseState) -> dict[str, Any]:
    paper_folder_name = _workflow_paper_folder_name(state)
    return {
        "paper_folder_name": paper_folder_name,
        "human_directives": normalize_human_feedback(state.get("human_directives")),
        "coder_instructions": str(state.get("coder_instructions") or "").strip(),
        "structured_paper": _load_structured_paper_for_state(state),
        "page_plan": _load_page_plan_for_state(state),
        "template_profile": _load_template_profile_for_state(state),
        "block_render_specs": [],
        "block_render_artifacts": [],
        "coder_feedback_history": [],
        "visual_feedback": [],
        "visual_screenshot_path": "",
        "visual_iterations": 0,
        "is_visually_approved": False,
        "visual_smoke_report": None,
        "coder_artifact": _load_coder_artifact_for_state(state),
        "coder_critic_passed": False,
        "coder_retry_count": 0,
    }


def _coder_phase_finalize_node(state: CoderPhaseState) -> dict[str, Any]:
    paper_folder_name = _workflow_paper_folder_name(state)
    coder_artifact = normalize_coder_artifact(state.get("coder_artifact"))
    page_plan = PagePlan.model_validate(state.get("page_plan"))
    visual_smoke_report = normalize_visual_smoke_report(state.get("visual_smoke_report"))
    if not coder_artifact or not bool(state.get("coder_critic_passed")):
        raise RuntimeError("Coder agent failed to build the final webpage.")

    _, _, planner_json_path, coder_json_path = get_output_paths(paper_folder_name)
    save_page_plan(planner_json_path, page_plan)
    save_coder_artifact(coder_json_path, coder_artifact)
    print(f"[Coder] Generated entry html at {coder_artifact.entry_html}")
    if not bool(state.get("is_visually_approved")) and int(state.get("visual_iterations", 0)) > 0:
        print("[PaperAlchemy-Coder] visual smoke test flagged issues; returning the latest artifact for human review.")
    updated_human_directives = _planner_recovery_feedback_from_visual_smoke(
        state.get("human_directives"),
        visual_smoke_report,
    )
    return {
        "structured_paper": None,
        "page_plan": None,
        "template_profile": None,
        "coder_artifact": None,
        "block_render_artifacts": [],
        "human_directives": updated_human_directives,
        "patch_error": "",
        "revision_plan": None,
        "targeted_replacement_plan": None,
        "css_revision_plan": None,
        "css_revision_summary": "",
        "patch_agent_output": "",
        "shell_binding_review": None,
        "shell_manual_selection": None,
        "layout_compose_session": None,
        "layout_compose_update": None,
        "visual_smoke_report": visual_smoke_report,
    }


def build_coder_phase_graph(max_retry: int = 2):
    workflow = StateGraph(CoderPhaseState)
    workflow.add_node("coder_prepare", _coder_phase_prepare_node)
    workflow.add_node("coder", coder_node)
    workflow.add_node("coder_critic", coder_critic_node)
    workflow.add_node("take_screenshot", take_screenshot_action)
    workflow.add_node("vision_critic", vision_critic_node)
    workflow.add_node("coder_finalize", _coder_phase_finalize_node)

    workflow.set_entry_point("coder_prepare")
    workflow.add_edge("coder_prepare", "coder")
    workflow.add_edge("coder", "coder_critic")
    workflow.add_conditional_edges(
        "coder_critic",
        build_coder_critic_router(max_retry=max_retry),
        {"retry": "coder", "visual_qa": "take_screenshot", "end": "coder_finalize"},
    )
    workflow.add_edge("take_screenshot", "vision_critic")
    workflow.add_conditional_edges(
        "vision_critic",
        build_vision_qa_router(),
        {"retry": "coder", "end": "coder_finalize"},
    )
    workflow.add_edge("coder_finalize", END)
    return workflow.compile(name="coder_phase")
