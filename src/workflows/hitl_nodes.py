from __future__ import annotations

from pathlib import Path
from typing import Any

from src.agents.coder import run_coder_agent_with_diagnostics
from src.agents.planner import run_planner_agent
from src.agents.reader import run_reader_agent
from src.contracts.schemas import (
    CoderArtifact,
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
from src.contracts.state import WorkflowState
from src.services.artifact_store import (
    get_output_paths,
    get_template_profile_output_path,
    load_block_render_artifacts_from_disk,
    load_cached_structured_data,
    save_coder_artifact,
    save_page_plan,
    save_structured_data,
    save_template_profile,
)
from src.services.human_feedback import extract_human_feedback_text
from src.template.compile import prepare_template_compile_bundle
from src.template.shell_resolver import build_layout_compose_session, resolve_page_plan_shells
from src.ui.formatters import (
    _planner_recovery_feedback_from_visual_smoke,
    format_page_plan_to_markdown,
    format_paper_to_markdown,
)
from src.utils.html_utils import read_text_with_fallback, resolve_template_entry_html_path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

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
            update={"selector_hint": str(manual_selection.selector_hint or "").strip()},
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
        "template_profile": template_profile,
        "template_profile_path": str(template_profile_path),
        "template_compile_cache_hit": cache_hit,
    }

def planner_phase_node(state: WorkflowState) -> dict[str, Any]:
    paper_folder_name = str(state.get("paper_folder_name") or "").strip()
    if not paper_folder_name:
        raise ValueError("paper_folder_name is missing for planner phase.")

    structured_data = StructuredPaper.model_validate(state.get("structured_paper"))
    generation_constraints = dict(state.get("generation_constraints") or {})
    user_constraints = dict(state.get("user_constraints") or {})
    template_candidates: list[TemplateCandidate] = []
    for candidate in (state.get("template_candidates") or []):
        try:
            template_candidates.append(TemplateCandidate.model_validate(candidate))
        except Exception:
            continue
    try:
        selected_template = (
            TemplateCandidate.model_validate(state.get("selected_template"))
            if state.get("selected_template")
            else None
        )
    except Exception:
        selected_template = None
    try:
        template_profile = (
            TemplateProfile.model_validate(state.get("template_profile"))
            if state.get("template_profile")
            else None
        )
    except Exception:
        template_profile = None
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
        "block_render_artifacts": [],
    }

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
    page_plan = PagePlan.model_validate(approved_page_plan)
    previous_coder_artifact = normalize_coder_artifact(state.get("coder_artifact"))
    template_profile = (
        TemplateProfile.model_validate(state.get("template_profile"))
        if state.get("template_profile")
        else None
    )

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
        "patch_agent_output": "",
        "shell_binding_review": None,
        "shell_manual_selection": None,
        "layout_compose_session": None,
        "layout_compose_update": None,
        "visual_smoke_report": visual_smoke_report,
        "block_render_artifacts": load_block_render_artifacts_from_disk(paper_folder_name),
    }

def layout_compose_prepare_node(state: WorkflowState) -> dict[str, Any]:
    paper_folder_name = str(state.get("paper_folder_name") or "").strip()
    if not paper_folder_name:
        raise ValueError("paper_folder_name is missing for layout compose preparation.")

    approved_page_plan = state.get("approved_page_plan") or state.get("page_plan")
    page_plan = PagePlan.model_validate(approved_page_plan)
    structured_data = StructuredPaper.model_validate(state.get("structured_paper"))
    template_profile = (
        TemplateProfile.model_validate(state.get("template_profile"))
        if state.get("template_profile")
        else None
    )
    compose_session = _build_layout_compose_session_for_plan(
        page_plan=page_plan,
        structured_data=structured_data,
        paper_folder_name=paper_folder_name,
        template_profile=template_profile,
    )
    print("[LayoutCompose] Prepared layout compose session for manual review.")
    return {
        "page_plan": page_plan,
        "approved_page_plan": page_plan,
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

    approved_page_plan = state.get("approved_page_plan") or state.get("page_plan")
    page_plan = PagePlan.model_validate(approved_page_plan)
    manual_selection = normalize_shell_manual_selection(state.get("shell_manual_selection"))
    template_profile = (
        TemplateProfile.model_validate(state.get("template_profile"))
        if state.get("template_profile")
        else None
    )
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
            "page_plan": resolved_page_plan,
            "approved_page_plan": resolved_page_plan,
            "shell_binding_review": binding_review,
            "shell_manual_selection": None,
            "visual_smoke_report": None,
        }

    print("[ShellResolver] All blocks resolved to template shells.")
    return {
        "page_plan": resolved_page_plan,
        "approved_page_plan": resolved_page_plan,
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
