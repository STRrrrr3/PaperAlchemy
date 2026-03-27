from __future__ import annotations

from pathlib import Path

from src.agents.coder import run_coder_agent_with_diagnostics
from src.agents.translator import edit_intent_router_node, translator_node
from src.contracts.schemas import PagePlan, StructuredPaper, TemplateProfile
from src.patching.patch_pipeline import patch_agent_node, patch_executor_node
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
from src.services.preview_service import PREVIEW_CACHE_DIR
from src.template.resources import ensure_autopage_template_assets
from src.ui.app_builder import APP_CSS, build_app as _build_app
from src.ui.constraints import (
    INPUT_DIR,
    OUTPUT_DIR,
    TEMPLATE_TOP_K,
    build_generation_constraints,
    build_planner_constraints,
    build_user_constraints,
    ensure_parsed_output,
    ensure_template_assets,
    get_default_pdf,
    list_available_pdfs,
)
from src.ui.formatters import (
    _coerce_string_list,
    _extract_front_matter_candidates,
    _planner_recovery_feedback_from_visual_smoke,
    _trim_review_text,
    _visual_smoke_feedback_text,
    append_log_lines,
    attach_candidate_labels,
    build_candidate_label,
    format_page_plan_to_markdown,
    format_paper_to_markdown,
    resolve_selected_candidate,
)
from src.ui.layout_compose_handlers import (
    _move_layout_compose_block,
    _persist_layout_compose_update,
    _require_layout_compose_snapshot,
    continue_layout_compose_to_draft,
    move_layout_compose_block_down,
    move_layout_compose_block_up,
    return_to_outline_review_from_layout_compose,
    save_layout_compose_block,
    select_layout_compose_block,
)
from src.ui.review_handlers import (
    approve_extraction_and_plan_outline,
    approve_outline_and_generate_draft,
    approve_webpage,
    find_templates,
    preview_selected_template,
    request_webpage_revision,
    revise_extraction,
    revise_outline,
    run_extraction,
)
from src.ui.updates import (
    _active_layout_compose_block,
    _binding_ui_active,
    _binding_ui_hidden,
    _build_shell_binding_preview_assets,
    _format_layout_compose_block_summary,
    _format_layout_compose_editor,
    _format_layout_compose_validation,
    _format_shell_binding_review,
    _hidden_preview_update,
    _layout_compose_figure_caption,
    _layout_compose_section_caption,
    _layout_compose_ui_active,
    _layout_compose_ui_hidden,
    _normalize_manual_layout_compose_enabled,
    _ordered_layout_compose_blocks,
    _review_accordion_updates,
    _stage_action_updates,
    _visible_preview_update,
)
from src.workflows.batch_runtime import (
    confirm_and_start_generation,
    render_current_workflow_preview,
    run_langgraph_batch,
)
from src.workflows.hitl_graph import (
    build_hitl_workflow as _build_hitl_workflow_impl,
    set_default_hitl_workflow,
)
from src.workflows.hitl_nodes import (
    _apply_shell_manual_selection_to_plan,
    _build_layout_compose_session_for_plan,
    _get_template_entry_path,
    _resolve_page_plan_shell_contracts,
    binding_review_node,
    layout_compose_prepare_node,
    layout_compose_review_node,
    non_patch_feedback_node,
    normalize_coder_artifact,
    normalize_layout_compose_session,
    normalize_layout_compose_update,
    normalize_shell_binding_review,
    normalize_shell_manual_selection,
    normalize_visual_smoke_report,
    outline_review_node,
    overview_node,
    planner_phase_node,
    reader_phase_node,
    shell_resolver_phase_node,
    template_compile_phase_node,
    webpage_review_node,
)
from src.workflows.hitl_routes import (
    draft_recovery_router,
    edit_intent_route_router,
    human_review_router,
    outline_review_router,
    webpage_review_router,
)

PROJECT_ROOT = Path(__file__).resolve().parent


def coder_phase_node(state):
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


def build_hitl_workflow():
    return _build_hitl_workflow_impl(
        reader_phase_node=reader_phase_node,
        overview_node=overview_node,
        template_compile_phase_node=template_compile_phase_node,
        planner_phase_node=planner_phase_node,
        outline_review_node=outline_review_node,
        layout_compose_prepare_node=layout_compose_prepare_node,
        layout_compose_review_node=layout_compose_review_node,
        webpage_review_node=webpage_review_node,
        translator_node=translator_node,
        edit_intent_router_node=edit_intent_router_node,
        non_patch_feedback_node=non_patch_feedback_node,
        patch_agent_node=patch_agent_node,
        patch_executor_node=patch_executor_node,
        coder_phase_node=coder_phase_node,
        human_review_router=human_review_router,
        outline_review_router=outline_review_router,
        draft_recovery_router=draft_recovery_router,
        webpage_review_router=webpage_review_router,
        edit_intent_route_router=edit_intent_route_router,
    )


HITL_WORKFLOW = build_hitl_workflow()
set_default_hitl_workflow(HITL_WORKFLOW)


def build_app():
    return _build_app()


def main() -> None:
    allowed_paths = [str(OUTPUT_DIR.resolve())]
    try:
        synced_assets = ensure_autopage_template_assets(PROJECT_ROOT)
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
