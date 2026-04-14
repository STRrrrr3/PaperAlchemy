from __future__ import annotations

from pathlib import Path

from src.revision.css_revision import arbiter_autofix_node, css_revision_agent_node, css_revision_executor_node
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
    _trim_review_text,
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
from src.review.reviewer_nodes import (
    capture_review_screenshots_node,
    layout_rhythm_reviewer_node,
    review_arbiter_node,
    semantic_visual_reviewer_node,
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
    build_coder_phase_graph,
    build_planner_phase_graph,
    build_reader_phase_graph,
    layout_compose_prepare_node,
    layout_compose_review_node,
    normalize_coder_artifact,
    normalize_layout_compose_session,
    normalize_layout_compose_update,
    normalize_shell_binding_review,
    normalize_shell_manual_selection,
    outline_review_node,
    overview_node,
    planner_phase_node,
    reader_phase_node,
    shell_resolver_phase_node,
    template_compile_phase_node,
    webpage_review_node,
)
from src.workflows.hitl_routes import (
    human_review_router,
    outline_review_router,
    post_arbiter_router,
    webpage_review_router,
)

PROJECT_ROOT = Path(__file__).resolve().parent


def build_hitl_workflow():
    return _build_hitl_workflow_impl(
        reader_phase_node=build_reader_phase_graph(),
        overview_node=overview_node,
        template_compile_phase_node=template_compile_phase_node,
        planner_phase_node=build_planner_phase_graph(),
        outline_review_node=outline_review_node,
        layout_compose_prepare_node=layout_compose_prepare_node,
        layout_compose_review_node=layout_compose_review_node,
        capture_review_screenshots_node=capture_review_screenshots_node,
        semantic_visual_reviewer_node=semantic_visual_reviewer_node,
        layout_rhythm_reviewer_node=layout_rhythm_reviewer_node,
        review_arbiter_node=review_arbiter_node,
        arbiter_autofix_node=arbiter_autofix_node,
        webpage_review_node=webpage_review_node,
        css_revision_agent_node=css_revision_agent_node,
        css_revision_executor_node=css_revision_executor_node,
        coder_phase_node=build_coder_phase_graph(),
        human_review_router=human_review_router,
        outline_review_router=outline_review_router,
        webpage_review_router=webpage_review_router,
        post_arbiter_router=post_arbiter_router,
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
