from __future__ import annotations

from typing import Any

from src.contracts.schemas import LayoutComposeSession, LayoutComposeUpdate, PagePlan, StructuredPaper
from src.services.artifact_store import get_output_paths, save_page_plan
from src.template.shell_resolver import apply_layout_compose_session_to_page_plan, apply_layout_compose_update
from src.ui.formatters import _visual_smoke_feedback_text, format_page_plan_to_markdown
from src.ui.updates import (
    _hidden_preview_update,
    _layout_compose_ui_active,
    _layout_compose_ui_hidden,
    _normalize_manual_layout_compose_enabled,
    _review_accordion_updates,
    _stage_action_updates,
    _visible_preview_update,
)
from src.utils.html_utils import read_text_with_fallback
from src.workflows.batch_runtime import render_current_workflow_preview
from src.workflows.hitl_graph import get_default_hitl_workflow
from src.workflows.hitl_nodes import (
    _get_template_entry_path,
    normalize_layout_compose_session,
    normalize_visual_smoke_report,
)

def _require_layout_compose_snapshot(
    workflow_thread_id: str,
 ) -> tuple[dict[str, Any], dict[str, Any], LayoutComposeSession]:
    thread_id = str(workflow_thread_id or "").strip()
    if not thread_id:
        raise ValueError("No paused workflow was found. Re-run outline approval before entering layout compose.")

    config = {"configurable": {"thread_id": thread_id}}
    snapshot = get_default_hitl_workflow().get_state(config)
    snapshot_values = dict(snapshot.values or {})
    if str(snapshot_values.get("review_stage") or "").strip().lower() != "layout_compose":
        raise ValueError("Layout compose actions are only available during the layout compose review stage.")

    compose_session = normalize_layout_compose_session(snapshot_values.get("layout_compose_session"))
    if compose_session is None:
        raise ValueError("No layout compose session payload was found for the current workflow state.")
    return config, snapshot_values, compose_session

def _persist_layout_compose_update(
    config: dict[str, Any],
    session: LayoutComposeSession,
    update: LayoutComposeUpdate,
) -> LayoutComposeSession:
    updated_session = apply_layout_compose_update(session, update)
    get_default_hitl_workflow().update_state(
        config,
        {
            "layout_compose_session": updated_session,
            "layout_compose_update": update,
            "visual_smoke_report": None,
        },
        as_node="layout_compose_review",
    )
    return updated_session

def select_layout_compose_block(
    active_block_id: str | None,
    workflow_thread_id: str,
    current_logs: str,
) -> tuple[Any, ...]:
    run_log_lines = [line for line in str(current_logs or "").splitlines() if line.strip()]

    def log(message: str) -> None:
        print(message)
        run_log_lines.append(message)

    try:
        config, _, compose_session = _require_layout_compose_snapshot(workflow_thread_id)
        target_block_id = str(active_block_id or "").strip()
        if not target_block_id:
            raise ValueError("Choose one block to edit in the layout compose panel.")

        updated_session = _persist_layout_compose_update(
            config,
            compose_session,
            LayoutComposeUpdate(
                active_block_id=target_block_id,
                action="select_block",
            ),
        )
        log(f"[LayoutCompose] Active block switched to `{target_block_id}`.")
        return ("\n".join(run_log_lines), *_layout_compose_ui_active(updated_session))
    except Exception as exc:
        log(f"[Error] {exc}")
        try:
            _, _, compose_session = _require_layout_compose_snapshot(workflow_thread_id)
            compose_ui = _layout_compose_ui_active(compose_session)
        except Exception:
            compose_ui = _layout_compose_ui_hidden()
        return ("\n".join(run_log_lines), *compose_ui)

def save_layout_compose_block(
    active_block_id: str | None,
    selected_selector_hint: str | None,
    selected_figure_paths: list[str] | None,
    workflow_thread_id: str,
    current_logs: str,
) -> tuple[Any, ...]:
    run_log_lines = [line for line in str(current_logs or "").splitlines() if line.strip()]

    def log(message: str) -> None:
        print(message)
        run_log_lines.append(message)

    try:
        config, _, compose_session = _require_layout_compose_snapshot(workflow_thread_id)
        target_block_id = str(active_block_id or compose_session.active_block_id or "").strip()
        if not target_block_id:
            raise ValueError("No active block is selected for layout compose.")

        updated_session = _persist_layout_compose_update(
            config,
            compose_session,
            LayoutComposeUpdate(
                active_block_id=target_block_id,
                selected_selector_hint=selected_selector_hint,
                selected_figure_paths=list(selected_figure_paths or []),
                action="save_block",
            ),
        )
        log(f"[LayoutCompose] Saved block `{target_block_id}`.")
        return ("\n".join(run_log_lines), *_layout_compose_ui_active(updated_session))
    except Exception as exc:
        log(f"[Error] {exc}")
        try:
            _, _, compose_session = _require_layout_compose_snapshot(workflow_thread_id)
            compose_ui = _layout_compose_ui_active(compose_session)
        except Exception:
            compose_ui = _layout_compose_ui_hidden()
        return ("\n".join(run_log_lines), *compose_ui)

def _move_layout_compose_block(
    order_action: str,
    active_block_id: str | None,
    selected_selector_hint: str | None,
    selected_figure_paths: list[str] | None,
    workflow_thread_id: str,
    current_logs: str,
) -> tuple[Any, ...]:
    run_log_lines = [line for line in str(current_logs or "").splitlines() if line.strip()]

    def log(message: str) -> None:
        print(message)
        run_log_lines.append(message)

    try:
        config, _, compose_session = _require_layout_compose_snapshot(workflow_thread_id)
        target_block_id = str(active_block_id or compose_session.active_block_id or "").strip()
        if not target_block_id:
            raise ValueError("No active block is selected for layout compose.")

        updated_session = _persist_layout_compose_update(
            config,
            compose_session,
            LayoutComposeUpdate(
                active_block_id=target_block_id,
                selected_selector_hint=selected_selector_hint,
                selected_figure_paths=list(selected_figure_paths or []),
                order_action=order_action,  # type: ignore[arg-type]
                action=order_action,
            ),
        )
        log(f"[LayoutCompose] `{target_block_id}` moved {order_action.replace('_', ' ')}.")
        return ("\n".join(run_log_lines), *_layout_compose_ui_active(updated_session))
    except Exception as exc:
        log(f"[Error] {exc}")
        try:
            _, _, compose_session = _require_layout_compose_snapshot(workflow_thread_id)
            compose_ui = _layout_compose_ui_active(compose_session)
        except Exception:
            compose_ui = _layout_compose_ui_hidden()
        return ("\n".join(run_log_lines), *compose_ui)

def move_layout_compose_block_up(
    active_block_id: str | None,
    selected_selector_hint: str | None,
    selected_figure_paths: list[str] | None,
    workflow_thread_id: str,
    current_logs: str,
) -> tuple[Any, ...]:
    return _move_layout_compose_block(
        "move_up",
        active_block_id,
        selected_selector_hint,
        selected_figure_paths,
        workflow_thread_id,
        current_logs,
    )

def move_layout_compose_block_down(
    active_block_id: str | None,
    selected_selector_hint: str | None,
    selected_figure_paths: list[str] | None,
    workflow_thread_id: str,
    current_logs: str,
) -> tuple[Any, ...]:
    return _move_layout_compose_block(
        "move_down",
        active_block_id,
        selected_selector_hint,
        selected_figure_paths,
        workflow_thread_id,
        current_logs,
    )

def continue_layout_compose_to_draft(
    active_block_id: str | None,
    selected_selector_hint: str | None,
    selected_figure_paths: list[str] | None,
    workflow_thread_id: str,
    current_logs: str,
    current_outline_overview: str,
    current_preview: str | None,
    current_html_path: str,
) -> tuple[Any, ...]:
    run_log_lines = [line for line in str(current_logs or "").splitlines() if line.strip()]

    def log(message: str) -> None:
        print(message)
        run_log_lines.append(message)

    try:
        config, snapshot_values, compose_session = _require_layout_compose_snapshot(workflow_thread_id)
        target_block_id = str(active_block_id or compose_session.active_block_id or "").strip()
        if not target_block_id:
            raise ValueError("No active block is selected for layout compose.")

        update = LayoutComposeUpdate(
            active_block_id=target_block_id,
            selected_selector_hint=selected_selector_hint,
            selected_figure_paths=list(selected_figure_paths or []),
            action="continue_to_draft",
        )
        updated_session = _persist_layout_compose_update(config, compose_session, update)
        if updated_session.validation_errors:
            log("[LayoutCompose] Resolve the compose validation errors before continuing to the first draft.")
            return (
                "\n".join(run_log_lines),
                current_outline_overview,
                *_review_accordion_updates("webpage"),
                _hidden_preview_update(),
                "",
                *_stage_action_updates("none"),
                *_layout_compose_ui_active(updated_session),
            )

        approved_page_plan = PagePlan.model_validate(snapshot_values.get("approved_page_plan") or snapshot_values.get("page_plan"))
        template_entry_path = _get_template_entry_path(approved_page_plan)
        composed_page_plan = apply_layout_compose_session_to_page_plan(
            approved_page_plan,
            updated_session,
            read_text_with_fallback(template_entry_path),
        )
        paper_folder_name = str(snapshot_values.get("paper_folder_name") or "").strip()
        if not paper_folder_name:
            raise ValueError("The paused workflow state is missing paper metadata. Re-run outline approval.")
        _, _, planner_json_path, _ = get_output_paths(paper_folder_name)
        save_page_plan(planner_json_path, composed_page_plan)
        get_default_hitl_workflow().update_state(
            config,
            {
                "page_plan": composed_page_plan,
                "approved_page_plan": composed_page_plan,
                "layout_compose_session": updated_session,
                "layout_compose_update": update,
                "visual_smoke_report": None,
            },
            as_node="layout_compose_review",
        )

        log("[Coder] Layout compose confirmed. Generating the first draft...")
        get_default_hitl_workflow().invoke(None, config=config)
        paused_state = get_default_hitl_workflow().get_state(config)
        paused_values = dict(paused_state.values or {})
        review_stage = str(paused_values.get("review_stage") or "").strip().lower()
        smoke_report = normalize_visual_smoke_report(paused_values.get("visual_smoke_report"))
        feedback_text = _visual_smoke_feedback_text(smoke_report)
        if feedback_text:
            log(feedback_text)

        if review_stage == "outline":
            outline_overview = str(paused_values.get("outline_overview") or "").strip()
            if not outline_overview:
                page_plan = PagePlan.model_validate(paused_values.get("page_plan"))
                structured_data = StructuredPaper.model_validate(paused_values.get("structured_paper"))
                outline_overview = format_page_plan_to_markdown(
                    page_plan.model_dump(),
                    structured_data.model_dump(),
                )
            manual_layout_compose_enabled = _normalize_manual_layout_compose_enabled(
                paused_values.get("manual_layout_compose_enabled")
            )
            log("[Planner] Visual smoke flagged a structural mismatch, so the workflow returned to outline planning instead of opening webpage patch review.")
            return (
                "\n".join(run_log_lines),
                outline_overview,
                *_review_accordion_updates("outline"),
                _hidden_preview_update(),
                "",
                *_stage_action_updates(
                    "outline",
                    manual_layout_compose_enabled=manual_layout_compose_enabled,
                ),
                *_layout_compose_ui_hidden(),
            )

        if review_stage != "webpage":
            raise RuntimeError(
                f"Workflow paused at unexpected stage '{review_stage or '(empty)'}' after layout compose."
            )

        preview_image_path, entry_html_path = render_current_workflow_preview(paused_values)
        log(f"[Preview] Rendered webpage draft screenshot from {entry_html_path}")
        log("[Webpage] First draft ready. Review the preview, then approve it or request a revision.")
        return (
            "\n".join(run_log_lines),
            current_outline_overview,
            *_review_accordion_updates("webpage"),
            _visible_preview_update(preview_image_path),
            entry_html_path,
            *_stage_action_updates(
                "webpage",
                feedback_text_value=feedback_text,
            ),
            *_layout_compose_ui_hidden(),
        )
    except Exception as exc:
        log(f"[Error] {exc}")
        try:
            _, _, compose_session = _require_layout_compose_snapshot(workflow_thread_id)
            compose_ui = _layout_compose_ui_active(compose_session)
        except Exception:
            compose_ui = _layout_compose_ui_hidden()
        return (
            "\n".join(run_log_lines),
            current_outline_overview,
            *_review_accordion_updates("webpage"),
            _hidden_preview_update(),
            str(current_html_path or ""),
            *_stage_action_updates("none"),
            *compose_ui,
        )

def return_to_outline_review_from_layout_compose(
    workflow_thread_id: str,
    current_logs: str,
    current_outline_overview: str,
    current_preview: str | None,
    current_html_path: str,
) -> tuple[Any, ...]:
    run_log_lines = [line for line in str(current_logs or "").splitlines() if line.strip()]

    def log(message: str) -> None:
        print(message)
        run_log_lines.append(message)

    try:
        config, _, _ = _require_layout_compose_snapshot(workflow_thread_id)
        get_default_hitl_workflow().update_state(
            config,
            {
                "approved_page_plan": None,
                "is_outline_approved": False,
                "layout_compose_session": None,
                "layout_compose_update": None,
                "shell_binding_review": None,
                "shell_manual_selection": None,
                "visual_smoke_report": None,
                "review_stage": "outline",
            },
            as_node="outline_review",
        )
        updated_state = get_default_hitl_workflow().get_state(config)
        updated_values = dict(updated_state.values or {})
        manual_layout_compose_enabled = _normalize_manual_layout_compose_enabled(
            updated_values.get("manual_layout_compose_enabled")
        )
        log("[LayoutCompose] Returned to outline review. Revise the outline if needed, then approve it again to rebuild compose suggestions.")
        return (
            "\n".join(run_log_lines),
            current_outline_overview,
            *_review_accordion_updates("outline"),
            _visible_preview_update(current_preview),
            "",
            *_stage_action_updates(
                "outline",
                manual_layout_compose_enabled=manual_layout_compose_enabled,
            ),
            *_layout_compose_ui_hidden(),
        )
    except Exception as exc:
        log(f"[Error] {exc}")
        try:
            _, _, compose_session = _require_layout_compose_snapshot(workflow_thread_id)
            compose_ui = _layout_compose_ui_active(compose_session)
        except Exception:
            compose_ui = _layout_compose_ui_hidden()
        return (
            "\n".join(run_log_lines),
            current_outline_overview,
            *_review_accordion_updates("webpage"),
            _hidden_preview_update(),
            str(current_html_path or ""),
            *_stage_action_updates("none"),
            *compose_ui,
        )
