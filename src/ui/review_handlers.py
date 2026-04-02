from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import gradio as gr

from src.contracts.schemas import PagePlan, StructuredPaper
from src.contracts.state import WorkflowState
from src.services.artifact_store import get_output_paths
from src.services.human_feedback import build_human_feedback_payload, empty_human_feedback, extract_human_feedback_text
from src.services.preview_service import build_template_preview_path, take_local_screenshot
from src.template.deterministic_selector import score_and_select_templates
from src.ui.constraints import (
    INPUT_DIR,
    TEMPLATE_TOP_K,
    build_generation_constraints,
    build_user_constraints,
    ensure_parsed_output,
    ensure_template_assets,
    get_default_pdf,
)
from src.ui.formatters import (
    _visual_smoke_feedback_text,
    append_log_lines,
    attach_candidate_labels,
    format_page_plan_to_markdown,
    format_paper_to_markdown,
    resolve_selected_candidate,
)
from src.ui.updates import (
    _hidden_preview_update,
    _layout_compose_ui_active,
    _layout_compose_ui_hidden,
    _normalize_manual_layout_compose_enabled,
    _review_accordion_updates,
    _stage_action_updates,
    _visible_preview_update,
)
from src.workflows.batch_runtime import render_current_workflow_preview
from src.workflows.hitl_graph import get_default_hitl_workflow
from src.workflows.hitl_nodes import normalize_layout_compose_session, normalize_visual_smoke_report

def find_templates(
    background_color: str,
    density: str,
    navigation: str,
    layout: str,
) -> tuple[Any, ...]:
    log_lines: list[str] = []
    preview_image_path: str | None = None

    try:
        user_constraints = build_user_constraints(
            background_color=background_color,
            density=density,
            navigation=navigation,
            layout=layout,
        )
        synced_assets = ensure_template_assets()
        candidates = score_and_select_templates(
            user_constraints=user_constraints,
            tags_json_path=synced_assets.tags_json_path,
            top_k=TEMPLATE_TOP_K,
        )
        candidates = attach_candidate_labels(candidates)

        if not candidates:
            raise ValueError("No templates matched the provided constraints.")

        search_state = {
            "user_constraints": user_constraints,
            "tags_json_path": str(synced_assets.tags_json_path),
            "candidates": candidates,
        }

        log_lines.append(f"[Search] user_constraints={json.dumps(user_constraints, ensure_ascii=False)}")
        for candidate in candidates:
            log_lines.append(f"[Candidate] {candidate['ui_label']} -> {candidate['template_path']}")

        return (
            gr.update(
                choices=[candidate["ui_label"] for candidate in candidates],
                value=None,
                interactive=True,
            ),
            search_state,
            None,
            "\n".join(log_lines),
            _visible_preview_update(preview_image_path),
            gr.update(interactive=False),
            "",
            "",
            *_review_accordion_updates("webpage"),
            "",
            "",
            *_stage_action_updates("none"),
            *_layout_compose_ui_hidden(),
        )
    except Exception as exc:
        log_lines.append(f"[Error] {exc}")
        return (
            gr.update(choices=[], value=None, interactive=False),
            {"user_constraints": {}, "tags_json_path": "", "candidates": []},
            None,
            "\n".join(log_lines),
            None,
            gr.update(interactive=False),
            "",
            "",
            *_review_accordion_updates("webpage"),
            "",
            "",
            *_stage_action_updates("none"),
            *_layout_compose_ui_hidden(),
        )

def preview_selected_template(
    selected_label: str | None,
    search_state: dict[str, Any] | None,
    current_logs: str,
    current_preview_path: str | None,
) -> tuple[Any, ...]:
    candidate = resolve_selected_candidate(selected_label, search_state)
    if not candidate:
        return (
            _visible_preview_update(None),
            None,
            current_logs,
            gr.update(interactive=False),
            "",
            "",
            *_review_accordion_updates("webpage"),
            "",
            "",
            *_stage_action_updates("none"),
            *_layout_compose_ui_hidden(),
        )

    try:
        entry_html_path = Path(str(candidate.get("entry_html_path") or "")).resolve()
        preview_image_path = take_local_screenshot(
            str(entry_html_path),
            str(build_template_preview_path(candidate)),
        )
        if not preview_image_path:
            raise RuntimeError(f"Failed to render screenshot for {entry_html_path}")
        updated_logs = append_log_lines(
            current_logs,
            [
                f"[Preview] Rendered template screenshot for {candidate['template_id']} from {entry_html_path}",
                f"[Preview] Screenshot path: {preview_image_path}",
            ],
        )
        return (
            _visible_preview_update(preview_image_path),
            candidate,
            updated_logs,
            gr.update(interactive=True),
            "",
            "",
            *_review_accordion_updates("webpage"),
            "",
            "",
            *_stage_action_updates("none"),
            *_layout_compose_ui_hidden(),
        )
    except Exception as exc:
        updated_logs = append_log_lines(current_logs, [f"[Error] Failed to render template preview: {exc}"])
        return (
            _visible_preview_update(current_preview_path),
            None,
            updated_logs,
            gr.update(interactive=False),
            "",
            "",
            *_review_accordion_updates("webpage"),
            "",
            "",
            *_stage_action_updates("none"),
            *_layout_compose_ui_hidden(),
        )

def run_extraction(
    pdf_filename: str | None,
    selected_label: str | None,
    search_state: dict[str, Any] | None,
    selected_candidate_state: dict[str, Any] | None,
    current_logs: str,
) -> tuple[Any, ...]:
    selected_candidate = selected_candidate_state or resolve_selected_candidate(selected_label, search_state)
    if not selected_candidate:
        raise gr.Error("Select and preview one of the Top 5 candidate templates before running Step 1.")

    chosen_pdf = str(pdf_filename or "").strip() or get_default_pdf()
    safe_pdf_filename = str(chosen_pdf).strip()
    if not safe_pdf_filename:
        raise gr.Error("A valid PDF filename is required.")

    input_path = INPUT_DIR / safe_pdf_filename
    if not input_path.exists():
        raise gr.Error(f"Input PDF not found: {input_path}")

    run_log_lines = [line for line in str(current_logs or "").splitlines() if line.strip()]

    def log(message: str) -> None:
        print(message)
        run_log_lines.append(message)

    try:
        template_id = str(selected_candidate.get("template_id") or "").strip()
        template_path = str(selected_candidate.get("template_path") or "").strip()
        entry_html = str(selected_candidate.get("entry_html") or "").strip()
        if not (template_id and template_path and entry_html):
            raise ValueError("The selected template is missing template_id, template_path, or entry_html.")

        user_constraints = dict((search_state or {}).get("user_constraints") or {})
        ranked_candidates = list((search_state or {}).get("candidates") or [])
        synced_assets = ensure_template_assets()
        paper_folder_name = Path(safe_pdf_filename).stem
        output_dir, _, _, _ = get_output_paths(paper_folder_name)

        log(f"[UI] Using input PDF: {safe_pdf_filename}")
        log(f"[UI] Confirmed template: {template_id}")
        log(f"[UI] Template root: {template_path}")
        log(f"[UI] Entry HTML: {entry_html}")
        log(f"[UI] User constraints: {json.dumps(user_constraints, ensure_ascii=False)}")
        log(f"[Assets] Template resources ready at {synced_assets.templates_dir}")

        if not ensure_parsed_output(safe_pdf_filename, output_dir):
            raise RuntimeError("PDF parsing failed. Please inspect the parser output logs.")

        generation_constraints = build_generation_constraints(
            synced_assets=synced_assets,
            selected_candidate=selected_candidate,
            ranked_candidates=ranked_candidates,
        )
        thread_id = f"hitl::{paper_folder_name}::{uuid4().hex}"
        config = {"configurable": {"thread_id": thread_id}}
        initial_state: WorkflowState = {
            "paper_folder_name": paper_folder_name,
            "user_constraints": user_constraints,
            "generation_constraints": generation_constraints,
            "manual_layout_compose_enabled": False,
            "human_directives": empty_human_feedback(),
            "coder_instructions": "",
            "edit_intent": None,
            "edit_intent_reason": "",
            "patch_agent_output": "",
            "revision_plan": None,
            "targeted_replacement_plan": None,
            "patch_error": "",
            "paper_overview": "",
            "outline_overview": "",
            "is_approved": False,
            "is_outline_approved": False,
            "is_webpage_approved": False,
            "review_stage": "overview",
            "template_candidates": [],
            "selected_template": None,
            "template_profile": None,
            "template_profile_path": "",
            "template_compile_cache_hit": False,
            "block_render_artifacts": [],
            "structured_paper": None,
            "page_plan": None,
            "approved_page_plan": None,
            "coder_artifact": None,
            "shell_binding_review": None,
            "shell_manual_selection": None,
            "layout_compose_session": None,
            "layout_compose_update": None,
            "visual_smoke_report": None,
        }

        log("[Reader] Running workflow until the extraction review checkpoint...")
        get_default_hitl_workflow().invoke(initial_state, config=config)
        paused_state = get_default_hitl_workflow().get_state(config)
        paused_values = dict(paused_state.values or {})
        paper_overview = str(paused_values.get("paper_overview") or "").strip()
        if not paper_overview:
            structured_data = StructuredPaper.model_validate(paused_values.get("structured_paper"))
            paper_overview = format_paper_to_markdown(structured_data.model_dump())
        log("[Overview] Reader extraction complete. Review the source pack, revise if needed, or approve it to plan the webpage outline.")
        return (
            "\n".join(run_log_lines),
            paper_overview,
            "",
            *_review_accordion_updates("overview"),
            thread_id,
            "",
            *_stage_action_updates("overview"),
            *_layout_compose_ui_hidden(),
        )
    except Exception as exc:
        log(f"[Error] {exc}")
        return (
            "\n".join(run_log_lines),
            "",
            "",
            *_review_accordion_updates("webpage"),
            "",
            "",
            *_stage_action_updates("none"),
            *_layout_compose_ui_hidden(),
        )

def revise_extraction(
    feedback_text: str,
    feedback_images: Any,
    workflow_thread_id: str,
    current_logs: str,
    current_overview: str,
    current_outline_overview: str,
) -> tuple[Any, ...]:
    run_log_lines = [line for line in str(current_logs or "").splitlines() if line.strip()]

    def log(message: str) -> None:
        print(message)
        run_log_lines.append(message)

    try:
        thread_id = str(workflow_thread_id or "").strip()
        if not thread_id:
            raise ValueError("No paused workflow was found. Run Step 1 before requesting a revision.")

        config = {"configurable": {"thread_id": thread_id}}
        snapshot = get_default_hitl_workflow().get_state(config)
        snapshot_values = dict(snapshot.values or {})
        if str(snapshot_values.get("review_stage") or "overview") != "overview":
            raise ValueError("Extraction revisions are only available during the overview review stage.")

        paper_folder_name = str(snapshot_values.get("paper_folder_name") or "").strip()
        if not paper_folder_name:
            raise ValueError("The paused workflow state is missing paper metadata. Run Step 1 again.")

        feedback_payload = build_human_feedback_payload(feedback_text, feedback_images)
        human_directives = extract_human_feedback_text(feedback_payload)
        if not human_directives:
            raise ValueError("Enter feedback text before requesting an extraction revision.")

        log(f"[HITL] Reader revision feedback captured: {human_directives}")

        get_default_hitl_workflow().update_state(
            config,
            {
                "human_directives": feedback_payload,
                "coder_instructions": "",
                "edit_intent": None,
                "edit_intent_reason": "",
                "patch_agent_output": "",
                "revision_plan": None,
                "targeted_replacement_plan": None,
                "patch_error": "",
                "is_approved": False,
                "shell_binding_review": None,
                "shell_manual_selection": None,
                "layout_compose_session": None,
                "layout_compose_update": None,
                "visual_smoke_report": None,
            },
            as_node="overview",
        )

        log("[Reader] Resuming workflow to revise extraction...")
        get_default_hitl_workflow().invoke(None, config=config)
        paused_state = get_default_hitl_workflow().get_state(config)
        paused_values = dict(paused_state.values or {})
        paper_overview = str(paused_values.get("paper_overview") or "").strip()
        if not paper_overview:
            structured_data = StructuredPaper.model_validate(paused_values.get("structured_paper"))
            paper_overview = format_paper_to_markdown(structured_data.model_dump())
        log("[Overview] Revised extraction complete. Review the updated source pack or approve it to plan the webpage outline.")
        return (
            "\n".join(run_log_lines),
            paper_overview,
            "",
            *_review_accordion_updates("overview"),
            *_stage_action_updates("overview"),
            *_layout_compose_ui_hidden(),
        )
    except Exception as exc:
        log(f"[Error] {exc}")
        return (
            "\n".join(run_log_lines),
            current_overview,
            current_outline_overview,
            *_review_accordion_updates("overview"),
            *_stage_action_updates("overview", feedback_text_value=feedback_text),
            *_layout_compose_ui_hidden(),
        )

def approve_extraction_and_plan_outline(
    workflow_thread_id: str,
    current_logs: str,
    current_overview: str,
    current_outline_overview: str,
) -> tuple[Any, ...]:
    run_log_lines = [line for line in str(current_logs or "").splitlines() if line.strip()]

    def log(message: str) -> None:
        print(message)
        run_log_lines.append(message)

    try:
        thread_id = str(workflow_thread_id or "").strip()
        if not thread_id:
            raise ValueError("No paused workflow was found. Run Step 1 before approving.")

        config = {"configurable": {"thread_id": thread_id}}
        snapshot = get_default_hitl_workflow().get_state(config)
        snapshot_values = dict(snapshot.values or {})
        if str(snapshot_values.get("review_stage") or "overview") != "overview":
            raise ValueError("The extraction review stage has already been completed.")

        paper_folder_name = str(snapshot_values.get("paper_folder_name") or "").strip()
        if not paper_folder_name:
            raise ValueError("The paused workflow state is missing paper metadata. Run Step 1 again.")

        get_default_hitl_workflow().update_state(
            config,
            {
                "human_directives": empty_human_feedback(),
                "coder_instructions": "",
                "patch_agent_output": "",
                "revision_plan": None,
                "targeted_replacement_plan": None,
                "patch_error": "",
                "is_approved": True,
                "is_outline_approved": False,
                "is_webpage_approved": False,
                "shell_binding_review": None,
                "shell_manual_selection": None,
                "layout_compose_session": None,
                "layout_compose_update": None,
                "visual_smoke_report": None,
            },
            as_node="overview",
        )

        log("[Planner] Extraction approved. Planning the webpage outline...")
        get_default_hitl_workflow().invoke(None, config=config)
        paused_state = get_default_hitl_workflow().get_state(config)
        paused_values = dict(paused_state.values or {})
        if str(paused_values.get("review_stage") or "") != "outline":
            raise RuntimeError("Workflow did not pause at the outline review stage.")
        outline_overview = str(paused_values.get("outline_overview") or "").strip()
        if not outline_overview:
            page_plan = PagePlan.model_validate(paused_values.get("page_plan"))
            structured_data = StructuredPaper.model_validate(paused_values.get("structured_paper"))
            outline_overview = format_page_plan_to_markdown(
                page_plan.model_dump(),
                structured_data.model_dump(),
            )
        paper_overview = str(paused_values.get("paper_overview") or "").strip() or current_overview
        manual_layout_compose_enabled = _normalize_manual_layout_compose_enabled(
            paused_values.get("manual_layout_compose_enabled")
        )
        log("[Outline] Planner outline ready. Review the planned webpage sections, revise them if needed, or approve the outline to generate the first draft.")
        return (
            "\n".join(run_log_lines),
            paper_overview,
            outline_overview,
            *_review_accordion_updates("outline"),
            "",
            *_stage_action_updates(
                "outline",
                manual_layout_compose_enabled=manual_layout_compose_enabled,
            ),
            *_layout_compose_ui_hidden(),
        )
    except Exception as exc:
        log(f"[Error] {exc}")
        return (
            "\n".join(run_log_lines),
            current_overview,
            current_outline_overview,
            *_review_accordion_updates("overview"),
            "",
            *_stage_action_updates("overview"),
            *_layout_compose_ui_hidden(),
        )

def revise_outline(
    feedback_text: str,
    feedback_images: Any,
    workflow_thread_id: str,
    current_logs: str,
    current_outline_overview: str,
) -> tuple[Any, ...]:
    run_log_lines = [line for line in str(current_logs or "").splitlines() if line.strip()]

    def log(message: str) -> None:
        print(message)
        run_log_lines.append(message)

    try:
        thread_id = str(workflow_thread_id or "").strip()
        if not thread_id:
            raise ValueError("No paused workflow was found. Approve extraction before revising the outline.")

        config = {"configurable": {"thread_id": thread_id}}
        snapshot = get_default_hitl_workflow().get_state(config)
        snapshot_values = dict(snapshot.values or {})
        if str(snapshot_values.get("review_stage") or "") != "outline":
            raise ValueError("Outline revisions are only available during the planner outline review stage.")

        feedback_payload = build_human_feedback_payload(feedback_text, feedback_images)
        human_directives = extract_human_feedback_text(feedback_payload)
        if not human_directives:
            raise ValueError("Enter feedback text before requesting an outline revision.")

        log(f"[HITL] Planner outline revision feedback captured: {human_directives}")

        get_default_hitl_workflow().update_state(
            config,
            {
                "human_directives": feedback_payload,
                "is_outline_approved": False,
                "approved_page_plan": None,
                "patch_agent_output": "",
                "revision_plan": None,
                "targeted_replacement_plan": None,
                "patch_error": "",
                "shell_binding_review": None,
                "shell_manual_selection": None,
                "layout_compose_session": None,
                "layout_compose_update": None,
                "visual_smoke_report": None,
            },
            as_node="outline_review",
        )

        log("[Planner] Resuming workflow to revise the webpage outline...")
        get_default_hitl_workflow().invoke(None, config=config)
        paused_state = get_default_hitl_workflow().get_state(config)
        paused_values = dict(paused_state.values or {})
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
        log("[Outline] Revised outline ready. Review the updated webpage structure or approve it to generate the first draft.")
        return (
            "\n".join(run_log_lines),
            outline_overview,
            *_review_accordion_updates("outline"),
            *_stage_action_updates(
                "outline",
                manual_layout_compose_enabled=manual_layout_compose_enabled,
            ),
            *_layout_compose_ui_hidden(),
        )
    except Exception as exc:
        log(f"[Error] {exc}")
        return (
            "\n".join(run_log_lines),
            current_outline_overview,
            *_review_accordion_updates("outline"),
            *_stage_action_updates(
                "outline",
                feedback_text_value=feedback_text,
            ),
            *_layout_compose_ui_hidden(),
        )

def approve_outline_and_generate_draft(
    workflow_thread_id: str,
    current_logs: str,
    current_outline_overview: str,
    manual_layout_compose_enabled: bool,
    current_preview: str | None,
    current_html_path: str,
) -> tuple[Any, ...]:
    run_log_lines = [line for line in str(current_logs or "").splitlines() if line.strip()]

    def log(message: str) -> None:
        print(message)
        run_log_lines.append(message)

    try:
        thread_id = str(workflow_thread_id or "").strip()
        if not thread_id:
            raise ValueError("No paused workflow was found. Approve extraction before generating the first draft.")

        config = {"configurable": {"thread_id": thread_id}}
        snapshot = get_default_hitl_workflow().get_state(config)
        snapshot_values = dict(snapshot.values or {})
        if str(snapshot_values.get("review_stage") or "") != "outline":
            raise ValueError("The first draft can only be generated from the outline review stage.")

        approved_page_plan = PagePlan.model_validate(snapshot_values.get("page_plan"))
        manual_layout_compose_enabled = _normalize_manual_layout_compose_enabled(manual_layout_compose_enabled)
        get_default_hitl_workflow().update_state(
            config,
            {
                "human_directives": empty_human_feedback(),
                "coder_instructions": "",
                "patch_agent_output": "",
                "revision_plan": None,
                "targeted_replacement_plan": None,
                "patch_error": "",
                "approved_page_plan": approved_page_plan,
                "manual_layout_compose_enabled": manual_layout_compose_enabled,
                "is_outline_approved": True,
                "is_webpage_approved": False,
                "shell_binding_review": None,
                "shell_manual_selection": None,
                "layout_compose_session": None,
                "layout_compose_update": None,
                "visual_smoke_report": None,
            },
            as_node="outline_review",
        )

        if manual_layout_compose_enabled:
            log("[LayoutCompose] Outline approved. Preparing the optional manual layout compose stage before draft generation...")
        else:
            log("[Coder] Outline approved. Skipping manual layout compose and generating the first draft directly...")
        get_default_hitl_workflow().invoke(None, config=config)
        paused_state = get_default_hitl_workflow().get_state(config)
        paused_values = dict(paused_state.values or {})
        review_stage = str(paused_values.get("review_stage") or "").strip().lower()

        if review_stage == "layout_compose":
            compose_session = normalize_layout_compose_session(paused_values.get("layout_compose_session"))
            if compose_session is None:
                raise RuntimeError("Layout compose stage is active but no compose session payload was found.")
            log("[LayoutCompose] Manual layout compose is ready. Choose sections, reorder blocks, select figures, then continue to the first draft.")
            return (
                "\n".join(run_log_lines),
                current_outline_overview,
                *_review_accordion_updates("webpage"),
                _hidden_preview_update(),
                "",
                *_stage_action_updates("none"),
                *_layout_compose_ui_active(compose_session),
            )

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
            restored_manual_layout_compose_enabled = _normalize_manual_layout_compose_enabled(
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
                    manual_layout_compose_enabled=restored_manual_layout_compose_enabled,
                ),
                *_layout_compose_ui_hidden(),
            )

        if review_stage != "webpage":
            raise RuntimeError(
                f"Workflow paused at unexpected stage '{review_stage or '(empty)'}' after outline approval."
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
        return (
            "\n".join(run_log_lines),
            current_outline_overview,
            *_review_accordion_updates("outline"),
            _visible_preview_update(current_preview),
            str(current_html_path or ""),
            *_stage_action_updates(
                "outline",
                manual_layout_compose_enabled=manual_layout_compose_enabled,
            ),
            *_layout_compose_ui_hidden(),
        )

def request_webpage_revision(
    feedback_text: str,
    feedback_images: Any,
    workflow_thread_id: str,
    current_logs: str,
    current_preview: str | None,
    current_html_path: str,
) -> tuple[Any, ...]:
    run_log_lines = [line for line in str(current_logs or "").splitlines() if line.strip()]

    def log(message: str) -> None:
        print(message)
        run_log_lines.append(message)

    try:
        thread_id = str(workflow_thread_id or "").strip()
        if not thread_id:
            raise ValueError("No paused workflow was found. Generate the first draft before requesting a revision.")

        config = {"configurable": {"thread_id": thread_id}}
        snapshot = get_default_hitl_workflow().get_state(config)
        snapshot_values = dict(snapshot.values or {})
        if str(snapshot_values.get("review_stage") or "") != "webpage":
            raise ValueError("Webpage revisions are only available after the first draft is generated.")

        feedback_payload = build_human_feedback_payload(feedback_text, feedback_images)
        if not (extract_human_feedback_text(feedback_payload) or feedback_payload["images"]):
            raise ValueError("Enter feedback text or upload at least one screenshot before requesting a webpage revision.")

        # Gradio uploads arrive as temporary file paths, so we convert them once into
        # data URLs here and resume LangGraph with a standard multimodal Gemini payload.
        log(
            "[HITL] Webpage revision feedback captured: "
            f"text_length={len(extract_human_feedback_text(feedback_payload))}, screenshots={len(feedback_payload['images'])}"
        )
        get_default_hitl_workflow().update_state(
            config,
            {
                "human_directives": feedback_payload,
                "coder_instructions": "",
                "patch_agent_output": "",
                "revision_plan": None,
                "targeted_replacement_plan": None,
                "patch_error": "",
                "is_webpage_approved": False,
                "visual_smoke_report": None,
            },
            as_node="webpage_review",
        )

        log("[Translator] Resuming workflow through Translator -> EditIntentRouter -> Patch pipeline...")
        get_default_hitl_workflow().invoke(None, config=config)
        paused_state = get_default_hitl_workflow().get_state(config)
        paused_values = dict(paused_state.values or {})
        edit_intent = str(paused_values.get("edit_intent") or "").strip().lower()
        edit_intent_reason = str(paused_values.get("edit_intent_reason") or "").strip()
        patch_error = str(paused_values.get("patch_error") or "").strip()
        patch_agent_output = str(paused_values.get("patch_agent_output") or "").strip()
        targeted_replacement_plan = paused_values.get("targeted_replacement_plan") or {}
        if hasattr(targeted_replacement_plan, "model_dump"):
            targeted_replacement_plan = targeted_replacement_plan.model_dump()
        if isinstance(targeted_replacement_plan, dict):
            replacements_count = len(targeted_replacement_plan.get("replacements") or [])
            style_change_count = len(targeted_replacement_plan.get("style_changes") or [])
            attribute_change_count = len(targeted_replacement_plan.get("attribute_changes") or [])
            override_rule_count = len(targeted_replacement_plan.get("override_css_rules") or [])
            fallback_count = len(targeted_replacement_plan.get("fallback_blocks") or [])
        else:
            replacements_count = 0
            style_change_count = 0
            attribute_change_count = 0
            override_rule_count = 0
            fallback_count = 0
        if edit_intent == "non_patch":
            if edit_intent_reason:
                log(f"[EditIntentRouter] Routed revision to non_patch: {edit_intent_reason}")
            if patch_error:
                log(f"[Patch] Skipped patch flow: {patch_error}")
        elif patch_error:
            log(f"[Patch] Safe fail: {patch_error}")
        elif replacements_count or style_change_count or attribute_change_count or override_rule_count or fallback_count:
            log(
                "[Patch] Anchored DOM revision applied: "
                f"{replacements_count} targeted replacement(s), "
                f"{style_change_count} style change(s), "
                f"{attribute_change_count} attribute change(s), "
                f"{override_rule_count} override css rule(s), "
                f"{fallback_count} regenerated block(s)."
            )
        elif patch_agent_output:
            log(f"[Patch] Anchored DOM revision applied: {patch_agent_output}")
        preview_image_path, entry_html_path = render_current_workflow_preview(paused_values)
        log(f"[Preview] Rendered revised webpage screenshot from {entry_html_path}")
        log("[Webpage] Revised draft ready. Review it, then approve it or request another revision.")
        return (
            "\n".join(run_log_lines),
            *_review_accordion_updates("webpage"),
            _visible_preview_update(preview_image_path),
            entry_html_path,
            *_stage_action_updates("webpage"),
            *_layout_compose_ui_hidden(),
        )
    except Exception as exc:
        log(f"[Error] {exc}")
        return (
            "\n".join(run_log_lines),
            *_review_accordion_updates("webpage"),
            _visible_preview_update(current_preview),
            str(current_html_path or ""),
            *_stage_action_updates(
                "webpage",
                feedback_text_value=feedback_text,
                feedback_images_value=feedback_images,
            ),
            *_layout_compose_ui_hidden(),
        )

def approve_webpage(
    workflow_thread_id: str,
    current_logs: str,
    current_preview: str | None,
    current_html_path: str,
) -> tuple[Any, ...]:
    run_log_lines = [line for line in str(current_logs or "").splitlines() if line.strip()]

    def log(message: str) -> None:
        print(message)
        run_log_lines.append(message)

    try:
        thread_id = str(workflow_thread_id or "").strip()
        if not thread_id:
            raise ValueError("No paused workflow was found. Generate the first draft before approving the webpage.")

        config = {"configurable": {"thread_id": thread_id}}
        snapshot = get_default_hitl_workflow().get_state(config)
        snapshot_values = dict(snapshot.values or {})
        if str(snapshot_values.get("review_stage") or "") != "webpage":
            raise ValueError("The webpage approval step is only available while reviewing a generated draft.")

        get_default_hitl_workflow().update_state(
            config,
            {
                "human_directives": empty_human_feedback(),
                "coder_instructions": "",
                "patch_agent_output": "",
                "revision_plan": None,
                "targeted_replacement_plan": None,
                "patch_error": "",
                "is_webpage_approved": True,
                "visual_smoke_report": None,
            },
            as_node="webpage_review",
        )

        log("[Webpage] Final approval received. Completing the workflow...")
        get_default_hitl_workflow().invoke(None, config=config)
        log("[Done] Webpage approved and workflow completed successfully.")
        return (
            "\n".join(run_log_lines),
            *_review_accordion_updates("webpage"),
            _visible_preview_update(current_preview),
            str(current_html_path or ""),
            *_stage_action_updates("none"),
            *_layout_compose_ui_hidden(),
        )
    except Exception as exc:
        log(f"[Error] {exc}")
        return (
            "\n".join(run_log_lines),
            *_review_accordion_updates("webpage"),
            _visible_preview_update(current_preview),
            str(current_html_path or ""),
            *_stage_action_updates("webpage"),
            *_layout_compose_ui_hidden(),
        )
