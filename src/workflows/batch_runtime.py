from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import gradio as gr

from src.agents.coder import run_coder_agent_with_diagnostics
from src.agents.planner import run_planner_agent
from src.agents.reader import run_reader_agent
from src.contracts.schemas import CoderArtifact
from src.services.artifact_store import (
    get_template_profile_output_path,
    load_cached_structured_data,
    save_coder_artifact,
    save_page_plan,
    save_structured_data,
    save_template_profile,
)
from src.services.human_feedback import empty_human_feedback
from src.services.preview_service import build_final_preview_path, take_local_screenshot
from src.ui.constraints import (
    INPUT_DIR,
    OUTPUT_DIR,
    build_generation_constraints,
    ensure_parsed_output,
    ensure_template_assets,
    get_default_pdf,
)
from src.ui.formatters import _visual_smoke_feedback_text
from src.workflows.hitl_nodes import normalize_coder_artifact
from src.template.compile import prepare_template_compile_bundle

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def run_langgraph_batch(
    pdf_filename: str,
    user_constraints: dict[str, str],
    selected_candidate: dict[str, Any],
    ranked_candidates: list[dict[str, Any]],
    log: Callable[[str], None],
) -> CoderArtifact:
    safe_pdf_filename = str(pdf_filename or "").strip()
    if not safe_pdf_filename:
        raise ValueError("A valid PDF filename is required.")

    input_path = INPUT_DIR / safe_pdf_filename
    if not input_path.exists():
        raise FileNotFoundError(f"Input PDF not found: {input_path}")

    template_id = str(selected_candidate.get("template_id") or "").strip()
    template_path = str(selected_candidate.get("template_path") or "").strip()
    entry_html = str(selected_candidate.get("entry_html") or "").strip()
    if not (template_id and template_path and entry_html):
        raise ValueError("The selected template is missing template_id, template_path, or entry_html.")

    log(f"[UI] Using input PDF: {safe_pdf_filename}")
    log(f"[UI] Confirmed template: {template_id}")
    log(f"[UI] Template root: {template_path}")
    log(f"[UI] Entry HTML: {entry_html}")
    log(f"[UI] User constraints: {json.dumps(user_constraints, ensure_ascii=False)}")

    synced_assets = ensure_template_assets()
    log(f"[Assets] Template resources ready at {synced_assets.templates_dir}")

    paper_folder_name = Path(safe_pdf_filename).stem
    output_dir = OUTPUT_DIR / paper_folder_name
    structured_json_path = output_dir / "structured_paper.json"
    planner_json_path = output_dir / "page_plan.json"
    coder_json_path = output_dir / "coder_artifact.json"

    if not ensure_parsed_output(safe_pdf_filename, output_dir):
        raise RuntimeError("PDF parsing failed. Please inspect the parser output logs.")

    structured_data = load_cached_structured_data(structured_json_path)
    if not structured_data:
        log("[Reader] Running reader agent...")
        structured_data = run_reader_agent(paper_folder_name)
        if not structured_data:
            raise RuntimeError("Reader agent failed to produce structured paper data.")
        save_structured_data(structured_json_path, structured_data)
        log(f"[Reader] Saved structured paper to {structured_json_path}")
    else:
        log(f"[Reader] Reused cached structured paper from {structured_json_path}")

    generation_constraints = build_generation_constraints(
        synced_assets=synced_assets,
        selected_candidate=selected_candidate,
        ranked_candidates=ranked_candidates,
    )
    template_candidates, selected_template, template_profile, _, cache_hit, _ = prepare_template_compile_bundle(
        project_root=PROJECT_ROOT,
        generation_constraints=generation_constraints,
        user_constraints=user_constraints,
        synced_assets=synced_assets,
        allow_llm=bool(generation_constraints.get("template_compile_use_llm", True)),
        force_recompile=bool(generation_constraints.get("force_template_recompile")),
    )
    template_profile_path = get_template_profile_output_path(paper_folder_name)
    save_template_profile(template_profile_path, template_profile)
    log(f"[TemplateCompile] Compiled profile for {selected_template.template_id} (cache_hit={cache_hit})")

    log("[Planner] Running template-first planner graph with designated template...")
    page_plan = run_planner_agent(
        paper_folder_name=paper_folder_name,
        structured_data=structured_data,
        generation_constraints=generation_constraints,
        user_constraints=user_constraints,
        max_retry=2,
        template_candidates=template_candidates,
        selected_template=selected_template,
        template_profile=template_profile,
    )
    if not page_plan:
        raise RuntimeError("Planner agent failed to produce a page plan.")

    save_page_plan(planner_json_path, page_plan)
    log(f"[Planner] Saved page plan to {planner_json_path}")

    log("[Coder] Running coder agent...")
    coder_artifact, visual_smoke_report, resolved_page_plan = run_coder_agent_with_diagnostics(
        paper_folder_name=paper_folder_name,
        structured_data=structured_data,
        page_plan=page_plan,
        human_directives=empty_human_feedback(),
        coder_instructions="",
        max_retry=2,
        template_profile=template_profile,
    )
    if not coder_artifact:
        raise RuntimeError("Coder agent failed to build the final webpage.")

    if resolved_page_plan is not None:
        page_plan = resolved_page_plan
        save_page_plan(planner_json_path, page_plan)
    save_coder_artifact(coder_json_path, coder_artifact)
    log(f"[Coder] Generated entry html at {coder_artifact.entry_html}")
    smoke_feedback = _visual_smoke_feedback_text(visual_smoke_report)
    if smoke_feedback:
        log(smoke_feedback)
    if visual_smoke_report and visual_smoke_report.suggested_recovery == "rerun_planner":
        log("[Planner] Visual smoke recommends rerunning planner before accepting this batch draft.")
    log("[Done] Generation completed successfully.")
    return coder_artifact

def render_current_workflow_preview(state_values: dict[str, Any]) -> tuple[str, str]:
    coder_artifact = normalize_coder_artifact(state_values.get("coder_artifact"))
    if not coder_artifact:
        raise RuntimeError("Workflow preview is unavailable because coder_artifact is missing.")
    entry_html_path = Path(coder_artifact.entry_html).resolve()
    preview_image_path = take_local_screenshot(
        str(entry_html_path),
        str(build_final_preview_path(entry_html_path)),
    )
    if not preview_image_path:
        raise RuntimeError(f"Failed to render workflow preview for {entry_html_path}")
    return preview_image_path, str(entry_html_path)

def confirm_and_start_generation(
    pdf_filename: str | None,
    selected_label: str | None,
    search_state: dict[str, Any] | None,
    selected_candidate_state: dict[str, Any] | None,
    current_logs: str,
) -> tuple[str, str | None]:
    selected_candidate = selected_candidate_state or resolve_selected_candidate(selected_label, search_state)
    if not selected_candidate:
        raise gr.Error("Select and preview one of the Top 5 candidate templates before starting generation.")

    chosen_pdf = str(pdf_filename or "").strip() or get_default_pdf()
    user_constraints = dict((search_state or {}).get("user_constraints") or {})
    ranked_candidates = list((search_state or {}).get("candidates") or [])

    run_log_lines = [line for line in str(current_logs or "").splitlines() if line.strip()]

    def log(message: str) -> None:
        print(message)
        run_log_lines.append(message)

    preview_image_path: str | None = None

    try:
        coder_artifact = run_langgraph_batch(
            pdf_filename=chosen_pdf,
            user_constraints=user_constraints,
            selected_candidate=selected_candidate,
            ranked_candidates=ranked_candidates,
            log=log,
        )
        entry_html_path = Path(coder_artifact.entry_html).resolve()
        preview_image_path = take_local_screenshot(
            str(entry_html_path),
            str(build_final_preview_path(entry_html_path)),
        )
        if not preview_image_path:
            raise RuntimeError(f"Failed to render final screenshot for {entry_html_path}")
        log(f"[Preview] Rendered generated webpage screenshot from {entry_html_path}")
    except Exception as exc:
        log(f"[Error] {exc}")
        preview_image_path = None

    return "\n".join(run_log_lines), preview_image_path
