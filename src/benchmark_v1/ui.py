from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import gradio as gr

from src.benchmark_v1.core import (
    CONTROL_MODEL_IDS,
    CONTROL_MODEL_LABELS,
    build_run_dir,
    build_e2e_input_package,
    capture_pa_final,
    create_run,
    ensure_blind_model_order,
    find_entry_html,
    get_case,
    ingest_model_changed_files,
    ingest_web_llm_result,
    latest_run_id,
    load_cases,
    load_run,
    list_paper_folder_names,
    list_run_ids,
    materialize_case_branch_instructions,
    record_model_failure,
    record_manual_score,
    restore_pa_baseline,
    save_cases,
)
from src.benchmark_v1.instruction_gen import (
    generate_instruction_candidates,
    normalize_candidate_cases,
    parse_candidate_payload,
)
from src.benchmark_v1.scoring import approve_judge_review_score, score_case_from_manual_inputs


def build_app() -> gr.Blocks:
    paper_choices = list_paper_folder_names()
    default_paper = paper_choices[0] if paper_choices else ""
    run_choices = list_run_ids(default_paper) if default_paper else []
    default_run = run_choices[0] if run_choices else ""

    with gr.Blocks(title="PaperAlchemy Benchmark V1") as demo:
        run_dir_state = gr.State("")
        gr.Markdown("# PaperAlchemy Benchmark V1")
        gr.Markdown(
            "Single-turn reset benchmark harness. Confirm the main PaperAlchemy App is idle before capturing PA final results."
        )

        with gr.Row():
            paper_folder = gr.Dropdown(
                choices=paper_choices,
                value=default_paper,
                label="Paper folder",
                allow_custom_value=True,
            )
            run_id = gr.Dropdown(
                choices=run_choices,
                value=default_run,
                label="Cached run ID",
                allow_custom_value=True,
            )
        with gr.Row():
            refresh_button = gr.Button("Refresh Output Cache")
            load_button = gr.Button("Load Cached Run", variant="primary")
            create_button = gr.Button("Create Fresh Run & Export Baseline")
        run_json = gr.Code(label="Run JSON", language="json")
        baseline_zip = gr.File(label="Baseline anonymous-anchor zip", type="filepath", interactive=False)
        baseline_screenshot = gr.Image(
            label="Baseline screenshot",
            type="filepath",
            interactive=False,
            visible=False,
            show_download_button=False,
        )
        logs = gr.Textbox(label="Logs", lines=10, interactive=False)

        gr.Markdown("## Case Generation")
        generate_button = gr.Button("Generate Candidate Instructions")
        candidates_json = gr.Code(label="Candidate / Cases JSON", language="json")
        save_cases_button = gr.Button("Save Cases")
        case_dropdown = gr.Dropdown(label="Active case", choices=[], interactive=True)
        case_details = gr.Code(label="Selected case instruction", language="json", interactive=False)

        gr.Markdown("## PA Branch")
        with gr.Row():
            restore_button = gr.Button("Restore PA Baseline")
            force_noop = gr.Checkbox(label="Confirm no-op, force PA capture", value=False)
            capture_button = gr.Button("Capture PA Final", variant="primary")

        gr.Markdown("## Control Model Branch")
        build_e2e_button = gr.Button("Build E2E Input Package")
        e2e_package = gr.File(label="E2E input package zip", type="filepath", interactive=False)
        e2e_prompt = gr.Textbox(label="E2E prompt", lines=10, interactive=False)
        model_changed_file_inputs: list[gr.File] = []
        model_changed_path_inputs: list[gr.Textbox] = []
        for model_id in CONTROL_MODEL_IDS:
            model_label = CONTROL_MODEL_LABELS[model_id]
            with gr.Row():
                model_changed_file_inputs.append(
                    gr.File(
                        label=f"{model_label} changed files",
                        file_count="multiple",
                        type="filepath",
                    )
                )
                model_changed_path_inputs.append(
                    gr.Textbox(
                        label=f"{model_label} relative paths",
                        lines=4,
                        placeholder="site/index.html\nsite/css/styles.css\nsite/js/homeSetup.js",
                    )
                )
        ingest_models_button = gr.Button("Ingest Uploaded Changed Files")
        with gr.Row():
            failed_model_dropdown = gr.Dropdown(
                label="Mark failed model",
                choices=list(CONTROL_MODEL_IDS),
                value=CONTROL_MODEL_IDS[-1] if CONTROL_MODEL_IDS else None,
                interactive=True,
            )
            failed_reason = gr.Textbox(
                label="Failure reason",
                value="Model did not provide a directly usable changed-file result.",
                lines=2,
            )
            mark_failed_button = gr.Button("Mark Model Failed")

        gr.Markdown("## Scoring")
        blind_model_id_state = gr.State("")
        gr.Markdown("### Blind LLM Scoring")
        blind_status = gr.Markdown("Ingest model results to start blind LLM scoring.")
        with gr.Row():
            with gr.Column(scale=1):
                blind_completion = gr.Number(label="Completion score", value=50)
                blind_visual = gr.Number(label="Visual score", value=50)
                blind_note = gr.Textbox(label="Evaluator note", lines=4)
                confirm_blind_score_button = gr.Button("Confirm Blind LLM Score", variant="primary")
            with gr.Column(scale=2):
                blind_screenshot = gr.Image(
                    label="LLM result screenshot",
                    type="filepath",
                    interactive=False,
                    show_download_button=False,
                )
                blind_site_link = gr.HTML("")
        gr.Markdown("### PA Manual Score")
        with gr.Row():
            pa_completion = gr.Number(label="PA Completion score", value=50)
            pa_visual = gr.Number(label="PA Visual score", value=50)
        pa_note = gr.Textbox(label="PA evaluator note", lines=2)
        confirm_pa_score_button = gr.Button("Confirm PA Score")
        score_json = gr.Code(label="Score / Judge Review JSON", language="json", interactive=True)
        recompute_score_button = gr.Button("Recompute Active Case Score")
        approve_judge_review_button = gr.Button("Approve Judge Review & Write Score", variant="primary")

        create_button.click(
            fn=_ui_create_run,
            inputs=[paper_folder, run_id],
            outputs=[
                paper_folder,
                run_id,
                run_dir_state,
                run_json,
                baseline_zip,
                baseline_screenshot,
                logs,
                case_dropdown,
                candidates_json,
                case_details,
            ],
            api_name="benchmark_v1_create_run",
        )
        load_button.click(
            fn=_ui_load_run,
            inputs=[paper_folder, run_id],
            outputs=[
                paper_folder,
                run_id,
                run_dir_state,
                run_json,
                baseline_zip,
                baseline_screenshot,
                logs,
                case_dropdown,
                candidates_json,
                case_details,
            ],
            api_name="benchmark_v1_load_run",
        )
        refresh_button.click(
            fn=_ui_refresh_cache,
            inputs=[],
            outputs=[
                paper_folder,
                run_id,
                run_dir_state,
                run_json,
                baseline_zip,
                baseline_screenshot,
                logs,
                case_dropdown,
                candidates_json,
                case_details,
            ],
            api_name="benchmark_v1_refresh_cache",
        )
        paper_folder.change(
            fn=_ui_select_paper_folder,
            inputs=[paper_folder],
            outputs=[
                run_id,
                run_dir_state,
                run_json,
                baseline_zip,
                baseline_screenshot,
                logs,
                case_dropdown,
                candidates_json,
                case_details,
            ],
        )
        generate_button.click(
            fn=_ui_generate_candidates,
            inputs=[run_dir_state],
            outputs=[candidates_json, logs, case_dropdown, case_details],
            api_name="benchmark_v1_generate_candidates",
        )
        save_cases_button.click(
            fn=_ui_save_cases,
            inputs=[run_dir_state, candidates_json],
            outputs=[case_dropdown, logs, case_details],
            api_name="benchmark_v1_save_cases",
        )
        restore_button.click(
            fn=_ui_restore,
            inputs=[run_dir_state],
            outputs=[logs],
            api_name="benchmark_v1_restore_pa_baseline",
        )
        capture_button.click(
            fn=_ui_capture,
            inputs=[run_dir_state, case_dropdown, force_noop],
            outputs=[logs, case_dropdown, candidates_json, case_details],
            api_name="benchmark_v1_capture_pa_final",
        )
        build_e2e_button.click(
            fn=_ui_build_e2e_package,
            inputs=[run_dir_state, case_dropdown],
            outputs=[e2e_package, e2e_prompt, logs, candidates_json, case_details],
            api_name="benchmark_v1_build_e2e_input_package",
        )
        ingest_models_button.click(
            fn=_ui_ingest_model_results,
            inputs=[
                run_dir_state,
                case_dropdown,
                *model_changed_file_inputs,
                *model_changed_path_inputs,
            ],
            outputs=[
                logs,
                candidates_json,
                case_details,
                blind_status,
                blind_model_id_state,
                blind_screenshot,
                blind_site_link,
                blind_completion,
                blind_visual,
                blind_note,
                score_json,
            ],
            api_name="benchmark_v1_ingest_model_results",
        )
        mark_failed_button.click(
            fn=_ui_mark_model_failed,
            inputs=[run_dir_state, case_dropdown, failed_model_dropdown, failed_reason],
            outputs=[
                logs,
                candidates_json,
                case_details,
                blind_status,
                blind_model_id_state,
                blind_screenshot,
                blind_site_link,
                blind_completion,
                blind_visual,
                blind_note,
                score_json,
            ],
            api_name="benchmark_v1_mark_model_failed",
        )
        confirm_pa_score_button.click(
            fn=_ui_confirm_pa_score,
            inputs=[
                run_dir_state,
                case_dropdown,
                pa_completion,
                pa_visual,
                pa_note,
            ],
            outputs=[
                logs,
                candidates_json,
                case_details,
                blind_status,
                blind_model_id_state,
                blind_screenshot,
                blind_site_link,
                blind_completion,
                blind_visual,
                blind_note,
                score_json,
            ],
            api_name="benchmark_v1_confirm_pa_score",
        )
        confirm_blind_score_button.click(
            fn=_ui_confirm_blind_score,
            inputs=[
                run_dir_state,
                case_dropdown,
                blind_model_id_state,
                blind_completion,
                blind_visual,
                blind_note,
            ],
            outputs=[
                logs,
                candidates_json,
                case_details,
                blind_status,
                blind_model_id_state,
                blind_screenshot,
                blind_site_link,
                blind_completion,
                blind_visual,
                blind_note,
                score_json,
            ],
            api_name="benchmark_v1_confirm_blind_score",
        )
        recompute_score_button.click(
            fn=_ui_recompute_score,
            inputs=[run_dir_state, case_dropdown],
            outputs=[
                logs,
                candidates_json,
                case_details,
                blind_status,
                blind_model_id_state,
                blind_screenshot,
                blind_site_link,
                blind_completion,
                blind_visual,
                blind_note,
                score_json,
            ],
            api_name="benchmark_v1_recompute_score",
        )
        approve_judge_review_button.click(
            fn=_ui_approve_judge_review,
            inputs=[run_dir_state, case_dropdown, score_json],
            outputs=[
                logs,
                candidates_json,
                case_details,
                blind_status,
                blind_model_id_state,
                blind_screenshot,
                blind_site_link,
                blind_completion,
                blind_visual,
                blind_note,
                score_json,
            ],
            api_name="benchmark_v1_approve_judge_review",
        )
        demo.load(
            fn=_ui_refresh_cache,
            inputs=[],
            outputs=[
                paper_folder,
                run_id,
                run_dir_state,
                run_json,
                baseline_zip,
                baseline_screenshot,
                logs,
                case_dropdown,
                candidates_json,
                case_details,
            ],
        )
        case_dropdown.change(
            fn=_ui_case_changed,
            inputs=[run_dir_state, case_dropdown],
            outputs=[
                case_details,
                blind_status,
                blind_model_id_state,
                blind_screenshot,
                blind_site_link,
                blind_completion,
                blind_visual,
                blind_note,
                score_json,
            ],
        )
    return demo


def _ui_refresh_cache() -> tuple[Any, ...]:
    papers = list_paper_folder_names()
    paper_folder_name = papers[0] if papers else ""
    if not paper_folder_name:
        return _empty_loaded_state(
            paper_update=gr.update(choices=[], value=None),
            run_update=gr.update(choices=[], value=None),
            log="No testable paper folders found under data/output.",
        )
    return _load_run_response(
        paper_folder_name,
        latest_run_id(paper_folder_name),
        paper_choices=papers,
        log_prefix="Loaded latest cached run.",
    )


def _ui_select_paper_folder(paper_folder_name: str) -> tuple[Any, ...]:
    return _load_run_response(
        paper_folder_name,
        latest_run_id(paper_folder_name),
        log_prefix="Selected paper folder.",
        include_paper_update=False,
    )[1:]


def _ui_create_run(paper_folder_name: str, run_id: str) -> tuple[Any, ...]:
    try:
        clean_paper = _resolve_paper_folder(paper_folder_name)
        requested_run_id = str(run_id or "").strip()
        using_auto_run_id = bool(requested_run_id and build_run_dir(clean_paper, requested_run_id).exists())
        run = create_run(clean_paper, None if using_auto_run_id else requested_run_id or None)
        run_dir = build_run_dir(str(run["paper_folder_name"]), str(run["run_id"]))
        baseline_zip, screenshot = _run_artifacts(run)
        cases = load_cases(run_dir)
        run_choices = list_run_ids(str(run["paper_folder_name"]))
        message = f"Created fresh run {run['run_id']}."
        if using_auto_run_id:
            message += " The selected cached run already existed, so a new run ID was generated."
        return (
            gr.update(choices=list_paper_folder_names(), value=str(run["paper_folder_name"])),
            gr.update(choices=run_choices, value=str(run["run_id"])),
            str(run_dir),
            _json(run),
            baseline_zip,
            screenshot,
            message,
            _case_dropdown_update(cases),
            _json(cases),
            _case_details_json(cases),
        )
    except Exception as exc:
        return _empty_loaded_state(log=f"[Error] {exc}")


def _ui_load_run(paper_folder_name: str, run_id: str) -> tuple[Any, ...]:
    return _load_run_response(paper_folder_name, run_id, log_prefix="Loaded cached run.")


def _load_run_response(
    paper_folder_name: str,
    run_id: str,
    *,
    paper_choices: list[str] | None = None,
    log_prefix: str,
    include_paper_update: bool = True,
) -> tuple[Any, ...]:
    try:
        clean_paper = _resolve_paper_folder(paper_folder_name)
        clean_run_id = str(run_id or "").strip() or latest_run_id(clean_paper)
        if not clean_run_id:
            return _empty_loaded_state(
                paper_update=(
                    gr.update(choices=paper_choices or list_paper_folder_names(), value=clean_paper)
                    if include_paper_update
                    else None
                ),
                run_update=gr.update(choices=[], value=None),
                log=f"No cached Benchmark V1 runs found for {clean_paper}. Create a fresh run first.",
            )
        run_dir = build_run_dir(clean_paper, clean_run_id)
        run = load_run(run_dir)
        cases = load_cases(run_dir)
        baseline_zip, screenshot = _run_artifacts(run)
        run_choices = list_run_ids(str(run["paper_folder_name"]))
        paper_update = (
            gr.update(choices=paper_choices or list_paper_folder_names(), value=str(run["paper_folder_name"]))
            if include_paper_update
            else None
        )
        log = f"{log_prefix} {run['run_id']}."
        if cases:
            log += f" Loaded {len(cases)} cached case(s)."
        return (
            paper_update,
            gr.update(choices=run_choices, value=str(run["run_id"])),
            str(run_dir),
            _json(run),
            baseline_zip,
            screenshot,
            log,
            _case_dropdown_update(cases),
            _json(cases),
            _case_details_json(cases),
        )
    except Exception as exc:
        return _empty_loaded_state(log=f"[Error] {exc}")


def _ui_generate_candidates(run_dir_value: str) -> tuple[str, str, Any, str]:
    try:
        run_dir = _require_run_dir(run_dir_value)
        run = load_run(run_dir)
        baseline_entry = find_entry_html(Path(str(run["baseline_export_dir"])) / "site")
        candidates = generate_instruction_candidates(
            paper_folder_name=str(run["paper_folder_name"]),
            baseline_entry_html=str(baseline_entry),
        )
        candidates = materialize_case_branch_instructions(run_dir, [dict(item) for item in candidates])
        save_cases(run_dir, [dict(item) for item in candidates])
        message = f"Generated and cached {len(candidates)} Vertex AI candidate instruction(s)."
        return (
            _json(candidates),
            message,
            _case_dropdown_update(candidates),
            _case_details_json(candidates),
        )
    except Exception as exc:
        return "", f"[Error] {exc}", gr.update(), ""


def _ui_save_cases(run_dir_value: str, raw_cases_json: str) -> tuple[Any, str, str]:
    try:
        run_dir = _require_run_dir(run_dir_value)
        cases = normalize_candidate_cases(parse_candidate_payload(str(raw_cases_json or "[]")))
        if not isinstance(cases, list):
            raise ValueError("Cases JSON must be a list.")
        cases = materialize_case_branch_instructions(run_dir, [dict(item) for item in cases])
        save_cases(run_dir, [dict(item) for item in cases])
        choices = [str(case["case_id"]) for case in cases]
        return (
            gr.update(choices=choices, value=choices[0] if choices else None),
            f"Saved {len(cases)} case(s).",
            _case_details_json(cases),
        )
    except Exception as exc:
        return gr.update(), f"[Error] {exc}", ""


def _ui_restore(run_dir_value: str) -> str:
    try:
        result = restore_pa_baseline(_require_run_dir(run_dir_value))
        return "Restored PA baseline.\n" + _json(result)
    except Exception as exc:
        return f"[Error] {exc}"


def _ui_capture(
    run_dir_value: str,
    case_id: str,
    force_noop: bool,
    progress: gr.Progress = gr.Progress(),
) -> tuple[str, Any, str, str]:
    try:
        progress(0.05, desc="Preparing PA capture")
        run_dir = _require_run_dir(run_dir_value)
        progress(0.25, desc="Checking PA live site and cooldown")
        result = capture_pa_final(run_dir, case_id, force_noop=bool(force_noop))
        progress(0.85, desc="Refreshing Benchmark case state")
        cases = load_cases(run_dir)
        choices = [str(case["case_id"]) for case in cases]
        return (
            "Captured PA final. Confirmed main App was idle before capture.\n" + _json(result),
            gr.update(choices=choices, value=case_id),
            _json(cases),
            _case_details_json(cases, case_id),
        )
    except Exception as exc:
        return f"[Error] {exc}", gr.update(), "", ""


def _ui_build_e2e_package(
    run_dir_value: str,
    case_id: str,
    progress: gr.Progress = gr.Progress(),
) -> tuple[str | None, str, str, str, str]:
    try:
        progress(0.05, desc="Preparing E2E package")
        run_dir = _require_run_dir(run_dir_value)
        progress(0.25, desc="Copying pruned baseline and LLM working sites")
        result = build_e2e_input_package(run_dir, case_id)
        progress(0.85, desc="Refreshing generated prompt and case metadata")
        cases = load_cases(run_dir)
        return (
            str(result.get("zip_path") or "") or None,
            str(result.get("e2e_prompt") or ""),
            (
                f"Built E2E input package and refreshed {len(CONTROL_MODEL_IDS)} LLM working sites.\n"
                f"Package: {result.get('zip_path')}\n"
                f"LLM site dirs: {len(dict(result.get('llm_site_dirs') or {}))}\n"
                + _json(result)
            ),
            _json(cases),
            _case_details_json(cases, case_id),
        )
    except Exception as exc:
        return None, "", f"[Error] {exc}", "", ""


def _ui_ingest_web(
    run_dir_value: str,
    case_id: str,
    web_zip_path: str,
    allow_partial_class_match: bool,
) -> tuple[str, str]:
    try:
        run_dir = _require_run_dir(run_dir_value)
        result = ingest_web_llm_result(
            run_dir,
            case_id,
            web_zip_path,
            allow_partial_class_match=bool(allow_partial_class_match),
        )
        return "Ingested Web-LLM result.\n" + _json(result), _json(load_cases(run_dir))
    except Exception as exc:
        return f"[Error] {exc}", ""


def _ui_ingest_model_results(
    run_dir_value: str,
    case_id: str,
    *upload_values: Any,
    progress: gr.Progress = gr.Progress(),
) -> tuple[Any, ...]:
    try:
        progress(0.05, desc="Preparing changed-file ingest")
        expected_count = len(CONTROL_MODEL_IDS) * 2
        if len(upload_values) != expected_count:
            raise ValueError("Unexpected model upload input count.")
        file_values = list(upload_values[: len(CONTROL_MODEL_IDS)])
        path_values = list(upload_values[len(CONTROL_MODEL_IDS) :])
        run_dir = _require_run_dir(run_dir_value)
        ingested: list[dict[str, Any]] = []
        selected = [
            (model_id, files, rel_paths)
            for model_id, files, rel_paths in zip(CONTROL_MODEL_IDS, file_values, path_values)
            if files or str(rel_paths or "").strip()
        ]
        for index, (model_id, files, rel_paths) in enumerate(selected, start=1):
            if not files and not str(rel_paths or "").strip():
                continue
            label = CONTROL_MODEL_LABELS[model_id]
            progress(
                0.1 + 0.75 * ((index - 1) / max(1, len(selected))),
                desc=f"Ingesting {label}: resetting site, applying changed files, generating screenshot",
            )
            ingested.append(
                ingest_model_changed_files(
                    run_dir,
                    case_id,
                    files,
                    str(rel_paths or ""),
                    model_id=model_id,
                    model_label=CONTROL_MODEL_LABELS[model_id],
                )
            )
        if not ingested:
            raise ValueError("Upload at least one model changed-file result before ingesting.")
        labels = ", ".join(str(item["model_label"]) for item in ingested)
        warning_count = sum(len(item.get("warnings") or []) for item in ingested)
        detail_lines = [
            (
                f"- {item['model_label']}: {len(item.get('applied_files') or [])} changed file(s), "
                f"screenshot={item.get('screenshot_path') or '(unavailable)'}"
            )
            for item in ingested
        ]
        message = (
            f"Ingested {len(ingested)} changed-file model result(s): {labels}.\n"
            + "\n".join(detail_lines)
        )
        if warning_count:
            message += f" {warning_count} validation warning(s) were recorded."
        progress(0.95, desc="Refreshing blind scoring state")
        return _score_action_response(run_dir, case_id, message)
    except Exception as exc:
        return _score_error_response(f"[Error] {exc}")


def _ui_mark_model_failed(
    run_dir_value: str,
    case_id: str,
    model_id: str,
    failure_reason: str,
) -> tuple[Any, ...]:
    try:
        run_dir = _require_run_dir(run_dir_value)
        clean_model_id = str(model_id or "").strip()
        if clean_model_id not in CONTROL_MODEL_IDS:
            raise ValueError("Select a supported control model to mark as failed.")
        result = record_model_failure(
            run_dir,
            case_id,
            clean_model_id,
            failure_reason=failure_reason,
            model_label=CONTROL_MODEL_LABELS[clean_model_id],
        )
        message = (
            f"Marked {CONTROL_MODEL_LABELS[clean_model_id]} as failed for this case. "
            "The blind scoring step will show no screenshot for this failed result; use 0/0 unless you have a documented reason otherwise."
        )
        return _score_action_response(run_dir, case_id, message + "\n" + _json(result))
    except Exception as exc:
        return _score_error_response(f"[Error] {exc}")


def _ui_confirm_pa_score(
    run_dir_value: str,
    case_id: str,
    pa_completion: float,
    pa_visual: float,
    pa_note: str,
    progress: gr.Progress = gr.Progress(),
) -> tuple[Any, ...]:
    try:
        progress(0.05, desc="Saving PA score")
        run_dir = _require_run_dir(run_dir_value)
        record_manual_score(
            run_dir,
            case_id,
            "paperalchemy",
            completion_score=pa_completion,
            visual_score=pa_visual,
            evaluator_note=pa_note,
        )
        progress(0.35, desc="Checking whether all manual scores are complete")
        payload = _finalize_score_if_complete(run_dir, case_id)
        if payload is not None:
            progress(0.9, desc="Judge review draft was generated")
            score_path = payload.get("score_json_path") or (Path(run_dir) / "scoring" / "_judge_review_drafts" / str(case_id) / "score.json")
            review = dict(payload.get("judge_review") or {})
            message = (
                "Saved PA manual score and generated a pending Judge review draft. "
                f"Review flagged {review.get('flagged_count', 0)} human/Judge gap(s). "
                f"Draft: {score_path}. "
                "Edit Judge values in the JSON if needed, then click Approve Judge Review & Write Score."
            )
            return _score_action_response(run_dir, case_id, message, score_payload=payload)
        return _score_action_response(run_dir, case_id, "Saved PA manual score.")
    except Exception as exc:
        return _score_error_response(f"[Error] {exc}")


def _ui_confirm_blind_score(
    run_dir_value: str,
    case_id: str,
    model_id: str,
    completion: float,
    visual: float,
    note: str,
    progress: gr.Progress = gr.Progress(),
) -> tuple[Any, ...]:
    try:
        progress(0.05, desc="Saving blind LLM score")
        run_dir = _require_run_dir(run_dir_value)
        clean_model_id = str(model_id or "").strip()
        if clean_model_id not in CONTROL_MODEL_IDS:
            raise ValueError("No active blind LLM result is selected. Ingest model results first.")
        record_manual_score(
            run_dir,
            case_id,
            clean_model_id,
            completion_score=completion,
            visual_score=visual,
            evaluator_note=note,
        )
        progress(0.35, desc="Checking whether all blind scores are complete")
        payload = _finalize_score_if_complete(run_dir, case_id)
        if payload is not None:
            progress(0.9, desc="Judge review draft was generated")
            score_path = payload.get("score_json_path") or (Path(run_dir) / "scoring" / "_judge_review_drafts" / str(case_id) / "score.json")
            review = dict(payload.get("judge_review") or {})
            message = (
                "Saved blind LLM score and generated a pending Judge review draft. "
                f"Review flagged {review.get('flagged_count', 0)} human/Judge gap(s). "
                f"Draft: {score_path}. "
                "Edit Judge values in the JSON if needed, then click Approve Judge Review & Write Score."
            )
            return _score_action_response(run_dir, case_id, message, score_payload=payload)
        return _score_action_response(run_dir, case_id, "Saved blind LLM score.")
    except Exception as exc:
        return _score_error_response(f"[Error] {exc}")


def _ui_recompute_score(
    run_dir_value: str,
    case_id: str,
    progress: gr.Progress = gr.Progress(),
) -> tuple[Any, ...]:
    try:
        progress(0.05, desc="Loading saved manual scores and model results")
        run_dir = _require_run_dir(run_dir_value)
        progress(0.25, desc="Recomputing screenshots and automatic metrics")
        payload = score_case_from_manual_inputs(run_dir, case_id, review_only=True)
        progress(0.9, desc="Judge review draft was generated")
        score_path = payload.get("score_json_path") or (Path(run_dir) / "scoring" / "_judge_review_drafts" / str(case_id) / "score.json")
        review = dict(payload.get("judge_review") or {})
        message = (
            f"Recomputed a pending Judge review draft from saved manual scores and uploaded model results. "
            f"Review flagged {review.get('flagged_count', 0)} human/Judge gap(s). "
            f"Draft: {score_path}. "
            "Edit Judge values in the JSON if needed, then click Approve Judge Review & Write Score."
        )
        return _score_action_response(run_dir, case_id, message, score_payload=payload)
    except Exception as exc:
        return _score_error_response(f"[Error] {exc}")


def _ui_approve_judge_review(
    run_dir_value: str,
    case_id: str,
    raw_score_json: str,
    progress: gr.Progress = gr.Progress(),
) -> tuple[Any, ...]:
    try:
        progress(0.05, desc="Validating edited Judge review JSON")
        run_dir = _require_run_dir(run_dir_value)
        progress(0.35, desc="Recomputing totals from approved Judge scores")
        payload = approve_judge_review_score(run_dir, case_id, str(raw_score_json or ""))
        progress(0.9, desc="Score JSON, summary, and archive were written")
        score_path = Path(str(payload.get("score_json_path") or Path(run_dir) / "scoring" / str(case_id) / "score.json"))
        message = (
            "Approved Judge review and finalized this case. "
            f"Score: {score_path}. "
            f"Updated run summary and paper archive under {run_dir.parent.parent / 'benchmark_v1_archive'}."
        )
        return _score_action_response(run_dir, case_id, message, score_payload=payload)
    except Exception as exc:
        return _score_error_response(f"[Error] {exc}")


def _score_action_response(
    run_dir: Path,
    case_id: str,
    log_message: str,
    *,
    score_payload: dict[str, Any] | None = None,
) -> tuple[Any, ...]:
    cases = load_cases(run_dir)
    status, active_model_id, screenshot, site_link, completion, visual, note, score_text = _blind_state_values(
        run_dir,
        case_id,
    )
    if score_payload is not None:
        score_text = _json(score_payload)
    return (
        log_message,
        _json(cases),
        _case_details_json(cases, case_id),
        status,
        active_model_id,
        screenshot,
        site_link,
        completion,
        visual,
        note,
        score_text,
    )


def _score_error_response(log_message: str) -> tuple[Any, ...]:
    return (
        log_message,
        "",
        "",
        "Scoring state unavailable.",
        "",
        None,
        "",
        50,
        50,
        "",
        "",
    )


def _ui_case_changed(run_dir_value: str, case_id: str) -> tuple[Any, ...]:
    try:
        run_dir = _require_run_dir(run_dir_value)
        cases = load_cases(run_dir)
        status, active_model_id, screenshot, site_link, completion, visual, note, score_text = _blind_state_values(
            run_dir,
            case_id,
        )
        return (
            _case_details_json(cases, case_id),
            status,
            active_model_id,
            screenshot,
            site_link,
            completion,
            visual,
            note,
            score_text,
        )
    except Exception:
        return "", "Scoring state unavailable.", "", None, "", 50, 50, "", ""


def _blind_state_values(run_dir: Path, case_id: str) -> tuple[str, str, str | None, str, float, float, str, str]:
    case = get_case(run_dir, case_id)
    order = ensure_blind_model_order(run_dir, case_id)
    case = get_case(run_dir, case_id)
    manual_scores = dict(case.get("manual_score_inputs") or {})
    control_results = dict(case.get("control_model_results") or {})
    score_text = _existing_score_json(case)

    for model_id in order:
        if model_id not in control_results or model_id in manual_scores:
            continue
        result = dict(control_results.get(model_id) or {})
        scored_before = sum(1 for previous in order[: order.index(model_id)] if previous in manual_scores)
        position = scored_before + 1
        failed = str(result.get("input_mode") or "").strip() == "failed"
        screenshot = None if failed else str(result.get("screenshot_path") or "").strip() or None
        site_link = "" if failed else _entry_html_link_html(result.get("entry_html"))
        reason = str(result.get("failure_reason") or "Model did not provide a valid changed-file result.").strip()
        return (
            (
                f"LLM Result {position}/{len(CONTROL_MODEL_IDS)} - failed result has no screenshot."
                if failed
                else f"LLM Result {position}/{len(CONTROL_MODEL_IDS)}"
            ),
            model_id,
            screenshot,
            site_link,
            0 if failed else 50,
            0 if failed else 50,
            reason if failed else "",
            score_text,
        )

    missing = [model_id for model_id in CONTROL_MODEL_IDS if model_id not in control_results]
    if missing:
        ingested_count = len(CONTROL_MODEL_IDS) - len(missing)
        return (
            f"Ingested {ingested_count}/{len(CONTROL_MODEL_IDS)} LLM results; upload remaining changed files before blind scoring.",
            "",
            None,
            "",
            50,
            50,
            "",
            score_text,
        )
    if "paperalchemy" not in manual_scores:
        return (
            "All blind LLM scores are saved. Score PA result to generate Judge review.",
            "",
            None,
            "",
            50,
            50,
            "",
            score_text,
        )
    if not str(case.get("pa_export_name") or "").strip():
        return (
            "All blind LLM scores are saved, but PA final has not been captured. Capture PA Final, then use Recompute Active Case Score.",
            "",
            None,
            "",
            50,
            50,
            "",
            score_text,
        )
    return "All blind LLM scores are saved.", "", None, "", 50, 50, "", score_text


def _entry_html_link_html(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    path = Path(raw)
    if not path.exists() or not path.is_file():
        return ""
    try:
        href = path.resolve().as_uri()
    except ValueError:
        return ""
    return (
        "<div style='margin-top:8px'>"
        f"<a href='{html.escape(href, quote=True)}' target='_blank' rel='noopener noreferrer'>"
        "Open Current LLM Result Site"
        "</a>"
        "</div>"
    )


def _finalize_score_if_complete(run_dir: Path, case_id: str) -> dict[str, Any] | None:
    case = get_case(run_dir, case_id)
    manual_scores = dict(case.get("manual_score_inputs") or {})
    control_results = dict(case.get("control_model_results") or {})
    required_systems = ["paperalchemy", *CONTROL_MODEL_IDS]
    if any(system_name not in manual_scores for system_name in required_systems):
        return None
    if any(model_id not in control_results for model_id in CONTROL_MODEL_IDS):
        return None
    if not str(case.get("pa_export_name") or "").strip():
        return None
    return score_case_from_manual_inputs(run_dir, case_id, review_only=True)


def _existing_score_json(case: dict[str, Any]) -> str:
    raw_pending_path = str(case.get("pending_judge_review_path") or "").strip()
    if raw_pending_path:
        pending_path = Path(raw_pending_path)
        if pending_path.exists() and pending_path.is_file():
            try:
                return pending_path.read_text(encoding="utf-8")
            except OSError:
                pass
    raw_score_path = str(case.get("score_json_path") or "").strip()
    if not raw_score_path:
        return ""
    score_path = Path(raw_score_path)
    if not score_path.exists() or not score_path.is_file():
        return ""
    try:
        return score_path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _ui_case_details(run_dir_value: str, case_id: str) -> str:
    try:
        return _case_details_json(load_cases(_require_run_dir(run_dir_value)), case_id)
    except Exception:
        return ""


def _run_artifacts(run: dict[str, Any]) -> tuple[str | None, str | None]:
    export_dir = Path(str(run["baseline_export_dir"]))
    zip_path = export_dir / str(run["baseline_export_metadata"].get("site_zip_path") or "")
    screenshot = export_dir / str(run["baseline_export_metadata"].get("screenshot_path") or "")
    return str(zip_path) if zip_path.exists() else None, str(screenshot) if screenshot.exists() else None


def _resolve_paper_folder(value: str) -> str:
    clean = str(value or "").strip()
    if clean:
        return clean
    papers = list_paper_folder_names()
    if not papers:
        raise FileNotFoundError("No testable paper folders found under data/output.")
    return papers[0]


def _require_run_dir(value: str) -> Path:
    run_dir = Path(str(value or "").strip())
    if not run_dir.exists():
        raise FileNotFoundError("Load or create a Benchmark V1 run first.")
    return run_dir


def _case_dropdown_update(cases: list[dict[str, Any]]) -> Any:
    choices = [str(case["case_id"]) for case in cases]
    return gr.update(choices=choices, value=choices[0] if choices else None)


def _case_details_json(cases: list[dict[str, Any]], case_id: str | None = None) -> str:
    if not cases:
        return ""
    clean_case_id = str(case_id or "").strip()
    selected = next((case for case in cases if str(case.get("case_id") or "") == clean_case_id), cases[0])
    detail = {
        "case_id": selected.get("case_id"),
        "task_type": selected.get("task_type"),
        "subcategory": selected.get("subcategory"),
        "difficulty": selected.get("difficulty"),
        "difficulty_reason": selected.get("difficulty_reason"),
        "category_label_zh": selected.get("category_label_zh"),
        "subcategory_label_zh": selected.get("subcategory_label_zh"),
        "instruction": selected.get("instruction"),
        "instruction_zh": selected.get("instruction_zh"),
        "paperalchemy_instruction": selected.get("pa_instruction"),
        "paperalchemy_instruction_zh": selected.get("pa_instruction_zh"),
        "e2e_instruction": selected.get("e2e_instruction"),
        "e2e_instruction_zh": selected.get("e2e_instruction_zh"),
        "target_hint": selected.get("target_hint"),
        "target_hint_zh": selected.get("target_hint_zh"),
        "expected_observable": selected.get("expected_observable"),
        "expected_observable_zh": selected.get("expected_observable_zh"),
        "pdf_evidence": selected.get("pdf_evidence"),
        "web_evidence": selected.get("web_evidence"),
        "forbidden_changes": selected.get("forbidden_changes") or [],
        "target_assets": selected.get("target_assets") or [],
        "paper_pdf_path": selected.get("paper_pdf_path"),
        "e2e_input_zip_path": selected.get("e2e_input_zip_path"),
        "llm_site_dirs": selected.get("llm_site_dirs") or {},
        "control_model_results": selected.get("control_model_results") or {},
        "manual_score_inputs": selected.get("manual_score_inputs") or {},
    }
    return _json(detail)


def _empty_loaded_state(
    *,
    paper_update: Any | None = None,
    run_update: Any | None = None,
    log: str,
) -> tuple[Any, ...]:
    return (
        paper_update if paper_update is not None else gr.update(),
        run_update if run_update is not None else gr.update(),
        "",
        "",
        None,
        None,
        log,
        gr.update(choices=[], value=None),
        "",
        "",
    )


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False)
