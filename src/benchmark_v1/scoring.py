from __future__ import annotations

import difflib
import csv
import html
import json
import math
import re
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from langchain_core.messages import HumanMessage, SystemMessage

from src.benchmark_v1.core import (
    CONTROL_MODEL_IDS,
    CONTROL_MODEL_LABELS,
    RUN_JSON,
    build_v1_runs_dir,
    find_entry_html,
    load_cases,
    load_run,
    safe_extract_zip,
    should_exclude_benchmark_member,
    update_case,
)
from src.benchmark_v1.render import benchmark_render_config, take_benchmark_screenshot
from src.services.human_feedback import build_human_feedback_payload, build_multimodal_message_content
from src.services.llm import get_llm
from src.services.experiment_export import build_experiment_export_dir
from src.utils.html_utils import read_text_with_fallback

TEXT_SUFFIXES = {".css", ".html", ".js", ".json", ".md", ".txt"}
SUMMARY_SYSTEM_IDS = {"paperalchemy", *CONTROL_MODEL_IDS}
MINIMALITY_CHANGE_MULTIPLIER = 1500.0
DEFAULT_WEIGHTS = {
    "completion": 0.30,
    "preservation": 0.20,
    "minimality": 0.20,
    "sanity": 0.10,
    "visual": 0.20,
}
PRESERVATION_SKIPPED_WEIGHTS = {
    "completion": 0.40,
    "minimality": 0.25,
    "sanity": 0.10,
    "visual": 0.25,
}
JUDGE_RESULT_VERSION = "gemini_judge_target_assets_v2"
JUDGE_SYSTEM_PROMPT = """You are an anonymized visual judge for Benchmark V1.
Evaluate whether a candidate paper webpage result satisfies the task compared with the baseline.
You must not infer or mention which system produced the candidate.
Return JSON only."""


def score_case(
    run_dir: Path,
    case_id: str,
    *,
    completion_scores: dict[str, float] | None = None,
    visual_scores: dict[str, float] | None = None,
    evaluator_notes: dict[str, str] | None = None,
    review_only: bool = False,
) -> dict[str, Any]:
    run = load_run(run_dir)
    case = _load_case(run_dir, case_id)
    workspace = (
        Path(run_dir) / "scoring" / "_judge_review_drafts" / str(case_id)
        if review_only
        else Path(run_dir) / "scoring" / str(case_id)
    )
    cached_judge_results = _load_cached_judge_results(run_dir, case_id)
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    baseline_zip = Path(str(run["baseline_zip_path"]))
    baseline_dir = workspace / "baseline"
    safe_extract_zip(baseline_zip, baseline_dir)
    baseline_entry = find_entry_html(baseline_dir)

    results: dict[str, Any] = {}
    completion_scores = completion_scores or {}
    visual_scores = visual_scores or {}
    evaluator_notes = evaluator_notes or {}
    for system_name, final_zip in _iter_available_final_zips(run, case):
        system_dir = workspace / system_name
        safe_extract_zip(final_zip, system_dir)
        result = score_final_site(
            baseline_dir=baseline_dir,
            baseline_entry_html=baseline_entry,
            final_dir=system_dir,
            case=case,
            screenshot_dir=workspace / "screenshots" / system_name,
            completion_score=completion_scores.get(system_name),
            visual_score=visual_scores.get(system_name),
            cached_judge_result=cached_judge_results.get(system_name),
        )
        result["system_label"] = _system_label(system_name)
        results[system_name] = result

    for system_name, failed_result in _iter_failed_model_results(case):
        result = score_failed_model(failed_result)
        result["system_label"] = _system_label(system_name)
        results[system_name] = result

    if not results:
        raise ValueError(f"Case {case_id} has no captured final result to score.")

    score_payload = {
        "schema_version": "paperalchemy-benchmark-v1-score",
        "score_status": "pending_judge_review" if review_only else "final",
        "paper_folder_name": run["paper_folder_name"],
        "run_id": run["run_id"],
        "case_id": case_id,
        "case": case,
        "results": results,
        "evaluator_notes": evaluator_notes,
        "judge_review": _build_judge_review(results),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    score_path = workspace / "score.json"
    score_path.write_text(json.dumps(score_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    review_html_path = write_review_bundle(workspace, score_payload)
    score_payload["score_json_path"] = str(score_path)
    score_payload["review_html_path"] = str(review_html_path)
    if review_only:
        score_payload["draft_workspace_path"] = str(workspace)
        update_case(
            run_dir,
            case_id,
            {
                "pending_judge_review_path": str(score_path),
                "pending_judge_review_created_at": score_payload["created_at"],
            },
        )
        score_path.write_text(json.dumps(score_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        update_case(
            run_dir,
            case_id,
            {
                "score_json_path": str(score_path),
                "review_html_path": str(review_html_path),
                "last_scored_at": score_payload["created_at"],
                "pending_judge_review_path": "",
            },
        )
        write_run_summary(run_dir)
        write_paper_archive(str(run["paper_folder_name"]))
    return score_payload


def score_case_from_manual_inputs(run_dir: Path, case_id: str, *, review_only: bool = False) -> dict[str, Any]:
    case = _load_case(run_dir, case_id)
    manual_scores = dict(case.get("manual_score_inputs") or {})
    control_model_results = dict(case.get("control_model_results") or {})
    required_systems = ["paperalchemy", *CONTROL_MODEL_IDS]
    missing_scores = [system_name for system_name in required_systems if system_name not in manual_scores]
    missing_results = [model_id for model_id in CONTROL_MODEL_IDS if model_id not in control_model_results]
    if missing_scores:
        raise ValueError(f"Cannot recompute score; missing manual score(s): {', '.join(missing_scores)}.")
    if missing_results:
        raise ValueError(f"Cannot recompute score; missing model result(s): {', '.join(missing_results)}.")
    if not str(case.get("pa_export_name") or "").strip():
        raise ValueError("Cannot recompute score; PA final has not been captured for this case.")

    completion_scores = {
        system_name: float(dict(manual_scores.get(system_name) or {}).get("completion", 50))
        for system_name in required_systems
    }
    visual_scores = {
        system_name: float(dict(manual_scores.get(system_name) or {}).get("visual", 50))
        for system_name in required_systems
    }
    evaluator_notes = {
        system_name: str(dict(manual_scores.get(system_name) or {}).get("note") or "")
        for system_name in required_systems
    }
    return score_case(
        run_dir,
        case_id,
        completion_scores=completion_scores,
        visual_scores=visual_scores,
        evaluator_notes=evaluator_notes,
        review_only=review_only,
    )


def approve_judge_review_score(run_dir: Path, case_id: str, raw_score_payload: str | dict[str, Any]) -> dict[str, Any]:
    run = load_run(run_dir)
    if isinstance(raw_score_payload, str):
        payload = json.loads(raw_score_payload)
    else:
        payload = dict(raw_score_payload)
    if str(payload.get("case_id") or "") != str(case_id):
        raise ValueError("Judge review payload does not match the active case.")
    if str(payload.get("run_id") or "") != str(run.get("run_id") or ""):
        raise ValueError("Judge review payload does not match the active run.")

    draft_workspace = _resolve_judge_review_draft_workspace(run_dir, case_id, payload)
    scoring_root = (Path(run_dir) / "scoring").resolve()
    final_workspace = (scoring_root / str(case_id)).resolve()
    if final_workspace.parent != scoring_root:
        raise ValueError(f"Refusing to write score outside scoring workspace: {final_workspace}")
    if draft_workspace.exists() and draft_workspace.is_dir():
        if final_workspace.exists():
            shutil.rmtree(final_workspace)
        shutil.copytree(draft_workspace, final_workspace)
        payload = _replace_path_prefix(payload, str(draft_workspace), str(final_workspace))
    else:
        final_workspace.mkdir(parents=True, exist_ok=True)

    payload["score_status"] = "final"
    payload["approved_judge_review_at"] = datetime.now(timezone.utc).isoformat()
    payload.pop("draft_workspace_path", None)
    _apply_judge_review_item_edits(payload)
    payload["judge_review"] = _build_judge_review(dict(payload.get("results") or {}))
    _recompute_payload_totals_after_judge_review(payload)

    score_path = final_workspace / "score.json"
    review_html_path = write_review_bundle(final_workspace, payload)
    payload["score_json_path"] = str(score_path)
    payload["review_html_path"] = str(review_html_path)
    score_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    update_case(
        run_dir,
        case_id,
        {
            "score_json_path": str(score_path),
            "review_html_path": str(review_html_path),
            "last_scored_at": payload["approved_judge_review_at"],
            "pending_judge_review_path": "",
        },
    )
    write_run_summary(run_dir)
    write_paper_archive(str(run["paper_folder_name"]))
    return payload


def _resolve_judge_review_draft_workspace(run_dir: Path, case_id: str, payload: dict[str, Any]) -> Path:
    expected = (Path(run_dir) / "scoring" / "_judge_review_drafts" / str(case_id)).resolve()
    raw_draft = str(payload.get("draft_workspace_path") or "").strip()
    if raw_draft:
        candidate = Path(raw_draft).resolve()
    else:
        case = _load_case(run_dir, case_id)
        raw_pending = str(case.get("pending_judge_review_path") or "").strip()
        if not raw_pending:
            raise ValueError("Judge review draft path is missing; recompute the active case score first.")
        candidate = Path(raw_pending).resolve().parent
    if candidate != expected:
        raise ValueError(f"Refusing to approve Judge review from unexpected draft workspace: {candidate}")
    if not candidate.exists() or not candidate.is_dir():
        raise FileNotFoundError(f"Judge review draft workspace does not exist: {candidate}")
    return candidate


def _apply_judge_review_item_edits(payload: dict[str, Any]) -> None:
    results = dict(payload.get("results") or {})
    review = dict(payload.get("judge_review") or {})
    for item in list(review.get("items") or []):
        if not isinstance(item, dict):
            continue
        system_name = str(item.get("system") or "").strip()
        if not system_name or system_name not in results:
            continue
        result = dict(results.get(system_name) or {})
        if int(result.get("gate") or 0) == 0:
            continue
        judge = dict(result.get("gemini_judge") or {})
        current_completion_item = _clamp_score((result.get("completion") or {}).get("judge_score"))
        current_visual_item = _clamp_score((result.get("visual") or {}).get("judge_score"))
        if "judge_completion" in item and _clamp_score(item.get("judge_completion")) != current_completion_item:
            judge["completion"] = _clamp_score(item.get("judge_completion"))
        if "judge_visual" in item and _clamp_score(item.get("judge_visual")) != current_visual_item:
            judge["visual"] = _clamp_score(item.get("judge_visual"))
        if str(item.get("judge_reason") or "").strip():
            judge["reason"] = str(item.get("judge_reason") or "").strip()
        result["gemini_judge"] = judge
        results[system_name] = result
    payload["results"] = results


def score_failed_model(model_result: dict[str, Any]) -> dict[str, Any]:
    reason = str(model_result.get("failure_reason") or "Model did not provide a valid changed-file result.").strip()
    return {
        "gate": 0,
        "gate_errors": [reason],
        "input_mode": "failed",
        "failure_reason": reason,
        "technical_failure": True,
        "total_score": 0.0,
        "uncapped_score": 0.0,
        "score_caps": [{"reason": "technical_failure", "cap": 0.0}],
        "gemini_judge": None,
        "screenshots": {
            "baseline": "",
            "final": "",
        },
    }


def judge_candidate_result(
    *,
    case: dict[str, Any],
    baseline_screenshot: str,
    final_screenshot: str,
) -> dict[str, Any]:
    instruction = str(case.get("instruction") or case.get("e2e_instruction") or "").strip()
    expected = str(case.get("expected_observable") or "").strip()
    target_asset_paths = _judge_target_asset_paths(case)
    target_asset_block = (
        "\n".join(f"- Image {index}: provided target asset {path.name}" for index, path in enumerate(target_asset_paths, start=2))
        if target_asset_paths
        else "(none)"
    )
    final_image_index = len(target_asset_paths) + 2
    prompt = f"""Judge this anonymized Benchmark V1 webpage edit from screenshots.

Task instruction:
{instruction or "(not provided)"}

Expected observable:
{expected or "(not provided)"}

Images:
1. Baseline screenshot
{target_asset_block}
{final_image_index}. Candidate Result screenshot

Important:
- If provided target asset images are listed above, judge whether the candidate uses those exact visual assets.
- Do not infer the expected target appearance only from words like table, figure, chart, or logo in the instruction.
- The final image is always the candidate result screenshot.
- Compare the candidate screenshot against the baseline only to identify what changed.
- Judge only the modified area and the direct visual consequences of the candidate edit.
- Ignore all pre-existing baseline content, layout, fixed footers, sticky UI elements, cropping, browser/screenshot rendering artifacts, compression artifacts, and unrelated visual issues.
- Do not penalize anything that appears unchanged from the baseline, even if it looks imperfect.
- Only penalize issues that are caused by the candidate modification, made worse by it, or directly interfere with the requested edited area.
- The Visual score should reflect the quality, stability, readability, and integration of the modified portion only, not the overall webpage.

Rubric:
Completion:
- 100 = the requested modification is fully completed in the correct area.
- 85 = the core modification is completed with minor scope/detail issues.
- 70 = the intended change is visible but incomplete, weak, or only partially satisfies the request.
- 50 = a related change exists, but it does not really solve the requested task.
- 25 = the change is in the wrong area, barely effective, or mostly unrelated.
- 0 = the requested change is not visible, invalid, impossible to judge, or clearly not completed.

Visual:
- 100 = the modified portion looks natural, stable, professional, and well integrated.
- 85 = the modified portion looks good with only minor visual issues.
- 70 = the modified portion is acceptable but visibly rough, weak, or slightly inconsistent.
- 50 = the modified portion looks awkward, unstable, or noticeably inconsistent.
- 25 = the candidate edit harms local readability, spacing, alignment, or layout.
- 0 = the candidate edit is invisible, broken, or impossible to judge.

Return strict JSON only:
{{"completion": <0-100>, "visual": <0-100>, "reason": "<short reason>", "confidence": <0-1>}}
"""
    image_payload = build_human_feedback_payload(
        "",
        [baseline_screenshot, *[str(path) for path in target_asset_paths], final_screenshot],
    )
    if len(image_payload["images"]) < 2:
        raise ValueError("Gemini judge requires both baseline and candidate result screenshots.")
    llm = get_llm(
        temperature=1.0,
        use_smart_model=False,
        request_timeout=180,
        retries=3,
        streaming=True,
    )
    response = llm.invoke(
        [
            SystemMessage(content=JUDGE_SYSTEM_PROMPT),
            HumanMessage(content=build_multimodal_message_content(prompt, image_payload["images"])),
        ]
    )
    raw_text = _response_text(response)
    return _normalize_judge_result(_extract_json_object(raw_text), raw_text=raw_text, cached=False)


def score_final_site(
    *,
    baseline_dir: Path,
    baseline_entry_html: Path,
    final_dir: Path,
    case: dict[str, Any],
    screenshot_dir: Path,
    completion_score: float | None = None,
    visual_score: float | None = None,
    cached_judge_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    gate_errors: list[str] = []
    sanity_warnings: list[str] = []
    final_entry: Path | None
    try:
        final_entry = find_entry_html(final_dir)
    except Exception as exc:
        final_entry = None
        gate_errors.append(str(exc))

    screenshot_dir.mkdir(parents=True, exist_ok=True)
    baseline_screenshot = take_benchmark_screenshot(
        str(baseline_entry_html),
        str(screenshot_dir / "baseline.png"),
    )
    final_screenshot = ""
    if final_entry is not None:
        final_screenshot = take_benchmark_screenshot(str(final_entry), str(screenshot_dir / "final.png"))
    if not baseline_screenshot:
        sanity_warnings.append("Baseline screenshot unavailable.")
    if not final_screenshot:
        gate_errors.append("Final screenshot unavailable.")

    if gate_errors:
        return {
            "gate": 0,
            "gate_errors": gate_errors,
            "technical_failure": True,
            "total_score": 0.0,
            "uncapped_score": 0.0,
            "score_caps": [{"reason": "technical_failure", "cap": 0.0}],
            "gemini_judge": None,
            "screenshots": {
                "baseline": baseline_screenshot,
                "final": final_screenshot,
            },
        }

    minimality = compute_minimality_score(
        baseline_dir,
        final_dir,
        baseline_entry_html=baseline_entry_html,
        final_entry_html=final_entry,
        target_selectors=_minimality_target_selectors(case),
    )
    preservation = compute_preservation_score(
        baseline_entry_html=baseline_entry_html,
        final_entry_html=final_entry,
        target_selectors=list(case.get("target_selectors") or []),
    )
    sanity = compute_sanity_score(final_dir, extra_warnings=sanity_warnings)
    human_completion = _score_or_default(completion_score)
    human_visual = _score_or_default(visual_score)
    if _should_skip_judge_for_case(case):
        judge = _skipped_interactivity_judge(
            human_completion=human_completion,
            human_visual=human_visual,
        )
    elif cached_judge_result is not None:
        judge = _normalize_judge_result(cached_judge_result, raw_text="", cached=True)
    else:
        judge = _normalize_judge_result(
            judge_candidate_result(
                case=case,
                baseline_screenshot=baseline_screenshot,
                final_screenshot=final_screenshot,
            ),
            raw_text="",
            cached=False,
        )
    judge_completion = _clamp_score(judge.get("completion"))
    judge_visual = _clamp_score(judge.get("visual"))
    completion = round((human_completion + judge_completion) / 2.0, 2)
    visual = round((human_visual + judge_visual) / 2.0, 2)
    total = compute_weighted_score(
        completion=completion,
        preservation=preservation["score"],
        minimality=minimality["score"],
        sanity=sanity["score"],
        visual=visual,
        preservation_skipped=bool(preservation["skipped"]),
    )
    return {
        "gate": 1,
        "human_completion": {"score": human_completion, "source": "manual_or_default"},
        "human_visual": {"score": human_visual, "source": "manual_or_default"},
        "gemini_judge": judge,
        "completion": {
            "score": completion,
            "source": "human_gemini_average",
            "human_score": human_completion,
            "judge_score": judge_completion,
        },
        "preservation": preservation,
        "minimality": minimality,
        "sanity": sanity,
        "visual": {
            "score": visual,
            "source": "human_gemini_average",
            "human_score": human_visual,
            "judge_score": judge_visual,
        },
        "total_score": total["total_score"],
        "uncapped_score": total["uncapped_score"],
        "score_caps": total["score_caps"],
        "weights": total["weights"],
        "render_config": benchmark_render_config(),
        "screenshots": {
            "baseline": baseline_screenshot,
            "final": final_screenshot,
        },
    }


def _should_skip_judge_for_case(case: dict[str, Any]) -> bool:
    return str(case.get("task_type") or "").strip() == "interactivity_functional"


def _minimality_target_selectors(case: dict[str, Any]) -> list[str]:
    if str(case.get("task_type") or "").strip() != "content_multimodal":
        return []
    return list(case.get("target_selectors") or [])


def _skipped_interactivity_judge(*, human_completion: float, human_visual: float) -> dict[str, Any]:
    return {
        "judge_version": JUDGE_RESULT_VERSION,
        "completion": _clamp_score(human_completion),
        "visual": _clamp_score(human_visual),
        "reason": (
            "Skipped Gemini Judge for interactivity_functional case; "
            "human score used as judge-equivalent score."
        ),
        "confidence": 1.0,
        "source": "skipped_interactivity",
    }


def compute_weighted_score(
    *,
    completion: float,
    preservation: float | None,
    minimality: float,
    sanity: float,
    visual: float,
    preservation_skipped: bool,
) -> dict[str, Any]:
    if preservation_skipped:
        weights = PRESERVATION_SKIPPED_WEIGHTS
        weighted = (
            weights["completion"] * _clamp_score(completion)
            + weights["minimality"] * _clamp_score(minimality)
            + weights["sanity"] * _clamp_score(sanity)
            + weights["visual"] * _clamp_score(visual)
        )
    else:
        weights = DEFAULT_WEIGHTS
        weighted = (
            weights["completion"] * _clamp_score(completion)
            + weights["preservation"] * _clamp_score(preservation)
            + weights["minimality"] * _clamp_score(minimality)
            + weights["sanity"] * _clamp_score(sanity)
            + weights["visual"] * _clamp_score(visual)
        )
    caps: list[dict[str, Any]] = []
    completion_value = _clamp_score(completion)
    visual_value = _clamp_score(visual)
    if completion_value == 0:
        caps.append({"reason": "completion_eq_0", "cap": 25.0})
    elif completion_value < 50:
        caps.append({"reason": "completion_lt_50", "cap": 55.0})
    elif completion_value < 75:
        caps.append({"reason": "completion_lt_75", "cap": 70.0})
    elif completion_value < 90:
        caps.append({"reason": "completion_lt_90", "cap": 85.0})
    if visual_value == 0:
        caps.append({"reason": "visual_eq_0", "cap": 35.0})
    elif visual_value < 50:
        caps.append({"reason": "visual_lt_50", "cap": 65.0})

    capped = min([weighted, *[float(cap["cap"]) for cap in caps]])
    return {
        "total_score": round(capped, 2),
        "uncapped_score": round(weighted, 2),
        "score_caps": caps,
        "weights": dict(weights),
    }


def _build_judge_review(results: dict[str, Any]) -> dict[str, Any]:
    threshold = 25.0
    items: list[dict[str, Any]] = []
    for system_name, result in dict(results or {}).items():
        if int(result.get("gate") or 0) == 0:
            continue
        judge = dict(result.get("gemini_judge") or {})
        human_completion = _clamp_score((result.get("human_completion") or {}).get("score"))
        human_visual = _clamp_score((result.get("human_visual") or {}).get("score"))
        judge_completion = _clamp_score(judge.get("completion"))
        judge_visual = _clamp_score(judge.get("visual"))
        completion_gap = round(abs(human_completion - judge_completion), 2)
        visual_gap = round(abs(human_visual - judge_visual), 2)
        items.append(
            {
                "system": system_name,
                "human_completion": human_completion,
                "judge_completion": judge_completion,
                "completion_gap": completion_gap,
                "human_visual": human_visual,
                "judge_visual": judge_visual,
                "visual_gap": visual_gap,
                "needs_review": completion_gap >= threshold or visual_gap >= threshold,
                "judge_reason": str(judge.get("reason") or ""),
                "judge_source": str(judge.get("source") or ""),
            }
        )
    flagged = [item for item in items if item["needs_review"]]
    return {
        "status": "needs_review" if flagged else "ok",
        "gap_threshold": threshold,
        "flagged_count": len(flagged),
        "items": items,
        "instructions": (
            "Review judge_completion and judge_visual before approving. "
            "If the Gemini Judge is wrong, edit those two fields under results.<system>.gemini_judge, then approve."
        ),
    }


def _recompute_payload_totals_after_judge_review(payload: dict[str, Any]) -> None:
    for result in dict(payload.get("results") or {}).values():
        if int(result.get("gate") or 0) == 0:
            continue
        judge = dict(result.get("gemini_judge") or {})
        human_completion = _clamp_score((result.get("human_completion") or {}).get("score"))
        human_visual = _clamp_score((result.get("human_visual") or {}).get("score"))
        judge_completion = _clamp_score(judge.get("completion"))
        judge_visual = _clamp_score(judge.get("visual"))
        completion = round((human_completion + judge_completion) / 2.0, 2)
        visual = round((human_visual + judge_visual) / 2.0, 2)
        result["completion"]["score"] = completion
        result["completion"]["judge_score"] = judge_completion
        result["visual"]["score"] = visual
        result["visual"]["judge_score"] = judge_visual
        total = compute_weighted_score(
            completion=completion,
            preservation=(result.get("preservation") or {}).get("score"),
            minimality=_clamp_score((result.get("minimality") or {}).get("score")),
            sanity=_clamp_score((result.get("sanity") or {}).get("score")),
            visual=visual,
            preservation_skipped=bool((result.get("preservation") or {}).get("skipped")),
        )
        result["total_score"] = total["total_score"]
        result["uncapped_score"] = total["uncapped_score"]
        result["score_caps"] = total["score_caps"]
        result["weights"] = total["weights"]


def _replace_path_prefix(value: Any, old_prefix: str, new_prefix: str) -> Any:
    if isinstance(value, dict):
        return {key: _replace_path_prefix(item, old_prefix, new_prefix) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_path_prefix(item, old_prefix, new_prefix) for item in value]
    if isinstance(value, str) and old_prefix and value.startswith(old_prefix):
        return new_prefix + value[len(old_prefix) :]
    return value


def compute_minimality_score(
    baseline_dir: Path,
    final_dir: Path,
    *,
    baseline_entry_html: Path | None = None,
    final_entry_html: Path | None = None,
    target_selectors: list[str] | None = None,
) -> dict[str, Any]:
    baseline_files, baseline_excluded = _collect_text_files(baseline_dir)
    final_files, final_excluded = _collect_text_files(final_dir)
    target_adjustment = _apply_minimality_target_scope(
        baseline_files,
        final_files,
        baseline_dir=baseline_dir,
        final_dir=final_dir,
        baseline_entry_html=baseline_entry_html,
        final_entry_html=final_entry_html,
        target_selectors=target_selectors or [],
    )
    paths = sorted(set(baseline_files) | set(final_files))
    changed_files = 0
    changed_chars = 0
    baseline_chars = 0
    for rel_path in paths:
        before = baseline_files.get(rel_path, "")
        after = final_files.get(rel_path, "")
        baseline_chars += len(before)
        if before == after:
            continue
        changed_files += 1
        matcher = difflib.SequenceMatcher(a=before, b=after, autojunk=False)
        equal_chars = sum(block.size for block in matcher.get_matching_blocks())
        changed_chars += max(len(before), len(after)) - equal_chars
    ratio = changed_chars / max(1, baseline_chars)
    score = max(0.0, 100.0 - min(100.0, ratio * MINIMALITY_CHANGE_MULTIPLIER))
    return {
        "score": round(score, 2),
        "changed_files": changed_files,
        "changed_chars": changed_chars,
        "baseline_chars": baseline_chars,
        "change_ratio": round(ratio, 4),
        "multiplier": MINIMALITY_CHANGE_MULTIPLIER,
        "excluded_file_count": baseline_excluded + final_excluded,
        **target_adjustment,
    }


def _apply_minimality_target_scope(
    baseline_files: dict[str, str],
    final_files: dict[str, str],
    *,
    baseline_dir: Path,
    final_dir: Path,
    baseline_entry_html: Path | None,
    final_entry_html: Path | None,
    target_selectors: list[str],
) -> dict[str, Any]:
    selectors = [str(selector or "").strip() for selector in target_selectors if str(selector or "").strip()]
    if not selectors:
        return {
            "target_adjusted": False,
            "target_selectors_excluded": [],
        }
    adjusted_files: set[str] = set()
    warnings: list[str] = []
    for rel_path in sorted(set(baseline_files) & set(final_files)):
        if not rel_path.lower().endswith(".html"):
            continue
        baseline_path = Path(baseline_dir) / rel_path
        final_path = Path(final_dir) / rel_path
        if not baseline_path.exists() or not final_path.exists():
            continue
        baseline_files[rel_path] = _html_without_targets(baseline_path, selectors)
        final_files[rel_path] = _html_without_targets(final_path, selectors)
        adjusted_files.add(rel_path)
    if not adjusted_files and baseline_entry_html is not None and final_entry_html is not None:
        try:
            baseline_rel = Path(baseline_entry_html).resolve().relative_to(Path(baseline_dir).resolve()).as_posix()
            final_rel = Path(final_entry_html).resolve().relative_to(Path(final_dir).resolve()).as_posix()
        except Exception:
            warnings.append("Could not resolve entry HTML paths relative to minimality roots.")
        else:
            if baseline_rel in baseline_files and final_rel in final_files:
                baseline_files[baseline_rel] = _html_without_targets(Path(baseline_entry_html), selectors)
                final_files[final_rel] = _html_without_targets(Path(final_entry_html), selectors)
                adjusted_files.update({baseline_rel, final_rel})
            else:
                warnings.append("Entry HTML was not included in text-file minimality collection.")
    if not adjusted_files:
        return {
            "target_adjusted": False,
            "target_selectors_excluded": [],
            "target_adjustment_warning": "; ".join(warnings) or "No HTML files were eligible for target adjustment.",
        }
    result = {
        "target_adjusted": True,
        "target_selectors_excluded": selectors,
        "target_adjusted_files": sorted(adjusted_files),
    }
    if warnings:
        result["target_adjustment_warning"] = "; ".join(warnings)
    return result


def compute_preservation_score(
    *,
    baseline_entry_html: Path,
    final_entry_html: Path,
    target_selectors: list[str],
) -> dict[str, Any]:
    selectors = [str(selector or "").strip() for selector in target_selectors if str(selector or "").strip()]
    if not selectors:
        return {
            "score": None,
            "skipped": True,
            "preservation_skipped": True,
            "reason": "No target_selectors were provided; V1 skips Preservation and redistributes its weight.",
        }
    before = _html_without_targets(baseline_entry_html, selectors)
    after = _html_without_targets(final_entry_html, selectors)
    matcher = difflib.SequenceMatcher(a=before, b=after, autojunk=False)
    similarity = matcher.ratio()
    return {
        "score": round(max(0.0, min(100.0, similarity * 100.0)), 2),
        "skipped": False,
        "target_selectors": selectors,
        "non_target_similarity": round(similarity, 4),
    }


def compute_sanity_score(final_dir: Path, *, extra_warnings: list[str] | None = None) -> dict[str, Any]:
    warnings = list(extra_warnings or [])
    html_files = sorted(Path(final_dir).rglob("*.html"))
    if not html_files:
        warnings.append("No HTML files found.")
    broken_local_refs = 0
    for html_path in html_files:
        text = read_text_with_fallback(html_path)
        for match in re.findall(r"""(?:src|href)=["']([^"']+)["']""", text):
            if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", match) or match.startswith("#"):
                continue
            candidate = (html_path.parent / match.split("#", 1)[0].split("?", 1)[0]).resolve()
            if not candidate.exists():
                broken_local_refs += 1
    if broken_local_refs:
        warnings.append(f"{broken_local_refs} local resource reference(s) appear missing.")
    score = max(0.0, 100.0 - 10.0 * len(warnings))
    return {"score": round(score, 2), "warnings": warnings, "broken_local_refs": broken_local_refs}


def write_review_bundle(workspace: Path, score_payload: dict[str, Any]) -> Path:
    review_path = Path(workspace) / "review.html"
    rows: list[str] = []
    for system_name, result in dict(score_payload.get("results") or {}).items():
        screenshots = dict(result.get("screenshots") or {})
        baseline = _relative_or_empty(review_path.parent, screenshots.get("baseline"))
        final = _relative_or_empty(review_path.parent, screenshots.get("final"))
        total = html.escape(str(result.get("total_score", "")))
        uncapped = html.escape(str(result.get("uncapped_score", "")))
        completion = dict(result.get("completion") or {})
        visual = dict(result.get("visual") or {})
        judge = dict(result.get("gemini_judge") or {})
        caps = result.get("score_caps") or []
        cap_text = html.escape(json.dumps(caps, ensure_ascii=False)) if caps else "none"
        judge_reason = html.escape(str(judge.get("reason") or ""))
        system_label = html.escape(str(result.get("system_label") or _system_label(system_name)))
        rows.append(
            "<section>"
            f"<h2>{system_label} - {total}</h2>"
            "<dl class='metrics'>"
            f"<dt>Uncapped</dt><dd>{uncapped}</dd>"
            f"<dt>Completion</dt><dd>{html.escape(str(completion.get('score', '')))} "
            f"(human {html.escape(str(completion.get('human_score', '')))}, "
            f"judge {html.escape(str(completion.get('judge_score', '')))})</dd>"
            f"<dt>Visual</dt><dd>{html.escape(str(visual.get('score', '')))} "
            f"(human {html.escape(str(visual.get('human_score', '')))}, "
            f"judge {html.escape(str(visual.get('judge_score', '')))})</dd>"
            f"<dt>Judge confidence</dt><dd>{html.escape(str(judge.get('confidence', '')))}</dd>"
            f"<dt>Judge reason</dt><dd>{judge_reason}</dd>"
            f"<dt>Caps</dt><dd>{cap_text}</dd>"
            "</dl>"
            "<div class='pair'>"
            f"<figure><figcaption>Baseline</figcaption><img src='{html.escape(baseline)}'></figure>"
            f"<figure><figcaption>Final</figcaption><img src='{html.escape(final)}'></figure>"
            "</div>"
            "</section>"
        )
    review_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<style>body{font-family:Arial,sans-serif;margin:24px}"
        ".pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}"
        ".metrics{display:grid;grid-template-columns:max-content 1fr;gap:4px 12px;font-size:13px}"
        ".metrics dt{font-weight:700}.metrics dd{margin:0}"
        "img{max-width:100%;border:1px solid #ccc}section{margin-bottom:32px}</style>"
        "</head><body>"
        f"<h1>Benchmark V1 Review - {html.escape(str(score_payload.get('case_id') or ''))}</h1>"
        + "".join(rows)
        + "</body></html>",
        encoding="utf-8",
    )
    return review_path


def write_run_summary(run_dir: Path) -> dict[str, Any]:
    cases = load_cases(run_dir)
    summary_rows: list[dict[str, Any]] = []
    for case in cases:
        raw_score_path = str(case.get("score_json_path") or "").strip()
        if not raw_score_path:
            continue
        score_path = Path(raw_score_path)
        if not score_path.exists() or not score_path.is_file():
            continue
        payload = json.loads(score_path.read_text(encoding="utf-8"))
        for system_name, result in dict(payload.get("results") or {}).items():
            if system_name not in SUMMARY_SYSTEM_IDS:
                continue
            judge = dict(result.get("gemini_judge") or {})
            summary_rows.append(
                {
                    "paper_folder_name": payload.get("paper_folder_name"),
                    "run_id": payload.get("run_id"),
                    "case_id": case.get("case_id"),
                    "task_type": case.get("task_type"),
                    "subcategory": case.get("subcategory"),
                    "instruction_zh": case.get("instruction_zh"),
                    "system": system_name,
                    "total_score": result.get("total_score"),
                    "uncapped_score": result.get("uncapped_score"),
                    "completion": ((result.get("completion") or {}).get("score")),
                    "human_completion": ((result.get("human_completion") or {}).get("score")),
                    "judge_completion": judge.get("completion"),
                    "preservation": ((result.get("preservation") or {}).get("score")),
                    "minimality": ((result.get("minimality") or {}).get("score")),
                    "sanity": ((result.get("sanity") or {}).get("score")),
                    "visual": ((result.get("visual") or {}).get("score")),
                    "human_visual": ((result.get("human_visual") or {}).get("score")),
                    "judge_visual": judge.get("visual"),
                    "judge_reason": judge.get("reason", ""),
                    "judge_confidence": judge.get("confidence"),
                    "score_caps": json.dumps(result.get("score_caps") or [], ensure_ascii=False),
                    "preservation_skipped": bool((result.get("preservation") or {}).get("skipped")),
                    "evaluator_note": dict(payload.get("evaluator_notes") or {}).get(system_name, ""),
                    "score_json_path": str(score_path),
                    "review_html_path": str(case.get("review_html_path") or ""),
                }
            )
    summary = {
        "schema_version": "paperalchemy-benchmark-v1-summary",
        "rows": summary_rows,
        "case_count": len(cases),
        "scored_result_count": len(summary_rows),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    summary_json = Path(run_dir) / "summary.json"
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_csv = Path(run_dir) / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "paper_folder_name",
            "run_id",
            "case_id",
            "task_type",
            "subcategory",
            "instruction_zh",
            "system",
            "total_score",
            "uncapped_score",
            "completion",
            "human_completion",
            "judge_completion",
            "preservation",
            "minimality",
            "sanity",
            "visual",
            "human_visual",
            "judge_visual",
            "judge_reason",
            "judge_confidence",
            "score_caps",
            "preservation_skipped",
            "evaluator_note",
            "score_json_path",
            "review_html_path",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    return summary


def write_paper_archive(paper_folder_name: str) -> dict[str, Any]:
    runs_dir = build_v1_runs_dir(paper_folder_name)
    archive_rows: list[dict[str, Any]] = []
    if runs_dir.exists():
        for run_json in sorted(runs_dir.glob(f"*/{RUN_JSON}")):
            run_dir = run_json.parent
            summary_path = run_dir / "summary.json"
            if not summary_path.exists():
                continue
            try:
                payload = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            archive_rows.extend(dict(row) for row in payload.get("rows") or [])

    archive_dir = runs_dir.parent / "benchmark_v1_archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive = {
        "schema_version": "paperalchemy-benchmark-v1-paper-archive",
        "paper_folder_name": paper_folder_name,
        "row_count": len(archive_rows),
        "rows": archive_rows,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    archive_json = archive_dir / "archive.json"
    archive_json.write_text(json.dumps(archive, indent=2, ensure_ascii=False), encoding="utf-8")
    archive_csv = archive_dir / "archive.csv"
    fieldnames = [
        "paper_folder_name",
        "run_id",
        "case_id",
        "task_type",
        "subcategory",
        "instruction_zh",
        "system",
        "total_score",
        "uncapped_score",
        "completion",
        "human_completion",
        "judge_completion",
        "preservation",
        "minimality",
        "sanity",
        "visual",
        "human_visual",
        "judge_visual",
        "judge_reason",
        "judge_confidence",
        "score_caps",
        "preservation_skipped",
        "evaluator_note",
        "score_json_path",
        "review_html_path",
    ]
    with archive_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(archive_rows)
    write_paper_archive_html(archive_dir, archive)
    return archive


def write_paper_archive_html(archive_dir: Path, archive: dict[str, Any]) -> Path:
    rows_html: list[str] = []
    for row in archive.get("rows") or []:
        review_path = html.escape(str(row.get("review_html_path") or ""))
        review_link = f"<a href='{review_path}'>review</a>" if review_path else ""
        rows_html.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('run_id', '')))}</td>"
            f"<td>{html.escape(str(row.get('case_id', '')))}</td>"
            f"<td>{html.escape(str(row.get('system', '')))}</td>"
            f"<td>{html.escape(str(row.get('total_score', '')))}</td>"
            f"<td>{html.escape(str(row.get('uncapped_score', '')))}</td>"
            f"<td>{html.escape(str(row.get('completion', '')))}</td>"
            f"<td>{html.escape(str(row.get('human_completion', '')))}</td>"
            f"<td>{html.escape(str(row.get('judge_completion', '')))}</td>"
            f"<td>{html.escape(str(row.get('preservation', '')))}</td>"
            f"<td>{html.escape(str(row.get('minimality', '')))}</td>"
            f"<td>{html.escape(str(row.get('sanity', '')))}</td>"
            f"<td>{html.escape(str(row.get('visual', '')))}</td>"
            f"<td>{html.escape(str(row.get('human_visual', '')))}</td>"
            f"<td>{html.escape(str(row.get('judge_visual', '')))}</td>"
            f"<td>{html.escape(str(row.get('judge_confidence', '')))}</td>"
            f"<td>{html.escape(str(row.get('judge_reason', '')))}</td>"
            f"<td>{html.escape(str(row.get('score_caps', '')))}</td>"
            f"<td>{html.escape(str(row.get('evaluator_note', '')))}</td>"
            f"<td>{review_link}</td>"
            "</tr>"
        )
    archive_html = archive_dir / "archive.html"
    archive_html.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<style>body{font-family:Arial,sans-serif;margin:24px}"
        "table{border-collapse:collapse;width:100%;font-size:13px}"
        "th,td{border:1px solid #ddd;padding:6px;vertical-align:top}"
        "th{background:#f3f4f6;text-align:left}</style>"
        "</head><body>"
        f"<h1>Benchmark V1 Archive - {html.escape(str(archive.get('paper_folder_name') or ''))}</h1>"
        "<table><thead><tr>"
        "<th>Run</th><th>Case</th><th>System</th><th>Total</th><th>Uncapped</th>"
        "<th>C</th><th>Human C</th><th>Judge C</th><th>P</th><th>M</th><th>S</th>"
        "<th>V</th><th>Human V</th><th>Judge V</th><th>Judge confidence</th><th>Judge reason</th><th>Caps</th>"
        "<th>Evaluator note</th><th>Review</th>"
        "</tr></thead><tbody>"
        + "".join(rows_html)
        + "</tbody></table></body></html>",
        encoding="utf-8",
    )
    return archive_html


def _load_cached_judge_results(run_dir: Path, case_id: str) -> dict[str, dict[str, Any]]:
    try:
        case = _load_case(run_dir, case_id)
    except Exception:
        return {}

    candidates: list[Path] = []
    raw_score_path = str(case.get("score_json_path") or "").strip()
    if raw_score_path:
        candidates.append(Path(raw_score_path))
    candidates.append(Path(run_dir) / "scoring" / str(case_id) / "score.json")

    for score_path in candidates:
        if not score_path.exists() or not score_path.is_file():
            continue
        try:
            payload = json.loads(score_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        cached: dict[str, dict[str, Any]] = {}
        for system_name, result in dict(payload.get("results") or {}).items():
            raw_judge = dict((result or {}).get("gemini_judge") or {})
            if not raw_judge:
                continue
            if raw_judge.get("judge_version") != JUDGE_RESULT_VERSION:
                continue
            if "completion" not in raw_judge or "visual" not in raw_judge:
                continue
            try:
                cached[str(system_name)] = _normalize_judge_result(raw_judge, raw_text="", cached=True)
            except ValueError:
                continue
        if cached:
            return cached
    return {}


def _response_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if text is not None:
                    parts.append(str(text))
        if parts:
            return "\n".join(parts)
    try:
        return json.dumps(content, ensure_ascii=False)
    except Exception:
        return str(content)


def _extract_json_object(raw_text: str) -> dict[str, Any]:
    text = str(raw_text or "").strip()
    if not text:
        raise ValueError("Gemini judge returned an empty response.")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError(f"Gemini judge did not return JSON: {text[:200]}")
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Gemini judge returned malformed JSON: {text[:200]}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Gemini judge JSON response must be an object.")
    return payload


def _normalize_judge_result(
    payload: dict[str, Any],
    *,
    raw_text: str = "",
    cached: bool = False,
) -> dict[str, Any]:
    result = {
        "judge_version": JUDGE_RESULT_VERSION,
        "completion": round(_coerce_judge_score(payload, "completion"), 2),
        "visual": round(_coerce_judge_score(payload, "visual"), 2),
        "reason": str(payload.get("reason") or "").strip(),
        "confidence": round(_clamp_unit(payload.get("confidence")), 3),
        "source": "gemini_cached" if cached else "gemini",
    }
    raw = str(raw_text or "").strip()
    if raw:
        result["raw_response"] = raw[:2000]
    return result


def _judge_target_asset_paths(case: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    seen: set[str] = set()
    for raw_asset in case.get("target_assets") or []:
        if not isinstance(raw_asset, dict):
            continue
        raw_path = str(raw_asset.get("source_path") or raw_asset.get("path") or "").strip()
        if not raw_path:
            continue
        try:
            path = Path(raw_path).resolve()
        except Exception:
            continue
        key = str(path).casefold()
        if key in seen or not path.exists() or not path.is_file():
            continue
        seen.add(key)
        paths.append(path)
    return paths


def _coerce_judge_score(payload: dict[str, Any], key: str) -> float:
    if key not in payload:
        raise ValueError(f"Gemini judge JSON response is missing {key!r}.")
    try:
        return _clamp_score(float(payload[key]))
    except Exception as exc:
        raise ValueError(f"Gemini judge field {key!r} must be numeric.") from exc


def _iter_available_final_zips(run: dict[str, Any], case: dict[str, Any]) -> list[tuple[str, Path]]:
    results: list[tuple[str, Path]] = []
    paper_folder_name = str(run["paper_folder_name"])
    pa_export_name = str(case.get("pa_export_name") or "").strip()
    if pa_export_name:
        export_dir = build_experiment_export_dir(paper_folder_name, pa_export_name)
        metadata_path = export_dir / "export_metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            results.append(("paperalchemy", export_dir / str(metadata["site_zip_path"])))

    seen_systems = {system_name for system_name, _ in results}
    control_model_results = dict(case.get("control_model_results") or {})
    for model_id in CONTROL_MODEL_IDS:
        result = dict(control_model_results.get(model_id) or {})
        zip_path = str(result.get("zip_path") or "").strip()
        if zip_path and model_id not in seen_systems:
            results.append((model_id, Path(zip_path)))
            seen_systems.add(model_id)

    web_zip = str(case.get("web_llm_zip_path") or "").strip()
    if web_zip and "web_llm" not in seen_systems:
        results.append(("web_llm", Path(web_zip)))
    return results


def _iter_failed_model_results(case: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    control_model_results = dict(case.get("control_model_results") or {})
    results: list[tuple[str, dict[str, Any]]] = []
    for model_id in CONTROL_MODEL_IDS:
        result = dict(control_model_results.get(model_id) or {})
        if str(result.get("input_mode") or "").strip() == "failed":
            results.append((model_id, result))
    return results


def _system_label(system_name: str) -> str:
    if system_name == "paperalchemy":
        return "PaperAlchemy"
    if system_name == "web_llm":
        return "Web-LLM"
    return CONTROL_MODEL_LABELS.get(system_name, system_name)


def _load_case(run_dir: Path, case_id: str) -> dict[str, Any]:
    for case in load_cases(run_dir):
        if str(case.get("case_id") or "") == str(case_id or "").strip():
            return case
    raise KeyError(f"Unknown case_id: {case_id}")


def _collect_text_files(root_dir: Path) -> tuple[dict[str, str], int]:
    root = Path(root_dir).resolve()
    files: dict[str, str] = {}
    excluded_file_count = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.resolve().relative_to(root).as_posix()
        if should_exclude_benchmark_member(rel):
            excluded_file_count += 1
            continue
        if path.suffix.lower() in TEXT_SUFFIXES:
            files[rel] = read_text_with_fallback(path)
    return files, excluded_file_count


def _relative_or_empty(root: Path, value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    path = Path(raw)
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return raw


def _html_without_targets(html_path: Path, selectors: list[str]) -> str:
    soup = BeautifulSoup(read_text_with_fallback(html_path), "html.parser")
    target_tokens = _target_selector_tokens(selectors)
    for selector in selectors:
        try:
            targets = soup.select(selector)
        except Exception:
            targets = []
        for target in targets:
            target.decompose()
    if target_tokens:
        for style in soup.find_all("style"):
            style.string = _css_without_target_rules(style.get_text("", strip=False), target_tokens)
    return soup.get_text(" ", strip=True) + "\n" + soup.decode(formatter="minimal")


def _target_selector_tokens(selectors: list[str]) -> list[str]:
    tokens: set[str] = set()
    for selector in selectors:
        for prefix, name in re.findall(r"([.#])([A-Za-z0-9_-]+)", str(selector or "")):
            tokens.add(prefix + name)
    return sorted(tokens, key=len, reverse=True)


def _css_without_target_rules(css_text: str, target_tokens: list[str]) -> str:
    if not css_text or not target_tokens:
        return css_text

    def replace_rule(match: re.Match[str]) -> str:
        selector_text = match.group(1)
        if any(token in selector_text for token in target_tokens):
            return ""
        return match.group(0)

    return re.sub(r"([^{}]+)\{[^{}]*\}", replace_rule, css_text)


def _score_or_default(value: float | None) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 50.0
    return _clamp_score(value)


def _clamp_score(value: float | None) -> float:
    try:
        numeric = float(value if value is not None else 0.0)
    except Exception:
        numeric = 0.0
    return max(0.0, min(100.0, numeric))


def _clamp_unit(value: float | None) -> float:
    try:
        numeric = float(value if value is not None else 0.0)
    except Exception:
        numeric = 0.0
    return max(0.0, min(1.0, numeric))
