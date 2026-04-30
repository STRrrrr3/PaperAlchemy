from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.contracts.schemas import ArbiterReport, CoderArtifact, PagePlan, ReviewerReport, StructuredPaper
from src.contracts.state import WorkflowState
from src.prompts import (
    LAYOUT_RHYTHM_REVIEWER_SYSTEM_PROMPT,
    LAYOUT_RHYTHM_REVIEWER_USER_PROMPT_TEMPLATE,
    REVIEW_ARBITER_SYSTEM_PROMPT,
    REVIEW_ARBITER_USER_PROMPT_TEMPLATE,
    SEMANTIC_VISUAL_REVIEWER_SYSTEM_PROMPT,
    SEMANTIC_VISUAL_REVIEWER_USER_PROMPT_TEMPLATE,
)
from src.services.artifact_store import get_output_paths, load_cached_structured_data, load_coder_artifact, load_page_plan
from src.services.human_feedback import build_human_feedback_payload, build_multimodal_message_content
from src.services.llm import get_llm
from src.services.preview_service import (
    build_layout_compose_template_preview_path,
    build_page_screenshot_path,
    build_template_preview_path,
    load_style_context_json,
    take_local_screenshot,
)
from src.utils.html_utils import resolve_template_entry_html_path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _normalize_coder_artifact(value: Any) -> CoderArtifact | None:
    if value is None:
        return None
    if isinstance(value, CoderArtifact):
        return value
    try:
        return CoderArtifact.model_validate(value)
    except Exception:
        return None


def _normalize_page_plan(value: Any) -> PagePlan | None:
    if value is None:
        return None
    if isinstance(value, PagePlan):
        return value
    try:
        return PagePlan.model_validate(value)
    except Exception:
        return None


def _normalize_structured_paper(value: Any) -> StructuredPaper | None:
    if value is None:
        return None
    if isinstance(value, StructuredPaper):
        return value
    try:
        return StructuredPaper.model_validate(value)
    except Exception:
        return None


def _normalize_reviewer_report(value: Any, reviewer: str) -> ReviewerReport:
    try:
        if isinstance(value, ReviewerReport):
            report = value
        else:
            report = ReviewerReport.model_validate(value)
        return report.model_copy(update={"reviewer": reviewer}, deep=True)
    except Exception:
        return ReviewerReport(reviewer=reviewer, items=[])


def _normalize_arbiter_report(value: Any) -> ArbiterReport:
    try:
        if isinstance(value, ArbiterReport):
            return value
        return ArbiterReport.model_validate(value)
    except Exception:
        return ArbiterReport(items=[])


def _workflow_paper_folder_name(state: WorkflowState) -> str:
    return str(state.get("paper_folder_name") or "").strip()


def _load_coder_artifact_for_state(state: WorkflowState) -> CoderArtifact | None:
    artifact = _normalize_coder_artifact(state.get("coder_artifact"))
    if artifact is not None:
        return artifact
    paper_folder_name = _workflow_paper_folder_name(state)
    if not paper_folder_name:
        return None
    _, _, _, coder_json_path = get_output_paths(paper_folder_name)
    return load_coder_artifact(coder_json_path)


def _load_page_plan_for_state(state: WorkflowState) -> PagePlan | None:
    for candidate in (state.get("approved_page_plan"), state.get("page_plan")):
        plan = _normalize_page_plan(candidate)
        if plan is not None:
            return plan
    paper_folder_name = _workflow_paper_folder_name(state)
    if not paper_folder_name:
        return None
    _, _, planner_json_path, _ = get_output_paths(paper_folder_name)
    return load_page_plan(planner_json_path)


def _load_structured_paper_for_state(state: WorkflowState) -> StructuredPaper | None:
    paper = _normalize_structured_paper(state.get("structured_paper"))
    if paper is not None:
        return paper
    paper_folder_name = _workflow_paper_folder_name(state)
    if not paper_folder_name:
        return None
    _, structured_json_path, _, _ = get_output_paths(paper_folder_name)
    return load_cached_structured_data(structured_json_path)


def _build_layout_intent_json(page_plan: PagePlan | None) -> str:
    if page_plan is None:
        return "{}"
    try:
        excerpt = {
            "global_design": page_plan.global_design.model_dump(),
            "adaptation_strategy": page_plan.adaptation_strategy.model_dump(),
            "page_outline": [
                {
                    "block_id": item.block_id,
                    "order": item.order,
                    "title": item.title,
                    "estimated_height": item.estimated_height,
                }
                for item in sorted(page_plan.page_outline, key=lambda x: x.order)
            ],
        }
        return json.dumps(excerpt, indent=2, ensure_ascii=False)
    except Exception:
        return "{}"


def _image_payloads_from_paths(*paths: str) -> list[dict[str, str]]:
    clean_paths = [path for path in paths if str(path or "").strip()]
    if not clean_paths:
        return []
    return build_human_feedback_payload("", clean_paths)["images"]


def _template_preview_cache_path(page_plan: PagePlan) -> Path:
    return build_template_preview_path(
        {
            "template_id": page_plan.template_selection.selected_template_id,
            "entry_html": page_plan.template_selection.selected_entry_html,
        }
    )


def _capture_template_screenshot(page_plan: PagePlan | None) -> str:
    if page_plan is None:
        return ""

    template_entry_path = resolve_template_entry_html_path(page_plan, project_root=PROJECT_ROOT)
    if template_entry_path is None or not template_entry_path.exists():
        return ""

    primary_cache_path = _template_preview_cache_path(page_plan)
    if primary_cache_path.exists():
        return str(primary_cache_path)

    compose_cache_path = build_layout_compose_template_preview_path(template_entry_path)
    if compose_cache_path.exists():
        return str(compose_cache_path)

    screenshot_path = take_local_screenshot(
        str(template_entry_path),
        str(primary_cache_path),
    )
    return str(screenshot_path or "").strip()


def capture_review_screenshots_node(state: WorkflowState) -> dict[str, Any]:
    artifact = _load_coder_artifact_for_state(state)
    if artifact is None:
        print("[DraftReview] Screenshot capture skipped: coder artifact is missing.")
        return {
            "review_current_screenshot_path": "",
            "review_template_screenshot_path": "",
        }

    entry_html_path = Path(artifact.entry_html).resolve()
    current_screenshot_path = take_local_screenshot(
        str(entry_html_path),
        str(build_page_screenshot_path(entry_html_path, "review_current.png")),
    )
    if not current_screenshot_path:
        print("[DraftReview] Screenshot capture failed for the generated page. Reviewers will fail open.")
        return {
            "review_current_screenshot_path": "",
            "review_template_screenshot_path": "",
        }

    template_screenshot_path = _capture_template_screenshot(_load_page_plan_for_state(state))
    if template_screenshot_path:
        print(f"[DraftReview] Reused or captured template screenshot at {template_screenshot_path}")
    else:
        print("[DraftReview] Template screenshot unavailable; layout reviewer will fail open.")

    print(f"[DraftReview] Captured current webpage screenshot at {current_screenshot_path}")
    return {
        "review_current_screenshot_path": str(current_screenshot_path),
        "review_template_screenshot_path": str(template_screenshot_path or ""),
    }


def semantic_visual_reviewer_node(state: WorkflowState) -> dict[str, Any]:
    screenshot_path = str(state.get("review_current_screenshot_path") or "").strip()
    structured_paper = _load_structured_paper_for_state(state)
    if not screenshot_path or structured_paper is None:
        print("[DraftReview] semantic_visual reviewer skipped: missing screenshot or structured paper.")
        return {"semantic_visual_review": ReviewerReport(reviewer="semantic_visual", items=[])}
    images = _image_payloads_from_paths(screenshot_path)
    if not images:
        print("[DraftReview] semantic_visual reviewer skipped: screenshot payload could not be loaded.")
        return {"semantic_visual_review": ReviewerReport(reviewer="semantic_visual", items=[])}

    try:
        llm = get_llm(temperature=0.1, use_smart_model=True, thinking_level="medium")
        structured_llm = llm.with_structured_output(ReviewerReport)
        response = structured_llm.invoke(
            [
                SystemMessage(content=SEMANTIC_VISUAL_REVIEWER_SYSTEM_PROMPT),
                HumanMessage(
                    content=build_multimodal_message_content(
                        text=SEMANTIC_VISUAL_REVIEWER_USER_PROMPT_TEMPLATE.format(
                            structured_paper_json=json.dumps(
                                structured_paper.model_dump(),
                                indent=2,
                                ensure_ascii=False,
                            )
                        ),
                        images=images,
                    )
                ),
            ]
        )
        report = _normalize_reviewer_report(response, "semantic_visual")
        print(f"[DraftReview] semantic_visual reviewer produced {len(report.items)} item(s).")
        return {"semantic_visual_review": report}
    except Exception as exc:
        print(f"[DraftReview] semantic_visual reviewer failed open: {exc}")
        return {"semantic_visual_review": ReviewerReport(reviewer="semantic_visual", items=[])}


def layout_rhythm_reviewer_node(state: WorkflowState) -> dict[str, Any]:
    current_screenshot_path = str(state.get("review_current_screenshot_path") or "").strip()
    template_screenshot_path = str(state.get("review_template_screenshot_path") or "").strip()
    if not current_screenshot_path or not template_screenshot_path:
        print("[DraftReview] layout_rhythm reviewer skipped: missing current or template screenshot.")
        return {"layout_rhythm_review": ReviewerReport(reviewer="layout_rhythm", items=[])}
    images = _image_payloads_from_paths(
        current_screenshot_path,
        template_screenshot_path,
    )
    if len(images) < 2:
        print("[DraftReview] layout_rhythm reviewer skipped: screenshot payloads could not be loaded.")
        return {"layout_rhythm_review": ReviewerReport(reviewer="layout_rhythm", items=[])}

    artifact = _load_coder_artifact_for_state(state)
    page_plan = _load_page_plan_for_state(state)

    style_context_json = "{}"
    if artifact is not None:
        try:
            style_context_json = load_style_context_json(Path(artifact.entry_html).resolve())
        except Exception:
            pass

    layout_intent_json = _build_layout_intent_json(page_plan)

    try:
        llm = get_llm(temperature=0.1, use_smart_model=True, thinking_level="medium")
        structured_llm = llm.with_structured_output(ReviewerReport)
        response = structured_llm.invoke(
            [
                SystemMessage(content=LAYOUT_RHYTHM_REVIEWER_SYSTEM_PROMPT),
                HumanMessage(
                    content=build_multimodal_message_content(
                        text=LAYOUT_RHYTHM_REVIEWER_USER_PROMPT_TEMPLATE.format(
                            style_context_json=style_context_json,
                            layout_intent_json=layout_intent_json,
                        ),
                        images=images,
                    )
                ),
            ]
        )
        report = _normalize_reviewer_report(response, "layout_rhythm")
        print(f"[DraftReview] layout_rhythm reviewer produced {len(report.items)} item(s).")
        return {"layout_rhythm_review": report}
    except Exception as exc:
        print(f"[DraftReview] layout_rhythm reviewer failed open: {exc}")
        return {"layout_rhythm_review": ReviewerReport(reviewer="layout_rhythm", items=[])}


def polish_reviewer_node(_: WorkflowState) -> dict[str, Any]:
    return {"polish_review": ReviewerReport(reviewer="polish", items=[])}


def review_arbiter_node(state: WorkflowState) -> dict[str, Any]:
    semantic_visual_review = _normalize_reviewer_report(state.get("semantic_visual_review"), "semantic_visual")
    layout_rhythm_review = _normalize_reviewer_report(state.get("layout_rhythm_review"), "layout_rhythm")

    reviewer_reports = [semantic_visual_review, layout_rhythm_review]
    polish_review = state.get("polish_review")
    if polish_review is not None:
        reviewer_reports.append(_normalize_reviewer_report(polish_review, "polish"))

    total_input_items = sum(len(report.items) for report in reviewer_reports)
    if total_input_items == 0:
        print("[DraftReview] review arbiter short-circuited: no reviewer items.")
        return {"arbiter_review": ArbiterReport(items=[])}

    try:
        llm = get_llm(temperature=0, use_smart_model=True, thinking_level="high")
        structured_llm = llm.with_structured_output(ArbiterReport)
        response = structured_llm.invoke(
            [
                SystemMessage(content=REVIEW_ARBITER_SYSTEM_PROMPT),
                HumanMessage(
                    content=REVIEW_ARBITER_USER_PROMPT_TEMPLATE.format(
                        reviewer_reports_json=json.dumps(
                            [report.model_dump() for report in reviewer_reports],
                            indent=2,
                            ensure_ascii=False,
                        )
                    )
                ),
            ]
        )
        report = _normalize_arbiter_report(response)
        print(f"[DraftReview] review arbiter produced {len(report.items)} unified item(s).")
        return {"arbiter_review": report}
    except Exception as exc:
        print(f"[DraftReview] review arbiter failed open: {exc}")
        return {"arbiter_review": ArbiterReport(items=[])}
