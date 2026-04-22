from __future__ import annotations

from src.ui.updates import _normalize_manual_layout_compose_enabled
from src.contracts.schemas import ArbiterReport
from src.contracts.state import WorkflowState
from src.revision.request_intent import classify_revision_request
from src.services.human_feedback import extract_human_feedback_images, extract_human_feedback_text

def human_review_router(state: WorkflowState) -> str:
    if bool(state.get("is_approved")):
        return "template_compile"
    return "reader"

def outline_review_router(state: WorkflowState) -> str:
    if not bool(state.get("is_outline_approved")):
        return "planner"
    if _normalize_manual_layout_compose_enabled(state.get("manual_layout_compose_enabled")):
        return "layout_compose_prepare"
    return "coder"

def webpage_review_router(state: WorkflowState) -> str:
    if bool(state.get("is_webpage_approved")):
        return "end"

    feedback = state.get("human_directives")
    feedback_text = str(extract_human_feedback_text(feedback) or "").strip()
    feedback_images = extract_human_feedback_images(feedback)
    if not feedback_text and not feedback_images:
        return "css_revision_agent"

    request_intent = classify_revision_request(feedback_text, bool(feedback_images))
    if request_intent == "content":
        return "translator"
    return "css_revision_agent"


def translated_revision_router(state: WorkflowState) -> str:
    if str(state.get("edit_intent") or "").strip() in {"patch", "asset_rebind"}:
        return "patch_agent"
    return "css_revision_agent"

def post_arbiter_router(state: WorkflowState) -> str:
    if bool(state.get("arbiter_autofix_applied")):
        return "webpage_review"
    raw = state.get("arbiter_review")
    if raw is None:
        return "webpage_review"
    try:
        report = raw if isinstance(raw, ArbiterReport) else ArbiterReport.model_validate(raw)
    except Exception:
        return "webpage_review"
    if not report.items:
        return "webpage_review"
    return "arbiter_autofix"
