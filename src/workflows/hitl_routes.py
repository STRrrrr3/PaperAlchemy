from __future__ import annotations

from src.ui.updates import _normalize_manual_layout_compose_enabled
from src.contracts.schemas import ArbiterReport, RevisionRouteDecision
from src.contracts.state import WorkflowState
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
        return "end"
    return "revision_classifier"


def _normalize_route_decision(value: object) -> RevisionRouteDecision:
    if isinstance(value, RevisionRouteDecision):
        return value
    try:
        return RevisionRouteDecision.model_validate(value)
    except Exception:
        return RevisionRouteDecision()


def revision_route_router(state: WorkflowState) -> str:
    decision = _normalize_route_decision(state.get("revision_route_decision"))
    if decision.route in {"patch", "mixed"}:
        return "patch_agent"
    if decision.route == "css":
        return "css_revision_agent"
    return "webpage_review"


def post_patch_router(state: WorkflowState) -> str:
    if str(state.get("patch_error") or "").strip():
        return "webpage_review"
    decision = _normalize_route_decision(state.get("revision_route_decision"))
    if decision.route == "mixed" and str(decision.css_text or "").strip():
        return "css_revision_agent"
    return "webpage_review"

def post_arbiter_router(state: WorkflowState) -> str:
    raw = state.get("arbiter_review")
    if raw is None:
        return "webpage_review"
    try:
        report = raw if isinstance(raw, ArbiterReport) else ArbiterReport.model_validate(raw)
    except Exception:
        return "webpage_review"
    if not report.items:
        return "webpage_review"
    return "revision_classifier"
