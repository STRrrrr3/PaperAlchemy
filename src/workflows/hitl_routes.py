from __future__ import annotations

from src.ui.updates import _normalize_manual_layout_compose_enabled
from src.contracts.schemas import ArbiterReport
from src.contracts.state import WorkflowState

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
