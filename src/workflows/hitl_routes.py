from __future__ import annotations

from src.contracts.state import WorkflowState
from src.ui.updates import _normalize_manual_layout_compose_enabled
from src.workflows.hitl_nodes import normalize_visual_smoke_report

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

def draft_recovery_router(state: WorkflowState) -> str:
    smoke_report = normalize_visual_smoke_report(state.get("visual_smoke_report"))
    if smoke_report is not None and smoke_report.suggested_recovery == "rerun_planner":
        return "planner"
    return "webpage_review"

def webpage_review_router(state: WorkflowState) -> str:
    if bool(state.get("is_webpage_approved")):
        return "end"
    return "translator"

def edit_intent_route_router(state: WorkflowState) -> str:
    if str(state.get("edit_intent") or "").strip().lower() == "non_patch":
        return "non_patch_feedback"
    return "patch_agent"
