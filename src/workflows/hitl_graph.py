from __future__ import annotations

from typing import Any, Callable

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from src.contracts.state import WorkflowState

_DefaultWorkflow = None


def build_hitl_workflow(
    *,
    reader_phase_node: Callable[[WorkflowState], dict[str, Any]],
    overview_node: Callable[[WorkflowState], dict[str, Any]],
    template_compile_phase_node: Callable[[WorkflowState], dict[str, Any]],
    planner_phase_node: Callable[[WorkflowState], dict[str, Any]],
    outline_review_node: Callable[[WorkflowState], dict[str, Any]],
    layout_compose_prepare_node: Callable[[WorkflowState], dict[str, Any]],
    layout_compose_review_node: Callable[[WorkflowState], dict[str, Any]],
    capture_review_screenshots_node: Callable[[WorkflowState], dict[str, Any]],
    semantic_visual_reviewer_node: Callable[[WorkflowState], dict[str, Any]],
    layout_rhythm_reviewer_node: Callable[[WorkflowState], dict[str, Any]],
    review_arbiter_node: Callable[[WorkflowState], dict[str, Any]],
    revision_classifier_node: Callable[[WorkflowState], dict[str, Any]],
    webpage_review_node: Callable[[WorkflowState], dict[str, Any]],
    patch_agent_node: Callable[[WorkflowState], dict[str, Any]],
    patch_executor_node: Callable[[WorkflowState], dict[str, Any]],
    css_revision_agent_node: Callable[[WorkflowState], dict[str, Any]],
    css_revision_executor_node: Callable[[WorkflowState], dict[str, Any]],
    coder_phase_node: Callable[[WorkflowState], dict[str, Any]],
    human_review_router: Callable[[WorkflowState], str],
    outline_review_router: Callable[[WorkflowState], str],
    webpage_review_router: Callable[[WorkflowState], str],
    revision_route_router: Callable[[WorkflowState], str],
    post_patch_router: Callable[[WorkflowState], str],
    post_arbiter_router: Callable[[WorkflowState], str],
):
    workflow = StateGraph(WorkflowState)
    workflow.add_node("reader", reader_phase_node)
    workflow.add_node("overview", overview_node)
    workflow.add_node("template_compile", template_compile_phase_node)
    workflow.add_node("planner", planner_phase_node)
    workflow.add_node("outline_review", outline_review_node)
    workflow.add_node("layout_compose_prepare", layout_compose_prepare_node)
    workflow.add_node("layout_compose_review", layout_compose_review_node)
    workflow.add_node("capture_review_screenshots", capture_review_screenshots_node)
    workflow.add_node("reviewer_semantic_visual", semantic_visual_reviewer_node)
    workflow.add_node("reviewer_layout_rhythm", layout_rhythm_reviewer_node)
    workflow.add_node("review_arbiter", review_arbiter_node)
    workflow.add_node("revision_classifier", revision_classifier_node)
    workflow.add_node("webpage_review", webpage_review_node)
    workflow.add_node("patch_agent", patch_agent_node)
    workflow.add_node("patch_executor", patch_executor_node)
    workflow.add_node("css_revision_agent", css_revision_agent_node)
    workflow.add_node("css_revision_executor", css_revision_executor_node)
    workflow.add_node("coder", coder_phase_node)

    workflow.set_entry_point("reader")
    workflow.add_edge("reader", "overview")
    workflow.add_conditional_edges(
        "overview",
        human_review_router,
        {
            "template_compile": "template_compile",
            "reader": "reader",
        },
    )
    workflow.add_edge("template_compile", "planner")
    workflow.add_edge("planner", "outline_review")
    workflow.add_conditional_edges(
        "outline_review",
        outline_review_router,
        {
            "planner": "planner",
            "coder": "coder",
            "layout_compose_prepare": "layout_compose_prepare",
        },
    )
    workflow.add_edge("layout_compose_prepare", "layout_compose_review")
    workflow.add_edge("layout_compose_review", "coder")
    workflow.add_edge("coder", "capture_review_screenshots")
    workflow.add_edge("capture_review_screenshots", "reviewer_semantic_visual")
    workflow.add_edge("reviewer_semantic_visual", "reviewer_layout_rhythm")
    workflow.add_edge("reviewer_layout_rhythm", "review_arbiter")
    workflow.add_conditional_edges(
        "review_arbiter",
        post_arbiter_router,
        {
            "revision_classifier": "revision_classifier",
            "webpage_review": "webpage_review",
        },
    )
    workflow.add_conditional_edges(
        "webpage_review",
        webpage_review_router,
        {
            "revision_classifier": "revision_classifier",
            "end": END,
        },
    )
    workflow.add_conditional_edges(
        "revision_classifier",
        revision_route_router,
        {
            "patch_agent": "patch_agent",
            "css_revision_agent": "css_revision_agent",
            "webpage_review": "webpage_review",
        },
    )
    workflow.add_edge("patch_agent", "patch_executor")
    workflow.add_conditional_edges(
        "patch_executor",
        post_patch_router,
        {
            "css_revision_agent": "css_revision_agent",
            "webpage_review": "webpage_review",
        },
    )
    workflow.add_edge("css_revision_agent", "css_revision_executor")
    workflow.add_edge("css_revision_executor", "webpage_review")

    memory = MemorySaver()
    return workflow.compile(
        checkpointer=memory,
        interrupt_after=["overview", "outline_review", "layout_compose_review", "webpage_review"],
    )


def set_default_hitl_workflow(workflow: Any) -> None:
    global _DefaultWorkflow
    _DefaultWorkflow = workflow


def get_default_hitl_workflow() -> Any:
    if _DefaultWorkflow is None:
        raise RuntimeError("Default HITL workflow has not been initialized.")
    return _DefaultWorkflow
