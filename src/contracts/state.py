import operator
from typing import Annotated, Any, List, Literal, Optional, TypedDict

from src.services.human_feedback import HumanFeedbackPayload
from src.contracts.schemas import (
    ArbiterReport,
    BlockRenderArtifact,
    BlockRenderSpec,
    CssRevisionPlan,
    CoderArtifact,
    LayoutComposeSession,
    LayoutComposeUpdate,
    PagePlan,
    ReviewerReport,
    RevisionRouteDecision,
    SemanticPagePlan,
    RevisionPlan,
    ShellBindingReview,
    ShellManualSelection,
    StructuredPaper,
    TargetedReplacementPlan,
    TemplateCandidate,
    TemplateProfile,
    RevisionIntent,
)


class ReaderState(TypedDict):
    raw_markdown: str
    assets_list: List[dict]
    human_directives: HumanFeedbackPayload
    previous_structured_paper: Optional[StructuredPaper]
    feedback_history: Annotated[List[str], operator.add]
    text_structured_paper: Optional[dict[str, Any]]
    text_reader_feedback: str
    asset_registry: List[dict[str, Any]]
    asset_binding_candidates: dict[str, list[dict[str, Any]]]
    asset_binding_feedback: str
    structured_paper: Optional[StructuredPaper]
    critic_passed: bool
    retry_count: int


class PlannerState(TypedDict):
    structured_paper: StructuredPaper
    previous_page_plan: Optional[PagePlan]
    template_catalog: List[dict[str, Any]]
    template_link_map: dict[str, str]
    module_index: dict[str, Any]
    generation_constraints: dict[str, Any]
    user_constraints: dict[str, Any]
    human_directives: HumanFeedbackPayload
    template_candidates: List[TemplateCandidate]
    selected_template: Optional[TemplateCandidate]
    template_profile: Optional[TemplateProfile]
    planner_feedback_history: Annotated[List[str], operator.add]
    semantic_page_plan: Optional[SemanticPagePlan]
    page_plan: Optional[PagePlan]
    planner_critic_passed: bool
    planner_retry_count: int


class CoderState(TypedDict):
    paper_folder_name: str
    human_directives: HumanFeedbackPayload
    coder_instructions: str
    structured_paper: StructuredPaper
    page_plan: PagePlan
    template_profile: Optional[TemplateProfile]
    block_render_specs: List[BlockRenderSpec]
    block_render_artifacts: List[BlockRenderArtifact]
    coder_feedback_history: Annotated[List[str], operator.add]
    coder_artifact: Optional[CoderArtifact]
    coder_critic_passed: bool
    coder_retry_count: int


class WorkflowState(TypedDict):
    paper_folder_name: str
    user_constraints: dict[str, str]
    generation_constraints: dict[str, Any]
    manual_layout_compose_enabled: bool
    human_directives: HumanFeedbackPayload
    coder_instructions: str
    edit_intent: RevisionIntent | None
    edit_intent_reason: str
    patch_agent_output: str
    patch_applied_summary: str
    revision_route_decision: RevisionRouteDecision | None
    revision_plan: RevisionPlan | None
    targeted_replacement_plan: TargetedReplacementPlan | None
    css_revision_plan: CssRevisionPlan | None
    css_revision_summary: str
    patch_error: str
    paper_overview: str
    outline_overview: str
    is_approved: bool
    is_outline_approved: bool
    is_webpage_approved: bool
    review_stage: str
    template_candidates: List[TemplateCandidate]
    selected_template: Optional[TemplateCandidate]
    template_profile: Optional[TemplateProfile]
    template_profile_path: str
    template_compile_cache_hit: bool
    block_render_artifacts: List[BlockRenderArtifact]
    shell_binding_review: ShellBindingReview | None
    shell_manual_selection: ShellManualSelection | None
    layout_compose_session: LayoutComposeSession | None
    layout_compose_update: LayoutComposeUpdate | None
    review_current_screenshot_path: str
    review_template_screenshot_path: str
    semantic_visual_review: ReviewerReport | None
    layout_rhythm_review: ReviewerReport | None
    polish_review: ReviewerReport | None
    arbiter_review: ArbiterReport | None
    arbiter_autofix_applied: bool
    structured_paper: Optional[StructuredPaper]
    semantic_page_plan: Optional[SemanticPagePlan]
    page_plan: Optional[PagePlan]
    approved_page_plan: Optional[PagePlan]
    coder_artifact: Optional[CoderArtifact]

