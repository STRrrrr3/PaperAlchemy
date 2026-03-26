import operator
from typing import Annotated, Any, List, Optional, TypedDict

from src.human_feedback import HumanFeedbackPayload
from src.schemas import (
    BlockRenderArtifact,
    BlockRenderSpec,
    CoderArtifact,
    LayoutComposeSession,
    LayoutComposeUpdate,
    PagePlan,
    RevisionPlan,
    ShellBindingReview,
    ShellManualSelection,
    StructuredPaper,
    TargetedReplacementPlan,
    TemplateCandidate,
    TemplateProfile,
    VisualSmokeReport,
)


class ReaderState(TypedDict):
    raw_markdown: str
    assets_list: List[dict]
    human_directives: HumanFeedbackPayload
    previous_structured_paper: Optional[StructuredPaper]
    feedback_history: Annotated[List[str], operator.add]
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
    visual_feedback: Annotated[List[str], operator.add]
    visual_screenshot_path: str
    visual_iterations: int
    is_visually_approved: bool
    visual_smoke_report: VisualSmokeReport | None
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
    patch_agent_output: str
    revision_plan: RevisionPlan | None
    targeted_replacement_plan: TargetedReplacementPlan | None
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
    visual_smoke_report: VisualSmokeReport | None
    structured_paper: Optional[StructuredPaper]
    page_plan: Optional[PagePlan]
    approved_page_plan: Optional[PagePlan]
    coder_artifact: Optional[CoderArtifact]
