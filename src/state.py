import operator
from typing import Annotated, Any, List, Optional, TypedDict

from src.human_feedback import HumanFeedbackPayload
from src.schemas import CoderArtifact, PagePlan, SemanticPlan, StructuredPaper, TemplateCandidate


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
    template_catalog: List[dict[str, Any]]
    template_link_map: dict[str, str]
    module_index: dict[str, Any]
    generation_constraints: dict[str, Any]
    user_constraints: dict[str, Any]
    human_directives: HumanFeedbackPayload
    semantic_plan: Optional[SemanticPlan]
    template_candidates: List[TemplateCandidate]
    selected_template: Optional[TemplateCandidate]
    selected_template_path: Optional[str]
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
    coder_feedback_history: Annotated[List[str], operator.add]
    visual_feedback: Annotated[List[str], operator.add]
    visual_screenshot_path: str
    visual_iterations: int
    is_visually_approved: bool
    coder_artifact: Optional[CoderArtifact]
    coder_critic_passed: bool
    coder_retry_count: int


class WorkflowState(TypedDict):
    paper_folder_name: str
    user_constraints: dict[str, str]
    generation_constraints: dict[str, Any]
    human_directives: HumanFeedbackPayload
    coder_instructions: str
    paper_overview: str
    is_approved: bool
    is_webpage_approved: bool
    review_stage: str
    structured_paper: Optional[StructuredPaper]
    page_plan: Optional[PagePlan]
    coder_artifact: Optional[CoderArtifact]
