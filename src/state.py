from typing import Annotated, Any, List, Optional, TypedDict
import operator

from src.schemas import CoderArtifact, PagePlan, SemanticPlan, StructuredPaper, TemplateCandidate


class ReaderState(TypedDict):
    raw_markdown: str
    assets_list: List[dict]
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
    human_directives: str
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
    structured_paper: StructuredPaper
    page_plan: PagePlan
    coder_feedback_history: Annotated[List[str], operator.add]
    coder_artifact: Optional[CoderArtifact]
    coder_critic_passed: bool
    coder_retry_count: int


class WorkflowState(TypedDict):
    paper_folder_name: str
    user_constraints: dict[str, str]
    generation_constraints: dict[str, Any]
    human_directives: str
    structured_paper: Optional[StructuredPaper]
    page_plan: Optional[PagePlan]
    coder_artifact: Optional[CoderArtifact]
