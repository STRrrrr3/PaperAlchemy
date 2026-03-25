from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class FigureInfo(BaseModel):
    image_path: str = Field(description="Relative path in assets folder, e.g., 'assets/element_1.png'.")
    caption: Optional[str] = Field(default=None, description="Caption text if available.")
    type: str = Field(description="Asset type, e.g., 'chart', 'table', 'photo'.")


class PaperSection(BaseModel):
    section_title: str = Field(description="Section title, e.g., 'Introduction' or '3. Methodology'.")
    rich_web_content: str = Field(
        description="Dense Markdown narrative for webpage generation, preserving core technical paragraphs, equations, and results."
    )
    related_figures: List[FigureInfo] = Field(description="Figures and tables linked to this section.")


class StructuredPaper(BaseModel):
    paper_title: str = Field(description="Paper title.")
    overall_summary: str = Field(description="Brief summary of the entire paper.")
    sections: List[PaperSection] = Field(
        description=(
            "Ordered list of selected landing-page sections. If an Abstract exists, it must be first."
        )
    )


class CriticReport(BaseModel):
    is_extraction_valid: bool = Field(description="Whether Reader extraction passed audit.")
    extraction_feedback: str = Field(description="Actionable feedback when extraction fails audit.")


class SemanticBlock(BaseModel):
    block_id: str
    order: int
    title: str
    objective: str
    source_sections: List[str]
    preferred_interaction: Literal["none", "tabs", "accordion", "carousel", "hover-detail", "comparison-slider"]
    media_intensity: Literal["low", "medium", "high"]


class SemanticPlan(BaseModel):
    # Deprecated runtime artifact kept temporarily for staged cleanup. The live planner
    # flow now produces PagePlan directly via unified_planner.
    plan_version: str = Field(description="Semantic planning schema version.")
    planning_mode: Literal["hybrid_two_stage"] = Field(description="Hybrid planner mode id.")
    design_intent: str = Field(description="High-level design objective.")
    style_keywords: List[str] = Field(description="Visual style keywords.")
    required_capabilities: List[str] = Field(description="Template capability requirements.")
    block_blueprint: List[SemanticBlock] = Field(description="Template-agnostic block structure.")
    novelty_points: List[str] = Field(description="Key innovation points to emphasize.")

    @field_validator("planning_mode", mode="before")
    @classmethod
    def normalize_planning_mode(cls, value: str) -> str:
        if not isinstance(value, str):
            return value

        normalized = value.strip().lower()
        alias_map = {
            "hybrid": "hybrid_two_stage",
            "hybrid-semantic": "hybrid_two_stage",
            "hybrid_semantic": "hybrid_two_stage",
            "hybrid two stage": "hybrid_two_stage",
            "hybrid-two-stage": "hybrid_two_stage",
            "two_stage": "hybrid_two_stage",
        }
        return alias_map.get(normalized, value)


class TemplateCandidate(BaseModel):
    template_id: str
    root_dir: str
    chosen_entry_html: str
    score: float
    reasons: List[str]


class PlanMeta(BaseModel):
    plan_version: str = Field(description="Planner schema version, e.g., '1.1'.")
    planning_mode: Literal["autopage_template_first", "hybrid_template_bind"] = Field(
        description="Planning mode identifier."
    )
    target_framework: str = Field(description="Target framework, e.g., 'static-html', 'react', 'vue'.")
    confidence: float = Field(description="Planner confidence in [0, 1].")


class TemplateSelection(BaseModel):
    selected_template_id: str = Field(description="Template id from local catalog.")
    selected_root_dir: str = Field(description="Template root directory path.")
    selected_entry_html: str = Field(description="Entry HTML path relative to template root.")
    fallback_template_id: Optional[str] = Field(default=None, description="Optional fallback template id.")
    selection_rationale: str = Field(description="Why this template was selected.")


class DecisionSummary(BaseModel):
    design_goal: str = Field(description="Core design goal for this page.")
    novelty_points: List[str] = Field(description="Where this plan differentiates itself.")
    tradeoffs: List[str] = Field(description="Accepted tradeoffs for speed and quality.")


class AdaptationStrategy(BaseModel):
    preserve_from_template: List[str] = Field(description="What parts should stay as-is from template.")
    replace_content_areas: List[str] = Field(description="Template regions to replace with paper content.")
    style_override_level: Literal["none", "light", "medium"] = Field(description="Visual override intensity.")
    asset_policy: Literal["reuse_template_assets", "replace_with_paper_assets", "mixed"] = Field(
        description="Policy for template and paper assets."
    )


class ColorStrategy(BaseModel):
    background: str
    surface: str
    text: str
    accent: str


class GlobalDesign(BaseModel):
    style_keywords: List[str]
    color_strategy: ColorStrategy
    typography_strategy: str
    motion_level: Literal["none", "low", "medium"]
    density: Literal["compact", "balanced", "airy"]


class PageOutlineItem(BaseModel):
    block_id: str
    order: int
    title: str
    objective: str
    source_sections: List[str]
    estimated_height: Literal["S", "M", "L"]


class TargetTemplateRegion(BaseModel):
    selector_hint: str = Field(description="CSS selector hint or DOM region description.")
    region_role: Literal["hero", "section", "gallery", "table", "footer", "nav"]
    operation: Literal["replace_text", "replace_media", "insert_after", "append_child"]


class ComponentRecipeItem(BaseModel):
    slot: Literal["container", "content", "media", "interaction"]
    module_id: Optional[str] = None
    component_id: Optional[str] = None
    style_id: Optional[str] = None
    token_set_id: Optional[str] = None
    reason: str


class ContentContract(BaseModel):
    headline: str
    body_points: List[str]
    cta: Optional[str] = None


class AssetBinding(BaseModel):
    figure_paths: List[str]
    template_asset_fallback: Optional[str] = None


class InteractionPlan(BaseModel):
    pattern: Literal["none", "tabs", "accordion", "carousel", "hover-detail", "comparison-slider"]
    behavior_note: str


class ResponsiveRules(BaseModel):
    mobile_order: int
    desktop_layout: str


class ShellWrapperSignature(BaseModel):
    tag: str
    required_classes: List[str] = Field(default_factory=list)
    preserve_ids: List[str] = Field(default_factory=list)


class BlockShellContract(BaseModel):
    root_tag: str
    required_classes: List[str] = Field(default_factory=list)
    preserve_ids: List[str] = Field(default_factory=list)
    wrapper_chain: List[ShellWrapperSignature] = Field(default_factory=list)
    actionable_root_selector: str


class BlockPlan(BaseModel):
    block_id: str
    target_template_region: TargetTemplateRegion
    component_recipe: List[ComponentRecipeItem]
    content_contract: ContentContract
    asset_binding: AssetBinding
    interaction: InteractionPlan
    responsive_rules: ResponsiveRules
    shell_contract: Optional[BlockShellContract] = None
    a11y_notes: List[str]
    acceptance_checks: List[str]


class FileTouchItem(BaseModel):
    path: str
    action: Literal["edit", "create", "copy"]
    reason: str


class CoderHandoff(BaseModel):
    implementation_order: List[str]
    file_touch_plan: List[FileTouchItem]
    hard_constraints: List[str]
    known_risks: List[str]


class QualityCheck(BaseModel):
    name: Literal["grounding_check", "consistency_check", "feasibility_check", "template_path_check"]
    passed: bool
    note: str


class PagePlan(BaseModel):
    plan_meta: PlanMeta
    template_selection: TemplateSelection
    decision_summary: DecisionSummary
    adaptation_strategy: AdaptationStrategy
    global_design: GlobalDesign
    page_outline: List[PageOutlineItem]
    blocks: List[BlockPlan]
    dom_mapping: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Mapping of existing template CSS selectors to the HTML/text snippets that should be "
            "injected into those DOM nodes."
        ),
    )
    selectors_to_remove: List[str] = Field(
        default_factory=list,
        description=(
            "CSS selectors for residual template elements that should be completely removed from the "
            "DOM, such as dummy text blocks, legacy paper content, placeholder images, or irrelevant widgets."
        ),
    )
    coder_handoff: CoderHandoff
    quality_checks: List[QualityCheck]
    open_questions: List[str]


class PlannerCriticReport(BaseModel):
    is_plan_valid: bool = Field(description="Whether planner output passed semantic review.")
    plan_feedback: str = Field(description="Actionable feedback when plan is invalid.")


class CoderArtifact(BaseModel):
    site_dir: str = Field(description="Generated site directory path.")
    entry_html: str = Field(description="Generated entry html path.")
    selected_template_id: str = Field(description="Template used for generation.")
    copied_assets: List[str] = Field(description="Copied paper asset paths relative to site_dir.")
    edited_files: List[str] = Field(description="Edited file paths relative to site_dir.")
    notes: str = Field(description="Short build summary.")


class CoderCriticReport(BaseModel):
    is_build_valid: bool = Field(description="Whether coder output passes checks.")
    build_feedback: str = Field(description="Actionable feedback when build fails.")


GlobalAnchorId = Literal["header_brand", "header_primary_action", "header_nav", "footer_meta"]
SlotId = Literal["title", "summary", "body", "media", "meta", "actions"]
StylePropertyName = Literal[
    "font-size",
    "line-height",
    "margin",
    "margin-top",
    "margin-bottom",
    "padding",
    "gap",
    "text-align",
    "max-width",
    "width",
]
AttributeName = Literal["class", "href", "target", "aria-label", "style", "id"]


class PageManifestSlot(BaseModel):
    slot_id: SlotId
    selector: str


class PageManifestBlock(BaseModel):
    block_id: str
    source_sections: List[str]
    selector: str
    slots: List[PageManifestSlot]
    root_tag: str
    root_classes: List[str] = Field(default_factory=list)
    preserve_ids: List[str] = Field(default_factory=list)
    wrapper_chain: List[ShellWrapperSignature] = Field(default_factory=list)
    actionable_root_selector: str


class PageManifestGlobal(BaseModel):
    global_id: GlobalAnchorId
    selector: str
    target_tag: str
    required_classes: List[str] = Field(default_factory=list)
    actionable_selector: str


class PageManifest(BaseModel):
    schema_version: str
    entry_html: str
    selected_template_id: str
    blocks: List[PageManifestBlock]
    globals: List[PageManifestGlobal] = Field(default_factory=list)


class ShellResolutionCandidate(BaseModel):
    selector_hint: str
    region_role: Literal["hero", "section", "gallery", "table", "footer", "nav"]
    score: float
    reason: str
    preview_image_path: str = ""


class ShellBindingReview(BaseModel):
    block_id: str
    block_title: str
    original_selector_hint: str
    failure_reason: str
    template_entry_html: str
    template_preview_path: str = ""
    candidates: List[ShellResolutionCandidate] = Field(default_factory=list)


class ShellManualSelection(BaseModel):
    block_id: str
    selector_hint: str


class LayoutSectionOption(BaseModel):
    selector_hint: str
    region_role: Literal["hero", "section", "gallery", "table", "footer", "nav"]
    dom_index: int
    score: float
    reason: str
    preview_image_path: str = ""
    overlay_label: str = ""


class LayoutFigureOption(BaseModel):
    image_path: str
    caption: Optional[str] = None
    type: str
    source_section: str
    preview_image_path: str = ""


class LayoutComposeBlock(BaseModel):
    block_id: str
    title: str
    source_sections: List[str] = Field(default_factory=list)
    current_order: int
    selected_selector_hint: str = ""
    selected_figure_paths: List[str] = Field(default_factory=list)
    section_options: List[LayoutSectionOption] = Field(default_factory=list)
    figure_options: List[LayoutFigureOption] = Field(default_factory=list)


class LayoutComposeSession(BaseModel):
    template_entry_html: str
    template_preview_path: str = ""
    blocks: List[LayoutComposeBlock] = Field(default_factory=list)
    active_block_id: Optional[str] = None
    validation_errors: List[str] = Field(default_factory=list)


class LayoutComposeUpdate(BaseModel):
    active_block_id: Optional[str] = None
    selected_selector_hint: Optional[str] = None
    selected_figure_paths: Optional[List[str]] = None
    order_action: Optional[Literal["move_up", "move_down"]] = None
    action: str = ""


class VisualSmokeReport(BaseModel):
    passed: bool = True
    issue_class: Literal["none", "cosmetic", "structure"] = "none"
    suggested_recovery: Literal["accept", "patch_or_review", "rerun_planner"] = "accept"
    issues: List[str] = Field(default_factory=list)
    selectors_to_remove: List[str] = Field(default_factory=list)
    css_rules_to_inject: List[str] = Field(default_factory=list)
    screenshot_path: str = ""


class RevisionEdit(BaseModel):
    block_id: Optional[str] = None
    slot_id: Optional[SlotId] = None
    global_id: Optional[GlobalAnchorId] = None
    scope: Literal["slot", "block", "global"]
    change_request: str
    preserve_requirements: List[str] = Field(default_factory=list)
    acceptance_hint: str = ""

    @model_validator(mode="after")
    def validate_scope(self) -> "RevisionEdit":
        if self.scope == "slot":
            if not self.block_id or not self.slot_id:
                raise ValueError("slot scope edits must provide block_id and slot_id.")
            self.global_id = None
        elif self.scope == "block":
            if not self.block_id:
                raise ValueError("block scope edits must provide block_id.")
            self.slot_id = None
            self.global_id = None
        elif self.scope == "global":
            if not self.global_id:
                raise ValueError("global scope edits must provide global_id.")
            self.block_id = None
            self.slot_id = None
        else:
            raise ValueError(f"Unsupported revision scope '{self.scope}'.")
        return self


class RevisionPlan(BaseModel):
    edits: List[RevisionEdit] = Field(default_factory=list)


class StyleChange(BaseModel):
    block_id: Optional[str] = None
    slot_id: Optional[SlotId] = None
    global_id: Optional[GlobalAnchorId] = None
    scope: Literal["slot", "block", "global"]
    declarations: Dict[StylePropertyName, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_scope(self) -> "StyleChange":
        if not self.declarations:
            raise ValueError("style changes must include at least one declaration.")
        if self.scope == "slot":
            if not self.block_id or not self.slot_id:
                raise ValueError("slot scope style changes must provide block_id and slot_id.")
            self.global_id = None
        elif self.scope == "block":
            if not self.block_id:
                raise ValueError("block scope style changes must provide block_id.")
            self.slot_id = None
            self.global_id = None
        elif self.scope == "global":
            if not self.global_id:
                raise ValueError("global scope style changes must provide global_id.")
            self.block_id = None
            self.slot_id = None
        else:
            raise ValueError(f"Unsupported style change scope '{self.scope}'.")
        return self


class AttributeChange(BaseModel):
    block_id: Optional[str] = None
    slot_id: Optional[SlotId] = None
    global_id: Optional[GlobalAnchorId] = None
    scope: Literal["slot", "block", "global"]
    attributes: Dict[AttributeName, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_scope(self) -> "AttributeChange":
        if not self.attributes:
            raise ValueError("attribute changes must include at least one attribute.")
        if self.scope == "slot":
            if not self.block_id or not self.slot_id:
                raise ValueError("slot scope attribute changes must provide block_id and slot_id.")
            self.global_id = None
        elif self.scope == "block":
            if not self.block_id:
                raise ValueError("block scope attribute changes must provide block_id.")
            self.slot_id = None
            self.global_id = None
        elif self.scope == "global":
            if not self.global_id:
                raise ValueError("global scope attribute changes must provide global_id.")
            self.block_id = None
            self.slot_id = None
        else:
            raise ValueError(f"Unsupported attribute change scope '{self.scope}'.")
        return self


class TargetedReplacement(BaseModel):
    block_id: Optional[str] = None
    slot_id: Optional[SlotId] = None
    global_id: Optional[GlobalAnchorId] = None
    scope: Literal["slot", "block", "global"]
    html: str

    @model_validator(mode="after")
    def validate_scope(self) -> "TargetedReplacement":
        if self.scope == "slot":
            if not self.block_id or not self.slot_id:
                raise ValueError("slot scope replacements must provide block_id and slot_id.")
            self.global_id = None
        elif self.scope == "block":
            if not self.block_id:
                raise ValueError("block scope replacements must provide block_id.")
            self.slot_id = None
            self.global_id = None
        elif self.scope == "global":
            if not self.global_id:
                raise ValueError("global scope replacements must provide global_id.")
            self.block_id = None
            self.slot_id = None
        else:
            raise ValueError(f"Unsupported replacement scope '{self.scope}'.")
        return self


class OverrideCssRule(BaseModel):
    selector: str
    declarations: Dict[StylePropertyName, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_rule(self) -> "OverrideCssRule":
        if not str(self.selector or "").strip():
            raise ValueError("override css rules must provide selector.")
        if not self.declarations:
            raise ValueError("override css rules must include at least one declaration.")
        return self


class FallbackBlock(BaseModel):
    block_id: str
    reason: str


class TargetedReplacementPlan(BaseModel):
    replacements: List[TargetedReplacement] = Field(default_factory=list)
    style_changes: List[StyleChange] = Field(default_factory=list)
    attribute_changes: List[AttributeChange] = Field(default_factory=list)
    override_css_rules: List[OverrideCssRule] = Field(default_factory=list)
    fallback_blocks: List[FallbackBlock] = Field(default_factory=list)
