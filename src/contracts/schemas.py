from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

GlobalAnchorId = Literal["header_brand", "header_primary_action", "header_nav", "footer_meta"]
SlotId = Literal["title", "summary", "body", "media", "meta", "actions"]


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


class SemanticPlanMeta(BaseModel):
    plan_version: str = Field(description="Semantic planner schema version, e.g., '2.0'.")
    planning_mode: Literal["semantic_only"] = Field(description="Semantic-only planner mode identifier.")
    confidence: float = Field(description="Planner confidence in [0, 1].")

    @field_validator("planning_mode", mode="before")
    @classmethod
    def normalize_planning_mode(cls, value: str) -> str:
        if not isinstance(value, str):
            return value

        normalized = value.strip().lower()
        alias_map = {
            "semantic": "semantic_only",
            "semantic_first": "semantic_only",
            "semantic-first": "semantic_only",
            "semantic planner": "semantic_only",
        }
        return alias_map.get(normalized, value)


class SemanticTemplateSelection(BaseModel):
    selection_rationale: str = Field(description="Why the upstream-selected template is a good semantic fit.")
    fallback_template_id: Optional[str] = Field(
        default=None,
        description="Optional fallback template id from the candidate list.",
    )


class SemanticBlockPlan(BaseModel):
    block_id: str
    narrative_role: Literal["hook", "evidence", "method", "result", "conclusion", "supplement"]
    preferred_region_role: Literal["hero", "section", "gallery", "table", "footer", "nav"]
    presentation_priority: Literal["primary", "secondary", "supporting"]
    component_recipe: List["ComponentRecipeItem"]
    content_contract: "ContentContract"
    asset_binding: "AssetBinding"
    interaction: "InteractionPlan"
    layout_notes: Optional[str] = Field(
        default=None,
        description="Optional semantic layout or presentation intent for deterministic binding.",
    )
    a11y_notes: List[str]
    acceptance_checks: List[str]


class SemanticPagePlan(BaseModel):
    plan_meta: SemanticPlanMeta
    template_selection: SemanticTemplateSelection
    decision_summary: "DecisionSummary"
    adaptation_strategy: "AdaptationStrategy"
    global_design: "GlobalDesign"
    page_outline: List["PageOutlineItem"]
    semantic_blocks: List[SemanticBlockPlan]
    open_questions: List[str]


class TemplateCandidate(BaseModel):
    template_id: str
    root_dir: str
    chosen_entry_html: str
    score: float
    reasons: List[str]


class TemplateShellCandidate(BaseModel):
    selector: str
    role: Literal["hero", "section", "gallery", "table", "footer", "nav"]
    root_tag: str
    required_classes: List[str] = Field(default_factory=list)
    preserve_ids: List[str] = Field(default_factory=list)
    wrapper_chain: List["ShellWrapperSignature"] = Field(default_factory=list)
    dom_index: int = 0
    match_index: int = 0
    confidence: float = 0.0
    signals: List[str] = Field(default_factory=list)


class CanonicalShellNode(BaseModel):
    shell_id: str
    selector: str
    root_tag: str
    required_classes: List[str] = Field(default_factory=list)
    preserve_ids: List[str] = Field(default_factory=list)
    wrapper_chain: List["ShellWrapperSignature"] = Field(default_factory=list)
    actionable_root_selector: str
    region_role: Literal["hero", "section", "gallery", "table", "footer", "nav"]
    dom_index: int = 0
    match_index: int = 0
    confidence: float = 0.0
    bindable: bool = True
    signals: List[str] = Field(default_factory=list)
    risk_flags: List[str] = Field(default_factory=list)


class CanonicalGlobalAnchor(BaseModel):
    global_id: GlobalAnchorId
    selector: str
    target_tag: str
    required_classes: List[str] = Field(default_factory=list)
    preserve_ids: List[str] = Field(default_factory=list)
    actionable_selector: str
    confidence: float = 0.0
    risk_flags: List[str] = Field(default_factory=list)


class TemplateIR(BaseModel):
    schema_version: str = "1.0"
    shell_nodes: List[CanonicalShellNode] = Field(default_factory=list)
    global_anchors: List[CanonicalGlobalAnchor] = Field(default_factory=list)


class TemplateWidget(BaseModel):
    selector: str
    widget_type: str
    required_selectors: List[str] = Field(default_factory=list)
    script_dependencies: List[str] = Field(default_factory=list)
    risk_flags: List[str] = Field(default_factory=list)
    optional: bool = True


class TemplateProfile(BaseModel):
    template_id: str
    template_root_dir: str
    entry_html: str
    archetype: str
    template_ir: TemplateIR = Field(default_factory=TemplateIR)
    global_preserve_selectors: List[str] = Field(default_factory=list)
    shell_candidates: List[TemplateShellCandidate] = Field(default_factory=list)
    optional_widgets: List[TemplateWidget] = Field(default_factory=list)
    removable_demo_selectors: List[str] = Field(default_factory=list)
    unsafe_selectors: List[str] = Field(default_factory=list)
    compile_confidence: float = 0.0
    risk_flags: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)
    source_fingerprint: str

    @model_validator(mode="after")
    def sync_template_ir(self) -> "TemplateProfile":
        if not self.template_ir.shell_nodes and self.shell_candidates:
            self.template_ir = _template_ir_from_compat_fields(self.shell_candidates, self.global_preserve_selectors)
        elif not self.template_ir.global_anchors and self.global_preserve_selectors:
            self.template_ir = self.template_ir.model_copy(
                update={
                    "global_anchors": _compat_global_anchors_from_selectors(self.global_preserve_selectors),
                },
                deep=True,
            )

        self.shell_candidates = _compat_shell_candidates_from_template_ir(self.template_ir)
        self.global_preserve_selectors = _compat_global_preserve_selectors_from_template_ir(self.template_ir)
        return self


class PlanMeta(BaseModel):
    plan_version: str = Field(description="Planner schema version, e.g., '1.1'.")
    planning_mode: Literal["autopage_template_first", "hybrid_template_bind"] = Field(
        description="Planning mode identifier."
    )
    target_framework: str = Field(description="Target framework, e.g., 'static-html', 'react', 'vue'.")
    confidence: float = Field(description="Planner confidence in [0, 1].")
    render_strategy: Literal["compiled_block_assembly", "legacy_fullpage"] = Field(
        default="compiled_block_assembly",
        description="Preferred coder execution path for this plan.",
    )


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
    shell_id: str = Field(default="", description="Canonical TemplateIR shell identifier.")
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
    shell_id: str = ""
    root_tag: str
    required_classes: List[str] = Field(default_factory=list)
    preserve_ids: List[str] = Field(default_factory=list)
    wrapper_chain: List[ShellWrapperSignature] = Field(default_factory=list)
    actionable_root_selector: str
    match_index: int = 0


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
            "Compatibility mapping for preserved global template anchors. The new compiled-block "
            "path does not use dom_mapping as the primary page-generation interface."
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
    render_mode: Optional[Literal["compiled_block_assembly", "legacy_fullpage"]] = Field(
        default=None,
        description="Actual coder render mode used for this artifact.",
    )
    template_profile_path: Optional[str] = Field(
        default=None,
        description="Saved TemplateProfile path used for the build.",
    )
    page_manifest_path: Optional[str] = Field(
        default=None,
        description="Saved page manifest path for anchored revisions.",
    )
    block_artifact_dir: Optional[str] = Field(
        default=None,
        description="Directory containing per-block render artifacts when block assembly is used.",
    )
    fullpage_context_dir: Optional[str] = Field(
        default=None,
        description="Directory containing structure-guided fullpage digest artifacts when fullpage rendering is used.",
    )


class CoderCriticReport(BaseModel):
    is_build_valid: bool = Field(description="Whether coder output passes checks.")
    build_feedback: str = Field(description="Actionable feedback when build fails.")


_SUGGESTED_STYLE_PROPERTIES = [
    "font-size", "line-height", "margin", "margin-top", "margin-bottom",
    "padding", "gap", "text-align", "max-width", "width",
]
AttributeName = Literal["class", "href", "target", "aria-label", "style", "id"]


class ResolvedBlockBinding(BaseModel):
    block_id: str
    shell_id: str = ""
    selector: str
    region_role: Literal["hero", "section", "gallery", "table", "footer", "nav"]
    root_tag: str
    required_classes: List[str] = Field(default_factory=list)
    preserve_ids: List[str] = Field(default_factory=list)
    wrapper_chain: List[ShellWrapperSignature] = Field(default_factory=list)
    actionable_root_selector: str
    dom_index: int = 0
    match_index: int = 0


class BlockRenderSpec(BaseModel):
    block_id: str
    order: int
    title: str
    source_sections: List[str] = Field(default_factory=list)
    binding: ResolvedBlockBinding
    content_contract: ContentContract
    asset_binding: AssetBinding
    interaction: InteractionPlan
    responsive_rules: ResponsiveRules
    shell_contract: Optional[BlockShellContract] = None
    shell_html: str = ""
    allowed_slots: List[SlotId] = Field(
        default_factory=lambda: ["title", "summary", "body", "media", "meta", "actions"]
    )


class BlockRenderArtifact(BaseModel):
    block_id: str
    order: int
    selector: str
    match_index: int = 0
    render_mode: Literal["compiled_block_assembly", "legacy_fullpage"] = "compiled_block_assembly"
    html: str = ""
    html_path: str = ""
    metadata_path: str = ""
    screenshot_path: str = ""
    validation_errors: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class PageManifestSlot(BaseModel):
    slot_id: SlotId
    selector: str


class PageManifestBlock(BaseModel):
    block_id: str
    shell_id: str = ""
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
    shell_id: str = ""
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
    shell_id: str = ""
    selector_hint: str


class LayoutSectionOption(BaseModel):
    shell_id: str = ""
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


class ReviewItem(BaseModel):
    severity: Literal["high", "medium", "low"]
    target: str
    advice: str


class ReviewerReport(BaseModel):
    reviewer: Literal["semantic_visual", "layout_rhythm", "polish"]
    items: List[ReviewItem] = Field(default_factory=list)


class ArbiterReport(BaseModel):
    items: List[ReviewItem] = Field(default_factory=list)


class AnchorChildStyle(BaseModel):
    tag: str
    classes: List[str] = Field(default_factory=list)
    key_styles: Dict[str, str] = Field(default_factory=dict)


class AnchorStyleSnapshot(BaseModel):
    anchor_type: Literal["block", "global"]
    anchor_id: str
    selector: str
    applied_classes: List[str] = Field(default_factory=list)
    key_styles: Dict[str, str] = Field(default_factory=dict)
    children: List[AnchorChildStyle] = Field(default_factory=list)


class TemplateStyleContext(BaseModel):
    template_id: str = ""
    framework_hint: str = "custom"
    css_custom_properties: Dict[str, str] = Field(default_factory=dict)
    anchor_snapshots: List[AnchorStyleSnapshot] = Field(default_factory=list)


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
    declarations: Dict[str, str] = Field(default_factory=dict)

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
    declarations: Dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_rule(self) -> "OverrideCssRule":
        if not str(self.selector or "").strip():
            raise ValueError("override css rules must provide selector.")
        if not self.declarations:
            raise ValueError("override css rules must include at least one declaration.")
        return self


class CssRevisionRule(BaseModel):
    selector: str
    declarations: Dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_rule(self) -> "CssRevisionRule":
        clean = str(self.selector or "").strip()
        if not clean:
            raise ValueError("CSS revision rules must provide a selector.")
        if not self.declarations:
            raise ValueError("CSS revision rules must include at least one declaration.")
        if any(char in clean for char in ("{", "}", "@", ";", "\n", "\r")):
            raise ValueError(f"Selector contains forbidden characters: '{clean}'")
        return self


class ContentReplacement(BaseModel):
    block_id: Optional[str] = None
    slot_id: Optional[SlotId] = None
    global_id: Optional[GlobalAnchorId] = None
    scope: Literal["slot", "block", "global"]
    html: str

    @model_validator(mode="after")
    def validate_scope(self) -> "ContentReplacement":
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


class CssRevisionPlan(BaseModel):
    css_rules: List[CssRevisionRule] = Field(default_factory=list)
    content_replacements: List[ContentReplacement] = Field(default_factory=list)
    revision_summary: str = ""
    not_possible_explanation: str = ""


class RevisionHistoryEntry(BaseModel):
    version_id: str
    created_at: str
    source: Literal["initial_draft", "webpage_revision"]
    summary: str = ""
    html_files: List[str] = Field(default_factory=list)


class RevisionHistory(BaseModel):
    current_version_id: str = ""
    versions: List[RevisionHistoryEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_current_version(self) -> "RevisionHistory":
        current = str(self.current_version_id or "").strip()
        if not current:
            return self
        version_ids = {entry.version_id for entry in self.versions}
        if current not in version_ids:
            raise ValueError(f"Unknown current_version_id '{current}'.")
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


def _guess_global_anchor_id_from_selector(selector: str) -> GlobalAnchorId | None:
    lowered = str(selector or "").strip().lower()
    if not lowered:
        return None
    if "footer" in lowered:
        return "footer_meta"
    if any(token in lowered for token in ("button-group", "button", "cta", "action")):
        return "header_primary_action"
    if "header" in lowered and any(token in lowered for token in ("h1", "logo", "brand")):
        return "header_brand"
    if "nav" in lowered or "navbar" in lowered:
        return "header_nav"
    if "header" in lowered:
        return "header_nav"
    if "menu" in lowered and any(token in lowered for token in ("a", "title", "brand")):
        return "header_brand"
    return None


def _compat_global_anchors_from_selectors(selectors: List[str]) -> List[CanonicalGlobalAnchor]:
    anchors: List[CanonicalGlobalAnchor] = []
    seen: set[str] = set()
    for selector in selectors:
        clean = str(selector or "").strip()
        global_id = _guess_global_anchor_id_from_selector(clean)
        if not clean or not global_id or global_id in seen:
            continue
        seen.add(global_id)
        anchors.append(
            CanonicalGlobalAnchor(
                global_id=global_id,
                selector=clean,
                target_tag="",
                required_classes=[],
                preserve_ids=[],
                actionable_selector=clean,
                confidence=0.0,
                risk_flags=[],
            )
        )
    return anchors


def _compat_shell_candidates_from_template_ir(template_ir: TemplateIR) -> List[TemplateShellCandidate]:
    shell_candidates: List[TemplateShellCandidate] = []
    for shell_node in sorted(template_ir.shell_nodes, key=lambda item: (item.dom_index, item.selector)):
        if not shell_node.bindable:
            continue
        shell_candidates.append(
            TemplateShellCandidate(
                selector=str(shell_node.selector or "").strip(),
                role=shell_node.region_role,
                root_tag=shell_node.root_tag,
                required_classes=list(shell_node.required_classes),
                preserve_ids=list(shell_node.preserve_ids),
                wrapper_chain=list(shell_node.wrapper_chain),
                dom_index=int(shell_node.dom_index),
                match_index=int(shell_node.match_index),
                confidence=float(shell_node.confidence or 0.0),
                signals=list(shell_node.signals),
            )
        )
    return shell_candidates


def _compat_global_preserve_selectors_from_template_ir(template_ir: TemplateIR) -> List[str]:
    selectors: List[str] = []
    seen: set[str] = set()
    for anchor in template_ir.global_anchors:
        selector = str(anchor.selector or "").strip()
        if selector and selector not in seen:
            seen.add(selector)
            selectors.append(selector)
    return selectors


def _template_ir_from_compat_fields(
    shell_candidates: List[TemplateShellCandidate],
    global_preserve_selectors: List[str],
) -> TemplateIR:
    shell_nodes: List[CanonicalShellNode] = []
    for candidate in shell_candidates:
        selector = str(candidate.selector or "").strip()
        if not selector:
            continue
        shell_nodes.append(
            CanonicalShellNode(
                shell_id=f"compat_shell_{len(shell_nodes) + 1:02d}",
                selector=selector,
                root_tag=candidate.root_tag,
                required_classes=list(candidate.required_classes),
                preserve_ids=list(candidate.preserve_ids),
                wrapper_chain=list(candidate.wrapper_chain),
                actionable_root_selector=selector,
                region_role=candidate.role,
                dom_index=int(candidate.dom_index),
                match_index=int(candidate.match_index),
                confidence=float(candidate.confidence or 0.0),
                bindable=True,
                signals=list(candidate.signals),
                risk_flags=[],
            )
        )

    return TemplateIR(
        shell_nodes=shell_nodes,
        global_anchors=_compat_global_anchors_from_selectors(global_preserve_selectors),
    )

