from __future__ import annotations

from typing import Any

from src.contracts.schemas import (
    BlockPlan,
    CoderHandoff,
    FileTouchItem,
    PagePlan,
    PlanMeta,
    QualityCheck,
    ResponsiveRules,
    SemanticBlockPlan,
    SemanticPagePlan,
    TargetTemplateRegion,
    TemplateCandidate,
    TemplateSelection,
    TemplateProfile,
)
from src.template.template_ir import build_shell_contract, canonical_shell_nodes

_RUNTIME_RISK_FLAGS = {"fetch_runtime_dependency", "chart_runtime_dependency", "math_runtime_dependency"}


def _dedupe_strings(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        clean = str(value or "").strip()
        if clean and clean not in seen:
            seen.add(clean)
            deduped.append(clean)
    return deduped


def _compat_dom_mapping(template_profile: TemplateProfile) -> dict[str, str]:
    return {
        selector: "preserve_global_anchor"
        for selector in template_profile.global_preserve_selectors
        if str(selector or "").strip()
    }


def _selectors_to_remove(template_profile: TemplateProfile) -> list[str]:
    protected_selectors = {
        selector
        for selector in template_profile.global_preserve_selectors
        if str(selector or "").strip()
    }
    protected_selectors.update(
        str(candidate.selector or "").strip()
        for candidate in canonical_shell_nodes(template_profile, bindable_only=True)
        if str(candidate.selector or "").strip()
    )
    return [
        selector
        for selector in _dedupe_strings(list(template_profile.removable_demo_selectors or []))
        if selector not in protected_selectors
    ]


def _render_strategy_risks(template_profile: TemplateProfile) -> list[str]:
    risks: list[str] = []
    if float(template_profile.compile_confidence or 0.0) < 0.7:
        risks.append(
            f"template_compile_confidence={template_profile.compile_confidence:.2f} is below compiled block threshold 0.70"
        )
    high_risk_widget_types = {
        widget.widget_type
        for widget in template_profile.optional_widgets
        if _RUNTIME_RISK_FLAGS & set(widget.risk_flags)
    }
    if high_risk_widget_types:
        risks.append(
            "template widgets require risky runtime handling: " + ", ".join(sorted(high_risk_widget_types))
        )
    for risk_flag in template_profile.risk_flags:
        if risk_flag in _RUNTIME_RISK_FLAGS and risk_flag not in risks:
            risks.append(risk_flag)
    return _dedupe_strings(risks)


def _role_compatibility_score(preferred_role: str, actual_role: str) -> float:
    if preferred_role == actual_role:
        return 8.0
    if preferred_role in {"hero", "section"} and actual_role in {"hero", "section"}:
        return 4.5
    if preferred_role == "gallery" and actual_role == "section":
        return 2.5
    if preferred_role == "table" and actual_role == "section":
        return 2.0
    if preferred_role == "section" and actual_role in {"gallery", "table"}:
        return 1.0
    return -2.5


def _media_compatibility_score(block: SemanticBlockPlan, shell_node: Any) -> float:
    has_media = bool(block.asset_binding.asset_ids or block.asset_binding.template_asset_fallback)
    if not has_media:
        return -0.8 if shell_node.region_role in {"gallery", "table"} else 0.0

    score = 0.0
    if shell_node.region_role == "gallery":
        score += 3.5
    elif shell_node.region_role == "table":
        score += 3.0
    elif shell_node.region_role in {"hero", "section"}:
        score += 1.5

    signal_blob = " ".join(str(signal or "").lower() for signal in shell_node.signals)
    if any(token in signal_blob for token in ("image", "media", "gallery", "chart", "figure", "table")):
        score += 1.5
    return score


def _heading_affordance_score(block: SemanticBlockPlan, shell_node: Any) -> float:
    score = 0.0
    if block.content_contract.headline:
        if shell_node.region_role in {"hero", "section"}:
            score += 1.2
        signal_blob = " ".join(str(signal or "").lower() for signal in shell_node.signals)
        if any(token in signal_blob for token in ("hero", "headline", "title", "masthead")):
            score += 0.8
    if block.interaction.pattern != "none":
        if shell_node.region_role == "section":
            score += 0.6
        signal_blob = " ".join(str(signal or "").lower() for signal in shell_node.signals)
        if block.interaction.pattern in signal_blob:
            score += 0.8
    return score


def _stability_score(shell_node: Any) -> float:
    score = float(shell_node.confidence or 0.0) * 4.0
    if not shell_node.risk_flags:
        score += 1.0
    score += min(1.0, 0.2 * len(shell_node.required_classes or []))
    if shell_node.preserve_ids:
        score += 0.5
    return score


def _priority_score(block: SemanticBlockPlan, shell_node: Any) -> float:
    score = {
        "primary": 2.5,
        "secondary": 1.2,
        "supporting": 0.3,
    }.get(block.presentation_priority, 0.0)
    if block.presentation_priority == "primary" and block.narrative_role in {"hook", "result", "evidence"}:
        if shell_node.region_role == "hero":
            score += 2.0
        elif shell_node.region_role in {"gallery", "table"}:
            score += 1.0
    return score


def _dom_order_score(expected_position: int, shell_position: int) -> float:
    return max(-3.0, 2.5 - abs(expected_position - shell_position) * 0.75)


def _choose_shell_node(
    block: SemanticBlockPlan,
    *,
    expected_position: int,
    bindable_shells: list[Any],
    used_shell_ids: set[str],
) -> Any:
    candidate_pool = [node for node in bindable_shells if node.shell_id not in used_shell_ids] or bindable_shells

    best_candidate = candidate_pool[0]
    best_score = float("-inf")
    for shell_position, candidate in enumerate(candidate_pool):
        score = 0.0
        score += _role_compatibility_score(block.preferred_region_role, str(candidate.region_role or "section"))
        score += _media_compatibility_score(block, candidate)
        score += _heading_affordance_score(block, candidate)
        score += _stability_score(candidate)
        score += _priority_score(block, candidate)
        score += _dom_order_score(expected_position, shell_position)
        if candidate.shell_id in used_shell_ids:
            score -= 5.0
        if score > best_score:
            best_candidate = candidate
            best_score = score
    return best_candidate


def _derive_operation(block: SemanticBlockPlan, shell_node: Any) -> str:
    if shell_node.region_role in {"gallery", "table"} and block.asset_binding.asset_ids:
        return "replace_media"
    if block.interaction.pattern != "none" and shell_node.region_role == "section":
        return "append_child"
    return "replace_text"


def _derive_desktop_layout(block: SemanticBlockPlan, shell_node: Any) -> str:
    layout_notes = str(block.layout_notes or "").lower()
    if "split" in layout_notes:
        return "split"
    if "grid" in layout_notes:
        return "grid"
    if shell_node.region_role == "gallery":
        return "gallery_grid"
    if shell_node.region_role == "table":
        return "table_focus"
    if block.interaction.pattern != "none":
        return "interactive_stack"
    if block.asset_binding.asset_ids:
        return "media_split"
    return "stack"


def _bound_quality_checks(
    *,
    semantic_validation_passed: bool,
    semantic_feedback_history: list[str] | None,
    bound_critiques: list[str] | None = None,
) -> list[QualityCheck]:
    semantic_feedback = _dedupe_strings(list(semantic_feedback_history or []))
    bound_critiques = _dedupe_strings(list(bound_critiques or []))

    semantic_note = semantic_feedback[-1] if semantic_feedback else "Semantic plan passed grounding and consistency review."
    template_path_critiques = [
        critique
        for critique in bound_critiques
        if any(token in critique for token in ("selected_root_dir", "selected_entry_html", "file_touch_plan"))
    ]
    feasibility_critiques = [
        critique for critique in bound_critiques if critique not in template_path_critiques
    ]

    return [
        QualityCheck(
            name="grounding_check",
            passed=semantic_validation_passed,
            note=semantic_note,
        ),
        QualityCheck(
            name="consistency_check",
            passed=semantic_validation_passed,
            note=semantic_note,
        ),
        QualityCheck(
            name="feasibility_check",
            passed=not feasibility_critiques,
            note="; ".join(feasibility_critiques) if feasibility_critiques else "Deterministic binding feasibility checks passed.",
        ),
        QualityCheck(
            name="template_path_check",
            passed=not template_path_critiques,
            note="; ".join(template_path_critiques) if template_path_critiques else "Bound template paths and file touch plan are valid.",
        ),
    ]


def bind_semantic_plan(
    semantic_plan: SemanticPagePlan,
    selected_template: TemplateCandidate,
    template_profile: TemplateProfile,
    generation_constraints: dict[str, Any] | None = None,
    template_candidates: list[TemplateCandidate] | None = None,
    semantic_validation_passed: bool = True,
    semantic_feedback_history: list[str] | None = None,
    bound_critiques: list[str] | None = None,
) -> PagePlan:
    constraints = dict(generation_constraints or {})
    outline_items = sorted(semantic_plan.page_outline, key=lambda item: (item.order, item.block_id))
    outline_lookup = {item.block_id: item for item in outline_items}
    semantic_blocks = {block.block_id: block for block in semantic_plan.semantic_blocks}
    bindable_shells = canonical_shell_nodes(template_profile, bindable_only=True)
    if not bindable_shells:
        raise ValueError("TemplateProfile has no bindable TemplateIR shell nodes.")

    used_shell_ids: set[str] = set()
    final_blocks: list[BlockPlan] = []
    for expected_position, outline_item in enumerate(outline_items):
        semantic_block = semantic_blocks.get(outline_item.block_id)
        if semantic_block is None:
            raise ValueError(f"Semantic plan is missing semantic block '{outline_item.block_id}'.")

        chosen_shell = _choose_shell_node(
            semantic_block,
            expected_position=expected_position,
            bindable_shells=bindable_shells,
            used_shell_ids=used_shell_ids,
        )
        used_shell_ids.add(str(chosen_shell.shell_id or "").strip())
        final_blocks.append(
            BlockPlan(
                block_id=semantic_block.block_id,
                target_template_region=TargetTemplateRegion(
                    shell_id=str(chosen_shell.shell_id or "").strip(),
                    selector_hint=str(chosen_shell.selector or "").strip(),
                    region_role=chosen_shell.region_role,
                    operation=_derive_operation(semantic_block, chosen_shell),
                ),
                component_recipe=list(semantic_block.component_recipe),
                content_contract=semantic_block.content_contract,
                asset_binding=semantic_block.asset_binding,
                interaction=semantic_block.interaction,
                responsive_rules=ResponsiveRules(
                    mobile_order=int(outline_item.order),
                    desktop_layout=_derive_desktop_layout(semantic_block, chosen_shell),
                ),
                shell_contract=build_shell_contract(chosen_shell),
                a11y_notes=list(semantic_block.a11y_notes),
                acceptance_checks=list(semantic_block.acceptance_checks),
            )
        )

    render_strategy_risks = _render_strategy_risks(template_profile)
    render_strategy = "legacy_fullpage"
    target_framework = str(
        constraints.get("target_framework")
        or constraints.get("framework")
        or constraints.get("preferred_framework")
        or "static-html"
    ).strip()

    fallback_template_id = str(semantic_plan.template_selection.fallback_template_id or "").strip() or None
    if fallback_template_id is None:
        for candidate in template_candidates or []:
            if candidate.template_id != selected_template.template_id:
                fallback_template_id = candidate.template_id
                break

    entry_path = "/".join(
        part.strip("/")
        for part in [str(selected_template.root_dir or "").strip(), str(selected_template.chosen_entry_html or "").strip()]
        if part
    )
    hard_constraints = [
        "Preserve canonical TemplateIR shell selectors and contracts.",
        "Preserve global anchors from dom_mapping.",
        "Use only grounded asset_ids from StructuredPaper.asset_registry.",
    ]

    return PagePlan(
        plan_meta=PlanMeta(
            plan_version=semantic_plan.plan_meta.plan_version,
            planning_mode="hybrid_template_bind",
            target_framework=target_framework or "static-html",
            confidence=round(
                min(
                    1.0,
                    (float(semantic_plan.plan_meta.confidence or 0.0) + float(template_profile.compile_confidence or 0.0))
                    / 2.0,
                ),
                4,
            ),
            render_strategy=render_strategy,
        ),
        template_selection=TemplateSelection(
            selected_template_id=selected_template.template_id,
            selected_root_dir=selected_template.root_dir,
            selected_entry_html=selected_template.chosen_entry_html,
            fallback_template_id=fallback_template_id,
            selection_rationale=semantic_plan.template_selection.selection_rationale,
        ),
        decision_summary=semantic_plan.decision_summary,
        adaptation_strategy=semantic_plan.adaptation_strategy,
        global_design=semantic_plan.global_design,
        page_outline=outline_items,
        blocks=final_blocks,
        dom_mapping=_compat_dom_mapping(template_profile),
        selectors_to_remove=_selectors_to_remove(template_profile),
        coder_handoff=CoderHandoff(
            implementation_order=[item.block_id for item in outline_items],
            file_touch_plan=[
                FileTouchItem(
                    path=entry_path or str(selected_template.chosen_entry_html or "").strip(),
                    action="edit",
                    reason="Bind semantic plan onto the selected template entry HTML.",
                )
            ],
            hard_constraints=hard_constraints,
            known_risks=_dedupe_strings(render_strategy_risks),
        ),
        quality_checks=_bound_quality_checks(
            semantic_validation_passed=semantic_validation_passed,
            semantic_feedback_history=semantic_feedback_history,
            bound_critiques=bound_critiques,
        ),
        open_questions=list(semantic_plan.open_questions),
    )
