from __future__ import annotations

from typing import Iterable

from bs4 import BeautifulSoup, Tag

from src.contracts.schemas import (
    BlockPlan,
    BlockShellContract,
    CanonicalGlobalAnchor,
    CanonicalShellNode,
    GlobalAnchorId,
    PagePlan,
    TemplateProfile,
)
from src.template.structural_core import build_unique_selector

PRIMARY_GLOBAL_TAGS: dict[str, set[str]] = {
    "header_brand": {"a", "button"},
    "header_primary_action": {"a", "button"},
    "header_nav": {"nav"},
    "footer_meta": {"footer"},
}
FALLBACK_GLOBAL_TAGS: dict[str, set[str]] = {
    "header_brand": {"h1", "h2", "h3", "div", "span"},
    "header_primary_action": set(),
    "header_nav": {"ul", "div"},
    "footer_meta": {"section", "div", "small", "p"},
}


def selector_global_id(selector: str) -> GlobalAnchorId | None:
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


def _find_ancestor_with_tags(tag: Tag, allowed_tags: set[str]) -> Tag | None:
    for ancestor in tag.parents:
        if not isinstance(ancestor, Tag):
            continue
        if (ancestor.name or "") in allowed_tags:
            return ancestor
        if (ancestor.name or "") in {"html", "body"}:
            break
    return None


def _find_unique_descendant_with_tags(tag: Tag, allowed_tags: set[str]) -> Tag | None:
    if not allowed_tags:
        return None
    matches = [candidate for candidate in tag.find_all(allowed_tags) if isinstance(candidate, Tag)]
    if len(matches) == 1:
        return matches[0]
    return None


def resolve_global_anchor_target(tag: Tag, global_id: str) -> Tag:
    primary_tags = PRIMARY_GLOBAL_TAGS.get(global_id, set())
    fallback_tags = FALLBACK_GLOBAL_TAGS.get(global_id, set())
    tag_name = str(tag.name or "")

    if tag_name in primary_tags:
        return tag

    ancestor = _find_ancestor_with_tags(tag, primary_tags)
    if ancestor is not None:
        return ancestor

    descendant = _find_unique_descendant_with_tags(tag, primary_tags)
    if descendant is not None:
        return descendant

    if tag_name in fallback_tags:
        return tag

    ancestor = _find_ancestor_with_tags(tag, fallback_tags)
    if ancestor is not None:
        return ancestor

    descendant = _find_unique_descendant_with_tags(tag, fallback_tags)
    if descendant is not None:
        return descendant

    return tag


def build_shell_contract(shell_node: CanonicalShellNode) -> BlockShellContract:
    return BlockShellContract(
        shell_id=str(shell_node.shell_id or "").strip(),
        root_tag=str(shell_node.root_tag or "div"),
        required_classes=list(shell_node.required_classes or []),
        preserve_ids=list(shell_node.preserve_ids or []),
        wrapper_chain=list(shell_node.wrapper_chain or []),
        actionable_root_selector=str(shell_node.actionable_root_selector or shell_node.selector or "").strip(),
        match_index=shell_node.match_index,
    )


def canonical_shell_nodes(
    template_profile: TemplateProfile,
    *,
    bindable_only: bool = False,
) -> list[CanonicalShellNode]:
    nodes = sorted(template_profile.template_ir.shell_nodes, key=lambda item: (item.dom_index, item.selector))
    if bindable_only:
        return [node for node in nodes if node.bindable]
    return nodes


def canonical_global_anchors(template_profile: TemplateProfile) -> list[CanonicalGlobalAnchor]:
    return list(template_profile.template_ir.global_anchors)


def shell_node_by_id(template_profile: TemplateProfile, shell_id: str) -> CanonicalShellNode | None:
    clean_shell_id = str(shell_id or "").strip()
    if not clean_shell_id:
        return None
    return next(
        (node for node in template_profile.template_ir.shell_nodes if str(node.shell_id or "").strip() == clean_shell_id),
        None,
    )


def shell_node_by_selector(template_profile: TemplateProfile, selector: str) -> CanonicalShellNode | None:
    clean_selector = str(selector or "").strip()
    if not clean_selector:
        return None
    return next(
        (node for node in template_profile.template_ir.shell_nodes if str(node.selector or "").strip() == clean_selector),
        None,
    )


def resolve_shell_node(
    template_profile: TemplateProfile,
    *,
    shell_id: str = "",
    selector: str = "",
    bindable_only: bool = False,
) -> CanonicalShellNode | None:
    node = shell_node_by_id(template_profile, shell_id) or shell_node_by_selector(template_profile, selector)
    if node is None:
        return None
    if bindable_only and not node.bindable:
        return None
    return node


def bind_block_to_shell(block: BlockPlan, shell_node: CanonicalShellNode) -> BlockPlan:
    target_region = block.target_template_region.model_copy(
        update={
            "shell_id": str(shell_node.shell_id or "").strip(),
            "selector_hint": str(shell_node.selector or "").strip(),
            "region_role": shell_node.region_role,
        },
        deep=True,
    )
    return block.model_copy(
        update={
            "target_template_region": target_region,
            "shell_contract": build_shell_contract(shell_node),
        },
        deep=True,
    )


def bind_page_plan_to_template_ir(
    page_plan: PagePlan,
    template_profile: TemplateProfile,
    *,
    fill_missing_only: bool = False,
) -> PagePlan:
    updated_blocks: list[BlockPlan] = []
    for block in page_plan.blocks:
        if (
            fill_missing_only
            and isinstance(block.shell_contract, BlockShellContract)
            and str(block.target_template_region.shell_id or "").strip()
        ):
            updated_blocks.append(block)
            continue
        shell_node = resolve_shell_node(
            template_profile,
            shell_id=str(block.target_template_region.shell_id or "").strip(),
            selector=str(block.target_template_region.selector_hint or "").strip(),
        )
        if shell_node is None:
            updated_blocks.append(block)
            continue
        updated_blocks.append(bind_block_to_shell(block, shell_node))
    return page_plan.model_copy(update={"blocks": updated_blocks}, deep=True)


def build_global_anchor_from_tag(
    selector: str,
    global_id: str,
    tag: Tag,
    soup: BeautifulSoup,
    *,
    confidence: float = 0.0,
    risk_flags: Iterable[str] | None = None,
) -> CanonicalGlobalAnchor:
    actionable_target = resolve_global_anchor_target(tag, global_id)
    actionable_selector = build_unique_selector(actionable_target, soup)
    return CanonicalGlobalAnchor(
        global_id=global_id,
        selector=str(selector or "").strip(),
        target_tag=str(actionable_target.name or ""),
        required_classes=[str(item).strip() for item in actionable_target.get("class", []) if str(item).strip()],
        preserve_ids=[str(actionable_target.get("id") or "").strip()] if str(actionable_target.get("id") or "").strip() else [],
        actionable_selector=actionable_selector,
        confidence=float(confidence or 0.0),
        risk_flags=[str(flag).strip() for flag in (risk_flags or []) if str(flag).strip()],
    )
