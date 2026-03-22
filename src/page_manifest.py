from __future__ import annotations

import json
from pathlib import Path

from bs4 import BeautifulSoup, Tag

from src.schemas import (
    BlockPlan,
    BlockShellContract,
    GlobalAnchorId,
    PageManifest,
    PageManifestBlock,
    PageManifestGlobal,
    PageManifestSlot,
    PagePlan,
    ShellWrapperSignature,
)

BLOCK_ATTR = "data-pa-block"
SLOT_ATTR = "data-pa-slot"
GLOBAL_ATTR = "data-pa-global"
ALLOWED_SLOT_IDS = {"title", "summary", "body", "media", "meta", "actions"}
ALLOWED_GLOBAL_IDS: set[str] = {"header_brand", "header_primary_action", "header_nav", "footer_meta"}
PAGE_MANIFEST_SCHEMA_VERSION = "1.2"
_MAX_WRAPPER_CHAIN_DEPTH = 2
_PRIMARY_GLOBAL_TAGS: dict[str, set[str]] = {
    "header_brand": {"a", "button"},
    "header_primary_action": {"a", "button"},
    "header_nav": {"nav"},
    "footer_meta": {"footer"},
}
_FALLBACK_GLOBAL_TAGS: dict[str, set[str]] = {
    "header_brand": {"h1", "h2", "h3", "div", "span"},
    "header_primary_action": set(),
    "header_nav": {"ul", "div"},
    "footer_meta": {"section", "div", "small", "p"},
}

_SHELL_CONTAINER_TAGS = {"section", "div", "article", "header", "main", "aside", "nav", "footer"}
_SHELL_GENERIC_TOKENS = {
    "content",
    "container",
    "wrapper",
    "box",
    "visual",
    "section",
    "main",
}
_REGION_ROLE_HINTS: dict[str, set[str]] = {
    "hero": {"hero", "lead", "intro", "header", "catchphrase", "headline"},
    "section": {"section", "content", "body", "copy", "swatch"},
    "gallery": {"gallery", "media", "figure", "carousel"},
    "table": {"table", "results", "metrics", "data"},
    "footer": {"footer", "meta", "citation"},
    "nav": {"nav", "menu", "brand", "action"},
}


def build_block_selector(block_id: str) -> str:
    return f'[{BLOCK_ATTR}="{block_id}"]'


def build_slot_selector(block_id: str, slot_id: str) -> str:
    return f'{build_block_selector(block_id)} [{SLOT_ATTR}="{slot_id}"]'


def build_global_selector(global_id: str) -> str:
    return f'[{GLOBAL_ATTR}="{global_id}"]'


def build_page_manifest_path(entry_html_path: str | Path) -> Path:
    return Path(entry_html_path).resolve().parent.parent / "page_manifest.json"


def save_page_manifest(path: Path, manifest: PageManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_page_manifest(path: Path) -> PageManifest | None:
    if not path.exists():
        return None
    try:
        return PageManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _selector_global_id(selector: str) -> GlobalAnchorId | None:
    lowered = str(selector or "").lower()
    if not lowered:
        return None

    if "footer" in lowered:
        return "footer_meta"
    if any(token in lowered for token in ("button-group", "button", "cta", "action")):
        return "header_primary_action"
    if "nav" in lowered or "navbar" in lowered:
        return "header_nav"
    if any(token in lowered for token in ("h1", "logo", "brand")) and "header" in lowered:
        return "header_brand"
    if "menu" in lowered and any(token in lowered for token in ("a", "title", "brand")):
        return "header_brand"
    return None


def _dedupe_strings(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        clean = str(value or "").strip()
        if clean and clean not in seen:
            seen.add(clean)
            deduped.append(clean)
    return deduped


def _tag_classes(tag: Tag) -> list[str]:
    return _dedupe_strings([str(item).strip() for item in tag.get("class", [])])


def _tag_ids(tag: Tag) -> list[str]:
    element_id = str(tag.get("id") or "").strip()
    return [element_id] if element_id else []


def _tag_tokens(tag: Tag) -> set[str]:
    raw_parts = [str(tag.name or "")]
    raw_parts.extend(_tag_classes(tag))
    raw_parts.extend(_tag_ids(tag))
    tokens: set[str] = set()
    for part in raw_parts:
        for token in str(part).replace("_", "-").split("-"):
            clean = token.strip().lower()
            if clean:
                tokens.add(clean)
    return tokens


def _is_meaningful_wrapper(tag: Tag) -> bool:
    return (tag.name or "") not in {"html", "body"} and bool(_tag_classes(tag) or _tag_ids(tag))


def _wrapper_signature_from_tag(tag: Tag) -> ShellWrapperSignature:
    return ShellWrapperSignature(
        tag=str(tag.name or "div"),
        required_classes=_tag_classes(tag),
        preserve_ids=_tag_ids(tag),
    )


def _capture_wrapper_chain(tag: Tag, max_depth: int = _MAX_WRAPPER_CHAIN_DEPTH) -> list[ShellWrapperSignature]:
    wrappers: list[ShellWrapperSignature] = []
    for ancestor in tag.parents:
        if not isinstance(ancestor, Tag):
            continue
        if not _is_meaningful_wrapper(ancestor):
            continue
        wrappers.append(_wrapper_signature_from_tag(ancestor))
        if len(wrappers) >= max_depth:
            break
    return wrappers


def _select_unique_tag(soup: BeautifulSoup, selector: str) -> Tag | None:
    try:
        matches = soup.select(str(selector or ""))
    except Exception:
        return None
    tags = [match for match in matches if isinstance(match, Tag)]
    if len(tags) != 1:
        return None
    return tags[0]


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


def _resolve_global_anchor_target(tag: Tag, global_id: str) -> Tag:
    primary_tags = _PRIMARY_GLOBAL_TAGS.get(global_id, set())
    fallback_tags = _FALLBACK_GLOBAL_TAGS.get(global_id, set())
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


def _expected_global_target_meta(reference_soup: BeautifulSoup | None, selector_hint: str, global_id: str) -> tuple[str, list[str]]:
    if reference_soup is None:
        return "", []
    match = _select_unique_tag(reference_soup, selector_hint)
    if match is None:
        return "", []
    target = _resolve_global_anchor_target(match, global_id)
    return str(target.name or ""), _tag_classes(target)


def build_expected_global_anchors(
    page_plan: PagePlan,
    reference_html_text: str | None = None,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    seen: set[str] = set()
    reference_soup = BeautifulSoup(str(reference_html_text or ""), "html.parser") if reference_html_text else None
    for selector in page_plan.dom_mapping:
        global_id = _selector_global_id(selector)
        if not global_id or global_id in seen:
            continue
        seen.add(global_id)
        target_tag, required_classes = _expected_global_target_meta(reference_soup, selector, global_id)
        results.append(
            {
                "global_id": global_id,
                "selector_hint": selector,
                "target_tag": target_tag,
                "required_classes": required_classes,
            }
        )
    return results


def annotate_global_anchors(html_text: str, page_plan: PagePlan) -> str:
    soup = BeautifulSoup(str(html_text or ""), "html.parser")
    expected_globals = build_expected_global_anchors(page_plan)

    for item in expected_globals:
        global_id = str(item["global_id"])
        selector_hint = str(item["selector_hint"])
        existing = [tag for tag in soup.select(build_global_selector(global_id)) if isinstance(tag, Tag)]

        if len(existing) > 1:
            return str(soup)

        if len(existing) == 1:
            current_target = existing[0]
            actionable_target = _resolve_global_anchor_target(current_target, global_id)
            if actionable_target is not current_target:
                current_target.attrs.pop(GLOBAL_ATTR, None)
                if actionable_target.get(GLOBAL_ATTR) not in (None, global_id):
                    continue
                actionable_target[GLOBAL_ATTR] = global_id
            continue

        match = _select_unique_tag(soup, selector_hint)
        if match is None:
            continue
        actionable_target = _resolve_global_anchor_target(match, global_id)
        if actionable_target.get(GLOBAL_ATTR) not in (None, global_id):
            continue
        actionable_target[GLOBAL_ATTR] = global_id

    return str(soup)


def _shell_candidate_score(tag: Tag, region_role: str, depth: int) -> int:
    tokens = _tag_tokens(tag)
    classes = _tag_classes(tag)
    score = 0

    if (tag.name or "") in _SHELL_CONTAINER_TAGS:
        score += 4
    else:
        score -= 4

    score += max(0, 4 - depth)
    score += min(3, len(_tag_ids(tag)) + len(classes))

    specific_tokens = {token for token in tokens if token not in _SHELL_GENERIC_TOKENS}
    score += min(3, len(specific_tokens))

    if tokens & _SHELL_GENERIC_TOKENS:
        score += 1
    if tokens & _REGION_ROLE_HINTS.get(str(region_role or ""), set()):
        score += 2

    if region_role != "nav" and tokens & {"menu", "nav"}:
        score -= 3
    if region_role != "footer" and "footer" in tokens:
        score -= 3

    direct_children = [child for child in tag.children if isinstance(child, Tag)]
    if direct_children:
        score += 1
    return score


def _candidate_shell_tags_for_match(match: Tag) -> list[tuple[int, Tag]]:
    candidates: list[tuple[int, Tag]] = [(0, match)]
    for depth, ancestor in enumerate(match.parents, start=1):
        if not isinstance(ancestor, Tag):
            continue
        if (ancestor.name or "") in {"html", "body"}:
            break
        if not _is_meaningful_wrapper(ancestor) and (ancestor.name or "") not in _SHELL_CONTAINER_TAGS:
            continue
        candidates.append((depth, ancestor))
        if depth >= 4:
            break
    return candidates


def _matched_tags_for_block(block: BlockPlan, template_soup: BeautifulSoup) -> list[Tag]:
    try:
        return [
            match
            for match in template_soup.select(str(block.target_template_region.selector_hint or ""))
            if isinstance(match, Tag)
        ]
    except Exception:
        return []


def _choose_shell_root(block: BlockPlan, matches: list[Tag]) -> Tag | None:
    if not matches:
        return None

    scored_candidates: list[tuple[int, int, Tag]] = []
    for match in matches:
        for depth, candidate in _candidate_shell_tags_for_match(match):
            scored_candidates.append(
                (
                    _shell_candidate_score(candidate, block.target_template_region.region_role, depth),
                    -depth,
                    candidate,
                )
            )

    if not scored_candidates:
        return None

    scored_candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return scored_candidates[0][2]


def _build_shell_contract_for_block(block: BlockPlan, template_soup: BeautifulSoup) -> BlockShellContract | None:
    matches = _matched_tags_for_block(block, template_soup)
    root = _choose_shell_root(block, matches)
    if root is None:
        return None
    preserve_ids = _tag_ids(root) if len(matches) == 1 else []
    return BlockShellContract(
        root_tag=str(root.name or "div"),
        required_classes=_tag_classes(root),
        preserve_ids=preserve_ids,
        wrapper_chain=_capture_wrapper_chain(root),
        actionable_root_selector=str(block.target_template_region.selector_hint or "").strip(),
    )


def enrich_page_plan_shell_contracts(page_plan: PagePlan, template_reference_html: str) -> PagePlan:
    if not str(template_reference_html or "").strip():
        return page_plan

    template_soup = BeautifulSoup(str(template_reference_html or ""), "html.parser")
    updated_blocks: list[BlockPlan] = []
    for block in page_plan.blocks:
        if block.shell_contract is not None:
            updated_blocks.append(block)
            continue
        updated_blocks.append(
            block.model_copy(
                deep=True,
                update={"shell_contract": _build_shell_contract_for_block(block, template_soup)},
            )
        )
    return page_plan.model_copy(update={"blocks": updated_blocks}, deep=True)


def missing_shell_contract_block_ids(page_plan: PagePlan) -> list[str]:
    return [block.block_id for block in page_plan.blocks if block.shell_contract is None]


def _matches_wrapper_signature(tag: Tag, signature: ShellWrapperSignature) -> bool:
    if str(tag.name or "") != str(signature.tag or ""):
        return False
    actual_classes = set(_tag_classes(tag))
    if any(required not in actual_classes for required in signature.required_classes):
        return False
    expected_ids = set(signature.preserve_ids)
    actual_ids = set(_tag_ids(tag))
    if expected_ids and expected_ids != actual_ids:
        return False
    return True


def validate_block_tag_against_shell_contract(
    block_tag: Tag,
    shell_contract: BlockShellContract | None,
    block_id: str,
) -> list[str]:
    if shell_contract is None:
        return [f"Block '{block_id}' is missing shell_contract in PagePlan."]

    errors: list[str] = []
    actual_tag_name = str(block_tag.name or "")
    if actual_tag_name != str(shell_contract.root_tag or ""):
        errors.append(
            f"Block '{block_id}' root tag '{actual_tag_name}' does not match shell_contract root '{shell_contract.root_tag}'."
        )

    actual_classes = set(_tag_classes(block_tag))
    missing_classes = [name for name in shell_contract.required_classes if name not in actual_classes]
    if missing_classes:
        errors.append(
            f"Block '{block_id}' is missing required shell classes {missing_classes}."
        )

    expected_root_ids = set(shell_contract.preserve_ids)
    actual_root_ids = set(_tag_ids(block_tag))
    if expected_root_ids and expected_root_ids != actual_root_ids:
        errors.append(
            f"Block '{block_id}' root id signature {sorted(actual_root_ids)} does not match required ids {sorted(expected_root_ids)}."
        )

    if shell_contract.wrapper_chain:
        meaningful_ancestors = [
            ancestor
            for ancestor in block_tag.parents
            if isinstance(ancestor, Tag) and _is_meaningful_wrapper(ancestor)
        ]
        ancestor_index = 0
        for wrapper_signature in shell_contract.wrapper_chain:
            matched = False
            while ancestor_index < len(meaningful_ancestors):
                ancestor = meaningful_ancestors[ancestor_index]
                ancestor_index += 1
                if _matches_wrapper_signature(ancestor, wrapper_signature):
                    matched = True
                    break
            if not matched:
                errors.append(
                    f"Block '{block_id}' does not preserve required wrapper '{wrapper_signature.tag}' with classes {wrapper_signature.required_classes}."
                )
                break

    return errors


def extract_page_manifest(
    html_text: str,
    entry_html: str | Path,
    selected_template_id: str,
    page_plan: PagePlan,
    require_expected_globals: bool = True,
) -> PageManifest:
    soup = BeautifulSoup(str(html_text or ""), "html.parser")
    expected_blocks = {block.block_id: block for block in page_plan.blocks}
    expected_globals = build_expected_global_anchors(page_plan)
    source_sections_lookup = {
        outline_item.block_id: list(outline_item.source_sections)
        for outline_item in page_plan.page_outline
    }

    errors: list[str] = []
    seen_block_ids: set[str] = set()
    manifest_blocks: list[PageManifestBlock] = []
    seen_global_ids: set[str] = set()
    manifest_globals: list[PageManifestGlobal] = []

    for block_tag in soup.select(f"[{BLOCK_ATTR}]"):
        if not isinstance(block_tag, Tag):
            continue

        block_id = str(block_tag.get(BLOCK_ATTR) or "").strip()
        if not block_id:
            errors.append("Encountered an empty data-pa-block attribute.")
            continue
        if block_tag.find_parent(attrs={BLOCK_ATTR: True}) is not None:
            errors.append(f"Block '{block_id}' is nested inside another data-pa-block.")
            continue
        if block_id in seen_block_ids:
            errors.append(f"Duplicate data-pa-block '{block_id}' found in HTML.")
            continue
        if block_id not in expected_blocks:
            errors.append(f"HTML contains unknown data-pa-block '{block_id}' not present in PagePlan.")
            continue

        seen_block_ids.add(block_id)
        seen_slot_ids: set[str] = set()
        slot_records: list[PageManifestSlot] = []

        for slot_tag in block_tag.select(f"[{SLOT_ATTR}]"):
            if not isinstance(slot_tag, Tag):
                continue

            ancestor_block = slot_tag.find_parent(attrs={BLOCK_ATTR: True})
            if ancestor_block is not block_tag:
                continue

            slot_id = str(slot_tag.get(SLOT_ATTR) or "").strip()
            if not slot_id:
                errors.append(f"Block '{block_id}' contains an empty data-pa-slot attribute.")
                continue
            if slot_id not in ALLOWED_SLOT_IDS:
                errors.append(
                    f"Block '{block_id}' contains unsupported slot '{slot_id}'. "
                    f"Allowed slots: {sorted(ALLOWED_SLOT_IDS)}."
                )
                continue
            if slot_id in seen_slot_ids:
                errors.append(f"Block '{block_id}' contains duplicate slot '{slot_id}'.")
                continue

            seen_slot_ids.add(slot_id)
            slot_records.append(
                PageManifestSlot(
                    slot_id=slot_id,
                    selector=build_slot_selector(block_id, slot_id),
                )
            )

        if not slot_records:
            errors.append(f"Block '{block_id}' must contain at least one data-pa-slot.")
            continue

        errors.extend(
            validate_block_tag_against_shell_contract(
                block_tag=block_tag,
                shell_contract=expected_blocks[block_id].shell_contract,
                block_id=block_id,
            )
        )

        manifest_blocks.append(
            PageManifestBlock(
                block_id=block_id,
                source_sections=source_sections_lookup.get(block_id, []),
                selector=build_block_selector(block_id),
                slots=slot_records,
                root_tag=str(block_tag.name or "div"),
                root_classes=_tag_classes(block_tag),
                preserve_ids=_tag_ids(block_tag),
                wrapper_chain=_capture_wrapper_chain(block_tag),
                actionable_root_selector=build_block_selector(block_id),
            )
        )

    missing_block_ids = [block_id for block_id in expected_blocks if block_id not in seen_block_ids]
    if missing_block_ids:
        errors.append(f"Missing required data-pa-block ids: {missing_block_ids}")

    for global_tag in soup.select(f"[{GLOBAL_ATTR}]"):
        if not isinstance(global_tag, Tag):
            continue

        global_id = str(global_tag.get(GLOBAL_ATTR) or "").strip()
        if not global_id:
            errors.append("Encountered an empty data-pa-global attribute.")
            continue
        if global_id not in ALLOWED_GLOBAL_IDS:
            errors.append(
                f"Encountered unsupported data-pa-global '{global_id}'. "
                f"Allowed globals: {sorted(ALLOWED_GLOBAL_IDS)}."
            )
            continue
        if global_id in seen_global_ids:
            errors.append(f"Duplicate data-pa-global '{global_id}' found in HTML.")
            continue

        actionable_target = _resolve_global_anchor_target(global_tag, global_id)
        if actionable_target is not global_tag:
            errors.append(
                f"Global anchor '{global_id}' is not attached to its actionable root node."
            )
            continue

        seen_global_ids.add(global_id)
        manifest_globals.append(
            PageManifestGlobal(
                global_id=global_id,
                selector=build_global_selector(global_id),
                target_tag=str(global_tag.name or ""),
                required_classes=_tag_classes(global_tag),
                actionable_selector=build_global_selector(global_id),
            )
        )

    if require_expected_globals:
        missing_global_ids = [
            item["global_id"]
            for item in expected_globals
            if item["global_id"] not in seen_global_ids
        ]
        if missing_global_ids:
            errors.append(f"Missing required data-pa-global ids: {missing_global_ids}")

    if errors:
        raise ValueError(" ; ".join(errors))

    return PageManifest(
        schema_version=PAGE_MANIFEST_SCHEMA_VERSION,
        entry_html=str(Path(entry_html).resolve()),
        selected_template_id=str(selected_template_id or "").strip(),
        blocks=manifest_blocks,
        globals=manifest_globals,
    )
