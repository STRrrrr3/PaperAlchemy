from __future__ import annotations

import json
from pathlib import Path

from bs4 import BeautifulSoup, Tag

from src.contracts.schemas import (
    BlockShellContract,
    PageManifest,
    PageManifestBlock,
    PageManifestGlobal,
    PageManifestSlot,
    PagePlan,
    TemplateProfile,
)
from src.template.structural_core import (
    build_unique_selector as _build_unique_selector,
    capture_wrapper_chain as _capture_wrapper_chain,
    is_meaningful_wrapper as _is_meaningful_wrapper,
    matches_wrapper_signature as _matches_wrapper_signature,
    select_unique_tag as _select_unique_tag,
    tag_classes as _tag_classes,
    tag_ids as _tag_ids,
    tag_tokens as _tag_tokens,
)
from src.template.template_ir import (
    bind_page_plan_to_template_ir,
    canonical_global_anchors,
    resolve_global_anchor_target as _resolve_global_anchor_target,
    selector_global_id as _selector_global_id,
)

BLOCK_ATTR = "data-pa-block"
SLOT_ATTR = "data-pa-slot"
GLOBAL_ATTR = "data-pa-global"
ALLOWED_SLOT_IDS = {"title", "summary", "body", "media", "meta", "actions"}
ALLOWED_GLOBAL_IDS: set[str] = {"header_brand", "header_primary_action", "header_nav", "footer_meta"}
PAGE_MANIFEST_SCHEMA_VERSION = "1.2"


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


def _expected_global_target_meta(reference_soup: BeautifulSoup | None, selector_hint: str, global_id: str) -> tuple[str, list[str], list[str], str]:
    if reference_soup is None:
        return "", [], [], selector_hint
    match = _select_unique_tag(reference_soup, selector_hint)
    if match is None:
        return "", [], [], selector_hint
    target = _resolve_global_anchor_target(match, global_id)
    return (
        str(target.name or ""),
        _tag_classes(target),
        _tag_ids(target),
        _build_unique_selector(target, reference_soup),
    )


def build_expected_global_anchors(
    page_plan: PagePlan,
    template_profile: TemplateProfile | None = None,
    reference_html_text: str | None = None,
) -> list[dict[str, object]]:
    if template_profile is not None:
        return [
            {
                "global_id": anchor.global_id,
                "selector_hint": anchor.selector,
                "target_tag": anchor.target_tag,
                "required_classes": list(anchor.required_classes),
                "preserve_ids": list(anchor.preserve_ids),
                "actionable_selector": anchor.actionable_selector,
            }
            for anchor in canonical_global_anchors(template_profile)
        ]

    results: list[dict[str, object]] = []
    seen: set[str] = set()
    reference_soup = BeautifulSoup(str(reference_html_text or ""), "html.parser") if reference_html_text else None
    for selector in page_plan.dom_mapping:
        global_id = _selector_global_id(selector)
        if not global_id or global_id in seen:
            continue
        seen.add(global_id)
        target_tag, required_classes, preserve_ids, actionable_selector = _expected_global_target_meta(
            reference_soup,
            selector,
            global_id,
        )
        results.append(
            {
                "global_id": global_id,
                "selector_hint": selector,
                "target_tag": target_tag,
                "required_classes": required_classes,
                "preserve_ids": preserve_ids,
                "actionable_selector": actionable_selector,
            }
        )
    return results


def annotate_global_anchors(
    html_text: str,
    page_plan: PagePlan,
    template_profile: TemplateProfile | None = None,
) -> str:
    soup = BeautifulSoup(str(html_text or ""), "html.parser")
    expected_globals = build_expected_global_anchors(page_plan, template_profile=template_profile)

    for item in expected_globals:
        global_id = str(item["global_id"])
        selector_hint = str(item.get("selector_hint") or "")
        actionable_selector = str(item.get("actionable_selector") or selector_hint)
        canonical_target_tag = str(item.get("target_tag") or "").strip()
        existing = [tag for tag in soup.select(build_global_selector(global_id)) if isinstance(tag, Tag)]

        if len(existing) > 1:
            return str(soup)

        if len(existing) == 1:
            continue

        actionable_target = _select_unique_tag(soup, actionable_selector)
        if actionable_target is None:
            base_match = _select_unique_tag(soup, selector_hint)
            if base_match is None:
                continue
            actionable_target = (
                base_match
                if canonical_target_tag
                else _resolve_global_anchor_target(base_match, global_id)
            )
        if actionable_target is None:
            continue
        if actionable_target.get(GLOBAL_ATTR) not in (None, global_id):
            continue
        actionable_target[GLOBAL_ATTR] = global_id

    return str(soup)


def enrich_page_plan_shell_contracts(
    page_plan: PagePlan,
    template_profile: TemplateProfile | None,
) -> PagePlan:
    if template_profile is None:
        return page_plan
    return bind_page_plan_to_template_ir(page_plan, template_profile, fill_missing_only=True)


def missing_shell_contract_block_ids(page_plan: PagePlan) -> list[str]:
    return [block.block_id for block in page_plan.blocks if block.shell_contract is None]


def validate_block_tag_against_shell_contract(
    block_tag: Tag,
    shell_contract: BlockShellContract | None,
    block_id: str,
) -> list[str]:
    if shell_contract is None:
        return [f"Block '{block_id}' is missing shell_contract in PagePlan."]
    if not isinstance(shell_contract, BlockShellContract):
        try:
            shell_contract = BlockShellContract.model_validate(shell_contract)
        except Exception:
            return [f"Block '{block_id}' has an invalid shell_contract in PagePlan."]

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
    template_profile: TemplateProfile | None = None,
) -> PageManifest:
    soup = BeautifulSoup(str(html_text or ""), "html.parser")
    expected_blocks = {block.block_id: block for block in page_plan.blocks}
    expected_globals = build_expected_global_anchors(page_plan, template_profile=template_profile)
    expected_globals_by_id = {str(item["global_id"]): item for item in expected_globals}
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

        expected_shell_contract = expected_blocks[block_id].shell_contract
        if expected_shell_contract is not None and not isinstance(expected_shell_contract, BlockShellContract):
            try:
                expected_shell_contract = BlockShellContract.model_validate(expected_shell_contract)
            except Exception:
                expected_shell_contract = None

        errors.extend(
            validate_block_tag_against_shell_contract(
                block_tag=block_tag,
                shell_contract=expected_shell_contract,
                block_id=block_id,
            )
        )

        manifest_blocks.append(
            PageManifestBlock(
                block_id=block_id,
                shell_id=str(
                    expected_blocks[block_id].target_template_region.shell_id
                    or (
                        expected_shell_contract.shell_id
                        if expected_shell_contract is not None
                        else ""
                    )
                ).strip(),
                source_sections=source_sections_lookup.get(block_id, []),
                selector=build_block_selector(block_id),
                slots=slot_records,
                root_tag=(
                    expected_shell_contract.root_tag
                    if expected_shell_contract is not None
                    else str(block_tag.name or "div")
                ),
                root_classes=(
                    expected_shell_contract.required_classes
                    if expected_shell_contract is not None
                    else _tag_classes(block_tag)
                ),
                preserve_ids=(
                    expected_shell_contract.preserve_ids
                    if expected_shell_contract is not None
                    else _tag_ids(block_tag)
                ),
                wrapper_chain=(
                    expected_shell_contract.wrapper_chain
                    if expected_shell_contract is not None
                    else _capture_wrapper_chain(block_tag)
                ),
                actionable_root_selector=(
                    expected_shell_contract.actionable_root_selector
                    if expected_shell_contract is not None
                    else build_block_selector(block_id)
                ),
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

        seen_global_ids.add(global_id)
        expected_global = expected_globals_by_id.get(global_id, {})
        target_tag = str(expected_global.get("target_tag") or "").strip()
        required_classes = [str(item).strip() for item in (expected_global.get("required_classes") or []) if str(item).strip()]
        preserve_ids = [str(item).strip() for item in (expected_global.get("preserve_ids") or []) if str(item).strip()]

        if target_tag and str(global_tag.name or "") != target_tag:
            errors.append(
                f"Global anchor '{global_id}' tag '{global_tag.name or ''}' does not match canonical target '{target_tag}'."
            )
            continue

        actual_classes = set(_tag_classes(global_tag))
        missing_classes = [name for name in required_classes if name not in actual_classes]
        if missing_classes:
            errors.append(
                f"Global anchor '{global_id}' is missing required classes {missing_classes}."
            )
            continue

        actual_ids = set(_tag_ids(global_tag))
        expected_ids = set(preserve_ids)
        if expected_ids and expected_ids != actual_ids:
            errors.append(
                f"Global anchor '{global_id}' id signature {sorted(actual_ids)} does not match canonical ids {sorted(expected_ids)}."
            )
            continue

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

