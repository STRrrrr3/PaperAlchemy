from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup, Tag
from langchain_core.messages import HumanMessage, SystemMessage

from src.html_utils import (
    extract_html_fragment,
    message_content_to_text,
    read_current_page_html,
    read_template_reference_html,
    read_text_with_fallback,
)
from src.json_utils import to_pretty_json
from src.llm import get_llm
from src.page_manifest import (
    GLOBAL_ATTR,
    SLOT_ATTR,
    build_block_selector,
    build_global_selector,
    build_page_manifest_path,
    build_slot_selector,
    extract_page_manifest,
    load_page_manifest,
    missing_shell_contract_block_ids,
    validate_block_tag_against_shell_contract,
)
from src.page_validation import (
    REVISION_OVERRIDE_STYLE_TAG_ID,
    collect_allowed_asset_web_paths,
    collect_local_image_sources,
    is_safe_anchored_selector,
    validate_fragment_local_image_sources,
    validate_local_image_references,
)
from src.prompts import (
    BLOCK_REGEN_SYSTEM_PROMPT,
    BLOCK_REGEN_USER_PROMPT_TEMPLATE,
    PATCH_AGENT_SYSTEM_PROMPT,
    PATCH_AGENT_USER_PROMPT_TEMPLATE,
)
from src.schemas import (
    AttributeChange,
    CoderArtifact,
    FallbackBlock,
    OverrideCssRule,
    PageManifest,
    PagePlan,
    RevisionPlan,
    StructuredPaper,
    StyleChange,
    TargetedReplacement,
    TargetedReplacementPlan,
)
from src.state import WorkflowState

LEGACY_PAGE_ERROR = (
    "This page was generated before anchored revisions were enabled. "
    "Generate a new first draft before requesting targeted revisions."
)


class PatchApplicationError(ValueError):
    """Raised when anchored DOM patching cannot be safely applied."""


def _normalize_page_plan(plan: Any) -> PagePlan | None:
    if isinstance(plan, PagePlan):
        return plan
    if plan is None:
        return None
    try:
        return PagePlan.model_validate(plan)
    except Exception:
        return None


def _normalize_structured_paper(paper: Any) -> StructuredPaper | None:
    if isinstance(paper, StructuredPaper):
        return paper
    if paper is None:
        return None
    try:
        return StructuredPaper.model_validate(paper)
    except Exception:
        return None


def _normalize_coder_artifact(artifact: Any) -> CoderArtifact | None:
    if isinstance(artifact, CoderArtifact):
        return artifact
    if artifact is None:
        return None
    try:
        return CoderArtifact.model_validate(artifact)
    except Exception:
        return None


def _normalize_revision_plan(plan: Any) -> RevisionPlan | None:
    if isinstance(plan, RevisionPlan):
        return plan
    if plan is None:
        return None
    try:
        return RevisionPlan.model_validate(plan)
    except Exception:
        return None


def _normalize_targeted_replacement_plan(plan: Any) -> TargetedReplacementPlan | None:
    if isinstance(plan, TargetedReplacementPlan):
        return plan
    if plan is None:
        return None

    try:
        return TargetedReplacementPlan.model_validate(plan)
    except Exception:
        return None


def _extract_json_object(text: str) -> dict[str, Any] | None:
    raw_text = str(text or "").strip()
    if not raw_text:
        return None

    fenced_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw_text, flags=re.IGNORECASE | re.DOTALL)
    if fenced_match:
        raw_text = fenced_match.group(1).strip()

    try:
        parsed = json.loads(raw_text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass

    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        parsed = json.loads(raw_text[start : end + 1])
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _normalize_identifier(value: Any) -> str | None:
    clean = str(value or "").strip()
    return clean or None


def _normalize_declarations(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}

    normalized: dict[str, str] = {}
    for name, raw_value in value.items():
        clean_name = str(name or "").strip()
        clean_value = str(raw_value or "").strip()
        if clean_name and clean_value:
            normalized[clean_name] = clean_value
    return normalized


def _normalize_attributes(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}

    normalized: dict[str, str] = {}
    for name, raw_value in value.items():
        clean_name = str(name or "").strip()
        if not clean_name:
            continue
        normalized[clean_name] = "" if raw_value is None else str(raw_value).strip()
    return normalized


def _normalize_replacement_item(item: dict[str, Any]) -> dict[str, Any] | None:
    scope = _normalize_identifier(item.get("scope"))
    html = str(item.get("html") or "").strip()
    block_id = _normalize_identifier(item.get("block_id"))
    slot_id = _normalize_identifier(item.get("slot_id"))
    global_id = _normalize_identifier(item.get("global_id"))

    if scope not in {"slot", "block", "global"} or not html:
        return None
    if scope == "slot" and (not block_id or not slot_id):
        return None
    if scope == "block" and not block_id:
        return None
    if scope == "global" and not global_id:
        return None

    return {
        "scope": scope,
        "block_id": block_id,
        "slot_id": slot_id,
        "global_id": global_id,
        "html": html,
    }


def _normalize_scoped_change_target(item: dict[str, Any]) -> dict[str, Any] | None:
    scope = _normalize_identifier(item.get("scope"))
    block_id = _normalize_identifier(item.get("block_id"))
    slot_id = _normalize_identifier(item.get("slot_id"))
    global_id = _normalize_identifier(item.get("global_id"))

    if scope not in {"slot", "block", "global"}:
        return None
    if scope == "slot" and (not block_id or not slot_id):
        return None
    if scope == "block" and not block_id:
        return None
    if scope == "global" and not global_id:
        return None

    return {
        "scope": scope,
        "block_id": block_id,
        "slot_id": slot_id,
        "global_id": global_id,
    }


def _sanitize_targeted_plan_payload(raw_payload: Any) -> tuple[dict[str, Any] | None, list[str]]:
    if not isinstance(raw_payload, dict):
        return None, ["patch agent returned a non-object payload"]

    warnings: list[str] = []
    sanitized: dict[str, Any] = {
        "replacements": [],
        "style_changes": [],
        "attribute_changes": [],
        "override_css_rules": [],
        "fallback_blocks": [],
    }

    for item in raw_payload.get("replacements") or []:
        if not isinstance(item, dict):
            warnings.append("dropped invalid replacement")
            continue
        normalized = _normalize_replacement_item(item)
        if normalized is None:
            warnings.append("dropped invalid replacement")
            continue
        sanitized["replacements"].append(normalized)

    for item in raw_payload.get("style_changes") or []:
        if not isinstance(item, dict):
            warnings.append("dropped invalid style change")
            continue
        target = _normalize_scoped_change_target(item)
        declarations = _normalize_declarations(item.get("declarations"))
        if not declarations:
            warnings.append("dropped empty style change")
            continue
        if target is None:
            warnings.append("dropped invalid style change")
            continue
        sanitized["style_changes"].append({**target, "declarations": declarations})

    for item in raw_payload.get("attribute_changes") or []:
        if not isinstance(item, dict):
            warnings.append("dropped invalid attribute change")
            continue
        target = _normalize_scoped_change_target(item)
        attributes = _normalize_attributes(item.get("attributes"))
        if not attributes:
            warnings.append("dropped empty attribute change")
            continue
        if target is None:
            warnings.append("dropped invalid attribute change")
            continue
        sanitized["attribute_changes"].append({**target, "attributes": attributes})

    for item in raw_payload.get("override_css_rules") or []:
        if not isinstance(item, dict):
            warnings.append("dropped invalid override css rule")
            continue
        selector = str(item.get("selector") or "").strip()
        declarations = _normalize_declarations(item.get("declarations"))
        if not selector or not declarations:
            warnings.append("dropped empty override css rule")
            continue
        sanitized["override_css_rules"].append(
            {
                "selector": selector,
                "declarations": declarations,
            }
        )

    for item in raw_payload.get("fallback_blocks") or []:
        if not isinstance(item, dict):
            warnings.append("dropped invalid fallback block")
            continue
        block_id = _normalize_identifier(item.get("block_id"))
        reason = str(item.get("reason") or "").strip()
        if not block_id or not reason:
            warnings.append("dropped invalid fallback block")
            continue
        sanitized["fallback_blocks"].append(
            {
                "block_id": block_id,
                "reason": reason,
            }
        )

    return sanitized, warnings


def _salvage_validated_items(
    items: list[dict[str, Any]],
    model: type[Any],
    warning_label: str,
) -> tuple[list[Any], list[str]]:
    validated_items: list[Any] = []
    warnings: list[str] = []
    for item in items:
        try:
            validated_items.append(model.model_validate(item))
        except Exception:
            warnings.append(warning_label)
    return validated_items, warnings


def _build_targeted_replacement_plan_from_response(response: Any) -> tuple[TargetedReplacementPlan | None, list[str]]:
    raw_payload: dict[str, Any] | None
    if isinstance(response, dict):
        raw_payload = response
    else:
        raw_payload = _extract_json_object(message_content_to_text(response))
    if raw_payload is None:
        return None, ["patch agent returned invalid targeted replacement plan JSON"]

    sanitized_payload, warnings = _sanitize_targeted_plan_payload(raw_payload)
    if sanitized_payload is None:
        return None, warnings

    replacements, replacement_warnings = _salvage_validated_items(
        sanitized_payload["replacements"],
        TargetedReplacement,
        "dropped invalid replacement",
    )
    style_changes, style_warnings = _salvage_validated_items(
        sanitized_payload["style_changes"],
        StyleChange,
        "dropped invalid style change",
    )
    attribute_changes, attribute_warnings = _salvage_validated_items(
        sanitized_payload["attribute_changes"],
        AttributeChange,
        "dropped invalid attribute change",
    )
    override_css_rules, override_rule_warnings = _salvage_validated_items(
        sanitized_payload["override_css_rules"],
        OverrideCssRule,
        "dropped invalid override css rule",
    )
    fallback_blocks, fallback_warnings = _salvage_validated_items(
        sanitized_payload["fallback_blocks"],
        FallbackBlock,
        "dropped invalid fallback block",
    )
    warnings.extend(replacement_warnings)
    warnings.extend(style_warnings)
    warnings.extend(attribute_warnings)
    warnings.extend(override_rule_warnings)
    warnings.extend(fallback_warnings)

    if warnings and replacements:
        warnings.append("salvaged replacements after sanitization")

    return (
        TargetedReplacementPlan(
            replacements=replacements,
            style_changes=style_changes,
            attribute_changes=attribute_changes,
            override_css_rules=override_css_rules,
            fallback_blocks=fallback_blocks,
        ),
        warnings,
    )


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(path)
    return deduped


def _build_patch_target_paths(artifact: CoderArtifact, page_plan: PagePlan) -> list[Path]:
    entry_html_path = Path(artifact.entry_html)
    site_dir = Path(artifact.site_dir)
    template_entry_rel = str(page_plan.template_selection.selected_entry_html or "").strip()

    targets = [entry_html_path]
    if template_entry_rel:
        mirrored_entry_path = site_dir / template_entry_rel
        if mirrored_entry_path.exists() and mirrored_entry_path.resolve() != entry_html_path.resolve():
            targets.append(mirrored_entry_path)

    return _dedupe_paths(targets)


def _merge_unique_strings(*groups: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for item in group:
            clean = str(item or "").strip()
            if clean and clean not in seen:
                seen.add(clean)
                merged.append(clean)
    return merged


def _update_patch_notes(
    existing_notes: str,
    replacements_count: int,
    regenerated_count: int,
    style_change_count: int,
    attribute_change_count: int,
    override_rule_count: int,
) -> str:
    base = str(existing_notes or "").strip()
    patch_summary = (
        "v7-shell-aware-targeted-revision: "
        f"applied {replacements_count} replacement(s), "
        f"{style_change_count} style change(s), "
        f"{attribute_change_count} attribute change(s), "
        f"{override_rule_count} override css rule(s), "
        f"and regenerated {regenerated_count} block(s)."
    )
    if not base:
        return patch_summary

    parts = [part.strip() for part in base.split("|") if part.strip()]
    filtered_parts = [
        part
        for part in parts
        if not part.startswith("v6-targeted-revision:")
        and not part.startswith("v7-shell-aware-targeted-revision:")
    ]
    filtered_parts.append(patch_summary)
    return " | ".join(filtered_parts)


def _to_site_relative_paths(paths: list[Path], site_dir: Path) -> list[str]:
    relative_paths: list[str] = []
    for path in paths:
        try:
            relative_paths.append(str(path.resolve().relative_to(site_dir.resolve())).replace("\\", "/"))
        except ValueError:
            continue
    return relative_paths


def _write_files_transaction(file_contents: dict[Path, str]) -> None:
    original_bytes: dict[Path, bytes | None] = {}
    temp_paths: dict[Path, Path] = {}

    try:
        for path, text in file_contents.items():
            original_bytes[path] = path.read_bytes() if path.exists() else None
            path.parent.mkdir(parents=True, exist_ok=True)

            fd, temp_name = tempfile.mkstemp(
                prefix=f".{path.name}.",
                suffix=".tmp",
                dir=str(path.parent),
            )
            os.close(fd)
            temp_path = Path(temp_name)
            temp_path.write_text(text, encoding="utf-8")
            temp_paths[path] = temp_path
    except Exception as exc:
        for temp_path in temp_paths.values():
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
        raise PatchApplicationError(f"Patch Executor failed preparing file writes: {exc}") from exc

    replaced_paths: list[Path] = []
    try:
        for path in file_contents:
            os.replace(temp_paths[path], path)
            replaced_paths.append(path)
    except Exception as exc:
        for path in reversed(replaced_paths):
            original = original_bytes.get(path)
            try:
                if original is None:
                    path.unlink(missing_ok=True)
                else:
                    path.write_bytes(original)
            except Exception:
                pass
        raise PatchApplicationError(f"Patch Executor failed writing patched files: {exc}") from exc
    finally:
        for temp_path in temp_paths.values():
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)


def _summarize_targeted_plan(plan: TargetedReplacementPlan | None) -> str:
    if not plan:
        return ""
    return (
        f"replacements={len(plan.replacements)}; "
        f"style_changes={len(plan.style_changes)}; "
        f"attribute_changes={len(plan.attribute_changes)}; "
        f"override_css_rules={len(plan.override_css_rules)}; "
        f"fallback_blocks={len(plan.fallback_blocks)}"
    )


def _extract_fragment_nodes(fragment: BeautifulSoup) -> list[Any]:
    if fragment.body is not None:
        return list(fragment.body.contents)
    return list(fragment.contents)


def _is_ignorable_node(node: Any) -> bool:
    if isinstance(node, str):
        return not node.strip()
    return False


def _extract_single_root_tag(html_fragment: str) -> Tag:
    fragment = BeautifulSoup(str(html_fragment or ""), "html.parser")
    nodes = [node for node in _extract_fragment_nodes(fragment) if not _is_ignorable_node(node)]
    if len(nodes) != 1 or not isinstance(nodes[0], Tag):
        raise PatchApplicationError("Expected exactly one root element in block replacement HTML.")
    return nodes[0]


def _normalize_slot_fragment(html_fragment: str, slot_id: str) -> BeautifulSoup:
    fragment = BeautifulSoup(str(html_fragment or ""), "html.parser")
    nodes = [node for node in _extract_fragment_nodes(fragment) if not _is_ignorable_node(node)]
    if len(nodes) == 1 and isinstance(nodes[0], Tag):
        root = nodes[0]
        if str(root.get(SLOT_ATTR) or "").strip() == slot_id:
            normalized = BeautifulSoup("", "html.parser")
            for child in list(root.contents):
                normalized.append(child.extract())
            return normalized
    return fragment


def _normalize_global_fragment(html_fragment: str, global_id: str) -> BeautifulSoup:
    fragment = BeautifulSoup(str(html_fragment or ""), "html.parser")
    nodes = [node for node in _extract_fragment_nodes(fragment) if not _is_ignorable_node(node)]
    if len(nodes) == 1 and isinstance(nodes[0], Tag):
        root = nodes[0]
        if str(root.get(GLOBAL_ATTR) or "").strip() == global_id:
            normalized = BeautifulSoup("", "html.parser")
            for child in list(root.contents):
                normalized.append(child.extract())
            return normalized
    return fragment


def _replace_tag_contents(target: Tag, fragment: BeautifulSoup) -> None:
    target.clear()
    for node in _extract_fragment_nodes(fragment):
        if _is_ignorable_node(node):
            continue
        target.append(node)


def _select_block_target(soup: BeautifulSoup, block_id: str) -> Tag:
    block_targets = soup.select(build_block_selector(block_id))
    if len(block_targets) != 1:
        raise PatchApplicationError(f"Expected exactly one block '{block_id}' in current HTML.")
    return block_targets[0]


def _select_slot_target(soup: BeautifulSoup, block_id: str, slot_id: str) -> Tag:
    block_target = _select_block_target(soup, block_id)
    slot_targets = block_target.select(f'[{SLOT_ATTR}="{slot_id}"]')
    if len(slot_targets) != 1:
        raise PatchApplicationError(
            f"Expected exactly one slot '{slot_id}' inside block '{block_id}', found {len(slot_targets)}."
        )
    return slot_targets[0]


def _select_global_target(soup: BeautifulSoup, global_id: str) -> Tag:
    global_targets = soup.select(build_global_selector(global_id))
    if len(global_targets) != 1:
        raise PatchApplicationError(f"Expected exactly one global anchor '{global_id}' in current HTML.")
    return global_targets[0]


def _apply_slot_replacement(soup: BeautifulSoup, block_id: str, slot_id: str, html_fragment: str) -> None:
    slot_target = _select_slot_target(soup, block_id, slot_id)
    slot_fragment = _normalize_slot_fragment(html_fragment, slot_id)
    _replace_tag_contents(slot_target, slot_fragment)


def _apply_block_replacement(
    soup: BeautifulSoup,
    block_id: str,
    html_fragment: str,
    page_plan: PagePlan | None = None,
) -> None:
    block_target = _select_block_target(soup, block_id)
    root = _extract_single_root_tag(html_fragment)
    root_block_id = str(root.get("data-pa-block") or "").strip()
    if not root_block_id:
        root["data-pa-block"] = block_id
    elif root_block_id != block_id:
        raise PatchApplicationError(
            f"Replacement block root data-pa-block '{root_block_id}' does not match target '{block_id}'."
        )

    original_html = str(block_target)
    block_target.replace_with(root)

    if page_plan is not None:
        block_lookup = _block_model_lookup(page_plan)
        block_plan = block_lookup.get(block_id)
        shell_errors = validate_block_tag_against_shell_contract(
            block_tag=root,
            shell_contract=getattr(block_plan, "shell_contract", None),
            block_id=block_id,
        )
        if shell_errors:
            restored_root = _extract_single_root_tag(original_html)
            root.replace_with(restored_root)
            raise PatchApplicationError("; ".join(shell_errors))


def _apply_global_replacement(soup: BeautifulSoup, global_id: str, html_fragment: str) -> None:
    global_target = _select_global_target(soup, global_id)
    global_fragment = _normalize_global_fragment(html_fragment, global_id)
    _replace_tag_contents(global_target, global_fragment)


def _parse_inline_style(style_text: str) -> dict[str, str]:
    declarations: dict[str, str] = {}
    for part in str(style_text or "").split(";"):
        if ":" not in part:
            continue
        prop, value = part.split(":", 1)
        clean_prop = str(prop or "").strip()
        clean_value = str(value or "").strip()
        if clean_prop and clean_value:
            declarations[clean_prop] = clean_value
    return declarations


def _serialize_inline_style(declarations: dict[str, str]) -> str:
    if not declarations:
        return ""
    return "; ".join(f"{prop}: {value}" for prop, value in declarations.items()) + ";"


def _apply_style_change_to_tag(target: Tag, declarations: dict[str, str]) -> None:
    merged = _parse_inline_style(str(target.get("style") or ""))
    for prop, value in declarations.items():
        clean_prop = str(prop or "").strip()
        clean_value = str(value or "").strip()
        if clean_prop and clean_value:
            merged[clean_prop] = clean_value
    serialized = _serialize_inline_style(merged)
    if serialized:
        target["style"] = serialized
    else:
        target.attrs.pop("style", None)


def _apply_style_change(soup: BeautifulSoup, style_change: StyleChange) -> None:
    if style_change.scope == "slot":
        target = _select_slot_target(soup, str(style_change.block_id or ""), str(style_change.slot_id or ""))
    elif style_change.scope == "block":
        target = _select_block_target(soup, str(style_change.block_id or ""))
    else:
        target = _select_global_target(soup, str(style_change.global_id or ""))
    _apply_style_change_to_tag(target, {str(k): v for k, v in style_change.declarations.items()})


def _apply_attribute_change_to_tag(target: Tag, attributes: dict[str, str]) -> None:
    for name, value in attributes.items():
        clean_name = str(name or "").strip()
        clean_value = str(value or "").strip()
        if not clean_name:
            continue
        if not clean_value:
            target.attrs.pop(clean_name, None)
            continue
        if clean_name == "class":
            target[clean_name] = [item for item in clean_value.split() if item.strip()]
        else:
            target[clean_name] = clean_value


def _apply_attribute_change(soup: BeautifulSoup, attribute_change: AttributeChange) -> None:
    if attribute_change.scope == "slot":
        target = _select_slot_target(
            soup,
            str(attribute_change.block_id or ""),
            str(attribute_change.slot_id or ""),
        )
    elif attribute_change.scope == "block":
        target = _select_block_target(soup, str(attribute_change.block_id or ""))
    else:
        target = _select_global_target(soup, str(attribute_change.global_id or ""))
    _apply_attribute_change_to_tag(
        target,
        {str(name): value for name, value in attribute_change.attributes.items()},
    )


def _ensure_override_style_tag(soup: BeautifulSoup) -> Tag:
    existing = soup.select_one(f'style[id="{REVISION_OVERRIDE_STYLE_TAG_ID}"]')
    if isinstance(existing, Tag):
        return existing

    if soup.head is None:
        head = soup.new_tag("head")
        if soup.html is not None:
            soup.html.insert(0, head)
        else:
            soup.insert(0, head)
    style_tag = soup.new_tag("style")
    style_tag["id"] = REVISION_OVERRIDE_STYLE_TAG_ID
    soup.head.append(style_tag)
    return style_tag


def _render_override_rule(selector: str, declarations: dict[str, str]) -> str:
    lines = [f"{selector} {{"]
    for prop, value in declarations.items():
        lines.append(f"  {prop}: {value};")
    lines.append("}")
    return "\n".join(lines)


def _apply_override_css_rules(soup: BeautifulSoup, rules: list[Any]) -> None:
    if not rules:
        return
    style_tag = _ensure_override_style_tag(soup)
    existing_text = style_tag.string or style_tag.get_text() or ""
    rendered_rules = [
        _render_override_rule(
            selector=str(rule.selector),
            declarations={str(prop): value for prop, value in rule.declarations.items()},
        )
        for rule in rules
    ]
    combined = "\n\n".join([part for part in [existing_text.strip(), *rendered_rules] if part])
    style_tag.string = combined


def _current_slot_html_map(block_tag: Tag) -> dict[str, str]:
    slot_html_map: dict[str, str] = {}
    for slot_tag in block_tag.select(f"[{SLOT_ATTR}]"):
        if not isinstance(slot_tag, Tag):
            continue
        slot_id = str(slot_tag.get(SLOT_ATTR) or "").strip()
        if not slot_id or slot_id in slot_html_map:
            continue
        slot_html_map[slot_id] = slot_tag.decode_contents()
    return slot_html_map


def _target_key(scope: str, block_id: str | None = None, slot_id: str | None = None, global_id: str | None = None) -> str:
    if scope == "slot":
        return f"slot:{block_id}:{slot_id}"
    if scope == "block":
        return f"block:{block_id}"
    if scope == "global":
        return f"global:{global_id}"
    return scope


def _target_key_label(key: str) -> str:
    scope, _, remainder = key.partition(":")
    if scope == "slot":
        block_id, _, slot_id = remainder.partition(":")
        return f"{block_id}.{slot_id}"
    return remainder


def _requested_target_keys(revision_plan: RevisionPlan) -> list[str]:
    keys: list[str] = []
    seen: set[str] = set()
    for edit in revision_plan.edits:
        key = _target_key(
            edit.scope,
            block_id=edit.block_id,
            slot_id=edit.slot_id,
            global_id=edit.global_id,
        )
        if key not in seen:
            seen.add(key)
            keys.append(key)
    return keys


def _snapshot_for_target_key(soup: BeautifulSoup, key: str) -> str:
    parts = key.split(":")
    scope = parts[0]
    if scope == "slot" and len(parts) == 3:
        tag = soup.select_one(build_slot_selector(parts[1], parts[2]))
    elif scope == "block" and len(parts) == 2:
        tag = soup.select_one(build_block_selector(parts[1]))
    elif scope == "global" and len(parts) == 2:
        tag = soup.select_one(build_global_selector(parts[1]))
    else:
        tag = None
    return str(tag) if isinstance(tag, Tag) else ""


def _target_snapshot_map(soup: BeautifulSoup, keys: list[str]) -> dict[str, str]:
    return {key: _snapshot_for_target_key(soup, key) for key in keys}


def _collect_allowed_anchor_selectors(manifest: PageManifest) -> set[str]:
    selectors: set[str] = set()
    for block in manifest.blocks:
        selectors.add(block.selector)
        for slot in block.slots:
            selectors.add(slot.selector)
    for global_anchor in manifest.globals:
        selectors.add(global_anchor.selector)
    return selectors


def _override_rule_target_keys(selector: str, manifest: PageManifest) -> set[str]:
    matched: set[str] = set()
    for block in manifest.blocks:
        if selector == block.selector or selector.startswith(block.selector):
            matched.add(_target_key("block", block_id=block.block_id))
        for slot in block.slots:
            if selector == slot.selector or selector.startswith(slot.selector):
                matched.add(_target_key("slot", block_id=block.block_id, slot_id=slot.slot_id))
    for global_anchor in manifest.globals:
        if selector == global_anchor.selector or selector.startswith(global_anchor.selector):
            matched.add(_target_key("global", global_id=global_anchor.global_id))
    return matched


def _build_target_anchor_context_json(
    current_html: str,
    revision_plan: RevisionPlan,
) -> str:
    soup = BeautifulSoup(current_html, "html.parser")
    payload: list[dict[str, Any]] = []
    seen_blocks: set[str] = set()
    seen_globals: set[str] = set()

    for edit in revision_plan.edits:
        if edit.scope in {"slot", "block"} and edit.block_id and edit.block_id not in seen_blocks:
            block_tag = soup.select_one(build_block_selector(edit.block_id))
            if isinstance(block_tag, Tag):
                payload.append(
                    {
                        "scope": "block",
                        "block_id": edit.block_id,
                        "current_block_html": str(block_tag),
                        "current_slot_html": _current_slot_html_map(block_tag),
                    }
                )
                seen_blocks.add(edit.block_id)
        if edit.scope == "global" and edit.global_id and edit.global_id not in seen_globals:
            global_tag = soup.select_one(build_global_selector(edit.global_id))
            if isinstance(global_tag, Tag):
                payload.append(
                    {
                        "scope": "global",
                        "global_id": edit.global_id,
                        "current_global_html": str(global_tag),
                    }
                )
                seen_globals.add(edit.global_id)

    return json.dumps(payload, indent=2, ensure_ascii=False)


def _load_current_manifest(artifact: CoderArtifact | None):
    if not artifact:
        return None
    return load_page_manifest(build_page_manifest_path(artifact.entry_html))


def _available_asset_manifest_from_artifact(artifact: CoderArtifact) -> list[dict[str, str]]:
    site_dir = Path(artifact.site_dir)
    entry_html_parent = Path(artifact.entry_html).parent
    manifest: list[dict[str, str]] = []
    for rel_path in artifact.copied_assets:
        asset_path = site_dir / rel_path
        if not asset_path.exists():
            continue
        web_path = os.path.relpath(asset_path, start=entry_html_parent).replace("\\", "/")
        if not web_path.startswith((".", "/")):
            web_path = f"./{web_path}"
        manifest.append(
            {
                "relative_path": rel_path.replace("\\", "/"),
                "web_path": web_path,
                "filename": asset_path.name,
            }
        )
    return manifest


def _block_model_lookup(page_plan: PagePlan) -> dict[str, Any]:
    return {block.block_id: block for block in page_plan.blocks}


def _block_plan_lookup(page_plan: PagePlan) -> dict[str, Any]:
    return {
        block.block_id: block.model_dump()
        for block in page_plan.blocks
    }


def _revision_edits_for_block(revision_plan: RevisionPlan, block_id: str) -> list[dict[str, Any]]:
    return [
        edit.model_dump()
        for edit in revision_plan.edits
        if edit.block_id == block_id
    ]


def _preserve_requirements_for_block(revision_plan: RevisionPlan, block_id: str) -> list[str]:
    requirements: list[str] = []
    seen: set[str] = set()
    for edit in revision_plan.edits:
        if edit.block_id != block_id:
            continue
        for requirement in edit.preserve_requirements:
            clean = str(requirement or "").strip()
            if clean and clean not in seen:
                seen.add(clean)
                requirements.append(clean)
    return requirements


def _regenerate_block_html(
    block_id: str,
    current_block_html: str,
    page_plan: PagePlan,
    structured_paper: StructuredPaper,
    revision_plan: RevisionPlan,
    template_reference_html: str,
    artifact: CoderArtifact,
) -> str:
    block_plan = _block_plan_lookup(page_plan).get(block_id)
    if not block_plan:
        raise PatchApplicationError(f"Could not find PagePlan block '{block_id}' for fallback regeneration.")

    source_sections_lookup = {
        outline_item.block_id: list(outline_item.source_sections)
        for outline_item in page_plan.page_outline
    }
    try:
        llm = get_llm(temperature=0.1, use_smart_model=True)
        response = llm.invoke(
            [
                SystemMessage(content=BLOCK_REGEN_SYSTEM_PROMPT),
                HumanMessage(
                    content=BLOCK_REGEN_USER_PROMPT_TEMPLATE.format(
                        block_id=block_id,
                        source_sections_json=json.dumps(
                            source_sections_lookup.get(block_id, []),
                            indent=2,
                            ensure_ascii=False,
                        ),
                        target_block_plan_json=json.dumps(block_plan, indent=2, ensure_ascii=False),
                        target_block_edits_json=json.dumps(
                            _revision_edits_for_block(revision_plan, block_id),
                            indent=2,
                            ensure_ascii=False,
                        ),
                        preserve_requirements_json=json.dumps(
                            _preserve_requirements_for_block(revision_plan, block_id),
                            indent=2,
                            ensure_ascii=False,
                        ),
                        available_paper_assets_json=json.dumps(
                            _available_asset_manifest_from_artifact(artifact),
                            indent=2,
                            ensure_ascii=False,
                        ),
                        structured_paper_json=to_pretty_json(structured_paper),
                        current_block_html=current_block_html,
                        template_reference_html=template_reference_html,
                    )
                ),
            ]
        )
    except Exception as exc:
        raise PatchApplicationError(f"Block regeneration failed for '{block_id}': {exc}") from exc

    regenerated_html = extract_html_fragment(message_content_to_text(response))
    if not regenerated_html:
        raise PatchApplicationError(f"Block regeneration returned empty HTML for '{block_id}'.")

    return regenerated_html


def _strict_revision_validation_enabled(manifest: PageManifest) -> bool:
    return str(manifest.schema_version or "").strip() != "1.0"


def patch_agent_node(state: WorkflowState) -> dict[str, Any]:
    artifact = _normalize_coder_artifact(state.get("coder_artifact"))
    page_plan = _normalize_page_plan(state.get("page_plan"))
    structured_paper = _normalize_structured_paper(state.get("structured_paper"))
    revision_plan = _normalize_revision_plan(state.get("revision_plan"))
    current_html = read_current_page_html(artifact, missing_value="")
    template_reference_html = read_template_reference_html(page_plan, missing_value="")
    manifest = _load_current_manifest(artifact)

    if not artifact or not page_plan or not structured_paper or not current_html or not template_reference_html:
        message = "Patch Agent could not run because artifact, page plan, structured paper, or current HTML is missing."
        print(f"[PatchAgent] {message}")
        return {"targeted_replacement_plan": None, "patch_agent_output": "", "patch_error": message}
    if manifest is None:
        print(f"[PatchAgent] {LEGACY_PAGE_ERROR}")
        return {"targeted_replacement_plan": None, "patch_agent_output": "", "patch_error": LEGACY_PAGE_ERROR}
    missing_shell_blocks = missing_shell_contract_block_ids(page_plan)
    if missing_shell_blocks:
        message = LEGACY_PAGE_ERROR + " Missing shell contracts for: " + ", ".join(missing_shell_blocks)
        print(f"[PatchAgent] {message}")
        return {"targeted_replacement_plan": None, "patch_agent_output": "", "patch_error": message}
    if not revision_plan or not revision_plan.edits:
        message = "Translator did not produce any actionable anchored edits from the feedback."
        print(f"[PatchAgent] {message}")
        return {"targeted_replacement_plan": None, "patch_agent_output": "", "patch_error": message}

    manifest_blocks = {block.block_id: block for block in manifest.blocks}
    manifest_globals = {item.global_id: item for item in manifest.globals}
    missing_targets: list[str] = []
    for edit in revision_plan.edits:
        if edit.scope == "global":
            if edit.global_id not in manifest_globals:
                missing_targets.append(f"unknown global_id '{edit.global_id}'")
            continue

        block = manifest_blocks.get(str(edit.block_id or ""))
        if block is None:
            missing_targets.append(f"unknown block_id '{edit.block_id}'")
            continue
        if edit.scope == "slot" and edit.slot_id not in {slot.slot_id for slot in block.slots}:
            missing_targets.append(
                f"block '{edit.block_id}' does not expose requested slot '{edit.slot_id}'"
            )

    if missing_targets:
        message = "Revision targets are not available in the current anchored page: " + "; ".join(missing_targets)
        print(f"[PatchAgent] {message}")
        return {"targeted_replacement_plan": None, "patch_agent_output": "", "patch_error": message}

    asset_manifest = _available_asset_manifest_from_artifact(artifact)
    allowed_asset_web_paths = collect_allowed_asset_web_paths(asset_manifest)
    allowed_existing_local_sources = set(collect_local_image_sources(current_html))

    print("[PatchAgent] Building targeted DOM replacement plan...")
    try:
        llm = get_llm(temperature=0.1, use_smart_model=True)
        response = llm.invoke(
            [
                SystemMessage(content=PATCH_AGENT_SYSTEM_PROMPT),
                HumanMessage(
                    content=PATCH_AGENT_USER_PROMPT_TEMPLATE.format(
                        revision_plan_json=to_pretty_json(revision_plan),
                        current_page_manifest_json=to_pretty_json(manifest),
                        target_anchor_context_json=_build_target_anchor_context_json(
                            current_html=current_html,
                            revision_plan=revision_plan,
                        ),
                        structured_paper_json=to_pretty_json(structured_paper),
                        template_reference_html=template_reference_html,
                        available_paper_assets_json=json.dumps(
                            asset_manifest,
                            indent=2,
                            ensure_ascii=False,
                        ),
                    )
                ),
            ]
        )
    except Exception as exc:
        message = f"Patch Agent failed generating a targeted replacement plan: {exc}"
        print(f"[PatchAgent] {message}")
        return {"targeted_replacement_plan": None, "patch_agent_output": "", "patch_error": message}

    targeted_plan, sanitize_warnings = _build_targeted_replacement_plan_from_response(response)
    for warning in sanitize_warnings:
        print(f"[PatchAgent] {warning}")
    if not targeted_plan:
        message = "Patch Agent returned invalid targeted replacement plan JSON."
        print(f"[PatchAgent] {message}")
        return {"targeted_replacement_plan": None, "patch_agent_output": "", "patch_error": message}
    if (
        not targeted_plan.replacements
        and not targeted_plan.style_changes
        and not targeted_plan.attribute_changes
        and not targeted_plan.override_css_rules
        and not targeted_plan.fallback_blocks
    ):
        message = "Patch Agent returned an empty targeted replacement plan."
        print(f"[PatchAgent] {message}")
        return {"targeted_replacement_plan": None, "patch_agent_output": "", "patch_error": message}

    valid_block_ids = {block.block_id for block in manifest.blocks}
    valid_global_ids = {item.global_id for item in manifest.globals}
    valid_slots_by_block = {
        block.block_id: {slot.slot_id for slot in block.slots}
        for block in manifest.blocks
    }
    allowed_anchor_selectors = _collect_allowed_anchor_selectors(manifest)

    normalized_replacements = []
    normalized_style_changes: list[StyleChange] = []
    normalized_attribute_changes: list[AttributeChange] = []
    normalized_override_rules = []
    normalized_fallbacks: list[FallbackBlock] = []
    seen_fallback_blocks: set[str] = set()

    for fallback in targeted_plan.fallback_blocks:
        if fallback.block_id not in valid_block_ids:
            message = f"Patch Agent referenced unknown fallback block_id '{fallback.block_id}'."
            print(f"[PatchAgent] {message}")
            return {"targeted_replacement_plan": None, "patch_agent_output": "", "patch_error": message}
        if fallback.block_id not in seen_fallback_blocks:
            normalized_fallbacks.append(fallback)
            seen_fallback_blocks.add(fallback.block_id)

    for replacement in targeted_plan.replacements:
        if replacement.scope == "global":
            if replacement.global_id not in valid_global_ids:
                message = f"Patch Agent referenced unknown global_id '{replacement.global_id}'."
                print(f"[PatchAgent] {message}")
                return {"targeted_replacement_plan": None, "patch_agent_output": "", "patch_error": message}
        else:
            if replacement.block_id not in valid_block_ids:
                message = f"Patch Agent referenced unknown block_id '{replacement.block_id}'."
                print(f"[PatchAgent] {message}")
                return {"targeted_replacement_plan": None, "patch_agent_output": "", "patch_error": message}
            if replacement.scope == "slot":
                available_slots = valid_slots_by_block.get(str(replacement.block_id or ""), set())
                if str(replacement.slot_id or "").strip() not in available_slots:
                    if replacement.block_id not in seen_fallback_blocks:
                        normalized_fallbacks.append(
                            FallbackBlock(
                                block_id=str(replacement.block_id),
                                reason=(
                                    f"requested slot '{replacement.slot_id}' is not exposed in the current "
                                    f"manifest for block '{replacement.block_id}'"
                                ),
                            )
                        )
                        seen_fallback_blocks.add(str(replacement.block_id))
                    continue

        fragment_critiques = validate_fragment_local_image_sources(
            html_text=replacement.html,
            allowed_asset_web_paths=allowed_asset_web_paths,
            allowed_existing_local_sources=allowed_existing_local_sources,
        )
        if fragment_critiques:
            message = "Patch Agent generated invalid local image references: " + "; ".join(fragment_critiques)
            print(f"[PatchAgent] {message}")
            return {"targeted_replacement_plan": None, "patch_agent_output": "", "patch_error": message}

        normalized_replacements.append(replacement)

    for style_change in targeted_plan.style_changes:
        if style_change.scope == "global":
            if style_change.global_id not in valid_global_ids:
                message = f"Patch Agent referenced unknown global_id '{style_change.global_id}' in style_changes."
                print(f"[PatchAgent] {message}")
                return {"targeted_replacement_plan": None, "patch_agent_output": "", "patch_error": message}
            normalized_style_changes.append(style_change)
            continue

        if style_change.block_id not in valid_block_ids:
            message = f"Patch Agent referenced unknown block_id '{style_change.block_id}' in style_changes."
            print(f"[PatchAgent] {message}")
            return {"targeted_replacement_plan": None, "patch_agent_output": "", "patch_error": message}

        if style_change.scope == "slot":
            available_slots = valid_slots_by_block.get(str(style_change.block_id or ""), set())
            if str(style_change.slot_id or "").strip() not in available_slots:
                if style_change.block_id not in seen_fallback_blocks:
                    normalized_fallbacks.append(
                        FallbackBlock(
                            block_id=str(style_change.block_id),
                            reason=(
                                f"requested style target slot '{style_change.slot_id}' is not exposed in the current "
                                f"manifest for block '{style_change.block_id}'"
                            ),
                        )
                    )
                    seen_fallback_blocks.add(str(style_change.block_id))
                continue

        normalized_style_changes.append(style_change)

    for attribute_change in targeted_plan.attribute_changes:
        if attribute_change.scope == "global":
            if attribute_change.global_id not in valid_global_ids:
                message = f"Patch Agent referenced unknown global_id '{attribute_change.global_id}' in attribute_changes."
                print(f"[PatchAgent] {message}")
                return {"targeted_replacement_plan": None, "patch_agent_output": "", "patch_error": message}
            normalized_attribute_changes.append(attribute_change)
            continue

        if attribute_change.block_id not in valid_block_ids:
            message = f"Patch Agent referenced unknown block_id '{attribute_change.block_id}' in attribute_changes."
            print(f"[PatchAgent] {message}")
            return {"targeted_replacement_plan": None, "patch_agent_output": "", "patch_error": message}

        if attribute_change.scope == "slot":
            available_slots = valid_slots_by_block.get(str(attribute_change.block_id or ""), set())
            if str(attribute_change.slot_id or "").strip() not in available_slots:
                if attribute_change.block_id not in seen_fallback_blocks:
                    normalized_fallbacks.append(
                        FallbackBlock(
                            block_id=str(attribute_change.block_id),
                            reason=(
                                f"requested attribute target slot '{attribute_change.slot_id}' is not exposed in the current "
                                f"manifest for block '{attribute_change.block_id}'"
                            ),
                        )
                    )
                    seen_fallback_blocks.add(str(attribute_change.block_id))
                continue

        normalized_attribute_changes.append(attribute_change)

    for rule in targeted_plan.override_css_rules:
        selector = str(rule.selector or "").strip()
        if not is_safe_anchored_selector(selector, allowed_anchor_selectors):
            message = f"Patch Agent produced an unsafe override selector '{selector}'."
            print(f"[PatchAgent] {message}")
            return {"targeted_replacement_plan": None, "patch_agent_output": "", "patch_error": message}
        if not _override_rule_target_keys(selector, manifest):
            message = f"Patch Agent produced an override selector that does not match any anchored target: '{selector}'."
            print(f"[PatchAgent] {message}")
            return {"targeted_replacement_plan": None, "patch_agent_output": "", "patch_error": message}
        normalized_override_rules.append(rule)

    targeted_plan = TargetedReplacementPlan(
        replacements=normalized_replacements,
        style_changes=normalized_style_changes,
        attribute_changes=normalized_attribute_changes,
        override_css_rules=normalized_override_rules,
        fallback_blocks=normalized_fallbacks,
    )
    if (
        not targeted_plan.replacements
        and not targeted_plan.style_changes
        and not targeted_plan.attribute_changes
        and not targeted_plan.override_css_rules
        and not targeted_plan.fallback_blocks
    ):
        message = "Patch Agent returned no valid anchored changes after manifest validation."
        print(f"[PatchAgent] {message}")
        return {"targeted_replacement_plan": None, "patch_agent_output": "", "patch_error": message}

    return {
        "targeted_replacement_plan": targeted_plan,
        "patch_agent_output": _summarize_targeted_plan(targeted_plan),
        "patch_error": "",
    }


def patch_executor_node(state: WorkflowState) -> dict[str, Any]:
    existing_error = str(state.get("patch_error") or "").strip()
    if existing_error:
        print(f"[PatchExecutor] upstream safe fail: {existing_error}")
        return {"patch_error": existing_error}

    artifact = _normalize_coder_artifact(state.get("coder_artifact"))
    page_plan = _normalize_page_plan(state.get("page_plan"))
    structured_paper = _normalize_structured_paper(state.get("structured_paper"))
    revision_plan = _normalize_revision_plan(state.get("revision_plan"))
    targeted_plan = _normalize_targeted_replacement_plan(state.get("targeted_replacement_plan"))

    if not artifact:
        message = "Patch Executor could not run because coder_artifact is missing."
        print(f"[PatchExecutor] {message}")
        return {"patch_error": message}
    if not page_plan:
        message = "Patch Executor could not run because page_plan is missing."
        print(f"[PatchExecutor] {message}")
        return {"patch_error": message}
    if not structured_paper:
        message = "Patch Executor could not run because structured_paper is missing."
        print(f"[PatchExecutor] {message}")
        return {"patch_error": message}
    if not revision_plan:
        message = "Patch Executor could not run because revision_plan is missing."
        print(f"[PatchExecutor] {message}")
        return {"patch_error": message}
    if not targeted_plan:
        message = "Patch Executor could not run because targeted_replacement_plan is missing."
        print(f"[PatchExecutor] {message}")
        return {"patch_error": message}
    missing_shell_blocks = missing_shell_contract_block_ids(page_plan)
    if missing_shell_blocks:
        message = LEGACY_PAGE_ERROR + " Missing shell contracts for: " + ", ".join(missing_shell_blocks)
        print(f"[PatchExecutor] {message}")
        return {"patch_error": message}

    entry_html_path = Path(artifact.entry_html)
    if not entry_html_path.exists():
        message = f"Patch Executor could not find current HTML: {entry_html_path}"
        print(f"[PatchExecutor] {message}")
        return {"patch_error": message}

    manifest_path = build_page_manifest_path(entry_html_path)
    manifest = load_page_manifest(manifest_path)
    if manifest is None:
        print(f"[PatchExecutor] {LEGACY_PAGE_ERROR}")
        return {"patch_error": LEGACY_PAGE_ERROR}

    try:
        current_html = read_text_with_fallback(entry_html_path)
    except Exception as exc:
        message = f"Patch Executor failed reading current HTML: {exc}"
        print(f"[PatchExecutor] {message}")
        return {"patch_error": message}

    template_reference_html = read_template_reference_html(page_plan, missing_value="")
    if not template_reference_html:
        message = "Patch Executor could not read template reference HTML for block fallback regeneration."
        print(f"[PatchExecutor] {message}")
        return {"patch_error": message}

    strict_validation = _strict_revision_validation_enabled(manifest)
    asset_manifest = _available_asset_manifest_from_artifact(artifact)
    allowed_asset_web_paths = collect_allowed_asset_web_paths(asset_manifest)
    allowed_existing_local_sources = set(collect_local_image_sources(current_html))

    soup = BeautifulSoup(current_html, "html.parser")
    requested_target_keys = _requested_target_keys(revision_plan)
    before_snapshots = _target_snapshot_map(soup, requested_target_keys)
    changed_target_keys: set[str] = set()

    fallback_reasons = {item.block_id: item.reason for item in targeted_plan.fallback_blocks}
    fallback_block_ids: list[str] = []
    applied_replacements = 0
    applied_style_changes = 0
    applied_attribute_changes = 0
    applied_override_rules = 0

    for replacement in targeted_plan.replacements:
        try:
            if replacement.scope == "slot":
                _apply_slot_replacement(
                    soup=soup,
                    block_id=str(replacement.block_id or ""),
                    slot_id=str(replacement.slot_id or ""),
                    html_fragment=replacement.html,
                )
                changed_target_keys.add(
                    _target_key("slot", block_id=str(replacement.block_id), slot_id=str(replacement.slot_id))
                )
            elif replacement.scope == "global":
                _apply_global_replacement(
                    soup=soup,
                    global_id=str(replacement.global_id or ""),
                    html_fragment=replacement.html,
                )
                changed_target_keys.add(_target_key("global", global_id=str(replacement.global_id)))
            else:
                _apply_block_replacement(
                    soup=soup,
                    block_id=str(replacement.block_id or ""),
                    html_fragment=replacement.html,
                    page_plan=page_plan,
                )
                changed_target_keys.add(_target_key("block", block_id=str(replacement.block_id)))
            applied_replacements += 1
        except PatchApplicationError as exc:
            if replacement.scope == "global":
                message = f"Targeted global replacement failed for '{replacement.global_id}': {exc}"
                print(f"[PatchExecutor] {message}")
                return {"patch_error": message}
            print(
                "[PatchExecutor] targeted replacement fell back to block regeneration: "
                f"{replacement.block_id} ({exc})"
            )
            if replacement.block_id and replacement.block_id not in fallback_reasons:
                fallback_reasons[str(replacement.block_id)] = str(exc)
            if replacement.block_id and replacement.block_id not in fallback_block_ids:
                fallback_block_ids.append(str(replacement.block_id))

    for style_change in targeted_plan.style_changes:
        try:
            _apply_style_change(soup, style_change)
            changed_target_keys.add(
                _target_key(
                    style_change.scope,
                    block_id=style_change.block_id,
                    slot_id=style_change.slot_id,
                    global_id=style_change.global_id,
                )
            )
            applied_style_changes += 1
        except PatchApplicationError as exc:
            if style_change.scope == "global":
                message = f"Style change failed for global '{style_change.global_id}': {exc}"
                print(f"[PatchExecutor] {message}")
                return {"patch_error": message}
            if style_change.block_id and style_change.block_id not in fallback_reasons:
                fallback_reasons[str(style_change.block_id)] = str(exc)
            if style_change.block_id and style_change.block_id not in fallback_block_ids:
                fallback_block_ids.append(str(style_change.block_id))

    for attribute_change in targeted_plan.attribute_changes:
        try:
            _apply_attribute_change(soup, attribute_change)
            changed_target_keys.add(
                _target_key(
                    attribute_change.scope,
                    block_id=attribute_change.block_id,
                    slot_id=attribute_change.slot_id,
                    global_id=attribute_change.global_id,
                )
            )
            applied_attribute_changes += 1
        except PatchApplicationError as exc:
            if attribute_change.scope == "global":
                message = f"Attribute change failed for global '{attribute_change.global_id}': {exc}"
                print(f"[PatchExecutor] {message}")
                return {"patch_error": message}
            if attribute_change.block_id and attribute_change.block_id not in fallback_reasons:
                fallback_reasons[str(attribute_change.block_id)] = str(exc)
            if attribute_change.block_id and attribute_change.block_id not in fallback_block_ids:
                fallback_block_ids.append(str(attribute_change.block_id))

    if targeted_plan.override_css_rules:
        _apply_override_css_rules(soup, targeted_plan.override_css_rules)
        applied_override_rules = len(targeted_plan.override_css_rules)
        for rule in targeted_plan.override_css_rules:
            changed_target_keys.update(_override_rule_target_keys(str(rule.selector), manifest))

    for fallback in targeted_plan.fallback_blocks:
        if fallback.block_id not in fallback_block_ids:
            fallback_block_ids.append(fallback.block_id)

    regenerated_count = 0
    for block_id in fallback_block_ids:
        try:
            target_block = _select_block_target(soup, block_id)
            regenerated_html = _regenerate_block_html(
                block_id=block_id,
                current_block_html=str(target_block),
                page_plan=page_plan,
                structured_paper=structured_paper,
                revision_plan=revision_plan,
                template_reference_html=template_reference_html,
                artifact=artifact,
            )
            fragment_critiques = validate_fragment_local_image_sources(
                html_text=regenerated_html,
                allowed_asset_web_paths=allowed_asset_web_paths,
                allowed_existing_local_sources=allowed_existing_local_sources,
            )
            if fragment_critiques:
                raise PatchApplicationError("; ".join(fragment_critiques))
            _apply_block_replacement(
                soup=soup,
                block_id=block_id,
                html_fragment=regenerated_html,
                page_plan=page_plan,
            )
            changed_target_keys.add(_target_key("block", block_id=block_id))
            regenerated_count += 1
        except PatchApplicationError as exc:
            reason = str(exc) or fallback_reasons.get(block_id) or "unknown regeneration error"
            message = f"Target block fallback regeneration failed for '{block_id}': {reason}"
            print(f"[PatchExecutor] {message}")
            return {"patch_error": message}

    if (
        applied_replacements == 0
        and applied_style_changes == 0
        and applied_attribute_changes == 0
        and applied_override_rules == 0
        and regenerated_count == 0
    ):
        message = "Patch Executor did not receive any actionable targeted revisions."
        print(f"[PatchExecutor] {message}")
        return {"patch_error": message}

    after_snapshots = _target_snapshot_map(soup, requested_target_keys)
    unchanged_target_keys = [
        key
        for key, before_html in before_snapshots.items()
        if before_html.strip() == after_snapshots.get(key, "").strip() and key not in changed_target_keys
    ]
    if unchanged_target_keys:
        message = (
            "Anchored revision completed but did not change the requested target(s): "
            + ", ".join(_target_key_label(key) for key in unchanged_target_keys)
        )
        print(f"[PatchExecutor] {message}")
        return {"patch_error": message}

    updated_html = str(soup)

    if strict_validation:
        asset_critiques = validate_local_image_references(
            html_text=updated_html,
            entry_html_path=entry_html_path,
            site_dir=Path(artifact.site_dir),
            allowed_asset_web_paths=allowed_asset_web_paths,
            enforce_paper_asset_whitelist=True,
        )
        if asset_critiques:
            message = "Post-revision asset validation failed: " + "; ".join(asset_critiques)
            print(f"[PatchExecutor] {message}")
            return {"patch_error": message}

    try:
        updated_manifest = extract_page_manifest(
            html_text=updated_html,
            entry_html=entry_html_path,
            selected_template_id=artifact.selected_template_id,
            page_plan=page_plan,
            require_expected_globals=strict_validation,
        )
    except Exception as exc:
        message = f"Anchored revision validation failed after applying DOM patch: {exc}"
        print(f"[PatchExecutor] {message}")
        return {"patch_error": message}

    updated_artifact = artifact.model_copy(deep=True)
    patch_target_paths = _build_patch_target_paths(updated_artifact, page_plan)
    site_dir = Path(updated_artifact.site_dir)
    edited_html_files = _to_site_relative_paths(patch_target_paths, site_dir)
    updated_artifact.edited_files = _merge_unique_strings(updated_artifact.edited_files, edited_html_files)
    updated_artifact.notes = _update_patch_notes(
        updated_artifact.notes,
        replacements_count=applied_replacements,
        regenerated_count=regenerated_count,
        style_change_count=applied_style_changes,
        attribute_change_count=applied_attribute_changes,
        override_rule_count=applied_override_rules,
    )

    artifact_json_path = entry_html_path.resolve().parent.parent / "coder_artifact.json"
    file_contents: dict[Path, str] = {path: updated_html for path in patch_target_paths}
    file_contents[manifest_path] = json.dumps(
        updated_manifest.model_dump(),
        indent=2,
        ensure_ascii=False,
    )
    file_contents[artifact_json_path] = json.dumps(
        updated_artifact.model_dump(),
        indent=2,
        ensure_ascii=False,
    )

    try:
        _write_files_transaction(file_contents)
    except PatchApplicationError as exc:
        message = str(exc)
        print(f"[PatchExecutor] {message}")
        return {"patch_error": message}

    print(
        "[PatchExecutor] applied targeted revision: "
        f"{applied_replacements} replacement(s), "
        f"{applied_style_changes} style change(s), "
        f"{applied_attribute_changes} attribute change(s), "
        f"{applied_override_rules} override rule(s), "
        f"{regenerated_count} regenerated block(s)."
    )
    return {
        "coder_artifact": updated_artifact,
        "patch_error": "",
        "patch_agent_output": _summarize_targeted_plan(targeted_plan),
    }
