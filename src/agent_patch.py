from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup, Tag
from langchain_core.messages import HumanMessage, SystemMessage

from src.json_utils import to_pretty_json
from src.llm import get_llm
from src.page_manifest import (
    SLOT_ATTR,
    build_block_selector,
    build_page_manifest_path,
    extract_page_manifest,
    load_page_manifest,
)
from src.prompts import (
    BLOCK_REGEN_SYSTEM_PROMPT,
    BLOCK_REGEN_USER_PROMPT_TEMPLATE,
    PATCH_AGENT_SYSTEM_PROMPT,
    PATCH_AGENT_USER_PROMPT_TEMPLATE,
)
from src.schemas import (
    CoderArtifact,
    FallbackBlock,
    PagePlan,
    RevisionPlan,
    StructuredPaper,
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


def _read_text_with_fallback(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def _message_content_to_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str) and item.strip():
                parts.append(item.strip())
                continue
            if isinstance(item, dict):
                text = str(item.get("text") or "").strip()
                if text:
                    parts.append(text)
        return "\n".join(parts).strip()

    return str(content or "").strip()


def _read_current_html(artifact: CoderArtifact | None) -> str:
    if not artifact:
        return ""

    entry_path = Path(artifact.entry_html)
    if not entry_path.exists():
        return ""

    try:
        return _read_text_with_fallback(entry_path)
    except Exception:
        return ""


def _read_template_reference_html(page_plan: PagePlan | None) -> str:
    if not page_plan:
        return ""

    selected_root = str(page_plan.template_selection.selected_root_dir or "").strip()
    selected_entry = str(page_plan.template_selection.selected_entry_html or "").strip()
    if not selected_root or not selected_entry:
        return ""

    project_root = Path(__file__).resolve().parent.parent
    template_entry_path = project_root / selected_root / selected_entry
    if not template_entry_path.exists():
        return ""

    try:
        return _read_text_with_fallback(template_entry_path)
    except Exception:
        return ""


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


def _update_patch_notes(existing_notes: str, replacements_count: int, regenerated_count: int) -> str:
    base = str(existing_notes or "").strip()
    patch_summary = (
        "v5-targeted-revision: "
        f"applied {replacements_count} targeted replacement(s) and regenerated {regenerated_count} block(s)."
    )
    if not base:
        return patch_summary

    parts = [part.strip() for part in base.split("|") if part.strip()]
    filtered_parts = [part for part in parts if not part.startswith("v5-targeted-revision:")]
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
        f"fallback_blocks={len(plan.fallback_blocks)}"
    )


def _extract_fragment_nodes(fragment: BeautifulSoup) -> list[Any]:
    if fragment.body is not None:
        return list(fragment.body.contents)
    return list(fragment.contents)


def _extract_single_root_tag(html_fragment: str) -> Tag:
    fragment = BeautifulSoup(str(html_fragment or ""), "html.parser")
    nodes = [node for node in _extract_fragment_nodes(fragment) if not _is_ignorable_node(node)]
    if len(nodes) != 1 or not isinstance(nodes[0], Tag):
        raise PatchApplicationError("Expected exactly one root element in block replacement HTML.")
    return nodes[0]


def _is_ignorable_node(node: Any) -> bool:
    if isinstance(node, str):
        return not node.strip()
    return False


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


def _replace_tag_contents(target: Tag, fragment: BeautifulSoup) -> None:
    target.clear()
    for node in _extract_fragment_nodes(fragment):
        if _is_ignorable_node(node):
            continue
        target.append(node)


def _apply_slot_replacement(soup: BeautifulSoup, block_id: str, slot_id: str, html_fragment: str) -> None:
    block_targets = soup.select(build_block_selector(block_id))
    if len(block_targets) != 1:
        raise PatchApplicationError(f"Expected exactly one block '{block_id}' in current HTML.")

    slot_targets = block_targets[0].select(f'[{SLOT_ATTR}="{slot_id}"]')
    if len(slot_targets) != 1:
        raise PatchApplicationError(
            f"Expected exactly one slot '{slot_id}' inside block '{block_id}', found {len(slot_targets)}."
        )

    slot_fragment = _normalize_slot_fragment(html_fragment, slot_id)
    _replace_tag_contents(slot_targets[0], slot_fragment)


def _apply_block_replacement(soup: BeautifulSoup, block_id: str, html_fragment: str) -> None:
    block_targets = soup.select(build_block_selector(block_id))
    if len(block_targets) != 1:
        raise PatchApplicationError(f"Expected exactly one block '{block_id}' in current HTML.")

    root = _extract_single_root_tag(html_fragment)
    root_block_id = str(root.get("data-pa-block") or "").strip()
    if not root_block_id:
        root["data-pa-block"] = block_id
    elif root_block_id != block_id:
        raise PatchApplicationError(
            f"Replacement block root data-pa-block '{root_block_id}' does not match target '{block_id}'."
        )

    block_targets[0].replace_with(root)


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


def _target_block_html_map(soup: BeautifulSoup, block_ids: list[str]) -> dict[str, str]:
    block_html_map: dict[str, str] = {}
    seen: set[str] = set()
    for block_id in block_ids:
        clean = str(block_id or "").strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        block_tag = soup.select_one(build_block_selector(clean))
        if isinstance(block_tag, Tag):
            block_html_map[clean] = str(block_tag)
    return block_html_map


def _build_target_block_context_json(
    current_html: str,
    revision_plan: RevisionPlan,
) -> str:
    soup = BeautifulSoup(current_html, "html.parser")
    target_block_ids: list[str] = []
    for edit in revision_plan.edits:
        if edit.block_id not in target_block_ids:
            target_block_ids.append(edit.block_id)

    payload: list[dict[str, Any]] = []
    for block_id in target_block_ids:
        block_tag = soup.select_one(build_block_selector(block_id))
        if not isinstance(block_tag, Tag):
            continue

        payload.append(
            {
                "block_id": block_id,
                "current_block_html": str(block_tag),
                "current_slot_html": _current_slot_html_map(block_tag),
            }
        )

    return json.dumps(payload, indent=2, ensure_ascii=False)


def _load_current_manifest(artifact: CoderArtifact | None):
    if not artifact:
        return None
    return load_page_manifest(build_page_manifest_path(artifact.entry_html))


def patch_agent_node(state: WorkflowState) -> dict[str, Any]:
    artifact = _normalize_coder_artifact(state.get("coder_artifact"))
    page_plan = _normalize_page_plan(state.get("page_plan"))
    structured_paper = _normalize_structured_paper(state.get("structured_paper"))
    revision_plan = _normalize_revision_plan(state.get("revision_plan"))
    current_html = _read_current_html(artifact)
    template_reference_html = _read_template_reference_html(page_plan)
    manifest = _load_current_manifest(artifact)

    if not artifact or not page_plan or not structured_paper or not current_html or not template_reference_html:
        message = "Patch Agent could not run because artifact, page plan, structured paper, or current HTML is missing."
        print(f"[PatchAgent] {message}")
        return {"targeted_replacement_plan": None, "patch_agent_output": "", "patch_error": message}
    if manifest is None:
        print(f"[PatchAgent] {LEGACY_PAGE_ERROR}")
        return {"targeted_replacement_plan": None, "patch_agent_output": "", "patch_error": LEGACY_PAGE_ERROR}
    if not revision_plan or not revision_plan.edits:
        message = "Translator did not produce any actionable anchored edits from the feedback."
        print(f"[PatchAgent] {message}")
        return {"targeted_replacement_plan": None, "patch_agent_output": "", "patch_error": message}

    manifest_lookup = {block.block_id: block for block in manifest.blocks}
    missing_targets: list[str] = []
    for edit in revision_plan.edits:
        block = manifest_lookup.get(edit.block_id)
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

    print("[PatchAgent] Building targeted DOM replacement plan...")
    try:
        llm = get_llm(temperature=0.1, use_smart_model=True)
        structured_llm = llm.with_structured_output(TargetedReplacementPlan)
        response = structured_llm.invoke(
            [
                SystemMessage(content=PATCH_AGENT_SYSTEM_PROMPT),
                HumanMessage(
                    content=PATCH_AGENT_USER_PROMPT_TEMPLATE.format(
                        revision_plan_json=to_pretty_json(revision_plan),
                        current_page_manifest_json=to_pretty_json(manifest),
                        target_block_context_json=_build_target_block_context_json(
                            current_html=current_html,
                            revision_plan=revision_plan,
                        ),
                        structured_paper_json=to_pretty_json(structured_paper),
                        template_reference_html=template_reference_html,
                    )
                ),
            ]
        )
    except Exception as exc:
        message = f"Patch Agent failed generating a targeted replacement plan: {exc}"
        print(f"[PatchAgent] {message}")
        return {"targeted_replacement_plan": None, "patch_agent_output": "", "patch_error": message}

    targeted_plan = _normalize_targeted_replacement_plan(response)
    if not targeted_plan or (not targeted_plan.replacements and not targeted_plan.fallback_blocks):
        message = "Patch Agent returned an empty targeted replacement plan."
        print(f"[PatchAgent] {message}")
        return {"targeted_replacement_plan": None, "patch_agent_output": "", "patch_error": message}

    valid_block_ids = {block.block_id for block in manifest.blocks}
    valid_slots_by_block = {
        block.block_id: {slot.slot_id for slot in block.slots}
        for block in manifest.blocks
    }
    for replacement in targeted_plan.replacements:
        if replacement.block_id not in valid_block_ids:
            message = f"Patch Agent referenced unknown block_id '{replacement.block_id}'."
            print(f"[PatchAgent] {message}")
            return {"targeted_replacement_plan": None, "patch_agent_output": "", "patch_error": message}
    for fallback in targeted_plan.fallback_blocks:
        if fallback.block_id not in valid_block_ids:
            message = f"Patch Agent referenced unknown fallback block_id '{fallback.block_id}'."
            print(f"[PatchAgent] {message}")
            return {"targeted_replacement_plan": None, "patch_agent_output": "", "patch_error": message}

    normalized_replacements = []
    normalized_fallbacks: list[FallbackBlock] = []
    seen_fallback_blocks: set[str] = set()
    for fallback in targeted_plan.fallback_blocks:
        if fallback.block_id not in seen_fallback_blocks:
            normalized_fallbacks.append(fallback)
            seen_fallback_blocks.add(fallback.block_id)
    for replacement in targeted_plan.replacements:
        if replacement.scope == "slot":
            available_slots = valid_slots_by_block.get(replacement.block_id, set())
            if str(replacement.slot_id or "").strip() not in available_slots:
                if replacement.block_id not in seen_fallback_blocks:
                    normalized_fallbacks.append(
                        FallbackBlock(
                            block_id=replacement.block_id,
                            reason=(
                                f"requested slot '{replacement.slot_id}' is not exposed in the current "
                                f"manifest for block '{replacement.block_id}'"
                            ),
                        )
                    )
                    seen_fallback_blocks.add(replacement.block_id)
                continue
        normalized_replacements.append(replacement)

    targeted_plan = TargetedReplacementPlan(
        replacements=normalized_replacements,
        fallback_blocks=normalized_fallbacks,
    )
    if not targeted_plan.replacements and not targeted_plan.fallback_blocks:
        message = "Patch Agent returned no valid anchored replacements after manifest validation."
        print(f"[PatchAgent] {message}")
        return {"targeted_replacement_plan": None, "patch_agent_output": "", "patch_error": message}

    return {
        "targeted_replacement_plan": targeted_plan,
        "patch_agent_output": _summarize_targeted_plan(targeted_plan),
        "patch_error": "",
    }


def _extract_html_fragment(text: str) -> str:
    raw_text = str(text or "").strip()
    if not raw_text:
        return ""

    fenced_match = re.search(r"```(?:html)?\s*(.*?)```", raw_text, flags=re.IGNORECASE | re.DOTALL)
    if fenced_match:
        raw_text = fenced_match.group(1).strip()

    return raw_text.strip()


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

    regenerated_html = _extract_html_fragment(_message_content_to_text(response))
    if not regenerated_html:
        raise PatchApplicationError(f"Block regeneration returned empty HTML for '{block_id}'.")

    return regenerated_html


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
        current_html = _read_text_with_fallback(entry_html_path)
    except Exception as exc:
        message = f"Patch Executor failed reading current HTML: {exc}"
        print(f"[PatchExecutor] {message}")
        return {"patch_error": message}

    template_reference_html = _read_template_reference_html(page_plan)
    if not template_reference_html:
        message = "Patch Executor could not read template reference HTML for block fallback regeneration."
        print(f"[PatchExecutor] {message}")
        return {"patch_error": message}

    soup = BeautifulSoup(current_html, "html.parser")
    targeted_block_ids = [edit.block_id for edit in revision_plan.edits]
    before_block_html = _target_block_html_map(soup, targeted_block_ids)
    fallback_reasons = {item.block_id: item.reason for item in targeted_plan.fallback_blocks}
    fallback_block_ids: list[str] = []
    applied_replacements = 0

    for replacement in targeted_plan.replacements:
        try:
            if replacement.scope == "slot":
                _apply_slot_replacement(
                    soup=soup,
                    block_id=replacement.block_id,
                    slot_id=str(replacement.slot_id or ""),
                    html_fragment=replacement.html,
                )
            else:
                _apply_block_replacement(
                    soup=soup,
                    block_id=replacement.block_id,
                    html_fragment=replacement.html,
                )
            applied_replacements += 1
        except PatchApplicationError as exc:
            print(
                "[PatchExecutor] targeted replacement fell back to block regeneration: "
                f"{replacement.block_id} ({exc})"
            )
            if replacement.block_id not in fallback_reasons:
                fallback_reasons[replacement.block_id] = str(exc)
            if replacement.block_id not in fallback_block_ids:
                fallback_block_ids.append(replacement.block_id)

    for fallback in targeted_plan.fallback_blocks:
        if fallback.block_id not in fallback_block_ids:
            fallback_block_ids.append(fallback.block_id)

    regenerated_count = 0
    for block_id in fallback_block_ids:
        target_block = soup.select_one(build_block_selector(block_id))
        if not isinstance(target_block, Tag):
            message = f"Fallback regeneration could not find block '{block_id}' in current HTML."
            print(f"[PatchExecutor] {message}")
            return {"patch_error": message}

        try:
            regenerated_html = _regenerate_block_html(
                block_id=block_id,
                current_block_html=str(target_block),
                page_plan=page_plan,
                structured_paper=structured_paper,
                revision_plan=revision_plan,
                template_reference_html=template_reference_html,
                artifact=artifact,
            )
            _apply_block_replacement(
                soup=soup,
                block_id=block_id,
                html_fragment=regenerated_html,
            )
            regenerated_count += 1
        except PatchApplicationError as exc:
            reason = fallback_reasons.get(block_id) or str(exc)
            message = f"Target block fallback regeneration failed for '{block_id}': {reason}"
            print(f"[PatchExecutor] {message}")
            return {"patch_error": message}

    if applied_replacements == 0 and regenerated_count == 0:
        message = "Patch Executor did not receive any actionable targeted replacements."
        print(f"[PatchExecutor] {message}")
        return {"patch_error": message}

    after_block_html = _target_block_html_map(soup, targeted_block_ids)
    unchanged_target_blocks = [
        block_id
        for block_id, before_html in before_block_html.items()
        if before_html.strip() == after_block_html.get(block_id, "").strip()
    ]
    if unchanged_target_blocks:
        message = (
            "Anchored revision completed but did not change the requested target block(s): "
            + ", ".join(unchanged_target_blocks)
        )
        print(f"[PatchExecutor] {message}")
        return {"patch_error": message}

    updated_html = str(soup)
    try:
        updated_manifest = extract_page_manifest(
            html_text=updated_html,
            entry_html=entry_html_path,
            selected_template_id=artifact.selected_template_id,
            page_plan=page_plan,
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
        f"{applied_replacements} replacement(s), {regenerated_count} regenerated block(s)."
    )
    return {
        "coder_artifact": updated_artifact,
        "patch_error": "",
        "patch_agent_output": _summarize_targeted_plan(targeted_plan),
    }
