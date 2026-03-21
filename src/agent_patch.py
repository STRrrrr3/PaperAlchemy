from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.human_feedback import extract_human_feedback_text
from src.llm import get_llm
from src.prompts import PATCH_AGENT_SYSTEM_PROMPT, PATCH_AGENT_USER_PROMPT_TEMPLATE
from src.schemas import CoderArtifact, PagePlan
from src.state import WorkflowState

FULL_REGENERATE_REQUIRED = "FULL_REGENERATE_REQUIRED"

_SEARCH_START = "<<<<<<< SEARCH\n"
_SEARCH_SEPARATOR = "\n=======\n"
_SEARCH_END = "\n>>>>>>> REPLACE"


@dataclass(frozen=True)
class SearchReplaceBlock:
    search: str
    replace: str


@dataclass(frozen=True)
class PatchOperation:
    start: int
    end: int
    replacement: str


class PatchParseError(ValueError):
    """Raised when Patch Agent output is not strict Search/Replace text."""


class PatchApplicationError(ValueError):
    """Raised when deterministic patch validation or file writes fail."""


def _normalize_page_plan(plan: Any) -> PagePlan | None:
    if isinstance(plan, PagePlan):
        return plan
    if plan is None:
        return None
    try:
        return PagePlan.model_validate(plan)
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


def is_full_regenerate_required(output: Any) -> bool:
    return str(output or "").strip() == FULL_REGENERATE_REQUIRED


def patch_agent_node(state: WorkflowState) -> dict[str, Any]:
    artifact = _normalize_coder_artifact(state.get("coder_artifact"))
    page_plan = _normalize_page_plan(state.get("page_plan"))
    current_html = _read_current_html(artifact)
    template_reference_html = _read_template_reference_html(page_plan)
    translated_instructions = str(state.get("coder_instructions") or "").strip()
    raw_human_feedback = extract_human_feedback_text(state.get("human_directives"))

    if not artifact or not page_plan or not current_html or not template_reference_html:
        print("[PatchAgent] missing artifact, page plan, current HTML, or template reference HTML; falling back to full regenerate.")
        return {"patch_agent_output": FULL_REGENERATE_REQUIRED}

    print("[PatchAgent] Generating grounded Search/Replace patch blocks...")
    try:
        llm = get_llm(temperature=0.1, use_smart_model=True)
        response = llm.invoke(
            [
                SystemMessage(content=PATCH_AGENT_SYSTEM_PROMPT),
                HumanMessage(
                    content=PATCH_AGENT_USER_PROMPT_TEMPLATE.format(
                        translated_instructions=translated_instructions or "(none)",
                        raw_human_feedback=raw_human_feedback or "(none)",
                        current_html=current_html,
                        template_reference_html=template_reference_html,
                    )
                ),
            ]
        )
    except Exception as exc:
        print(f"[PatchAgent] patch generation failed: {exc}. Falling back to full regenerate.")
        return {"patch_agent_output": FULL_REGENERATE_REQUIRED}

    patch_agent_output = _message_content_to_text(response)
    if is_full_regenerate_required(patch_agent_output):
        return {"patch_agent_output": FULL_REGENERATE_REQUIRED}

    candidate_output = patch_agent_output.strip()
    try:
        candidate_blocks = parse_search_replace_blocks(candidate_output)
        build_patch_operations(current_html, candidate_blocks)
    except (PatchParseError, PatchApplicationError) as exc:
        print(f"[PatchAgent] generated invalid patch output: {exc}. Falling back to full regenerate.")
        return {"patch_agent_output": FULL_REGENERATE_REQUIRED}

    return {"patch_agent_output": candidate_output}


def parse_search_replace_blocks(raw_text: str) -> list[SearchReplaceBlock]:
    normalized = str(raw_text or "").replace("\r\n", "\n").replace("\r", "\n")
    stripped = normalized.strip()
    if not stripped:
        raise PatchParseError("Patch Agent returned empty output.")
    if "```" in stripped:
        raise PatchParseError("Patch Agent output must not contain code fences.")
    if stripped == FULL_REGENERATE_REQUIRED:
        raise PatchParseError("Patch Agent requested full regenerate instead of patch blocks.")

    blocks: list[SearchReplaceBlock] = []
    position = 0
    while position < len(stripped):
        if not stripped.startswith(_SEARCH_START, position):
            raise PatchParseError("Patch Agent output contained text outside Search/Replace blocks.")

        search_start = position + len(_SEARCH_START)
        separator_index = stripped.find(_SEARCH_SEPARATOR, search_start)
        if separator_index < 0:
            raise PatchParseError("Patch Agent output is missing a SEARCH/REPLACE separator.")

        replace_start = separator_index + len(_SEARCH_SEPARATOR)
        end_index = stripped.find(_SEARCH_END, replace_start)
        if end_index < 0:
            raise PatchParseError("Patch Agent output is missing a REPLACE terminator.")

        search_text = stripped[search_start:separator_index]
        replace_text = stripped[replace_start:end_index]
        if not search_text:
            raise PatchParseError(f"Patch block {len(blocks) + 1} SEARCH snippet is empty.")

        blocks.append(SearchReplaceBlock(search=search_text, replace=replace_text))
        position = end_index + len(_SEARCH_END)
        while position < len(stripped) and stripped[position].isspace():
            position += 1

    return blocks


def _find_all_occurrences(text: str, snippet: str) -> list[int]:
    matches: list[int] = []
    start = 0
    while True:
        index = text.find(snippet, start)
        if index < 0:
            return matches
        matches.append(index)
        start = index + 1


def build_patch_operations(current_html: str, blocks: list[SearchReplaceBlock]) -> list[PatchOperation]:
    if not current_html:
        raise PatchApplicationError("Patch Executor received empty current HTML.")
    if not blocks:
        raise PatchApplicationError("Patch Executor received no patch blocks to apply.")

    operations: list[PatchOperation] = []
    matched_ranges: list[tuple[int, int, int]] = []

    for block_index, block in enumerate(blocks, start=1):
        match_positions = _find_all_occurrences(current_html, block.search)
        if not match_positions:
            raise PatchApplicationError(
                f"Patch Executor could not find an exact SEARCH match for block {block_index}."
            )
        if len(match_positions) > 1:
            raise PatchApplicationError(
                f"Patch Executor found multiple exact SEARCH matches for block {block_index}; add more context."
            )

        start = match_positions[0]
        end = start + len(block.search)
        matched_ranges.append((start, end, block_index))
        operations.append(PatchOperation(start=start, end=end, replacement=block.replace))

    matched_ranges.sort(key=lambda item: item[0])
    for previous, current in zip(matched_ranges, matched_ranges[1:]):
        prev_start, prev_end, prev_index = previous
        current_start, _, current_index = current
        if current_start < prev_end:
            raise PatchApplicationError(
                f"Patch Executor detected overlapping SEARCH ranges between blocks {prev_index} and {current_index}."
            )

    return operations


def apply_search_replace_blocks(current_html: str, blocks: list[SearchReplaceBlock]) -> str:
    updated_html = current_html
    operations = build_patch_operations(current_html, blocks)
    for operation in sorted(operations, key=lambda item: item.start, reverse=True):
        updated_html = (
            updated_html[: operation.start]
            + operation.replacement
            + updated_html[operation.end :]
        )
    return updated_html


def apply_patch_output(current_html: str, raw_patch_output: str) -> tuple[str, list[SearchReplaceBlock]]:
    blocks = parse_search_replace_blocks(raw_patch_output)
    return apply_search_replace_blocks(current_html, blocks), blocks


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


def _update_patch_notes(existing_notes: str, block_count: int) -> str:
    base = str(existing_notes or "").strip()
    patch_summary = f"v4-search-replace-patch: last successful patch applied {block_count} block(s)."
    if not base:
        return patch_summary

    parts = [part.strip() for part in base.split("|") if part.strip()]
    filtered_parts = [part for part in parts if not part.startswith("v4-search-replace-patch:")]
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


def patch_executor_node(state: WorkflowState) -> dict[str, Any]:
    artifact = _normalize_coder_artifact(state.get("coder_artifact"))
    page_plan = _normalize_page_plan(state.get("page_plan"))
    paper_folder_name = str(state.get("paper_folder_name") or "").strip()
    raw_patch_output = str(state.get("patch_agent_output") or "")

    if not artifact:
        message = "Patch Executor could not run because coder_artifact is missing."
        print(f"[PatchExecutor] {message}")
        return {"patch_error": message}
    if not page_plan:
        message = "Patch Executor could not run because page_plan is missing."
        print(f"[PatchExecutor] {message}")
        return {"patch_error": message}
    if not paper_folder_name:
        message = "Patch Executor could not run because paper_folder_name is missing."
        print(f"[PatchExecutor] {message}")
        return {"patch_error": message}

    entry_html_path = Path(artifact.entry_html)
    if not entry_html_path.exists():
        message = f"Patch Executor could not find current HTML: {entry_html_path}"
        print(f"[PatchExecutor] {message}")
        return {"patch_error": message}

    try:
        current_html = _read_text_with_fallback(entry_html_path)
    except Exception as exc:
        message = f"Patch Executor failed reading current HTML: {exc}"
        print(f"[PatchExecutor] {message}")
        return {"patch_error": message}

    try:
        patched_html, blocks = apply_patch_output(current_html, raw_patch_output)
    except (PatchParseError, PatchApplicationError) as exc:
        message = str(exc)
        print(f"[PatchExecutor] {message}")
        return {"patch_error": message}

    updated_artifact = artifact.model_copy(deep=True)
    patch_target_paths = _build_patch_target_paths(updated_artifact, page_plan)
    site_dir = Path(updated_artifact.site_dir)
    edited_html_files = _to_site_relative_paths(patch_target_paths, site_dir)
    updated_artifact.edited_files = _merge_unique_strings(updated_artifact.edited_files, edited_html_files)
    updated_artifact.notes = _update_patch_notes(updated_artifact.notes, len(blocks))

    project_root = Path(__file__).resolve().parent.parent
    artifact_json_path = project_root / "data" / "output" / paper_folder_name / "coder_artifact.json"
    file_contents: dict[Path, str] = {path: patched_html for path in patch_target_paths}
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

    print(f"[PatchExecutor] applied {len(blocks)} patch block(s) to {entry_html_path}")
    return {"coder_artifact": updated_artifact, "patch_error": ""}

