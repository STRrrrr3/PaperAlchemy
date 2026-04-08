from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.contracts.schemas import CoderArtifact, PagePlan, RevisionHistory, RevisionHistoryEntry
from src.patching.patch_pipeline import _build_patch_target_paths, _to_site_relative_paths
from src.services.artifact_store import get_output_paths, load_coder_artifact
from src.validators.page_manifest import build_page_manifest_path

REVISION_HISTORY_DIRNAME = "revisions"
REVISION_HISTORY_FILENAME = "history.json"


def build_revision_history_dir(paper_folder_name: str) -> Path:
    output_dir, _, _, _ = get_output_paths(paper_folder_name)
    return output_dir / REVISION_HISTORY_DIRNAME


def build_revision_history_path(paper_folder_name: str) -> Path:
    return build_revision_history_dir(paper_folder_name) / REVISION_HISTORY_FILENAME


def load_revision_history(paper_folder_name: str) -> RevisionHistory | None:
    history_path = build_revision_history_path(paper_folder_name)
    if not history_path.exists():
        return None
    try:
        payload = json.loads(history_path.read_text(encoding="utf-8"))
        return RevisionHistory.model_validate(payload)
    except Exception:
        return None


def save_revision_history(paper_folder_name: str, history: RevisionHistory) -> None:
    history_path = build_revision_history_path(paper_folder_name)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(
        json.dumps(history.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def build_revision_history_state(
    paper_folder_name: str = "",
    history: RevisionHistory | None = None,
) -> dict[str, Any]:
    normalized_history = history or RevisionHistory()
    version_ids = [entry.version_id for entry in normalized_history.versions]
    current_version_id = str(normalized_history.current_version_id or "").strip()
    current_index = version_ids.index(current_version_id) if current_version_id in version_ids else -1
    return {
        "paper_folder_name": str(paper_folder_name or "").strip(),
        "current_version_id": current_version_id,
        "version_ids": version_ids,
        "current_index": current_index,
        "total_versions": len(version_ids),
    }


def empty_revision_history_state() -> dict[str, Any]:
    return build_revision_history_state()


def _snapshot_dir(paper_folder_name: str, version_id: str) -> Path:
    return build_revision_history_dir(paper_folder_name) / version_id


def _entry_by_version_id(history: RevisionHistory, version_id: str) -> RevisionHistoryEntry:
    for entry in history.versions:
        if entry.version_id == version_id:
            return entry
    raise ValueError(f"Unknown revision history version '{version_id}'.")


def _current_index(history: RevisionHistory) -> int:
    current = str(history.current_version_id or "").strip()
    if not current:
        return len(history.versions) - 1
    for index, entry in enumerate(history.versions):
        if entry.version_id == current:
            return index
    raise ValueError(f"Unknown current revision version '{current}'.")


def _next_version_id(version_count: int) -> str:
    return f"v{version_count + 1:03d}"


def _delete_snapshot_dir(paper_folder_name: str, version_id: str) -> None:
    shutil.rmtree(_snapshot_dir(paper_folder_name, version_id), ignore_errors=True)


def _truncate_after_current(history: RevisionHistory, paper_folder_name: str) -> RevisionHistory:
    if not history.versions:
        history.current_version_id = ""
        return history

    current_index = _current_index(history)
    if current_index >= len(history.versions) - 1:
        return history

    for entry in history.versions[current_index + 1 :]:
        _delete_snapshot_dir(paper_folder_name, entry.version_id)
    history.versions = history.versions[: current_index + 1]
    history.current_version_id = history.versions[-1].version_id
    return history


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _snapshot_live_files(
    *,
    paper_folder_name: str,
    version_id: str,
    artifact: CoderArtifact,
    page_plan: PagePlan,
) -> list[str]:
    patch_target_paths = _build_patch_target_paths(artifact, page_plan)
    site_dir = Path(artifact.site_dir).resolve()
    html_files = _to_site_relative_paths(patch_target_paths, site_dir)
    snapshot_site_dir = _snapshot_dir(paper_folder_name, version_id) / "site"

    for live_path, relative_path in zip(patch_target_paths, html_files):
        _copy_file(Path(live_path).resolve(), snapshot_site_dir / relative_path)

    entry_html_path = Path(artifact.entry_html).resolve()
    _copy_file(build_page_manifest_path(entry_html_path), _snapshot_dir(paper_folder_name, version_id) / "page_manifest.json")
    _copy_file(entry_html_path.parent.parent / "coder_artifact.json", _snapshot_dir(paper_folder_name, version_id) / "coder_artifact.json")
    return html_files


def append_revision_version(
    *,
    paper_folder_name: str,
    artifact: CoderArtifact,
    page_plan: PagePlan,
    source: str,
    summary: str,
) -> RevisionHistory:
    history = load_revision_history(paper_folder_name) or RevisionHistory()
    history = _truncate_after_current(history, paper_folder_name)

    version_id = _next_version_id(len(history.versions))
    html_files = _snapshot_live_files(
        paper_folder_name=paper_folder_name,
        version_id=version_id,
        artifact=artifact,
        page_plan=page_plan,
    )
    history.versions.append(
        RevisionHistoryEntry(
            version_id=version_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            source=str(source).strip() or "webpage_revision",
            summary=str(summary or "").strip(),
            html_files=html_files,
        )
    )
    history.current_version_id = version_id
    save_revision_history(paper_folder_name, history)
    return history


def reset_revision_history_for_draft(
    *,
    paper_folder_name: str,
    artifact: CoderArtifact,
    page_plan: PagePlan,
    summary: str,
) -> RevisionHistory:
    shutil.rmtree(build_revision_history_dir(paper_folder_name), ignore_errors=True)
    return append_revision_version(
        paper_folder_name=paper_folder_name,
        artifact=artifact,
        page_plan=page_plan,
        source="initial_draft",
        summary=summary,
    )


def ensure_revision_history_bootstrapped(
    *,
    paper_folder_name: str,
    artifact: CoderArtifact,
    page_plan: PagePlan,
    summary: str,
) -> RevisionHistory:
    history = load_revision_history(paper_folder_name)
    if history is not None and history.versions:
        return history
    return reset_revision_history_for_draft(
        paper_folder_name=paper_folder_name,
        artifact=artifact,
        page_plan=page_plan,
        summary=summary,
    )


def restore_revision_version(
    *,
    paper_folder_name: str,
    version_id: str,
) -> tuple[RevisionHistory, CoderArtifact]:
    history = load_revision_history(paper_folder_name)
    if history is None or not history.versions:
        raise ValueError("No saved webpage revision history was found for the current paper.")

    entry = _entry_by_version_id(history, version_id)
    snapshot_dir = _snapshot_dir(paper_folder_name, version_id)
    snapshot_artifact_path = snapshot_dir / "coder_artifact.json"
    artifact = load_coder_artifact(snapshot_artifact_path)
    if artifact is None:
        raise ValueError(f"Saved webpage version '{version_id}' is missing coder_artifact.json.")

    site_dir = Path(artifact.site_dir).resolve()
    for relative_path in entry.html_files:
        source_path = snapshot_dir / "site" / relative_path
        if not source_path.exists():
            raise ValueError(f"Saved webpage version '{version_id}' is missing HTML snapshot '{relative_path}'.")
        _copy_file(source_path, site_dir / relative_path)

    output_dir, _, _, coder_json_path = get_output_paths(paper_folder_name)
    _copy_file(snapshot_dir / "page_manifest.json", output_dir / "page_manifest.json")
    _copy_file(snapshot_artifact_path, coder_json_path)

    history.current_version_id = version_id
    save_revision_history(paper_folder_name, history)
    restored_artifact = load_coder_artifact(coder_json_path)
    if restored_artifact is None:
        raise ValueError(f"Failed to reload restored webpage version '{version_id}'.")
    return history, restored_artifact
