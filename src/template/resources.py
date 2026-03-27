from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


AUTO_PAGE_ROOT_REL = Path("AutoPage")
AUTO_PAGE_TAGS_REL = AUTO_PAGE_ROOT_REL / "tags.json"
AUTO_PAGE_TEMPLATES_REL = AUTO_PAGE_ROOT_REL / "templates"
AUTO_PAGE_TEMPLATE_LINK_REL = AUTO_PAGE_TEMPLATES_REL / "template_link.json"

RESOURCE_ROOT_REL = Path("data") / "templates" / "autopage"
RESOURCE_TEMPLATES_REL = RESOURCE_ROOT_REL / "templates"
RESOURCE_TAGS_REL = RESOURCE_ROOT_REL / "tags.json"
RESOURCE_TEMPLATE_LINK_REL = RESOURCE_ROOT_REL / "template_link.json"


@dataclass(frozen=True)
class SyncedTemplateAssets:
    resource_root: Path
    tags_json_path: Path
    templates_dir: Path
    template_link_json_path: Path
    synced_template_ids: list[str]
    missing_template_ids: list[str]


def load_template_tags(tags_json_path: str | Path) -> dict[str, dict[str, str]]:
    path = Path(tags_json_path)
    if not path.exists():
        raise FileNotFoundError(f"Template tags file not found: {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Template tags file is not valid JSON: {path}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Template tags payload must be a JSON object: {path}")

    normalized: dict[str, dict[str, str]] = {}
    for template_id, raw_tags in payload.items():
        clean_id = str(template_id or "").strip()
        if not clean_id or not isinstance(raw_tags, dict):
            continue

        normalized[clean_id] = {
            str(feature): str(value).strip().lower()
            for feature, value in raw_tags.items()
            if str(feature).strip() and str(value).strip()
        }

    if not normalized:
        raise ValueError(f"No usable template tags found in: {path}")

    return normalized


def _copy_file_if_needed(source: Path, target: Path, force: bool = False) -> None:
    if not source.exists():
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    if force or not target.exists() or source.stat().st_mtime > target.stat().st_mtime:
        shutil.copy2(source, target)


def _build_assets_from_existing_resources(project_root: Path) -> SyncedTemplateAssets | None:
    resource_root = project_root / RESOURCE_ROOT_REL
    destination_tags = project_root / RESOURCE_TAGS_REL
    destination_templates_dir = project_root / RESOURCE_TEMPLATES_REL
    destination_template_link = project_root / RESOURCE_TEMPLATE_LINK_REL

    if not destination_templates_dir.exists():
        return None

    synced_template_ids = sorted(
        path.name for path in destination_templates_dir.iterdir() if path.is_dir()
    )
    if not synced_template_ids:
        return None

    missing_template_ids: list[str] = []
    if destination_tags.exists():
        try:
            tagged_ids = sorted(load_template_tags(destination_tags))
            missing_template_ids = [
                template_id
                for template_id in tagged_ids
                if not (destination_templates_dir / template_id).exists()
            ]
        except Exception:
            missing_template_ids = []
    else:
        destination_tags.parent.mkdir(parents=True, exist_ok=True)
        destination_tags.write_text("{}", encoding="utf-8")

    if not destination_template_link.exists():
        destination_template_link.write_text("{}", encoding="utf-8")

    return SyncedTemplateAssets(
        resource_root=resource_root,
        tags_json_path=destination_tags,
        templates_dir=destination_templates_dir,
        template_link_json_path=destination_template_link,
        synced_template_ids=synced_template_ids,
        missing_template_ids=missing_template_ids,
    )


def ensure_autopage_template_assets(
    project_root: Path,
    force: bool = False,
) -> SyncedTemplateAssets:
    source_tags = project_root / AUTO_PAGE_TAGS_REL
    source_templates_dir = project_root / AUTO_PAGE_TEMPLATES_REL
    source_template_link = project_root / AUTO_PAGE_TEMPLATE_LINK_REL

    existing_assets = _build_assets_from_existing_resources(project_root)
    if not source_templates_dir.exists():
        if existing_assets is not None:
            return existing_assets
        raise FileNotFoundError(f"AutoPage templates directory not found: {source_templates_dir}")

    resource_root = project_root / RESOURCE_ROOT_REL
    destination_tags = project_root / RESOURCE_TAGS_REL
    destination_templates_dir = project_root / RESOURCE_TEMPLATES_REL
    destination_template_link = project_root / RESOURCE_TEMPLATE_LINK_REL

    destination_templates_dir.mkdir(parents=True, exist_ok=True)

    _copy_file_if_needed(source_tags, destination_tags, force=force)
    _copy_file_if_needed(source_template_link, destination_template_link, force=force)

    template_ids_to_sync: list[str]
    if source_tags.exists():
        try:
            template_ids_to_sync = sorted(load_template_tags(source_tags))
        except Exception:
            template_ids_to_sync = sorted(
                path.name for path in source_templates_dir.iterdir() if path.is_dir()
            )
    else:
        template_ids_to_sync = sorted(
            path.name for path in source_templates_dir.iterdir() if path.is_dir()
        )

    synced_template_ids: list[str] = []
    missing_template_ids: list[str] = []

    for template_id in template_ids_to_sync:
        source_dir = source_templates_dir / template_id
        destination_dir = destination_templates_dir / template_id

        if not source_dir.exists():
            missing_template_ids.append(template_id)
            continue

        if force and destination_dir.exists():
            shutil.rmtree(destination_dir)

        if force or not destination_dir.exists():
            shutil.copytree(source_dir, destination_dir, dirs_exist_ok=False)

        synced_template_ids.append(template_id)

    if not synced_template_ids:
        raise FileNotFoundError(
            "No AutoPage templates were synchronized into PaperAlchemy resources."
        )

    if not destination_tags.exists():
        destination_tags.write_text("{}", encoding="utf-8")

    if not destination_template_link.exists():
        destination_template_link.write_text("{}", encoding="utf-8")

    return SyncedTemplateAssets(
        resource_root=resource_root,
        tags_json_path=destination_tags,
        templates_dir=destination_templates_dir,
        template_link_json_path=destination_template_link,
        synced_template_ids=synced_template_ids,
        missing_template_ids=missing_template_ids,
    )

