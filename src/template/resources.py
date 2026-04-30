from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TEMPLATE_LIBRARY_ROOT_REL = Path("data") / "templates" / "template_library"
TEMPLATE_LIBRARY_TEMPLATES_REL = TEMPLATE_LIBRARY_ROOT_REL / "templates"
TEMPLATE_LIBRARY_TAGS_REL = TEMPLATE_LIBRARY_ROOT_REL / "tags.json"
TEMPLATE_LIBRARY_TEMPLATE_LINK_REL = TEMPLATE_LIBRARY_ROOT_REL / "template_link.json"


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


def _build_assets_from_resource_root(
    project_root: Path,
    resource_root_rel: Path,
) -> SyncedTemplateAssets | None:
    resource_root = project_root / resource_root_rel
    destination_tags = resource_root / "tags.json"
    destination_templates_dir = resource_root / "templates"
    destination_template_link = resource_root / "template_link.json"

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


def ensure_template_library_assets(
    project_root: Path,
    force: bool = False,
) -> SyncedTemplateAssets:
    del force

    assets = _build_assets_from_resource_root(project_root, TEMPLATE_LIBRARY_ROOT_REL)
    if assets is not None:
        return assets

    raise FileNotFoundError(
        f"Template library templates directory not found: {project_root / TEMPLATE_LIBRARY_TEMPLATES_REL}"
    )

