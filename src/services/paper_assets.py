from __future__ import annotations

from pathlib import Path

from src.contracts.schemas import (
    ASSET_CONFIRMATION_NONE_ID,
    PagePlan,
    PaperAsset,
    PaperSection,
    StructuredPaper,
)
from src.validators.page_validation import collect_local_image_sources


def asset_lookup(structured_paper: StructuredPaper) -> dict[str, PaperAsset]:
    return {
        str(asset.asset_id or "").strip(): asset
        for asset in structured_paper.asset_registry
        if str(asset.asset_id or "").strip()
    }


def section_assets(structured_paper: StructuredPaper, section: PaperSection) -> list[PaperAsset]:
    lookup = asset_lookup(structured_paper)
    results: list[PaperAsset] = []
    seen: set[str] = set()
    for binding in section.asset_bindings:
        asset_id = str(binding.asset_id or "").strip()
        if not asset_id or asset_id in seen:
            continue
        asset = lookup.get(asset_id)
        if asset is None:
            continue
        results.append(asset)
        seen.add(asset_id)
    return results


def section_asset_ids(structured_paper: StructuredPaper, section_title: str) -> list[str]:
    for section in structured_paper.sections:
        if str(section.section_title or "").strip() != str(section_title or "").strip():
            continue
        return [
            str(binding.asset_id or "").strip()
            for binding in section.asset_bindings
            if str(binding.asset_id or "").strip()
        ]
    return []


def collect_page_plan_asset_ids(page_plan: PagePlan, structured_paper: StructuredPaper) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for block in page_plan.blocks:
        for asset_id in block.asset_binding.asset_ids:
            clean = str(asset_id or "").strip()
            if clean and clean not in seen:
                seen.add(clean)
                ordered.append(clean)
    if ordered:
        return ordered

    for section in structured_paper.sections:
        for binding in section.asset_bindings:
            clean = str(binding.asset_id or "").strip()
            if clean and clean not in seen:
                seen.add(clean)
                ordered.append(clean)
    return ordered


def resolved_asset_source_path(project_root: Path, paper_folder_name: str, asset: PaperAsset) -> Path:
    return (project_root / "data" / "output" / paper_folder_name / str(asset.image_path or "").strip()).resolve()


def asset_target_filename(asset: PaperAsset) -> str:
    source_suffix = Path(str(asset.image_path or "").strip()).suffix or ".png"
    return f"{str(asset.asset_id or '').strip()}{source_suffix}"


def build_asset_manifest(
    *,
    project_root: Path,
    paper_folder_name: str,
    structured_paper: StructuredPaper,
    site_dir: Path,
    entry_html_path: Path,
) -> list[dict[str, str]]:
    entry_html_parent = entry_html_path.parent
    section_by_asset_id: dict[str, str] = {}
    for section in structured_paper.sections:
        for binding in section.asset_bindings:
            asset_id = str(binding.asset_id or "").strip()
            if asset_id and asset_id not in section_by_asset_id:
                section_by_asset_id[asset_id] = str(section.section_title or "").strip()

    manifest: list[dict[str, str]] = []
    for asset in structured_paper.asset_registry:
        target_path = site_dir / "assets" / "paper" / asset_target_filename(asset)
        rel_path = str(target_path.relative_to(site_dir)).replace("\\", "/")
        try:
            web_path = target_path.relative_to(entry_html_parent).as_posix()
        except ValueError:
            web_path = rel_path
        if not web_path.startswith((".", "/")):
            web_path = f"./{web_path}"
        source_path = resolved_asset_source_path(project_root, paper_folder_name, asset)
        manifest.append(
            {
                "asset_id": str(asset.asset_id or "").strip(),
                "source_path": str(asset.image_path or "").strip(),
                "absolute_source_path": str(source_path),
                "relative_path": rel_path,
                "web_path": web_path,
                "filename": target_path.name,
                "caption": str(asset.caption or "").strip(),
                "type": str(asset.type or "").strip(),
                "section_title": section_by_asset_id.get(str(asset.asset_id or "").strip(), ""),
                "page_number": str(asset.page_number or ""),
            }
        )
    return manifest


def ensure_manifest_assets_present(
    *,
    html_text: str,
    asset_manifest: list[dict[str, str]],
    site_dir: Path,
) -> None:
    manifest_by_web_path = {
        str(item.get("web_path") or "").strip(): item
        for item in asset_manifest
        if str(item.get("web_path") or "").strip()
    }
    for src in collect_local_image_sources(html_text):
        item = manifest_by_web_path.get(src)
        if item is None:
            continue
        relative_path = str(item.get("relative_path") or "").strip()
        absolute_source_path = str(item.get("absolute_source_path") or "").strip()
        if not relative_path or not absolute_source_path:
            continue
        target_path = (site_dir / relative_path).resolve()
        if target_path.exists():
            continue
        source_path = Path(absolute_source_path)
        if not source_path.exists() or not source_path.is_file():
            continue
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(source_path.read_bytes())


def confirmation_pending(structured_paper: StructuredPaper | None) -> bool:
    if structured_paper is None or structured_paper.asset_confirmation_session is None:
        return False
    return not bool(structured_paper.asset_confirmation_session.is_complete)


def is_none_asset(asset_id: str | None) -> bool:
    return str(asset_id or "").strip() == ASSET_CONFIRMATION_NONE_ID
