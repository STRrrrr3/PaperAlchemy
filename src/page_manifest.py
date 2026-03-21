from __future__ import annotations

import json
from pathlib import Path

from bs4 import BeautifulSoup, Tag

from src.schemas import PageManifest, PageManifestBlock, PageManifestSlot, PagePlan

BLOCK_ATTR = "data-pa-block"
SLOT_ATTR = "data-pa-slot"
ALLOWED_SLOT_IDS = {"title", "summary", "body", "media", "meta", "actions"}
PAGE_MANIFEST_SCHEMA_VERSION = "1.0"


def build_block_selector(block_id: str) -> str:
    return f'[{BLOCK_ATTR}="{block_id}"]'


def build_slot_selector(block_id: str, slot_id: str) -> str:
    return f'{build_block_selector(block_id)} [{SLOT_ATTR}="{slot_id}"]'


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


def extract_page_manifest(
    html_text: str,
    entry_html: str | Path,
    selected_template_id: str,
    page_plan: PagePlan,
) -> PageManifest:
    soup = BeautifulSoup(str(html_text or ""), "html.parser")
    expected_blocks = {block.block_id: block for block in page_plan.blocks}
    source_sections_lookup = {
        outline_item.block_id: list(outline_item.source_sections)
        for outline_item in page_plan.page_outline
    }

    errors: list[str] = []
    seen_block_ids: set[str] = set()
    manifest_blocks: list[PageManifestBlock] = []

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

        manifest_blocks.append(
            PageManifestBlock(
                block_id=block_id,
                source_sections=source_sections_lookup.get(block_id, []),
                selector=build_block_selector(block_id),
                slots=slot_records,
            )
        )

    missing_block_ids = [block_id for block_id in expected_blocks if block_id not in seen_block_ids]
    if missing_block_ids:
        errors.append(f"Missing required data-pa-block ids: {missing_block_ids}")

    if errors:
        raise ValueError(" ; ".join(errors))

    return PageManifest(
        schema_version=PAGE_MANIFEST_SCHEMA_VERSION,
        entry_html=str(Path(entry_html).resolve()),
        selected_template_id=str(selected_template_id or "").strip(),
        blocks=manifest_blocks,
    )
