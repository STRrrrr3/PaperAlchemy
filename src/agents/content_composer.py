import json
import re
from difflib import SequenceMatcher
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.contracts.schemas import (
    PageContentBlock,
    PageContentPlan,
    PageOutlineItem,
    PagePlan,
    PaperSection,
    StructuredPaper,
)
from src.prompts import CONTENT_COMPOSER_SYSTEM_PROMPT, CONTENT_COMPOSER_USER_PROMPT_TEMPLATE
from src.services.llm import get_llm
from src.utils.json_utils import to_pretty_json

_METRIC_RE = re.compile(
    r"(?:\b\d+(?:\.\d+)?\s?(?:%|ms|s|x|×|K|M|TPS|nodes?|steps?|f\s?=\s?\d+)\b|\b\d+(?:\.\d+)?[×x]\b)",
    re.IGNORECASE,
)

_HEIGHT_WEIGHT = {"S": 1.0, "M": 1.45, "L": 2.0}
_COMPRESSION_RATIO = {
    "teaser": 0.14,
    "compact": 0.24,
    "balanced": 0.40,
    "dense": 0.58,
    "near_full": 0.74,
}
_COMPRESSION_FLOOR = {
    "teaser": 120,
    "compact": 220,
    "balanced": 360,
    "dense": 560,
    "near_full": 780,
}


def _clean_text(value: object) -> str:
    return " ".join(str(value or "").split()).strip()


def _normalize_title(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _section_lookup(structured_paper: StructuredPaper) -> dict[str, PaperSection]:
    return {_normalize_title(section.section_title): section for section in structured_paper.sections}


def _best_section_match(title: str, structured_paper: StructuredPaper) -> PaperSection | None:
    normalized = _normalize_title(title)
    if not normalized:
        return None
    lookup = _section_lookup(structured_paper)
    if normalized in lookup:
        return lookup[normalized]

    best_section = None
    best_score = 0.0
    for section in structured_paper.sections:
        candidate = _normalize_title(section.section_title)
        if not candidate:
            continue
        if normalized in candidate or candidate in normalized:
            score = 0.92
        else:
            score = SequenceMatcher(None, normalized, candidate).ratio()
        if score > best_score:
            best_section = section
            best_score = score
    return best_section if best_score >= 0.58 else None


def _matched_sections(source_sections: list[str], structured_paper: StructuredPaper) -> list[PaperSection]:
    matched: list[PaperSection] = []
    seen: set[str] = set()
    for title in source_sections:
        section = _best_section_match(title, structured_paper)
        if section is None:
            continue
        key = str(section.section_title or "").strip()
        if key and key not in seen:
            seen.add(key)
            matched.append(section)
    return matched


def _outline_lookup(page_plan: PagePlan) -> dict[str, PageOutlineItem]:
    return {item.block_id: item for item in page_plan.page_outline}


def _block_order(page_plan: PagePlan) -> dict[str, int]:
    outline = _outline_lookup(page_plan)
    return {
        block.block_id: int(outline.get(block.block_id).order if outline.get(block.block_id) else block.responsive_rules.mobile_order)
        for block in page_plan.blocks
    }


def _extract_metrics(text: str) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for match in _METRIC_RE.finditer(text):
        value = match.group(0).strip()
        key = value.lower()
        if value and key not in seen:
            seen.add(key)
            values.append(value)
    return values[:12]


def _extract_tables(text: str) -> list[str]:
    tables: list[str] = []
    current: list[str] = []
    for line in str(text or "").splitlines():
        if "|" in line and line.strip().startswith("|"):
            current.append(line)
            continue
        if current:
            tables.append("\n".join(current))
            current = []
    if current:
        tables.append("\n".join(current))
    return tables[:3]


def _split_paragraphs(text: str) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", str(text or "")) if part.strip()]
    if paragraphs:
        return paragraphs
    clean = str(text or "").strip()
    return [clean] if clean else []


def _weighted_segments(text: str, weights: list[float]) -> list[str]:
    if not weights:
        return []
    paragraphs = _split_paragraphs(text)
    if len(paragraphs) >= len(weights):
        segments = ["" for _ in weights]
        totals = [0 for _ in weights]
        target_total = max(1, sum(len(paragraph) for paragraph in paragraphs))
        target_weights = [weight / sum(weights) * target_total for weight in weights]
        cursor = 0
        for paragraph in paragraphs:
            segments[cursor] = (segments[cursor] + "\n\n" + paragraph).strip()
            totals[cursor] += len(paragraph)
            if cursor < len(weights) - 1 and totals[cursor] >= target_weights[cursor]:
                cursor += 1
        return segments

    chunk_size = max(1, len(text) // len(weights))
    return [text[index * chunk_size : (index + 1) * chunk_size].strip() for index in range(len(weights))]


def _height_for_block(page_plan: PagePlan, block_id: str) -> str:
    outline_item = _outline_lookup(page_plan).get(block_id)
    return str(outline_item.estimated_height if outline_item is not None else "M")


def _source_titles_for_block(page_plan: PagePlan, block_id: str) -> list[str]:
    outline_item = _outline_lookup(page_plan).get(block_id)
    return list(outline_item.source_sections if outline_item is not None else [])


def _infer_compression_level(page_plan: PagePlan, block_id: str) -> str:
    block = next(block for block in page_plan.blocks if block.block_id == block_id)
    outline_item = _outline_lookup(page_plan).get(block_id)
    blob = " ".join(
        [
            str(block.block_id or ""),
            str(block.target_template_region.region_role or ""),
            str(block.target_template_region.operation or ""),
            str(block.responsive_rules.desktop_layout or ""),
            str(block.content_contract.headline or ""),
            str(outline_item.title if outline_item else ""),
            str(outline_item.objective if outline_item else ""),
        ]
    ).lower()
    has_assets = bool(block.asset_binding.asset_ids or block.asset_binding.template_asset_fallback)
    height = _height_for_block(page_plan, block_id)
    layout = str(block.responsive_rules.desktop_layout or "").lower()
    region = str(block.target_template_region.region_role or "").lower()

    if "hero" in blob or region == "hero" or int(_block_order(page_plan).get(block_id, 99)) == 1:
        return "compact"
    if "abstract" in blob or "overview" in blob:
        return "balanced" if height != "S" else "compact"
    if any(token in blob for token in ("evaluation", "result", "performance", "throughput", "latency", "overhead", "table")):
        return "near_full" if height == "L" or layout in {"grid", "table_focus", "gallery_grid"} else "dense"
    if region in {"table", "gallery"} or layout in {"grid", "table_focus", "gallery_grid"}:
        return "near_full" if height == "L" else "dense"
    if has_assets:
        return "dense"
    if any(token in blob for token in ("method", "design", "component", "operation", "recovery", "mechanism")):
        return "dense"
    return "balanced"


def _min_visible_chars(level: str, source_chars: int, height: str) -> int:
    ratio = _COMPRESSION_RATIO.get(level, 0.4)
    floor = int(_COMPRESSION_FLOOR.get(level, 360) * _HEIGHT_WEIGHT.get(height, 1.45))
    return max(80, min(max(floor, int(source_chars * ratio)), max(120, source_chars)))


def _clip_to_chars(text: str, target_chars: int) -> str:
    clean = str(text or "").strip()
    if len(clean) <= target_chars:
        return clean
    cutoff = max(target_chars, clean.rfind("\n\n", 0, target_chars + 240))
    if cutoff <= 0:
        cutoff = target_chars
    return clean[:cutoff].rstrip()


def _source_group_key(sections: list[PaperSection]) -> str:
    return "||".join(str(section.section_title or "").strip() for section in sections)


def _source_segments_by_block(page_plan: PagePlan, structured_paper: StructuredPaper) -> dict[str, str]:
    grouped_blocks: dict[str, list[str]] = {}
    source_text_by_key: dict[str, str] = {}
    for block in page_plan.blocks:
        sections = _matched_sections(_source_titles_for_block(page_plan, block.block_id), structured_paper)
        if not sections and structured_paper.sections:
            sections = [structured_paper.sections[0]]
        key = _source_group_key(sections)
        grouped_blocks.setdefault(key, []).append(block.block_id)
        source_text_by_key[key] = "\n\n".join(str(section.rich_web_content or "").strip() for section in sections if str(section.rich_web_content or "").strip())

    order = _block_order(page_plan)
    result: dict[str, str] = {}
    for key, block_ids in grouped_blocks.items():
        ordered_ids = sorted(block_ids, key=lambda block_id: order.get(block_id, 999))
        if len(ordered_ids) == 1:
            result[ordered_ids[0]] = source_text_by_key.get(key, "")
            continue
        weights = [_HEIGHT_WEIGHT.get(_height_for_block(page_plan, block_id), 1.45) for block_id in ordered_ids]
        for block_id, segment in zip(ordered_ids, _weighted_segments(source_text_by_key.get(key, ""), weights)):
            result[block_id] = segment
    return result


def _asset_ids_for_block(page_plan: PagePlan, block_id: str) -> list[str]:
    for block in page_plan.blocks:
        if block.block_id == block_id:
            return [str(asset_id).strip() for asset_id in block.asset_binding.asset_ids if str(asset_id).strip()]
    return []


def _body_points_for_block(page_plan: PagePlan, block_id: str) -> list[str]:
    for block in page_plan.blocks:
        if block.block_id == block_id:
            return [str(item).strip() for item in block.content_contract.body_points if str(item).strip()]
    return []


def build_fallback_page_content_plan(
    *,
    structured_paper: StructuredPaper,
    page_plan: PagePlan,
) -> PageContentPlan:
    source_segments = _source_segments_by_block(page_plan, structured_paper)
    outline = _outline_lookup(page_plan)
    blocks: list[PageContentBlock] = []
    total_source_chars = len(str(structured_paper.overall_summary or "")) + sum(
        len(str(section.rich_web_content or "")) for section in structured_paper.sections
    )
    for block in sorted(page_plan.blocks, key=lambda item: _block_order(page_plan).get(item.block_id, 999)):
        outline_item = outline.get(block.block_id)
        matched = _matched_sections(list(outline_item.source_sections if outline_item else []), structured_paper)
        actual_source_titles = [section.section_title for section in matched]
        source_text = source_segments.get(block.block_id, "")
        body_points = _body_points_for_block(page_plan, block.block_id)
        level = _infer_compression_level(page_plan, block.block_id)
        min_chars = _min_visible_chars(level, len(source_text), _height_for_block(page_plan, block.block_id))
        headline = str(block.content_contract.headline or (outline_item.title if outline_item else block.block_id)).strip()
        lead = body_points[0] if body_points else _clean_text(source_text)[:240]
        narrative_parts = []
        if body_points:
            narrative_parts.append("\n".join(f"- {point}" for point in body_points))
        if source_text:
            narrative_parts.append(_clip_to_chars(source_text, max(min_chars + 260, int(min_chars * 1.45))))
        required_narrative = "\n\n".join(part for part in narrative_parts if part).strip() or lead or headline
        blocks.append(
            PageContentBlock(
                block_id=block.block_id,
                source_sections=actual_source_titles,
                headline=headline,
                lead=lead,
                compression_level=level,
                compression_rationale=(
                    f"Derived from outline height {_height_for_block(page_plan, block.block_id)}, "
                    f"layout {block.responsive_rules.desktop_layout}, role {block.target_template_region.region_role}, "
                    f"and {len(_asset_ids_for_block(page_plan, block.block_id))} bound asset(s)."
                ),
                required_narrative_markdown=required_narrative,
                must_include_claims=body_points[:6],
                must_include_metrics=_extract_metrics(required_narrative),
                must_include_tables=_extract_tables(required_narrative),
                asset_ids=_asset_ids_for_block(page_plan, block.block_id),
                min_visible_chars=min_chars,
                rendering_notes="Render this block's markdown directly; preserve metrics and tables.",
            )
        )

    target_visible_chars = sum(block.min_visible_chars for block in blocks)
    coverage_ratio = round(target_visible_chars / total_source_chars, 3) if total_source_chars else 0.0
    return PageContentPlan(
        paper_title=structured_paper.paper_title,
        source_char_count=total_source_chars,
        target_visible_chars=target_visible_chars,
        coverage_ratio=coverage_ratio,
        blocks=blocks,
    )


def validate_page_content_plan(
    content_plan: PageContentPlan,
    *,
    structured_paper: StructuredPaper,
    page_plan: PagePlan,
) -> PageContentPlan:
    expected_ids = [str(block.block_id or "").strip() for block in page_plan.blocks if str(block.block_id or "").strip()]
    actual_ids = [str(block.block_id or "").strip() for block in content_plan.blocks]
    missing = sorted(set(expected_ids) - set(actual_ids))
    extra = sorted(set(actual_ids) - set(expected_ids))
    if missing or extra:
        details = []
        if missing:
            details.append("missing block(s): " + ", ".join(missing))
        if extra:
            details.append("unknown block(s): " + ", ".join(extra))
        raise ValueError("PageContentPlan block coverage mismatch: " + "; ".join(details))
    if len(actual_ids) != len(expected_ids):
        raise ValueError("PageContentPlan must contain exactly one content block per PagePlan block.")

    valid_sections = {str(section.section_title or "").strip() for section in structured_paper.sections}
    valid_assets = {
        str(asset.asset_id or "").strip()
        for asset in structured_paper.asset_registry
        if str(asset.asset_id or "").strip()
    }
    for block in page_plan.blocks:
        valid_assets.update(str(asset_id or "").strip() for asset_id in block.asset_binding.asset_ids if str(asset_id or "").strip())

    for block in content_plan.blocks:
        if not str(block.required_narrative_markdown or "").strip():
            raise ValueError(f"PageContentPlan block '{block.block_id}' has empty required_narrative_markdown.")
        unknown_sections = [section for section in block.source_sections if section not in valid_sections]
        if unknown_sections:
            raise ValueError(
                f"PageContentPlan block '{block.block_id}' references unknown source section(s): "
                + ", ".join(unknown_sections)
            )
        unknown_assets = [asset_id for asset_id in block.asset_ids if asset_id not in valid_assets]
        if unknown_assets:
            raise ValueError(
                f"PageContentPlan block '{block.block_id}' references unknown asset_id(s): "
                + ", ".join(unknown_assets)
            )
    return content_plan


def run_content_composer_agent(
    *,
    structured_paper: StructuredPaper,
    page_plan: PagePlan,
    human_directives: str,
    coder_instructions: str,
) -> PageContentPlan:
    try:
        llm = get_llm(temperature=0.2, use_smart_model=True, thinking_level="high")
        structured_llm = llm.with_structured_output(PageContentPlan)
        response = structured_llm.invoke(
            [
                SystemMessage(content=CONTENT_COMPOSER_SYSTEM_PROMPT),
                HumanMessage(
                    content=CONTENT_COMPOSER_USER_PROMPT_TEMPLATE.format(
                        structured_paper_json=to_pretty_json(structured_paper),
                        page_plan_json=json.dumps(page_plan.model_dump(), indent=2, ensure_ascii=False),
                        human_directives=human_directives or "(none)",
                        coder_instructions=coder_instructions or "(none)",
                    )
                ),
            ]
        )
        return validate_page_content_plan(
            PageContentPlan.model_validate(response),
            structured_paper=structured_paper,
            page_plan=page_plan,
        )
    except Exception as exc:
        print(f"[PaperAlchemy-ContentComposer] falling back to deterministic composition: {exc}")
        return validate_page_content_plan(
            build_fallback_page_content_plan(structured_paper=structured_paper, page_plan=page_plan),
            structured_paper=structured_paper,
            page_plan=page_plan,
        )
