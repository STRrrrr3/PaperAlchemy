from __future__ import annotations

from typing import Any

from src.contracts.schemas import SemanticPagePlan, TemplateCandidate


def _text_blob(template_item: dict[str, Any]) -> str:
    fields = [
        str(template_item.get("template_id") or ""),
        str(template_item.get("root_dir") or ""),
        " ".join(str(x) for x in (template_item.get("entry_html_candidates") or [])),
        " ".join(str(x) for x in (template_item.get("style_files") or [])),
        " ".join(str(x) for x in (template_item.get("script_files") or [])),
    ]
    return " ".join(fields).lower()


def _has_capability(template_item: dict[str, Any], capability: str) -> bool:
    scripts = template_item.get("script_files") or []
    styles = template_item.get("style_files") or []
    has_image_info = bool(template_item.get("has_image_info"))
    has_video_info = bool(template_item.get("has_video_info"))

    if capability == "needs_interactivity":
        return len(scripts) > 0
    if capability == "needs_media_gallery":
        return has_image_info or has_video_info
    if capability == "needs_rich_styling":
        return len(styles) >= 1
    if capability == "needs_video_support":
        return has_video_info or any("video" in str(x).lower() for x in scripts)
    if capability == "needs_table_friendly_layout":
        return len(styles) >= 1
    return True


def _score_template(template_item: dict[str, Any], semantic_plan: SemanticPagePlan) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []

    entry_candidates = template_item.get("entry_html_candidates") or []
    if entry_candidates:
        score += 0.3
        reasons.append("has_entry_html")
    else:
        score -= 1.0
        reasons.append("missing_entry_html")

    blob = _text_blob(template_item)
    for kw in semantic_plan.global_design.style_keywords:
        key = kw.strip().lower()
        if not key:
            continue
        if key in blob:
            score += 0.12
            reasons.append(f"style_kw_match:{key}")

    needs_interactivity = any(block.interaction.pattern != "none" for block in semantic_plan.semantic_blocks)
    if needs_interactivity:
        if _has_capability(template_item, "needs_interactivity"):
            score += 0.18
            reasons.append("cap_ok:needs_interactivity")
        else:
            score -= 0.2
            reasons.append("cap_miss:needs_interactivity")

    high_media_blocks = [
        blk for blk in semantic_plan.semantic_blocks if blk.asset_binding.figure_paths
    ]
    if high_media_blocks and bool(template_item.get("has_image_info")):
        score += 0.15
        reasons.append("high_media_compatible")

    if any(block.preferred_region_role == "table" for block in semantic_plan.semantic_blocks):
        if _has_capability(template_item, "needs_table_friendly_layout"):
            score += 0.12
            reasons.append("cap_ok:needs_table_friendly_layout")
        else:
            score -= 0.12
            reasons.append("cap_miss:needs_table_friendly_layout")

    block_count = len(semantic_plan.page_outline)
    if block_count >= 8 and len(template_item.get("style_files") or []) > 0:
        score += 0.08
        reasons.append("long_page_style_support")

    return round(score, 4), reasons


def select_template_candidates(
    template_catalog: list[dict[str, Any]],
    semantic_plan: SemanticPagePlan,
    top_k: int,
) -> list[TemplateCandidate]:
    if not template_catalog:
        return []

    k = max(1, min(int(top_k), 8))
    scored: list[TemplateCandidate] = []

    for template_item in template_catalog:
        template_id = str(template_item.get("template_id") or "").strip()
        root_dir = str(template_item.get("root_dir") or "").strip()
        entry_candidates = template_item.get("entry_html_candidates") or []
        chosen_entry = str(entry_candidates[0]) if entry_candidates else ""
        if not template_id or not root_dir or not chosen_entry:
            continue

        score, reasons = _score_template(template_item, semantic_plan)
        scored.append(
            TemplateCandidate(
                template_id=template_id,
                root_dir=root_dir,
                chosen_entry_html=chosen_entry,
                score=score,
                reasons=reasons[:8],
            )
        )

    scored.sort(key=lambda x: (x.score, x.template_id), reverse=True)
    return scored[:k]

