from __future__ import annotations

from collections import defaultdict
import re
from typing import Any

from src.contracts.schemas import ArbiterReport, PagePlan

def append_log_lines(existing_text: str, new_lines: list[str]) -> str:
    merged = [line for line in str(existing_text or "").splitlines() if line.strip()]
    merged.extend(line for line in new_lines if str(line).strip())
    return "\n".join(merged)

def _coerce_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        cleaned = value.strip()
        return [cleaned] if cleaned else []

    if isinstance(value, list):
        results: list[str] = []
        for item in value:
            if isinstance(item, str):
                cleaned = item.strip()
                if cleaned:
                    results.append(cleaned)
        return results

    return []

def _extract_front_matter_candidates(structured_paper_dict: dict[str, Any]) -> tuple[list[str], list[str]]:
    authors = _coerce_string_list(
        structured_paper_dict.get("authors")
        or structured_paper_dict.get("author_names")
        or structured_paper_dict.get("paper_authors")
    )
    affiliations = _coerce_string_list(
        structured_paper_dict.get("affiliations")
        or structured_paper_dict.get("institutions")
        or structured_paper_dict.get("paper_affiliations")
    )

    if authors and affiliations:
        return authors, affiliations

    candidate_texts: list[str] = []
    overall_summary = str(structured_paper_dict.get("overall_summary") or "").strip()
    if overall_summary:
        candidate_texts.append(overall_summary)

    for section in list(structured_paper_dict.get("sections") or [])[:2]:
        rich_web_content = str(section.get("rich_web_content") or "").strip()
        if rich_web_content:
            candidate_texts.append(rich_web_content)

    author_keywords = ("author", "authors", "corresponding author", "affiliated")
    affiliation_keywords = (
        "university",
        "institute",
        "college",
        "school",
        "department",
        "lab",
        "laboratory",
        "company",
        "research",
        "affiliation",
        "affiliations",
    )

    for raw_text in candidate_texts:
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", raw_text):
            normalized = " ".join(sentence.split()).strip(" -")
            if not normalized:
                continue

            lowered = normalized.lower()
            if any(keyword in lowered for keyword in author_keywords) and normalized not in authors:
                authors.append(normalized)
            if any(keyword in lowered for keyword in affiliation_keywords) and normalized not in affiliations:
                affiliations.append(normalized)

    return authors[:3], affiliations[:4]

def _trim_review_text(text: Any, max_chars: int = 700) -> str:
    normalized = " ".join(str(text or "").split()).strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."

def format_paper_to_markdown(structured_paper_dict: dict[str, Any]) -> str:
    paper_title = str(structured_paper_dict.get("paper_title") or "Untitled Paper").strip()
    overall_summary = str(structured_paper_dict.get("overall_summary") or "").strip()
    sections = list(structured_paper_dict.get("sections") or [])
    asset_registry = {
        str(asset.get("asset_id") or "").strip(): asset
        for asset in list(structured_paper_dict.get("asset_registry") or [])
        if isinstance(asset, dict) and str(asset.get("asset_id") or "").strip()
    }
    authors, affiliations = _extract_front_matter_candidates(structured_paper_dict)

    lines = [
        f"# {paper_title}",
        "",
        "This is the Reader extraction source pack that the Planner will use as input.",
        "It is not the final webpage outline.",
        "Use Revise Extraction to add missing source material, remove incorrect content, or fix metadata and asset grounding before webpage planning.",
        "",
        "## Authors",
    ]

    if authors:
        lines.extend(f"- {author}" for author in authors)
    else:
        lines.append("- Not explicitly extracted in the current StructuredPaper output.")

    lines.extend(["", "## Affiliations"])
    if affiliations:
        lines.extend(f"- {affiliation}" for affiliation in affiliations)
    else:
        lines.append("- Not explicitly extracted in the current StructuredPaper output.")

    lines.extend(["", "## Overall Summary"])
    lines.append(overall_summary or "No overall summary was extracted.")

    lines.extend(["", "## Source Sections"])
    if not sections:
        lines.append("- No sections were extracted.")
        return "\n".join(lines)

    for index, section in enumerate(sections, start=1):
        section_title = str(section.get("section_title") or f"Section {index}").strip()
        rich_web_content = str(section.get("rich_web_content") or "").strip()
        asset_bindings = list(section.get("asset_bindings") or [])

        lines.extend(["", f"### {index}. {section_title}"])
        lines.append(_trim_review_text(rich_web_content) or "No rich web content was extracted.")

        if asset_bindings:
            lines.extend(["", "Linked assets:"])
            for binding in asset_bindings[:3]:
                asset_id = str(binding.get("asset_id") or "").strip()
                asset = asset_registry.get(asset_id, {})
                image_path = str(asset.get("image_path") or "").strip()
                caption = str(asset.get("caption") or "").strip()
                figure_label = f"`{asset_id}`" if asset_id else "`(missing asset id)`"
                if caption:
                    lines.append(f"- {figure_label}: {caption} ({image_path})")
                else:
                    lines.append(f"- {figure_label}: {image_path or '(missing path)'}")

    confirmation_session = structured_paper_dict.get("asset_confirmation_session") or {}
    pending_items = list(confirmation_session.get("items") or []) if isinstance(confirmation_session, dict) else []
    if pending_items:
        lines.extend(["", "## Pending Asset Confirmations"])
        for item in pending_items:
            section_title = str(item.get("section_title") or "").strip() or "(unknown section)"
            proposed_asset_id = str(item.get("proposed_asset_id") or "").strip() or "(none)"
            selection_reason = str(item.get("selection_reason") or "").strip() or "Low-confidence asset binding."
            lines.append(f"- {section_title}: `{proposed_asset_id}` — {selection_reason}")

    return "\n".join(lines)

def format_page_plan_to_markdown(
    page_plan_dict: dict[str, Any],
    structured_paper_dict: dict[str, Any] | None = None,
) -> str:
    page_plan = PagePlan.model_validate(page_plan_dict)
    outline_items = sorted(page_plan.page_outline, key=lambda item: item.order)
    structured_sections = list((structured_paper_dict or {}).get("sections") or [])
    source_section_titles = [
        str(section.get("section_title") or "").strip()
        for section in structured_sections
        if str(section.get("section_title") or "").strip()
    ]

    usage_by_section: dict[str, list[str]] = defaultdict(list)
    merged_blocks: list[tuple[str, list[str]]] = []
    for item in outline_items:
        for source_section in item.source_sections:
            usage_by_section[source_section].append(item.block_id)
        if len(item.source_sections) > 1:
            merged_blocks.append((item.block_id, list(item.source_sections)))

    unused_sections = [
        section_title
        for section_title in source_section_titles
        if section_title not in usage_by_section
    ]
    split_sections = [
        (section_title, block_ids)
        for section_title, block_ids in usage_by_section.items()
        if len(block_ids) > 1
    ]

    lines = [
        "# Planned Webpage Outline",
        "",
        "This is the planner-stage webpage outline that the first draft will follow.",
        "Use Revise Outline to add, remove, merge, split, rename, or reorder webpage sections before generating the draft.",
        "",
        "## Ordered Webpage Sections",
    ]

    if not outline_items:
        lines.append("- No webpage sections are currently planned.")
        return "\n".join(lines)

    for item in outline_items:
        lines.extend(
            [
                "",
                f"### {item.order}. {item.title}",
                f"- `block_id`: `{item.block_id}`",
                f"- Objective: {item.objective}",
                "- Source sections: "
                + (", ".join(item.source_sections) if item.source_sections else "(none)"),
                f"- Estimated height: {item.estimated_height}",
            ]
        )

    lines.extend(["", "## Mapping Diagnostics"])
    lines.append(
        "- Unused source sections: "
        + (", ".join(unused_sections) if unused_sections else "None")
    )
    if split_sections:
        lines.append(
            "- Split source sections: "
            + "; ".join(
                f"`{section_title}` -> {', '.join(block_ids)}"
                for section_title, block_ids in split_sections
            )
        )
    else:
        lines.append("- Split source sections: None")

    if merged_blocks:
        lines.append(
            "- Merged webpage blocks: "
            + "; ".join(
                f"`{block_id}` <- {', '.join(source_sections)}"
                for block_id, source_sections in merged_blocks
            )
        )
    else:
        lines.append("- Merged webpage blocks: None")

    return "\n".join(lines)

def build_candidate_label(candidate: dict[str, Any]) -> str:
    template_name = str(candidate.get("display_name") or candidate.get("template_id") or "unknown-template")
    rank = int(candidate.get("rank") or 0)
    score = float(candidate.get("score") or 0.0)
    max_score = float(candidate.get("max_possible_score") or 0.0)
    return f"{rank}. {template_name} (score {score:.2f}/{max_score:.2f})"

def attach_candidate_labels(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labeled: list[dict[str, Any]] = []
    for candidate in candidates:
        enriched = dict(candidate)
        enriched["ui_label"] = build_candidate_label(enriched)
        labeled.append(enriched)
    return labeled

def resolve_selected_candidate(
    selected_label: str | None,
    search_state: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not selected_label:
        return None

    for candidate in (search_state or {}).get("candidates", []):
        if str(candidate.get("ui_label") or "") == str(selected_label):
            return candidate
    return None

def _arbiter_review_feedback_text(report: Any) -> str:
    if report is None:
        return ""
    try:
        arbiter_report = report if isinstance(report, ArbiterReport) else ArbiterReport.model_validate(report)
    except Exception:
        return ""
    if not arbiter_report.items:
        return ""
    suggestions = [
        f"{item.severity}: {item.target} -> {item.advice}"
        for item in arbiter_report.items
    ]
    return "[Draft Review] " + " | ".join(suggestions)

def format_arbiter_autofix_prompt(report: ArbiterReport) -> str:
    if not report.items:
        return ""
    lines = ["[Auto-fix from Arbiter Review]",
             "The automated review found the following issues to fix.",
             "Each item includes a target and specific changes.",
             "Apply these fixes precisely as described:\n"]
    for i, item in enumerate(report.items, 1):
        lines.append(f"{i}. [{item.severity.upper()}] {item.target}: {item.advice}")
    lines.append("\nApply both CSS fixes and content replacements as needed. "
                 "Use the exact selectors and values specified above. "
                 "For wrong images, replace with the correct asset path from the available paper assets.")
    return "\n".join(lines)
