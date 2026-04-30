import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup, Tag
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from src.agents.coder_critic import (
    build_coder_critic_router,
    coder_critic_node,
)
from src.agents.content_composer import run_content_composer_agent
from src.utils.html_utils import (
    extract_html_document,
    extract_html_fragment,
    message_content_to_text,
    normalize_html_document_whitespace,
    read_current_page_html,
    read_text_with_fallback,
)
from src.services.human_feedback import extract_human_feedback_text, normalize_human_feedback
from src.services.artifact_store import save_page_content_plan
from src.services.paper_assets import asset_lookup, asset_target_filename, collect_page_plan_asset_ids, resolved_asset_source_path
from src.utils.json_utils import to_pretty_json
from src.services.llm import get_llm
from src.validators.page_manifest import (
    annotate_global_anchors,
    build_expected_global_anchors,
    build_page_manifest_path,
    enrich_page_plan_shell_contracts,
    extract_page_manifest,
    missing_shell_contract_block_ids,
    save_page_manifest,
)
from src.validators.page_validation import (
    collect_allowed_asset_web_paths,
    validate_fragment_local_image_sources,
    validate_local_image_references,
)
from src.prompts import (
    BLOCK_RENDER_SYSTEM_PROMPT,
    BLOCK_RENDER_USER_PROMPT_TEMPLATE,
    CODER_SYSTEM_PROMPT,
    CODER_USER_PROMPT_TEMPLATE,
)
from src.contracts.schemas import (
    BlockRenderArtifact,
    BlockRenderSpec,
    BlockShellContract,
    CoderArtifact,
    FULLPAGE_RENDER_STRATEGY,
    PaperAsset,
    PagePlan,
    PageContentPlan,
    ResolvedBlockBinding,
    StructuredPaper,
    TemplateProfile,
)
from src.contracts.state import CoderState
from src.template.template_ir import resolve_shell_node

_AFFILIATION_KEYWORDS = (
    "university",
    "institute",
    "college",
    "school",
    "department",
    "lab",
    "laboratory",
    "research",
    "academy",
    "company",
    "center",
    "centre",
)

_BLOCK_LAYOUT_BASELINE_STYLE_ID = "paperalchemy-block-layout-baseline"
MIN_FULLPAGE_CONTENT_COVERAGE_RATIO = 0.52
MIN_FULLPAGE_VISIBLE_TEXT_CHARS = 4200
MIN_SOURCE_CHARS_FOR_FULLPAGE_VISIBLE_FLOOR = 5000
_BLOCK_LAYOUT_BASELINE_CSS = """[data-pa-block] {
  box-sizing: border-box;
  width: 100%;
  max-width: 1000px;
  margin-left: auto;
  margin-right: auto;
}"""
_BLOCK_ROOT_INLINE_LAYOUT_PROPS = {
    "width",
    "min-width",
    "max-width",
    "inline-size",
    "min-inline-size",
    "max-inline-size",
    "margin-left",
    "margin-right",
    "margin-inline",
    "margin-inline-start",
    "margin-inline-end",
}


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


def _normalize_template_profile(profile: Any) -> TemplateProfile | None:
    if isinstance(profile, TemplateProfile):
        return profile
    if profile is None:
        return None
    try:
        return TemplateProfile.model_validate(profile)
    except Exception:
        return None


def _safe_slug(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", text.strip())
    slug = slug.strip("-").lower()
    return slug or "asset"


def _dedupe_clean_strings(values: list[str]) -> list[str]:
    results: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = " ".join(str(value or "").split()).strip(" ,;")
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            results.append(cleaned)
    return results


def _split_metadata_segments(blob: str) -> list[str]:
    text = re.sub(r"\s+(?:and|&)\s+", ", ", str(blob or "").strip(), flags=re.IGNORECASE)
    if not text:
        return []

    segments: list[str] = []
    current: list[str] = []
    depth = 0
    for char in text:
        if char == "(":
            depth += 1
        elif char == ")" and depth > 0:
            depth -= 1

        if char in {",", ";"} and depth == 0:
            segment = "".join(current).strip()
            if segment:
                segments.append(segment)
            current = []
            continue
        current.append(char)

    tail = "".join(current).strip()
    if tail:
        segments.append(tail)
    return segments


def _extract_labeled_metadata(text: str, labels: tuple[str, ...]) -> str:
    normalized = " ".join(str(text or "").split())
    if not normalized:
        return ""
    label_pattern = "|".join(re.escape(label) for label in labels)
    next_label_pattern = r"(?:authors?|affiliations?|institutions?)\s*:"
    match = re.search(
        rf"(?:{label_pattern})\s*:\s*(.+?)(?=\b{next_label_pattern}|\Z)",
        normalized,
        flags=re.IGNORECASE,
    )
    if not match:
        return ""
    return match.group(1).strip(" .;")


def _looks_like_author_name(value: str) -> bool:
    words = [item for item in str(value or "").split() if item]
    if not 2 <= len(words) <= 5:
        return False
    lowered = value.lower()
    if any(keyword in lowered for keyword in _AFFILIATION_KEYWORDS):
        return False
    return all(re.match(r"^[A-Z][A-Za-z.'`-]*$", word) for word in words)


def _parse_author_blob(blob: str) -> tuple[list[str], list[str]]:
    authors: list[str] = []
    affiliations: list[str] = []
    for segment in _split_metadata_segments(blob):
        normalized = " ".join(segment.split()).strip(" .;")
        if not normalized:
            continue

        for paren_value in re.findall(r"\(([^)]+)\)", normalized):
            candidate_affiliation = " ".join(paren_value.split()).strip(" .;")
            if candidate_affiliation:
                affiliations.append(candidate_affiliation)

        candidate = re.sub(r"\([^)]*\)", "", normalized)
        candidate = re.sub(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", "", candidate)
        candidate = re.sub(r"^[^A-Za-z]+|[^A-Za-z]+$", "", candidate).strip()
        if not candidate:
            continue
        if any(keyword in candidate.lower() for keyword in _AFFILIATION_KEYWORDS):
            affiliations.append(candidate)
            continue
        if _looks_like_author_name(candidate):
            authors.append(candidate)

    return _dedupe_clean_strings(authors), _dedupe_clean_strings(affiliations)


def _extract_front_matter_metadata(structured_paper: StructuredPaper | None) -> tuple[list[str], list[str]]:
    if structured_paper is None:
        return [], []

    overall_summary = str(structured_paper.overall_summary or "").strip()
    early_section_text = "\n".join(
        str(section.rich_web_content or "").strip()
        for section in list(structured_paper.sections or [])[:2]
        if str(section.rich_web_content or "").strip()
    )
    combined_text = "\n".join(part for part in [overall_summary, early_section_text] if part)

    explicit_authors = _extract_labeled_metadata(combined_text, ("Authors", "Author"))
    explicit_affiliations = _extract_labeled_metadata(combined_text, ("Affiliations", "Affiliation", "Institutions"))
    authors, author_affiliations = _parse_author_blob(explicit_authors) if explicit_authors else ([], [])
    affiliations = _dedupe_clean_strings(_split_metadata_segments(explicit_affiliations)) if explicit_affiliations else []
    affiliations.extend(author_affiliations)

    paper_title = str(structured_paper.paper_title or "").strip()
    short_title = paper_title.split(":")[0].strip() if paper_title else ""
    if not authors:
        contributor_patterns = [
            rf"\b(?:Developed|Authored|Written|Proposed|Presented|Introduced|Created)\s+by\s+(.+?)(?=(?:,\s*)?{re.escape(short_title)}\b|\b(?:this paper|the paper)\b|\b(?:addresses|presents|proposes|introduces|studies|explores|describes|shows|evaluates)\b|\Z)",
            r"\bAuthors?\s*[:\-]\s*(.+?)(?=\b(?:Affiliations?|Institutions?)\s*:|\Z)",
        ]
        for pattern in contributor_patterns:
            match = re.search(pattern, overall_summary, flags=re.IGNORECASE)
            if not match:
                continue
            parsed_authors, parsed_affiliations = _parse_author_blob(match.group(1))
            authors.extend(parsed_authors)
            affiliations.extend(parsed_affiliations)
            if authors:
                break

    if not affiliations:
        for text in (overall_summary, early_section_text):
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
                normalized = " ".join(sentence.split()).strip(" .;")
                if normalized and any(keyword in normalized.lower() for keyword in _AFFILIATION_KEYWORDS):
                    affiliations.append(normalized)

    return _dedupe_clean_strings(authors)[:8], _dedupe_clean_strings(affiliations)[:6]


def _to_html_relative_path(target_path: Path, base_dir: Path) -> str:
    rel_path = os.path.relpath(target_path, start=base_dir)
    web_path = str(rel_path).replace("\\", "/")
    if not web_path.startswith((".", "/")):
        web_path = f"./{web_path}"
    return web_path


def _ensure_body_markers(html_text: str) -> str:
    result = str(html_text or "")
    if "PaperAlchemy Generated Body Start" not in result:
        result = re.sub(
            r"(<body\b[^>]*>)",
            r"\1\n<!-- PaperAlchemy Generated Body Start -->",
            result,
            count=1,
            flags=re.IGNORECASE,
        )
    if "PaperAlchemy Generated Body End" not in result:
        result = re.sub(
            r"(</body>)",
            r"<!-- PaperAlchemy Generated Body End -->\n\1",
            result,
            count=1,
            flags=re.IGNORECASE,
        )
    return result


def _ensure_doctype(html_text: str) -> str:
    normalized = str(html_text or "").strip()
    if normalized and "<!doctype" not in normalized.lower():
        normalized = "<!DOCTYPE html>\n" + normalized
    return normalized


def _normalize_asset_key(value: str) -> str:
    return str(value or "").strip().replace("\\", "/")


def _collect_asset_ids(page_plan: PagePlan, structured_paper: StructuredPaper) -> list[str]:
    return collect_page_plan_asset_ids(page_plan, structured_paper)


def _build_asset_lookup(structured_paper: StructuredPaper) -> dict[str, dict[str, str]]:
    registry_lookup = asset_lookup(structured_paper)
    section_by_asset_id: dict[str, str] = {}
    for section in structured_paper.sections:
        for binding in section.asset_bindings:
            asset_id = str(binding.asset_id or "").strip()
            if asset_id and asset_id not in section_by_asset_id:
                section_by_asset_id[asset_id] = str(section.section_title or "").strip()

    lookup: dict[str, dict[str, str]] = {}
    for asset_id, asset in registry_lookup.items():
        lookup[asset_id] = {
            "asset_id": asset_id,
            "caption": str(asset.caption or "").strip(),
            "type": str(asset.type or "").strip(),
            "section_title": section_by_asset_id.get(asset_id, ""),
            "image_path": str(asset.image_path or "").strip(),
            "page_number": str(asset.page_number or ""),
        }
    return lookup


def _copy_paper_assets(
    project_root: Path,
    paper_folder_name: str,
    site_dir: Path,
    entry_html_path: Path,
    structured_paper: StructuredPaper,
    asset_ids: list[str],
) -> tuple[list[dict[str, str]], list[str]]:
    asset_manifest: list[dict[str, str]] = []
    copied_assets: list[str] = []
    if not asset_ids:
        return asset_manifest, copied_assets

    target_dir = site_dir / "assets" / "paper"
    target_dir.mkdir(parents=True, exist_ok=True)
    asset_metadata_lookup = _build_asset_lookup(structured_paper)
    registry_lookup = asset_lookup(structured_paper)
    used_names: set[str] = set()

    for asset_id in asset_ids:
        clean_asset_id = str(asset_id or "").strip()
        if not clean_asset_id:
            continue
        asset = registry_lookup.get(clean_asset_id)
        metadata = asset_metadata_lookup.get(clean_asset_id, {})
        if asset is None:
            continue
        source_path = resolved_asset_source_path(project_root, paper_folder_name, asset)
        if not source_path.exists() or not source_path.is_file():
            continue

        target_name = asset_target_filename(asset)
        base_name = _safe_slug(Path(target_name).stem)[:60]
        suffix = Path(target_name).suffix or source_path.suffix or ".png"
        disambiguation = 2
        while target_name in used_names:
            target_name = f"{base_name}-{disambiguation}{suffix}"
            disambiguation += 1
        used_names.add(target_name)

        target_path = target_dir / target_name
        shutil.copy2(source_path, target_path)
        copied_rel_path = str(target_path.relative_to(site_dir)).replace("\\", "/")
        web_path = _to_html_relative_path(target_path, entry_html_path.parent)
        asset_manifest.append(
            {
                "asset_id": clean_asset_id,
                "source_path": str(asset.image_path or "").strip(),
                "web_path": web_path,
                "filename": target_name,
                "caption": str(metadata.get("caption") or ""),
                "type": str(metadata.get("type") or ""),
                "section_title": str(metadata.get("section_title") or ""),
                "page_number": str(metadata.get("page_number") or ""),
            }
        )
        copied_assets.append(copied_rel_path)

    return asset_manifest, copied_assets


def _format_feedback_block(value: Any) -> str:
    if not isinstance(value, list):
        return "(none)"
    lines: list[str] = []
    for index, item in enumerate(value, start=1):
        clean = str(item or "").strip()
        if clean:
            lines.append(f"{index}. {clean}")
    return "\n".join(lines) if lines else "(none)"


def _visible_text_char_count(html_text: str) -> int:
    soup = BeautifulSoup(str(html_text or ""), "html.parser")
    text = " ".join(soup.get_text(" ", strip=True).split())
    return len(text)


def _structured_source_char_count(structured_paper: StructuredPaper | None) -> int:
    if structured_paper is None:
        return 0
    parts = [str(structured_paper.overall_summary or "").strip()]
    parts.extend(
        str(section.rich_web_content or "").strip()
        for section in list(structured_paper.sections or [])
        if str(section.rich_web_content or "").strip()
    )
    return sum(len(part) for part in parts if part)


def _validate_fullpage_content_density(
    html_text: str,
    *,
    structured_paper: StructuredPaper | None,
    page_plan: PagePlan,
    page_content_plan: PageContentPlan | None = None,
) -> list[str]:
    critiques: list[str] = []
    soup = BeautifulSoup(str(html_text or ""), "html.parser")
    rendered_blocks = {
        str(tag.get("data-pa-block") or "").strip()
        for tag in soup.select("[data-pa-block]")
        if str(tag.get("data-pa-block") or "").strip()
    }
    expected_blocks = {
        str(block.block_id or "").strip()
        for block in page_plan.blocks
        if str(block.block_id or "").strip()
    }
    missing_blocks = sorted(expected_blocks - rendered_blocks)
    if missing_blocks:
        critiques.append("Fullpage draft omitted planned block(s): " + ", ".join(missing_blocks) + ".")

    if page_content_plan is not None:
        block_tags = {
            str(tag.get("data-pa-block") or "").strip(): tag
            for tag in soup.select("[data-pa-block]")
            if str(tag.get("data-pa-block") or "").strip()
        }
        for content_block in page_content_plan.blocks:
            block_id = str(content_block.block_id or "").strip()
            block_tag = block_tags.get(block_id)
            if block_tag is None:
                continue
            block_text = " ".join(block_tag.get_text(" ", strip=True).split())
            if len(block_text) < int(content_block.min_visible_chars or 0):
                critiques.append(
                    f"Block '{block_id}' is too compressed: visible text has {len(block_text)} chars, "
                    f"but PageContentPlan requires at least {content_block.min_visible_chars}."
                )
            lowered = block_text.lower()
            missing_metrics = [
                metric
                for metric in content_block.must_include_metrics
                if str(metric or "").strip() and str(metric or "").strip().lower() not in lowered
            ]
            if missing_metrics:
                critiques.append(
                    f"Block '{block_id}' is missing required metric(s): "
                    + ", ".join(missing_metrics[:8])
                    + "."
                )
            missing_tables = []
            for table in content_block.must_include_tables:
                clean_table = str(table or "").strip()
                if not clean_table:
                    continue
                if clean_table.lower() in lowered:
                    continue
                numeric_tokens = re.findall(r"\d+(?:\.\d+)?", clean_table)
                if block_tag.find("table") or any(token in lowered for token in numeric_tokens[:6]):
                    continue
                missing_tables.append(clean_table)
            if missing_tables:
                critiques.append(f"Block '{block_id}' is missing required table evidence from PageContentPlan.")
        return critiques

    source_chars = _structured_source_char_count(structured_paper)
    if source_chars <= 0:
        return critiques

    visible_chars = _visible_text_char_count(html_text)
    expected_chars = int(source_chars * MIN_FULLPAGE_CONTENT_COVERAGE_RATIO)
    if source_chars >= MIN_SOURCE_CHARS_FOR_FULLPAGE_VISIBLE_FLOOR:
        expected_chars = max(MIN_FULLPAGE_VISIBLE_TEXT_CHARS, expected_chars)
    if visible_chars < expected_chars:
        critiques.append(
            "Fullpage draft is too compressed: "
            f"visible text has {visible_chars} chars, but source narrative has {source_chars} chars. "
            f"Regenerate with at least {expected_chars} visible chars by expanding each planned block "
            "with concrete material from STRUCTURED_PAPER_JSON.sections[*].rich_web_content. "
            "Do not treat content_contract.body_points as a maximum length; preserve mechanisms, "
            "numeric results, tables, equations, and comparisons in polished HTML."
        )

    return critiques


def _read_previous_generated_html(state: CoderState) -> str:
    artifact = _normalize_coder_artifact(state.get("coder_artifact"))
    if not artifact:
        return "(none)"
    previous_html = read_current_page_html(artifact, missing_value="").strip()
    return previous_html or "(none)"


def _sanitized_page_plan_for_prompt(page_plan: PagePlan) -> dict[str, Any]:
    payload = page_plan.model_dump()
    payload["dom_mapping"] = {
        selector: "[compat_global_anchor]"
        for selector in page_plan.dom_mapping
    }
    return payload


def _with_shell_enriched_page_plan(page_plan: PagePlan, template_profile: TemplateProfile | None) -> PagePlan:
    enriched_plan = enrich_page_plan_shell_contracts(page_plan, template_profile)
    missing_blocks = missing_shell_contract_block_ids(enriched_plan)
    if missing_blocks:
        raise ValueError("Template shell extraction failed for block(s): " + ", ".join(missing_blocks))
    return enriched_plan


def _output_dir(project_root: Path, paper_folder_name: str) -> Path:
    return project_root / "data" / "output" / paper_folder_name


def _template_profile_output_path(output_dir: Path) -> Path:
    return output_dir / "template_profile.json"


def _block_specs_dir(output_dir: Path) -> Path:
    return output_dir / "block_specs"


def _block_renders_dir(output_dir: Path) -> Path:
    return output_dir / "block_renders"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _selector_tag(soup: BeautifulSoup, selector: str, match_index: int = 0) -> Tag:
    try:
        matches = [match for match in soup.select(str(selector or "").strip()) if isinstance(match, Tag)]
    except Exception as exc:
        raise ValueError(f"Invalid selector '{selector}': {exc}") from exc
    if not matches:
        raise ValueError(f"No matches for selector '{selector}'.")
    if match_index >= len(matches):
        raise ValueError(
            f"match_index {match_index} out of range for selector '{selector}' with {len(matches)} match(es)."
        )
    return matches[match_index]


def _extract_single_root_tag(html_fragment: str) -> Tag:
    fragment = BeautifulSoup(str(html_fragment or ""), "html.parser")
    nodes = [
        node
        for node in (list(fragment.body.contents) if fragment.body is not None else list(fragment.contents))
        if not (isinstance(node, str) and not node.strip())
    ]
    if len(nodes) != 1 or not isinstance(nodes[0], Tag):
        raise ValueError("Expected exactly one root element in block fragment HTML.")
    return nodes[0]


def _allowed_slot_ids() -> set[str]:
    return {"title", "summary", "body", "media", "meta", "actions"}


def _build_render_specs(
    *,
    page_plan: PagePlan,
    template_profile: TemplateProfile,
    template_reference_html: str,
) -> tuple[list[BlockRenderSpec], PagePlan]:
    template_soup = BeautifulSoup(str(template_reference_html or ""), "html.parser")
    outline_lookup = {item.block_id: item for item in page_plan.page_outline}
    updated_blocks = []
    specs: list[BlockRenderSpec] = []

    for block in page_plan.blocks:
        selector = str(block.target_template_region.selector_hint or "").strip()
        if not selector:
            raise ValueError(f"Block '{block.block_id}' is missing selector_hint for block assembly.")

        mi = block.shell_contract.match_index if block.shell_contract else 0
        shell_tag = _selector_tag(template_soup, selector, match_index=mi)
        candidate = resolve_shell_node(
            template_profile,
            shell_id=str(block.target_template_region.shell_id or "").strip(),
            selector=selector,
        )
        shell_contract = block.shell_contract
        if shell_contract is None:
            raise ValueError(f"Block '{block.block_id}' is missing shell_contract for block assembly.")

        updated_block = block.model_copy(update={"shell_contract": shell_contract}, deep=True)
        updated_blocks.append(updated_block)

        outline_item = outline_lookup.get(block.block_id)
        order = int(outline_item.order if outline_item is not None else updated_block.responsive_rules.mobile_order)
        title = str(outline_item.title if outline_item is not None else updated_block.block_id)
        source_sections = (
            list(outline_item.source_sections)
            if outline_item is not None
            else []
        )
        specs.append(
            BlockRenderSpec(
                block_id=updated_block.block_id,
                order=order,
                title=title,
                source_sections=source_sections,
                binding=ResolvedBlockBinding(
                    block_id=updated_block.block_id,
                    shell_id=str(updated_block.target_template_region.shell_id or shell_contract.shell_id or "").strip(),
                    selector=selector,
                    region_role=updated_block.target_template_region.region_role,
                    root_tag=shell_contract.root_tag,
                    required_classes=list(shell_contract.required_classes),
                    preserve_ids=list(shell_contract.preserve_ids),
                    wrapper_chain=list(shell_contract.wrapper_chain),
                    actionable_root_selector=shell_contract.actionable_root_selector,
                    dom_index=int(candidate.dom_index if candidate is not None else order),
                    match_index=mi,
                ),
                content_contract=updated_block.content_contract,
                asset_binding=updated_block.asset_binding,
                interaction=updated_block.interaction,
                responsive_rules=updated_block.responsive_rules,
                shell_contract=shell_contract,
                shell_html=str(shell_tag),
            )
        )

    specs.sort(key=lambda item: (item.order, item.block_id))
    updated_plan = page_plan.model_copy(update={"blocks": updated_blocks}, deep=True)
    return specs, updated_plan


def _render_block_fragment(
    *,
    spec: BlockRenderSpec,
    structured_paper: StructuredPaper,
    template_reference_html: str,
    available_paper_assets: list[dict[str, str]],
    coder_instructions: str,
    human_directives: str,
    retry_feedback: str = "",
) -> str:
    llm = get_llm(temperature=0.15, use_smart_model=True, thinking_level="high")
    effective_instructions = coder_instructions
    if retry_feedback:
        effective_instructions = (effective_instructions + "\n\nRetry feedback:\n" + retry_feedback).strip()
    response = llm.invoke(
        [
            SystemMessage(content=BLOCK_RENDER_SYSTEM_PROMPT),
            HumanMessage(
                content=BLOCK_RENDER_USER_PROMPT_TEMPLATE.format(
                    block_render_spec_json=to_pretty_json(spec),
                    structured_paper_json=to_pretty_json(structured_paper),
                    template_shell_html=spec.shell_html,
                    template_reference_html=template_reference_html,
                    available_paper_assets_json=to_pretty_json(available_paper_assets),
                    coder_instructions=effective_instructions or "(none)",
                    human_directives=human_directives or "(none)",
                )
            ),
        ]
    )
    return extract_html_fragment(message_content_to_text(response))


def _validate_block_fragment(
    *,
    fragment_html: str,
    spec: BlockRenderSpec,
    allowed_asset_web_paths: set[str],
) -> list[str]:
    critiques: list[str] = []
    try:
        root = _extract_single_root_tag(fragment_html)
    except ValueError as exc:
        return [str(exc)]

    block_id = str(root.get("data-pa-block") or "").strip()
    if block_id != spec.block_id:
        critiques.append(
            f"Fragment root data-pa-block must equal '{spec.block_id}', got '{block_id or '(missing)'}'."
        )
    if str(root.name or "") != spec.binding.root_tag:
        critiques.append(
            f"Fragment root tag '{root.name}' does not match required shell root '{spec.binding.root_tag}'."
        )
    actual_classes = {item for item in root.get("class", []) if str(item).strip()}
    missing_classes = [item for item in spec.binding.required_classes if item not in actual_classes]
    if missing_classes:
        critiques.append(f"Fragment root is missing required shell classes {missing_classes}.")
    if spec.binding.preserve_ids:
        actual_id = str(root.get("id") or "").strip()
        if actual_id not in set(spec.binding.preserve_ids):
            critiques.append(
                f"Fragment root id '{actual_id or '(missing)'}' must preserve one of {spec.binding.preserve_ids}."
            )

    slot_ids: list[str] = []
    for slot_tag in root.select("[data-pa-slot]"):
        slot_id = str(slot_tag.get("data-pa-slot") or "").strip()
        if slot_id:
            slot_ids.append(slot_id)
    if not slot_ids:
        critiques.append("Fragment must expose at least one data-pa-slot.")
    unsupported_slots = sorted(set(slot_ids) - _allowed_slot_ids())
    if unsupported_slots:
        critiques.append(f"Fragment uses unsupported data-pa-slot values: {unsupported_slots}.")

    critiques.extend(
        validate_fragment_local_image_sources(
            html_text=str(root),
            allowed_asset_web_paths=allowed_asset_web_paths,
            allowed_existing_local_sources=set(),
        )
    )
    return critiques


def _write_block_render_artifact(
    *,
    output_dir: Path,
    spec: BlockRenderSpec,
    fragment_html: str,
    validation_errors: list[str],
    notes: list[str],
) -> BlockRenderArtifact:
    specs_dir = _block_specs_dir(output_dir)
    renders_dir = _block_renders_dir(output_dir)
    specs_dir.mkdir(parents=True, exist_ok=True)
    renders_dir.mkdir(parents=True, exist_ok=True)

    spec_path = specs_dir / f"{spec.block_id}.json"
    render_html_path = renders_dir / f"{spec.block_id}.html"
    render_meta_path = renders_dir / f"{spec.block_id}.json"
    _write_json(spec_path, spec.model_dump())
    render_html_path.write_text(fragment_html, encoding="utf-8")
    _write_json(
        render_meta_path,
        {
            "block_id": spec.block_id,
            "order": spec.order,
            "selector": spec.binding.selector,
            "render_mode": "compiled_block_assembly",
            "html_path": str(render_html_path),
            "metadata_path": str(render_meta_path),
            "validation_errors": validation_errors,
            "notes": notes,
        },
    )
    return BlockRenderArtifact(
        block_id=spec.block_id,
        order=spec.order,
        selector=spec.binding.selector,
        match_index=spec.binding.match_index,
        render_mode="compiled_block_assembly",
        html=fragment_html,
        html_path=str(render_html_path),
        metadata_path=str(render_meta_path),
        validation_errors=validation_errors,
        notes=notes,
    )


def _render_blocks(
    *,
    output_dir: Path,
    specs: list[BlockRenderSpec],
    structured_paper: StructuredPaper,
    template_reference_html: str,
    available_paper_assets: list[dict[str, str]],
    coder_instructions: str,
    human_directives: str,
) -> list[BlockRenderArtifact]:
    allowed_asset_web_paths = collect_allowed_asset_web_paths(available_paper_assets)
    artifacts: list[BlockRenderArtifact] = []

    for spec in specs:
        fragment_html = ""
        last_errors: list[str] = []
        notes: list[str] = []
        for attempt in range(1, 3):
            fragment_html = _render_block_fragment(
                spec=spec,
                structured_paper=structured_paper,
                template_reference_html=template_reference_html,
                available_paper_assets=available_paper_assets,
                coder_instructions=coder_instructions,
                human_directives=human_directives,
                retry_feedback="\n".join(last_errors) if last_errors else "",
            )
            last_errors = _validate_block_fragment(
                fragment_html=fragment_html,
                spec=spec,
                allowed_asset_web_paths=allowed_asset_web_paths,
            )
            if not last_errors:
                notes.append(f"rendered_on_attempt_{attempt}")
                break
        if last_errors:
            raise ValueError(f"Block renderer failed for '{spec.block_id}': {'; '.join(last_errors)}")
        artifacts.append(
            _write_block_render_artifact(
                output_dir=output_dir,
                spec=spec,
                fragment_html=fragment_html,
                validation_errors=last_errors,
                notes=notes,
            )
        )

    artifacts.sort(key=lambda item: (item.order, item.block_id))
    return artifacts


def _find_main_content_container(soup: BeautifulSoup, template_profile: TemplateProfile) -> Tag:
    """Find the main content container that holds all section-level content."""
    # Try common structural selectors in order of specificity
    for selector in (
        "section.main-container",
        "main",
        "div#website-body > section",
        "div#website-body",
        "div#content",
        "div.container",
    ):
        try:
            matches = [m for m in soup.select(selector) if isinstance(m, Tag)]
        except Exception:
            continue
        if matches:
            return matches[0]
    # Fallback: body itself
    return soup.body or soup


def _wrap_block_in_pattern(
    block_tag: Tag,
    shell_contract: BlockShellContract,
    soup: BeautifulSoup,
) -> Tag:
    """Wrap a rendered block in cloned wrapper elements from the shell contract."""
    result = block_tag
    for wrapper_sig in shell_contract.wrapper_chain:
        outer = soup.new_tag(str(wrapper_sig.tag or "div"))
        classes = [c for c in wrapper_sig.required_classes if c]
        if classes:
            outer["class"] = classes
        outer.append(result)
        result = outer
    # Ensure consistent centering for all blocks
    outermost = result
    outermost["style"] = str(outermost.get("style") or "") + "; max-width: 1100px; margin-left: auto; margin-right: auto;"
    return result


def _find_global_element(soup: BeautifulSoup, template_profile: TemplateProfile, global_id: str) -> Tag | None:
    """Find a global element using its canonical anchor selector from the template profile."""
    for anchor in template_profile.template_ir.global_anchors:
        if anchor.global_id != global_id:
            continue
        for sel in (anchor.actionable_selector, anchor.selector):
            sel = str(sel or "").strip()
            if not sel:
                continue
            try:
                matches = [m for m in soup.select(sel) if isinstance(m, Tag)]
            except Exception:
                continue
            if matches:
                return matches[0]
    return None


def _update_global_elements(
    soup: BeautifulSoup,
    structured_paper: StructuredPaper | None,
    page_plan: PagePlan,
    template_profile: TemplateProfile,
) -> None:
    """Update header brand, nav links, and footer with paper-specific content."""
    if structured_paper is None:
        return
    paper_title = str(structured_paper.paper_title or "").strip()
    short_title = paper_title.split(":")[0].strip() if paper_title else ""
    if not short_title:
        return

    # Update header brand via global anchor
    brand = _find_global_element(soup, template_profile, "header_brand")
    if brand is not None:
        brand.string = short_title
        brand["data-pa-global"] = "header_brand"

    # Update <title> tag
    title_tag = soup.find("title")
    if isinstance(title_tag, Tag):
        title_tag.string = short_title

    # Update nav via global anchor — replace with paper section links
    nav = _find_global_element(soup, template_profile, "header_nav")
    if nav is not None:
        nav["data-pa-global"] = "header_nav"
        nav.clear()
        outline = sorted(page_plan.page_outline, key=lambda x: x.order)
        for item in outline[:6]:
            # Extract a short label from the section title
            raw = str(item.title or item.block_id)
            label = raw.split(":")[-1].split("/")[-1].split("and")[0].strip()[:18]
            a_tag = soup.new_tag("a", href=f"#{item.block_id}")
            btn = soup.new_tag("button", attrs={"class": "outline"})
            span = soup.new_tag("span", attrs={"class": "outline"})
            span.string = label
            btn.append(span)
            a_tag.append(btn)
            nav.append(a_tag)

    # Update footer via global anchor
    footer = _find_global_element(soup, template_profile, "footer_meta")
    if footer is not None:
        footer["data-pa-global"] = "footer_meta"
        footer_text = f"(c) 2025. {short_title} Authors."
        text_el = footer.find("a") or footer.find(string=True)
        if text_el and isinstance(text_el, Tag):
            text_el.string = footer_text
        elif text_el:
            text_el.replace_with(footer_text)


def _split_inline_style_declarations(style_text: str) -> list[tuple[str, str]]:
    declarations: list[tuple[str, str]] = []
    for raw_declaration in str(style_text or "").split(";"):
        if ":" not in raw_declaration:
            continue
        raw_name, raw_value = raw_declaration.split(":", 1)
        name = raw_name.strip()
        value = raw_value.strip()
        if name and value:
            declarations.append((name, value))
    return declarations


def _split_css_value_tokens(value: str) -> list[str]:
    tokens: list[str] = []
    current: list[str] = []
    depth = 0
    for char in str(value or "").strip():
        if char == "(":
            depth += 1
        elif char == ")" and depth > 0:
            depth -= 1

        if char.isspace() and depth == 0:
            token = "".join(current).strip()
            if token:
                tokens.append(token)
            current = []
            continue
        current.append(char)

    token = "".join(current).strip()
    if token:
        tokens.append(token)
    return tokens


def _vertical_margin_declarations(value: str) -> list[tuple[str, str]]:
    tokens = _split_css_value_tokens(value)
    if not tokens:
        return []
    if len(tokens) == 1:
        top = bottom = tokens[0]
    elif len(tokens) == 2:
        top = bottom = tokens[0]
    else:
        top = tokens[0]
        bottom = tokens[2]
    return [("margin-top", top), ("margin-bottom", bottom)]


def _normalize_block_root_inline_layout(soup: BeautifulSoup) -> None:
    for block in soup.select("[data-pa-block]"):
        if not isinstance(block, Tag):
            continue
        declarations = _split_inline_style_declarations(str(block.get("style") or ""))
        if not declarations:
            continue

        kept: list[tuple[str, str]] = []
        for name, value in declarations:
            normalized_name = name.lower()
            if normalized_name in _BLOCK_ROOT_INLINE_LAYOUT_PROPS:
                continue
            if normalized_name == "margin":
                kept.extend(_vertical_margin_declarations(value))
                continue
            kept.append((name, value))

        if kept:
            block["style"] = " ".join(f"{name}: {value};" for name, value in kept)
        else:
            block.attrs.pop("style", None)


def _inject_block_layout_baseline_style(soup: BeautifulSoup) -> None:
    if soup.head is None:
        head = soup.new_tag("head")
        if soup.html is not None:
            soup.html.insert(0, head)
        else:
            soup.insert(0, head)

    existing = soup.head.select_one(f'style[id="{_BLOCK_LAYOUT_BASELINE_STYLE_ID}"]')
    if isinstance(existing, Tag):
        existing.string = _BLOCK_LAYOUT_BASELINE_CSS
        return

    baseline_tag = soup.new_tag("style", attrs={"id": _BLOCK_LAYOUT_BASELINE_STYLE_ID})
    baseline_tag.string = _BLOCK_LAYOUT_BASELINE_CSS
    # Keep template links before this baseline; revision overrides are appended later and must win by source order.
    soup.head.append(baseline_tag)


def _inject_front_matter_metadata(
    soup: BeautifulSoup,
    structured_paper: StructuredPaper | None,
) -> None:
    authors, affiliations = _extract_front_matter_metadata(structured_paper)
    if not authors and not affiliations:
        return

    page_text = soup.get_text(" ", strip=True)
    if authors and authors[0] in page_text:
        return

    first_block = next((tag for tag in soup.select("[data-pa-block]") if isinstance(tag, Tag)), None)
    if first_block is None:
        return

    existing_meta = first_block.select_one('[data-pa-slot="meta"]')
    if isinstance(existing_meta, Tag) and "authors:" in existing_meta.get_text(" ", strip=True).lower():
        return

    meta_block = soup.new_tag(
        "div",
        attrs={
            "data-pa-slot": "meta",
            "class": "paperalchemy-front-matter",
            "style": "margin: 0.75rem 0 1.25rem; color: #4b5563; font-size: 0.95rem; line-height: 1.6;",
        },
    )
    if authors:
        authors_line = soup.new_tag("p", attrs={"style": "margin: 0;"})
        authors_line.append(soup.new_tag("strong"))
        authors_line.strong.string = "Authors: "
        authors_line.append(", ".join(authors))
        meta_block.append(authors_line)
    if affiliations:
        affiliations_line = soup.new_tag("p", attrs={"style": "margin: 0.35rem 0 0;"})
        affiliations_line.append(soup.new_tag("strong"))
        affiliations_line.strong.string = "Affiliations: "
        affiliations_line.append("; ".join(affiliations))
        meta_block.append(affiliations_line)

    title_slot = first_block.select_one('[data-pa-slot="title"]')
    if isinstance(title_slot, Tag):
        title_slot.insert_after(meta_block)
    elif first_block.contents:
        first_block.insert(0, meta_block)
    else:
        first_block.append(meta_block)


def _inject_math_fallback_script(soup: BeautifulSoup) -> None:
    if soup.find("script", attrs={"id": "paperalchemy-math-fallback"}):
        return

    html_blob = str(soup)
    if "$" not in html_blob and "\\(" not in html_blob and "MathJax" not in html_blob and "katex" not in html_blob.lower():
        return

    mathjax_script = soup.find("script", attrs={"id": "MathJax-script"})
    if isinstance(mathjax_script, Tag):
        mathjax_script.attrs.pop("async", None)
        mathjax_script["defer"] = ""

    fallback_script = soup.new_tag("script", attrs={"id": "paperalchemy-math-fallback"})
    fallback_script.string = """
(function () {
  function normalizeMathExpression(expr) {
    return String(expr || "")
      .replace(/\\\\mathcal\\{([^}]+)\\}/g, "$1")
      .replace(/\\\\mathbb\\{([^}]+)\\}/g, "$1")
      .replace(/\\\\mathrm\\{([^}]+)\\}/g, "$1")
      .replace(/\\\\operatorname\\{([^}]+)\\}/g, "$1")
      .replace(/\\\\text\\{([^}]+)\\}/g, "$1")
      .replace(/\\\\langle/g, "<")
      .replace(/\\\\rangle/g, ">")
      .replace(/\\\\leq/g, "≤")
      .replace(/\\\\geq/g, "≥")
      .replace(/\\\\neq/g, "≠")
      .replace(/\\\\cdot/g, "·")
      .replace(/\\\\times/g, "×")
      .replace(/\\\\rightarrow/g, "→")
      .replace(/\\\\to/g, "→")
      .replace(/\\\\sigma/g, "sigma")
      .replace(/\\\\tau/g, "tau")
      .replace(/\\\\phi/g, "phi")
      .replace(/\\\\psi/g, "psi")
      .replace(/\\\\lambda/g, "lambda")
      .replace(/\\\\alpha/g, "alpha")
      .replace(/\\\\beta/g, "beta")
      .replace(/\\\\gamma/g, "gamma")
      .replace(/\\\\,/g, " ")
      .replace(/\\^\\{([^}]+)\\}/g, "^$1")
      .replace(/_\\{([^}]+)\\}/g, "_$1")
      .replace(/\\\\([A-Za-z]+)/g, "$1")
      .replace(/[{}]/g, "")
      .replace(/\\s+/g, " ")
      .trim();
  }

  function replaceInlineMath(text) {
    return String(text || "").replace(/\\$\\$([^$]+)\\$\\$|\\$([^$]+)\\$/g, function (_, blockExpr, inlineExpr) {
      return normalizeMathExpression(blockExpr || inlineExpr);
    });
  }

  function applyFallback() {
    if (window.__paperalchemyMathFallbackApplied || !document.body) {
      return;
    }
    var walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, {
      acceptNode: function (node) {
        if (!node || !node.nodeValue || node.nodeValue.indexOf("$") === -1) {
          return NodeFilter.FILTER_REJECT;
        }
        var parent = node.parentElement;
        if (!parent) {
          return NodeFilter.FILTER_REJECT;
        }
        var tagName = parent.tagName;
        if (tagName === "SCRIPT" || tagName === "STYLE" || tagName === "TEXTAREA") {
          return NodeFilter.FILTER_REJECT;
        }
        if (parent.closest("mjx-container, .MathJax")) {
          return NodeFilter.FILTER_REJECT;
        }
        return NodeFilter.FILTER_ACCEPT;
      }
    });
    var nodes = [];
    var current;
    while ((current = walker.nextNode())) {
      nodes.push(current);
    }
    nodes.forEach(function (node) {
      node.nodeValue = replaceInlineMath(node.nodeValue);
    });
    window.__paperalchemyMathFallbackApplied = true;
  }

  function tryTypesetOrFallback() {
    if (window.MathJax && typeof window.MathJax.typesetPromise === "function") {
      window.MathJax.typesetPromise().catch(function () {
        applyFallback();
      });
      return;
    }
    applyFallback();
  }

  window.addEventListener("load", function () {
    var mathScript = document.getElementById("MathJax-script");
    if (mathScript) {
      mathScript.addEventListener("error", applyFallback, { once: true });
    }
    window.setTimeout(tryTypesetOrFallback, 1200);
  });
})();
""".strip()

    target = soup.body or soup
    target.append(fallback_script)


def _postprocess_generated_html(
    html_text: str,
    *,
    structured_paper: StructuredPaper | None,
    page_plan: PagePlan,
    template_profile: TemplateProfile | None,
) -> str:
    soup = BeautifulSoup(str(html_text or ""), "html.parser")
    if template_profile is not None and structured_paper is not None:
        _update_global_elements(soup, structured_paper, page_plan, template_profile)
    _normalize_block_root_inline_layout(soup)
    _inject_block_layout_baseline_style(soup)
    _inject_front_matter_metadata(soup, structured_paper)
    _inject_math_fallback_script(soup)
    return normalize_html_document_whitespace(_ensure_doctype(str(soup)))


def _assemble_page(
    *,
    page_plan: PagePlan,
    template_profile: TemplateProfile,
    template_reference_html: str,
    block_artifacts: list[BlockRenderArtifact],
    block_specs: list[BlockRenderSpec] | None = None,
    structured_paper: StructuredPaper | None = None,
) -> str:
    soup = BeautifulSoup(str(template_reference_html or ""), "html.parser")
    container = _find_main_content_container(soup, template_profile)

    # Identify global elements that must be preserved
    global_tag_ids: set[int] = set()
    for anchor in template_profile.template_ir.global_anchors:
        try:
            for match in soup.select(str(anchor.selector or "").strip()):
                if isinstance(match, Tag):
                    global_tag_ids.add(id(match))
                    # Also protect all ancestors up to container
                    for parent in match.parents:
                        if isinstance(parent, Tag):
                            global_tag_ids.add(id(parent))
        except Exception:
            pass

    # Clear non-global content from container
    for child in list(container.children):
        if isinstance(child, Tag) and id(child) not in global_tag_ids:
            child.decompose()

    # Build spec lookup for wrapper info
    spec_lookup: dict[str, BlockRenderSpec] = {}
    if block_specs:
        spec_lookup = {s.block_id: s for s in block_specs}

    # Clone wrapper pattern + insert each block sequentially
    for artifact in sorted(block_artifacts, key=lambda item: (item.order, item.block_id)):
        block_root = _extract_single_root_tag(artifact.html)
        spec = spec_lookup.get(artifact.block_id)
        if spec and spec.shell_contract and spec.shell_contract.wrapper_chain:
            wrapped = _wrap_block_in_pattern(block_root, spec.shell_contract, soup)
        else:
            wrapped = block_root
        container.append(wrapped)

    # Update global elements (header brand, nav, footer) with paper content
    _update_global_elements(soup, structured_paper, page_plan, template_profile)
    _inject_front_matter_metadata(soup, structured_paper)
    _inject_math_fallback_script(soup)

    # Remove unsafe widget selectors not bound to any block
    bound_selectors = {artifact.selector for artifact in block_artifacts}
    for selector in template_profile.unsafe_selectors:
        if selector in bound_selectors or selector in set(template_profile.global_preserve_selectors):
            continue
        try:
            matches = [match for match in soup.select(str(selector or "").strip()) if isinstance(match, Tag)]
        except Exception:
            matches = []
        for match in matches:
            match.decompose()

    html_text = _ensure_doctype(str(soup))
    html_text = _ensure_body_markers(html_text)
    html_text = annotate_global_anchors(html_text, page_plan, template_profile=template_profile)
    return normalize_html_document_whitespace(html_text)


def _persist_generated_page(
    *,
    output_dir: Path,
    site_dir: Path,
    generated_entry_html_path: Path,
    template_entry_rel: str,
    template_profile: TemplateProfile | None,
    generated_html: str,
    page_manifest: Any,
) -> tuple[str | None, str]:
    generated_entry_html_path.parent.mkdir(parents=True, exist_ok=True)
    generated_entry_html_path.write_text(generated_html, encoding="utf-8")
    manifest_path = build_page_manifest_path(generated_entry_html_path)
    save_page_manifest(manifest_path, page_manifest)

    mirrored_entry_path = site_dir / template_entry_rel if template_entry_rel else generated_entry_html_path
    if mirrored_entry_path.resolve() != generated_entry_html_path.resolve():
        mirrored_entry_path.parent.mkdir(parents=True, exist_ok=True)
        mirrored_entry_path.write_text(generated_html, encoding="utf-8")

    template_profile_path = None
    if template_profile is not None:
        template_profile_path = str(_template_profile_output_path(output_dir))
        _write_json(Path(template_profile_path), template_profile.model_dump())
    return template_profile_path, str(manifest_path)


def _build_compiled_artifact(
    *,
    output_dir: Path,
    site_dir: Path,
    generated_entry_html_path: Path,
    template_entry_rel: str,
    page_plan: PagePlan,
    template_profile_path: str | None,
    page_manifest_path: str,
    copied_assets: list[str],
    paper_asset_manifest: list[dict[str, str]],
) -> CoderArtifact:
    edited_files = ["index.html"]
    mirrored_entry_path = site_dir / template_entry_rel if template_entry_rel else generated_entry_html_path
    if mirrored_entry_path.resolve() != generated_entry_html_path.resolve():
        edited_files.append(str(mirrored_entry_path.relative_to(site_dir)).replace("\\", "/"))
    return CoderArtifact(
        site_dir=str(site_dir),
        entry_html=str(generated_entry_html_path),
        selected_template_id=page_plan.template_selection.selected_template_id,
        copied_assets=copied_assets,
        paper_asset_manifest=paper_asset_manifest,
        edited_files=edited_files,
        notes=(
            "v8-compiled-block-assembly: rendered block specs, assembled template shells programmatically, "
            "preserved global anchors, and validated page manifest compatibility."
        ),
        render_mode="compiled_block_assembly",
        template_profile_path=template_profile_path,
        page_manifest_path=page_manifest_path,
        block_artifact_dir=str(_block_renders_dir(output_dir)),
    )


def _run_compiled_block_assembly(
    *,
    paper_folder_name: str,
    structured_paper: StructuredPaper,
    page_plan: PagePlan,
    template_profile: TemplateProfile,
    human_directives: str,
    coder_instructions: str,
) -> tuple[CoderArtifact, PagePlan, list[BlockRenderSpec], list[BlockRenderArtifact]]:
    project_root = Path(__file__).resolve().parents[2]
    template_root = project_root / page_plan.template_selection.selected_root_dir
    template_entry_rel = str(page_plan.template_selection.selected_entry_html or "").strip()
    template_entry_path = template_root / template_entry_rel
    output_dir = _output_dir(project_root, paper_folder_name)
    site_dir = output_dir / "site"
    generated_entry_html_path = site_dir / "index.html"

    if site_dir.exists():
        shutil.rmtree(site_dir)
    if not template_root.exists():
        raise FileNotFoundError(f"Template root not found: {template_root}")
    if not template_entry_path.exists():
        raise FileNotFoundError(f"Template entry html not found: {template_entry_path}")

    template_reference_html = read_text_with_fallback(template_entry_path)
    shutil.copytree(template_root, site_dir)
    asset_ids = _collect_asset_ids(page_plan, structured_paper)
    available_paper_assets, copied_assets = _copy_paper_assets(
        project_root=project_root,
        paper_folder_name=paper_folder_name,
        site_dir=site_dir,
        entry_html_path=generated_entry_html_path,
        structured_paper=structured_paper,
        asset_ids=asset_ids,
    )

    block_render_specs, updated_page_plan = _build_render_specs(
        page_plan=page_plan,
        template_profile=template_profile,
        template_reference_html=template_reference_html,
    )
    block_render_artifacts = _render_blocks(
        output_dir=output_dir,
        specs=block_render_specs,
        structured_paper=structured_paper,
        template_reference_html=template_reference_html,
        available_paper_assets=available_paper_assets,
        coder_instructions=coder_instructions,
        human_directives=human_directives,
    )
    generated_html = _assemble_page(
        page_plan=updated_page_plan,
        template_profile=template_profile,
        template_reference_html=template_reference_html,
        block_artifacts=block_render_artifacts,
        block_specs=block_render_specs,
        structured_paper=structured_paper,
    )
    asset_critiques = validate_local_image_references(
        html_text=generated_html,
        entry_html_path=generated_entry_html_path,
        site_dir=site_dir,
        allowed_asset_web_paths=collect_allowed_asset_web_paths(available_paper_assets),
        enforce_paper_asset_whitelist=True,
    )
    if asset_critiques:
        raise ValueError("Compiled assembly failed local image validation: " + " | ".join(asset_critiques))

    page_manifest = extract_page_manifest(
        html_text=generated_html,
        entry_html=generated_entry_html_path,
        selected_template_id=page_plan.template_selection.selected_template_id,
        page_plan=updated_page_plan,
        template_profile=template_profile,
    )
    template_profile_path, page_manifest_path = _persist_generated_page(
        output_dir=output_dir,
        site_dir=site_dir,
        generated_entry_html_path=generated_entry_html_path,
        template_entry_rel=template_entry_rel,
        template_profile=template_profile,
        generated_html=generated_html,
        page_manifest=page_manifest,
    )
    artifact = _build_compiled_artifact(
        output_dir=output_dir,
        site_dir=site_dir,
        generated_entry_html_path=generated_entry_html_path,
        template_entry_rel=template_entry_rel,
        page_plan=updated_page_plan,
        template_profile_path=template_profile_path,
        page_manifest_path=page_manifest_path,
        copied_assets=copied_assets,
        paper_asset_manifest=available_paper_assets,
    )
    return artifact, updated_page_plan, block_render_specs, block_render_artifacts


def _run_template_guided_fullpage_render(
    *,
    paper_folder_name: str,
    structured_paper: StructuredPaper,
    page_plan: PagePlan,
    human_directives: str,
    coder_instructions: str,
    state: CoderState,
    template_profile: TemplateProfile | None,
) -> tuple[CoderArtifact, PagePlan]:
    previous_generated_html = _read_previous_generated_html(state)
    project_root = Path(__file__).resolve().parents[2]
    template_root = project_root / page_plan.template_selection.selected_root_dir
    template_entry_rel = str(page_plan.template_selection.selected_entry_html or "").strip()
    template_entry_path = template_root / template_entry_rel
    output_dir = _output_dir(project_root, paper_folder_name)
    site_dir = output_dir / "site"
    generated_entry_html_path = site_dir / "index.html"

    if site_dir.exists():
        shutil.rmtree(site_dir)
    if not template_root.exists():
        raise FileNotFoundError(f"Template root not found: {template_root}")
    if not template_entry_path.exists():
        raise FileNotFoundError(f"Template entry html not found: {template_entry_path}")

    template_reference_html = read_text_with_fallback(template_entry_path)
    page_plan = _with_shell_enriched_page_plan(page_plan, template_profile)
    shutil.copytree(template_root, site_dir)

    asset_ids = _collect_asset_ids(page_plan, structured_paper)
    asset_manifest, copied_assets = _copy_paper_assets(
        project_root=project_root,
        paper_folder_name=paper_folder_name,
        site_dir=site_dir,
        entry_html_path=generated_entry_html_path,
        structured_paper=structured_paper,
        asset_ids=asset_ids,
    )
    page_content_plan = run_content_composer_agent(
        structured_paper=structured_paper,
        page_plan=page_plan,
        human_directives=human_directives,
        coder_instructions=coder_instructions,
    )
    page_content_plan_path = output_dir / "page_content_plan.json"
    save_page_content_plan(page_content_plan_path, page_content_plan)

    llm = get_llm(temperature=0.2, use_smart_model=True, thinking_level="high")
    density_critiques: list[str] = []
    generated_html = ""
    for attempt in range(1, 3):
        prior_feedback_parts = [_format_feedback_block(state.get("coder_feedback_history"))]
        if density_critiques:
            prior_feedback_parts.append("\n".join(density_critiques))
        response = llm.invoke(
            [
                SystemMessage(content=CODER_SYSTEM_PROMPT),
                HumanMessage(
                    content=CODER_USER_PROMPT_TEMPLATE.format(
                        structured_paper_json=to_pretty_json(structured_paper),
                        page_plan_json=json.dumps(
                            _sanitized_page_plan_for_prompt(page_plan),
                            indent=2,
                            ensure_ascii=False,
                        ),
                        page_content_plan_json=to_pretty_json(page_content_plan),
                        template_reference_html=template_reference_html,
                        coder_instructions=coder_instructions or "(none)",
                        human_directives=human_directives or "(none)",
                        available_paper_assets_json=to_pretty_json(asset_manifest),
                        global_anchor_requirements_json=json.dumps(
                            build_expected_global_anchors(
                                page_plan,
                                template_profile=template_profile,
                                reference_html_text=template_reference_html,
                            ),
                            indent=2,
                            ensure_ascii=False,
                        ),
                        prior_coder_feedback="\n\n".join(
                            part for part in prior_feedback_parts if str(part or "").strip() and part != "(none)"
                        )
                        or "(none)",
                        previous_generated_html=previous_generated_html,
                    )
                ),
            ]
        )

        generated_html = extract_html_document(message_content_to_text(response))
        if not generated_html:
            raise ValueError("Template-guided fullpage coder did not return a valid HTML document.")
        generated_html = _ensure_body_markers(generated_html)
        generated_html = normalize_html_document_whitespace(generated_html)
        generated_html = _postprocess_generated_html(
            generated_html,
            structured_paper=structured_paper,
            page_plan=page_plan,
            template_profile=template_profile,
        )
        generated_html = annotate_global_anchors(generated_html, page_plan, template_profile=template_profile)
        density_critiques = _validate_fullpage_content_density(
            generated_html,
            structured_paper=structured_paper,
            page_plan=page_plan,
            page_content_plan=page_content_plan,
        )
        if not density_critiques:
            break
        print(
            "[PaperAlchemy-Coder] template-guided fullpage content density retry "
            f"{attempt}/2: {' | '.join(density_critiques)}"
        )
    if density_critiques:
        raise ValueError("Template-guided fullpage coder failed content density validation: " + " | ".join(density_critiques))

    asset_critiques = validate_local_image_references(
        html_text=generated_html,
        entry_html_path=generated_entry_html_path,
        site_dir=site_dir,
        allowed_asset_web_paths=collect_allowed_asset_web_paths(asset_manifest),
        enforce_paper_asset_whitelist=True,
    )
    if asset_critiques:
        raise ValueError("Template-guided fullpage coder failed local image validation: " + " | ".join(asset_critiques))

    page_manifest = extract_page_manifest(
        html_text=generated_html,
        entry_html=generated_entry_html_path,
        selected_template_id=page_plan.template_selection.selected_template_id,
        page_plan=page_plan,
        template_profile=template_profile,
    )
    template_profile_path, page_manifest_path = _persist_generated_page(
        output_dir=output_dir,
        site_dir=site_dir,
        generated_entry_html_path=generated_entry_html_path,
        template_entry_rel=template_entry_rel,
        template_profile=template_profile,
        generated_html=generated_html,
        page_manifest=page_manifest,
    )
    edited_files = ["index.html"]
    mirrored_entry_path = site_dir / template_entry_rel if template_entry_rel else generated_entry_html_path
    if mirrored_entry_path.resolve() != generated_entry_html_path.resolve():
        edited_files.append(str(mirrored_entry_path.relative_to(site_dir)).replace("\\", "/"))
    artifact = CoderArtifact(
        site_dir=str(site_dir),
        entry_html=str(generated_entry_html_path),
        selected_template_id=page_plan.template_selection.selected_template_id,
        copied_assets=copied_assets,
        paper_asset_manifest=asset_manifest,
        edited_files=edited_files,
        notes=(
            "v8-template-guided-fullpage: generated shell-constrained HTML via fullpage coder and "
            "validated stable data-pa-block, data-pa-slot, and data-pa-global anchors."
        ),
        render_mode=FULLPAGE_RENDER_STRATEGY,
        template_profile_path=template_profile_path,
        page_manifest_path=page_manifest_path,
        block_artifact_dir=None,
        fullpage_context_dir=None,
        page_content_plan_path=str(page_content_plan_path),
    )
    return artifact, page_plan


def _run_legacy_fullpage_render(
    *,
    paper_folder_name: str,
    structured_paper: StructuredPaper,
    page_plan: PagePlan,
    human_directives: str,
    coder_instructions: str,
    state: CoderState,
    template_profile: TemplateProfile | None,
) -> tuple[CoderArtifact, PagePlan]:
    return _run_template_guided_fullpage_render(
        paper_folder_name=paper_folder_name,
        structured_paper=structured_paper,
        page_plan=page_plan,
        human_directives=human_directives,
        coder_instructions=coder_instructions,
        state=state,
        template_profile=template_profile,
    )


def coder_node(state: CoderState) -> dict[str, Any]:
    print(
        f"[PaperAlchemy-Coder] building site "
        f"(attempt {state.get('coder_retry_count', 0) + 1})..."
    )
    page_plan = _normalize_page_plan(state.get("page_plan"))
    structured_paper = _normalize_structured_paper(state.get("structured_paper"))
    template_profile = _normalize_template_profile(state.get("template_profile"))
    paper_folder_name = str(state.get("paper_folder_name") or "").strip()
    human_directives = extract_human_feedback_text(state.get("human_directives"))
    coder_instructions = str(state.get("coder_instructions") or "").strip()
    if not page_plan or not structured_paper or not paper_folder_name:
        print("[PaperAlchemy-Coder] missing page_plan/structured_paper/paper_folder_name.")
        return {}

    requested_strategy = str(page_plan.plan_meta.render_strategy or FULLPAGE_RENDER_STRATEGY).strip()
    try:
        if requested_strategy == "compiled_block_assembly" and template_profile is not None:
            artifact, resolved_page_plan, block_specs, block_artifacts = _run_compiled_block_assembly(
                paper_folder_name=paper_folder_name,
                structured_paper=structured_paper,
                page_plan=page_plan,
                template_profile=template_profile,
                human_directives=human_directives,
                coder_instructions=coder_instructions,
            )
            return {
                "coder_artifact": artifact,
                "page_plan": resolved_page_plan,
                "block_render_specs": block_specs,
                "block_render_artifacts": block_artifacts,
            }

        artifact, resolved_page_plan = _run_template_guided_fullpage_render(
            paper_folder_name=paper_folder_name,
            structured_paper=structured_paper,
            page_plan=page_plan,
            human_directives=human_directives,
            coder_instructions=coder_instructions,
            state=state,
            template_profile=template_profile,
        )
        return {
            "coder_artifact": artifact,
            "page_plan": resolved_page_plan,
            "block_render_specs": [],
            "block_render_artifacts": [],
        }
    except Exception as exc:
        if requested_strategy == "compiled_block_assembly":
            print(
                "[PaperAlchemy-Coder] compiled block assembly failed, "
                f"falling back to template-guided fullpage: {exc}"
            )
            try:
                artifact, resolved_page_plan = _run_template_guided_fullpage_render(
                    paper_folder_name=paper_folder_name,
                    structured_paper=structured_paper,
                    page_plan=page_plan,
                    human_directives=human_directives,
                    coder_instructions=coder_instructions,
                    state=state,
                    template_profile=template_profile,
                )
                return {
                    "coder_artifact": artifact,
                    "page_plan": resolved_page_plan,
                    "block_render_specs": [],
                    "block_render_artifacts": [],
                }
            except Exception as fallback_exc:
                print(f"[PaperAlchemy-Coder] template-guided fullpage fallback failed: {fallback_exc}")
                return {}
        print(f"[PaperAlchemy-Coder] build failed: {exc}")
        return {}


def build_coder_graph(max_retry: int = 1):
    workflow = StateGraph(CoderState)
    workflow.add_node("coder", coder_node)
    workflow.add_node("coder_critic", coder_critic_node)

    workflow.set_entry_point("coder")
    workflow.add_edge("coder", "coder_critic")
    workflow.add_conditional_edges(
        "coder_critic",
        build_coder_critic_router(max_retry=max_retry),
        {"retry": "coder", "end": END},
    )

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


def run_coder_agent(
    paper_folder_name: str,
    structured_data: StructuredPaper,
    page_plan: PagePlan,
    human_directives: str | dict = "",
    coder_instructions: str = "",
    previous_coder_artifact: CoderArtifact | None = None,
    max_retry: int = 2,
    template_profile: TemplateProfile | None = None,
) -> CoderArtifact | None:
    artifact, _ = run_coder_agent_with_diagnostics(
        paper_folder_name=paper_folder_name,
        structured_data=structured_data,
        page_plan=page_plan,
        human_directives=human_directives,
        coder_instructions=coder_instructions,
        previous_coder_artifact=previous_coder_artifact,
        max_retry=max_retry,
        template_profile=template_profile,
    )
    return artifact


def run_coder_agent_with_diagnostics(
    paper_folder_name: str,
    structured_data: StructuredPaper,
    page_plan: PagePlan,
    human_directives: str | dict = "",
    coder_instructions: str = "",
    previous_coder_artifact: CoderArtifact | None = None,
    max_retry: int = 2,
    template_profile: TemplateProfile | None = None,
) -> tuple[CoderArtifact | None, PagePlan | None]:
    app = build_coder_graph(max_retry=max_retry)
    thread = {"configurable": {"thread_id": f"coder_{paper_folder_name}"}}
    initial_state: CoderState = {
        "paper_folder_name": paper_folder_name,
        "human_directives": normalize_human_feedback(human_directives),
        "coder_instructions": str(coder_instructions or "").strip(),
        "structured_paper": structured_data,
        "page_plan": page_plan,
        "template_profile": template_profile,
        "block_render_specs": [],
        "block_render_artifacts": [],
        "coder_feedback_history": [],
        "coder_artifact": previous_coder_artifact,
        "coder_critic_passed": False,
        "coder_retry_count": 0,
    }

    print("[PaperAlchemy-Coder] running Coder + CoderCritic graph...")
    for _ in app.stream(initial_state, thread):
        pass

    final_state = app.get_state(thread)
    artifact_result = final_state.values.get("coder_artifact")
    resolved_page_plan = _normalize_page_plan(final_state.values.get("page_plan"))
    normalized_artifact = _normalize_coder_artifact(artifact_result)

    if not normalized_artifact or not final_state.values.get("coder_critic_passed"):
        print("[PaperAlchemy-Coder] coder completed but critic did not fully pass.")
        return None, resolved_page_plan

    print(f"[PaperAlchemy-Coder] build completed: {normalized_artifact.entry_html}")
    return normalized_artifact, resolved_page_plan

