import json
import re
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from src.agents.reader_critic import build_critic_router, critic_node
from src.contracts.schemas import (
    ASSET_CONFIRMATION_NONE_ID,
    AssetConfirmationItem,
    AssetConfirmationSession,
    PaperAsset,
    PaperSection,
    SectionAssetBinding,
    StructuredPaper,
    STRUCTURED_PAPER_SCHEMA_VERSION,
)
from src.contracts.state import ReaderState
from src.services.human_feedback import (
    build_human_feedback_payload,
    build_multimodal_message_content,
    extract_human_feedback_text,
    normalize_human_feedback,
)
from src.services.llm import get_llm
from src.prompts import READER_SYSTEM_PROMPT

TEXT_READER_SYSTEM_PROMPT = """You are running the text-only substage of the legacy PaperAlchemy Reader.

Preserve the legacy Reader's extraction behavior, editorial judgment, and quality bar from the prompt below.
However, multimodal capability is now decoupled into a separate visual Reader phase.

Text-only stage rules:
- do not bind figures or tables yet;
- do not output `asset_registry`, `asset_confirmation_session`, or section-level `asset_bindings`;
- return only `paper_title`, `overall_summary`, and `sections` matching the `TextReaderOutput` schema;
- preserve the same front-matter recovery behavior as the legacy Reader, including explicit `Authors:` / `Affiliations:` recovery and an early-section echo of that metadata.

Legacy Reader contract follows:
""" + READER_SYSTEM_PROMPT

TEXT_READER_USER_PROMPT_TEMPLATE = """Extract the text-only content pack from the paper markdown.

This is the text-only substage of the legacy Reader.
Preserve the same extraction behavior as the original Reader, but skip multimodal asset understanding because a separate visual Reader will handle it after the text pass completes.

### HUMAN_DIRECTIVES
{human_directives}

### PREVIOUS_TEXT_READER_JSON
{previous_text_reader_json}

### ASSETS LIST
{assets_context}

### FULL RAW MARKDOWN
{md_content}
"""

ASSET_VISION_SYSTEM_PROMPT = """You inspect one cropped academic paper asset.

Return strict JSON only:
{
  "caption": "string",
  "visual_summary": "string",
  "is_probably_decorative": boolean
}

Rules:
- Describe what is visibly shown in the asset itself.
- If this looks like a title-page icon, publisher badge, license mark, logo, or other decorative front matter, set is_probably_decorative=true.
- If the provided candidate caption is plausible, you may keep it. Otherwise correct it conservatively.
"""

ASSET_VISION_USER_PROMPT_TEMPLATE = """Inspect this cropped asset.

### ASSET_ID
{asset_id}

### ASSET_TYPE
{asset_type}

### PAGE_NUMBER
{page_number}

### CANDIDATE_CAPTION
{candidate_caption}

### NEARBY_MARKDOWN_EXCERPT
{nearby_markdown_excerpt}
"""

ASSET_BINDING_SYSTEM_PROMPT = """You bind extracted paper assets to already-extracted paper sections.

Return strict JSON only:
{
  "sections": [
    {
      "section_title": "string",
      "asset_bindings": [
        {
          "asset_id": "string",
          "confidence": 0.0,
          "rationale": "string"
        }
      ],
      "review_candidates": [
        {
          "section_title": "string",
          "proposed_asset_id": "string or null",
          "candidate_asset_ids": ["string"],
          "selection_reason": "string",
          "selected_asset_id": null
        }
      ]
    }
  ]
}

Rules:
- Use only asset_id values from ASSET_REGISTRY_JSON.
- Prefer empty asset_bindings over guessing.
- Do not bind decorative assets unless the section is explicitly about front matter or badges.
- Create review_candidates only when the best choice is ambiguous or low confidence.
- Candidate lists should contain 2-4 asset ids when possible.
"""

ASSET_BINDING_USER_PROMPT_TEMPLATE = """Bind assets to sections.

### TEXT_READER_JSON
{text_reader_json}

### ASSET_REGISTRY_JSON
{asset_registry_json}

### HUMAN_DIRECTIVES
{human_directives}
"""

ASSET_BINDING_CHECK_SYSTEM_PROMPT = """You verify whether a selected paper asset matches a section.

Return strict JSON only:
{
  "status": "ok" | "ambiguous" | "wrong",
  "reason": "string",
  "candidate_asset_ids": ["string"]
}

Rules:
- Judge the match between the section meaning and the visible content of the asset.
- Mark decorative icons, badges, and license marks as wrong if used as method/result figures.
- Use candidate_asset_ids only when you can name plausible alternatives from the provided list.
"""

ASSET_BINDING_CHECK_USER_PROMPT_TEMPLATE = """Verify whether this asset matches the section.

### SECTION_TITLE
{section_title}

### SECTION_EXCERPT
{section_excerpt}

### SELECTED_ASSET_ID
{asset_id}

### SELECTED_ASSET_METADATA_JSON
{asset_metadata_json}

### CANDIDATE_ASSET_IDS
{candidate_asset_ids}
"""

LOW_CONFIDENCE_THRESHOLD = 0.78
MAX_ASSET_CANDIDATES = 4


class TextPaperSection(BaseModel):
    section_title: str
    rich_web_content: str


class TextReaderOutput(BaseModel):
    paper_title: str
    overall_summary: str
    sections: list[TextPaperSection] = Field(default_factory=list)


class AssetVisionOutput(BaseModel):
    caption: str = ""
    visual_summary: str = ""
    is_probably_decorative: bool = False


class SectionBindingSuggestion(BaseModel):
    section_title: str
    asset_bindings: list[SectionAssetBinding] = Field(default_factory=list)
    review_candidates: list[AssetConfirmationItem] = Field(default_factory=list)


class AssetBindingPlan(BaseModel):
    sections: list[SectionBindingSuggestion] = Field(default_factory=list)


class AssetBindingCheck(BaseModel):
    status: str = "ok"
    reason: str = ""
    candidate_asset_ids: list[str] = Field(default_factory=list)


def _normalize_structured_paper(value: Any) -> StructuredPaper | None:
    if isinstance(value, StructuredPaper):
        return value
    if value is None:
        return None
    try:
        return StructuredPaper.model_validate(value)
    except Exception:
        return None


def _previous_text_reader_json(previous_structured_paper: Any) -> str:
    paper = _normalize_structured_paper(previous_structured_paper)
    if paper is None:
        return "null"
    try:
        return json.dumps(
            {
                "paper_title": paper.paper_title,
                "overall_summary": paper.overall_summary,
                "sections": [
                    {
                        "section_title": section.section_title,
                        "rich_web_content": section.rich_web_content,
                    }
                    for section in paper.sections
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
    except Exception:
        return "null"


def _asset_sort_key(item: dict[str, Any]) -> tuple[int, int, float, float, str]:
    bbox = item.get("bbox") or [0, 0, 0, 0]
    top = float(bbox[1]) if len(bbox) > 1 and bbox[1] is not None else 0.0
    left = float(bbox[0]) if len(bbox) > 0 and bbox[0] is not None else 0.0
    return (
        int(item.get("page_number") or 0),
        int(item.get("asset_order_on_page") or 0),
        top,
        left,
        str(item.get("image_path") or ""),
    )


def _is_probably_decorative_by_heuristic(asset: dict[str, Any]) -> bool:
    page_number = int(asset.get("page_number") or 0)
    bbox = asset.get("bbox") or []
    if not isinstance(bbox, list) or len(bbox) != 4:
        return False
    width = abs(float(bbox[2]) - float(bbox[0]))
    height = abs(float(bbox[3]) - float(bbox[1]))
    return page_number == 1 and ((width < 240 and height < 140) or (width < 420 and height < 160))


def _caption_candidates_from_markdown(raw_markdown: str) -> dict[str, list[dict[str, str]]]:
    candidates: dict[str, list[dict[str, str]]] = {"figure": [], "table": []}
    pattern = re.compile(r"^(Figure|Table)\s+[A-Za-z0-9.\-]+\.\s*(.+)$", flags=re.MULTILINE)
    for match in pattern.finditer(str(raw_markdown or "")):
        kind = "table" if match.group(1).strip().lower() == "table" else "figure"
        caption = " ".join(match.group(2).split()).strip()
        start = match.start()
        excerpt = raw_markdown[max(0, start - 240) : min(len(raw_markdown), match.end() + 320)].strip()
        candidates[kind].append({"caption": caption, "nearby_markdown_excerpt": excerpt})
    return candidates


def _seed_asset_registry(raw_markdown: str, assets_list: list[dict[str, Any]]) -> list[PaperAsset]:
    ordered_assets = sorted(
        [dict(item) for item in assets_list if isinstance(item, dict)],
        key=_asset_sort_key,
    )
    caption_candidates = _caption_candidates_from_markdown(raw_markdown)
    caption_indexes = {"figure": 0, "table": 0}
    registry: list[PaperAsset] = []

    for item in ordered_assets:
        asset_type = str(item.get("type") or "figure").strip() or "figure"
        asset_kind = "table" if asset_type == "table" else "figure"
        asset_id = str(item.get("asset_id") or "").strip()
        page_number = int(item.get("page_number") or 0)
        asset_order_on_page = int(item.get("asset_order_on_page") or 0)
        decorative = _is_probably_decorative_by_heuristic(item)
        candidate_caption = ""
        nearby_excerpt = ""
        if not decorative:
            idx = caption_indexes[asset_kind]
            if idx < len(caption_candidates[asset_kind]):
                candidate = caption_candidates[asset_kind][idx]
                candidate_caption = str(candidate.get("caption") or "").strip()
                nearby_excerpt = str(candidate.get("nearby_markdown_excerpt") or "").strip()
                caption_indexes[asset_kind] += 1

        registry.append(
            PaperAsset(
                asset_id=asset_id or f"{asset_kind}_p{page_number}_{asset_order_on_page or 1}",
                image_path=str(item.get("image_path") or "").strip(),
                page_image=str(item.get("page_image") or "").strip(),
                page_number=page_number,
                asset_order_on_page=asset_order_on_page,
                bbox=list(item.get("bbox") or []),
                caption=candidate_caption or str(item.get("caption") or "").strip() or None,
                type=asset_type,
                nearby_markdown_excerpt=nearby_excerpt,
                visual_summary="",
                is_probably_decorative=decorative,
            )
        )

    return registry


def _image_payload_for_asset(output_dir: Path, asset: PaperAsset) -> list[dict[str, str]]:
    image_path = (output_dir / str(asset.image_path or "").strip()).resolve()
    return build_human_feedback_payload("", [str(image_path)])["images"] if image_path.exists() else []


def _run_text_reader_extraction(
    raw_markdown: str,
    *,
    assets_list: list[dict[str, Any]] | None,
    human_directives: str,
    previous_structured_paper: Any,
    feedback_history: list[str] | None = None,
) -> TextReaderOutput:
    llm = get_llm(temperature=0.4, use_smart_model=True, thinking_level="high")
    structured_llm = llm.with_structured_output(TextReaderOutput)
    user_msg = TEXT_READER_USER_PROMPT_TEMPLATE.format(
        human_directives=human_directives or "(none)",
        previous_text_reader_json=_previous_text_reader_json(previous_structured_paper),
        assets_context=json.dumps(list(assets_list or []), indent=2, ensure_ascii=False),
        md_content=raw_markdown,
    )
    system_msg = TEXT_READER_SYSTEM_PROMPT
    if feedback_history:
        system_msg += "\n\nPrevious retry feedback:\n" + "\n".join(f"- {item}" for item in feedback_history if str(item).strip())

    response = structured_llm.invoke(
        [
            SystemMessage(content=system_msg),
            HumanMessage(content=user_msg),
        ]
    )
    return TextReaderOutput.model_validate(response)


def _run_asset_vision_pass(output_dir: Path, seeded_registry: list[PaperAsset]) -> list[PaperAsset]:
    updated_registry: list[PaperAsset] = []
    llm = get_llm(temperature=0.1, use_smart_model=True, thinking_level="high")
    structured_llm = llm.with_structured_output(AssetVisionOutput)

    for asset in seeded_registry:
        image_payloads = _image_payload_for_asset(output_dir, asset)
        if not image_payloads:
            updated_registry.append(asset)
            continue
        try:
            response = structured_llm.invoke(
                [
                    SystemMessage(content=ASSET_VISION_SYSTEM_PROMPT),
                    HumanMessage(
                        content=build_multimodal_message_content(
                            text=ASSET_VISION_USER_PROMPT_TEMPLATE.format(
                                asset_id=asset.asset_id,
                                asset_type=asset.type,
                                page_number=asset.page_number,
                                candidate_caption=str(asset.caption or "").strip() or "(none)",
                                nearby_markdown_excerpt=str(asset.nearby_markdown_excerpt or "").strip() or "(none)",
                            ),
                            images=image_payloads,
                        )
                    ),
                ]
            )
            vision = AssetVisionOutput.model_validate(response)
            updated_registry.append(
                asset.model_copy(
                    update={
                        "caption": str(vision.caption or "").strip() or asset.caption,
                        "visual_summary": str(vision.visual_summary or "").strip(),
                        "is_probably_decorative": bool(vision.is_probably_decorative),
                    },
                    deep=True,
                )
            )
        except Exception as exc:
            print(f"[PaperAlchemy-Reader] asset vision pass failed for {asset.asset_id}: {exc}")
            updated_registry.append(asset)

    return updated_registry


def _run_visual_asset_reader(output_dir: Path, seeded_registry: list[PaperAsset]) -> list[PaperAsset]:
    """Visual Reader: inspect cropped assets without changing text extraction behavior."""
    return _run_asset_vision_pass(output_dir, seeded_registry)


def _build_asset_registry_json(asset_registry: list[PaperAsset]) -> str:
    return json.dumps([asset.model_dump() for asset in asset_registry], indent=2, ensure_ascii=False)


def _run_asset_binding_planner(
    text_output: TextReaderOutput,
    asset_registry: list[PaperAsset],
    *,
    human_directives: str,
) -> AssetBindingPlan:
    llm = get_llm(temperature=0.2, use_smart_model=True, thinking_level="high")
    structured_llm = llm.with_structured_output(AssetBindingPlan)
    response = structured_llm.invoke(
        [
            SystemMessage(content=ASSET_BINDING_SYSTEM_PROMPT),
            HumanMessage(
                content=ASSET_BINDING_USER_PROMPT_TEMPLATE.format(
                    text_reader_json=json.dumps(text_output.model_dump(), indent=2, ensure_ascii=False),
                    asset_registry_json=_build_asset_registry_json(asset_registry),
                    human_directives=human_directives or "(none)",
                )
            ),
        ]
    )
    return AssetBindingPlan.model_validate(response)


def _run_visual_asset_binding_reader(
    text_output: TextReaderOutput,
    asset_registry: list[PaperAsset],
    *,
    human_directives: str,
) -> AssetBindingPlan:
    """Visual Reader: bind visually reviewed assets onto the already-extracted text pack."""
    return _run_asset_binding_planner(
        text_output,
        asset_registry,
        human_directives=human_directives,
    )


def _dedupe_confirmation_items(items: list[AssetConfirmationItem]) -> list[AssetConfirmationItem]:
    deduped: list[AssetConfirmationItem] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        proposed = str(item.proposed_asset_id or "").strip()
        key = (str(item.section_title or "").strip(), proposed)
        if key in seen:
            continue
        seen.add(key)
        candidate_asset_ids = [
            asset_id
            for asset_id in dict.fromkeys(
                [
                    *(item.candidate_asset_ids or []),
                    proposed,
                ]
            )
            if str(asset_id or "").strip()
        ][:MAX_ASSET_CANDIDATES]
        deduped.append(
            item.model_copy(
                update={
                    "candidate_asset_ids": candidate_asset_ids,
                    "selected_asset_id": item.selected_asset_id,
                },
                deep=True,
            )
        )
    return deduped


def _merge_reader_outputs(
    text_output: TextReaderOutput,
    asset_registry: list[PaperAsset],
    binding_plan: AssetBindingPlan,
) -> StructuredPaper:
    section_lookup = {item.section_title: item for item in binding_plan.sections}
    confirmation_items: list[AssetConfirmationItem] = []
    sections: list[PaperSection] = []

    for section in text_output.sections:
        suggestion = section_lookup.get(section.section_title)
        asset_bindings = list(suggestion.asset_bindings) if suggestion is not None else []
        if suggestion is not None:
            confirmation_items.extend(list(suggestion.review_candidates))
        for binding in asset_bindings:
            if float(binding.confidence or 0.0) < LOW_CONFIDENCE_THRESHOLD:
                confirmation_items.append(
                    AssetConfirmationItem(
                        section_title=section.section_title,
                        proposed_asset_id=binding.asset_id,
                        candidate_asset_ids=[binding.asset_id],
                        selection_reason=str(binding.rationale or "").strip() or "Low-confidence asset binding.",
                        selected_asset_id=None,
                    )
                )
        sections.append(
            PaperSection(
                section_title=section.section_title,
                rich_web_content=section.rich_web_content,
                asset_bindings=asset_bindings,
            )
        )

    confirmation_items = _dedupe_confirmation_items(confirmation_items)
    confirmation_session = (
        AssetConfirmationSession(items=confirmation_items)
        if confirmation_items
        else None
    )

    return StructuredPaper(
        schema_version=STRUCTURED_PAPER_SCHEMA_VERSION,
        paper_title=text_output.paper_title,
        overall_summary=text_output.overall_summary,
        asset_registry=asset_registry,
        asset_confirmation_session=confirmation_session,
        sections=sections,
    )


def _asset_lookup(structured_paper: StructuredPaper) -> dict[str, PaperAsset]:
    return {asset.asset_id: asset for asset in structured_paper.asset_registry if str(asset.asset_id or "").strip()}


def _candidate_asset_ids_for_section(structured_paper: StructuredPaper, section_title: str, proposed_asset_id: str) -> list[str]:
    asset_ids: list[str] = []
    for section in structured_paper.sections:
        if str(section.section_title or "").strip() != str(section_title or "").strip():
            continue
        asset_ids.extend(str(binding.asset_id or "").strip() for binding in section.asset_bindings if str(binding.asset_id or "").strip())
    if proposed_asset_id and proposed_asset_id not in asset_ids:
        asset_ids.append(proposed_asset_id)
    if len(asset_ids) >= MAX_ASSET_CANDIDATES:
        return asset_ids[:MAX_ASSET_CANDIDATES]

    registry_lookup = _asset_lookup(structured_paper)
    proposed_asset = registry_lookup.get(proposed_asset_id)
    if proposed_asset is not None:
        for asset in structured_paper.asset_registry:
            if asset.asset_id in asset_ids or asset.is_probably_decorative:
                continue
            if asset.type == proposed_asset.type:
                asset_ids.append(asset.asset_id)
            if len(asset_ids) >= MAX_ASSET_CANDIDATES:
                break
    return asset_ids[:MAX_ASSET_CANDIDATES]


def _run_asset_binding_check(
    output_dir: Path,
    structured_paper: StructuredPaper,
) -> StructuredPaper:
    registry_lookup = _asset_lookup(structured_paper)
    pending_items: list[AssetConfirmationItem] = list(
        structured_paper.asset_confirmation_session.items if structured_paper.asset_confirmation_session else []
    )
    llm = get_llm(temperature=0, use_smart_model=False, thinking_level="high")
    structured_llm = llm.with_structured_output(AssetBindingCheck)

    for section in structured_paper.sections:
        section_excerpt = " ".join(str(section.rich_web_content or "").split())[:900]
        for binding in section.asset_bindings:
            asset_id = str(binding.asset_id or "").strip()
            asset = registry_lookup.get(asset_id)
            if asset is None:
                continue
            image_payloads = _image_payload_for_asset(output_dir, asset)
            if not image_payloads:
                continue
            candidate_asset_ids = _candidate_asset_ids_for_section(structured_paper, section.section_title, asset_id)
            try:
                response = structured_llm.invoke(
                    [
                        SystemMessage(content=ASSET_BINDING_CHECK_SYSTEM_PROMPT),
                        HumanMessage(
                            content=build_multimodal_message_content(
                                text=ASSET_BINDING_CHECK_USER_PROMPT_TEMPLATE.format(
                                    section_title=section.section_title,
                                    section_excerpt=section_excerpt or "(none)",
                                    asset_id=asset_id,
                                    asset_metadata_json=json.dumps(asset.model_dump(), indent=2, ensure_ascii=False),
                                    candidate_asset_ids=json.dumps(candidate_asset_ids, ensure_ascii=False),
                                ),
                                images=image_payloads,
                            )
                        ),
                    ]
                )
                check = AssetBindingCheck.model_validate(response)
            except Exception as exc:
                print(f"[PaperAlchemy-Reader] asset binding critic failed for {asset_id}: {exc}")
                continue

            if (
                str(check.status or "").strip().lower() in {"ambiguous", "wrong"}
                or asset.is_probably_decorative
                or float(binding.confidence or 0.0) < LOW_CONFIDENCE_THRESHOLD
            ):
                pending_items.append(
                    AssetConfirmationItem(
                        section_title=section.section_title,
                        proposed_asset_id=asset_id,
                        candidate_asset_ids=[
                            asset_candidate
                            for asset_candidate in dict.fromkeys(
                                [*(check.candidate_asset_ids or []), *candidate_asset_ids]
                            )
                            if str(asset_candidate or "").strip()
                        ][:MAX_ASSET_CANDIDATES],
                        selection_reason=str(check.reason or "").strip()
                        or str(binding.rationale or "").strip()
                        or "Asset binding requires confirmation.",
                        selected_asset_id=None,
                    )
                )

    pending_items = _dedupe_confirmation_items(pending_items)
    return structured_paper.model_copy(
        update={
            "asset_confirmation_session": (
                AssetConfirmationSession(items=pending_items)
                if pending_items
                else None
            )
        },
        deep=True,
    )


def reader_node(state: ReaderState):
    print(f"[PaperAlchemy-Reader]Gemini 正在阅读全文提取结构 (第 {state.get('retry_count', 0)} 次尝试)...")
    raw_markdown = str(state.get("raw_markdown") or "")
    human_directives = extract_human_feedback_text(state.get("human_directives"))
    previous_structured_paper = state.get("previous_structured_paper")
    assets_list = state.get("assets_list")
    if not isinstance(assets_list, list):
        assets_list = []

    output_dir = Path(raw_markdown and state.get("paper_output_dir") or "")
    try:
        text_output = _run_text_reader_extraction(
            raw_markdown,
            assets_list=assets_list,
            human_directives=human_directives or "",
            previous_structured_paper=previous_structured_paper,
            feedback_history=list(state.get("feedback_history") or []),
        )
        seeded_registry = _seed_asset_registry(raw_markdown, assets_list)
        if not output_dir:
            paper_folder_name = str(state.get("paper_folder_name") or "").strip()
            if paper_folder_name:
                output_dir = Path(__file__).resolve().parents[2] / "data" / "output" / paper_folder_name
        asset_registry = _run_visual_asset_reader(output_dir, seeded_registry) if output_dir else seeded_registry
        binding_plan = _run_visual_asset_binding_reader(
            text_output,
            asset_registry,
            human_directives=human_directives or "",
        )
        structured_paper = _merge_reader_outputs(text_output, asset_registry, binding_plan)
        structured_paper = _run_asset_binding_check(output_dir, structured_paper) if output_dir else structured_paper
        return {
            "text_structured_paper": text_output.model_dump(),
            "asset_registry": [asset.model_dump() for asset in asset_registry],
            "structured_paper": structured_paper,
        }
    except Exception as e:
        print(f"[PaperAlchemy-Reader] Error: {e}")
        return {}


def build_reader_graph(max_retry: int = 3):
    workflow = StateGraph(ReaderState)
    workflow.add_node("reader", reader_node)
    workflow.add_node("critic", critic_node)

    workflow.set_entry_point("reader")
    workflow.add_edge("reader", "critic")

    workflow.add_conditional_edges(
        "critic",
        build_critic_router(max_retry=max_retry),
        {
            "retry": "reader",
            "end": END,
        },
    )

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


def _load_reader_inputs(output_dir: Path) -> tuple[str, list[dict]]:
    with open(output_dir / "full_paper.md", "r", encoding="utf-8") as f:
        raw_md = f.read()

    with open(output_dir / "parsed_data.json", "r", encoding="utf-8") as f:
        full_json = json.load(f)

    asset_registry = full_json.get("asset_registry")
    if isinstance(asset_registry, list) and asset_registry:
        return raw_md, [dict(item) for item in asset_registry if isinstance(item, dict)]

    assets: list[dict] = []
    for page in full_json.get("pages", []):
        page_number = page.get("page_number")
        page_image = page.get("page_image")
        page_items: list[dict] = []
        for item in list(page.get("figures", [])) + list(page.get("tables", [])):
            if isinstance(item, dict):
                page_items.append(dict(item))

        page_items.sort(key=_asset_sort_key)

        for asset_order_on_page, item in enumerate(page_items, start=1):
            asset_type = "table" if str(item.get("type") or "").strip() == "table" else "figure"
            assets.append(
                {
                    **item,
                    "page_number": page_number,
                    "page_image": page_image,
                    "asset_order_on_page": asset_order_on_page,
                    "asset_id": str(item.get("asset_id") or f"{asset_type}_p{page_number}_{asset_order_on_page}"),
                    "nearby_markdown_excerpt": str(item.get("nearby_markdown_excerpt") or ""),
                    "visual_summary": str(item.get("visual_summary") or ""),
                    "is_probably_decorative": bool(item.get("is_probably_decorative")),
                }
            )

    return raw_md, assets


def run_reader_agent(
    paper_folder_name: str,
    human_directives: str | dict = "",
    previous_structured_paper: StructuredPaper | None = None,
    max_retry: int = 3,
):
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[2]
    output_dir = project_root / "data" / "output" / paper_folder_name

    print(f"[PaperAlchemy]启动 Reader Agent，读取数据: {output_dir}")

    try:
        raw_md, assets = _load_reader_inputs(output_dir)
    except FileNotFoundError:
        print("[PaperAlchemy-Reader]🤡错误：找不到解析数据。请确保 parser.py 已运行🤡")
        return None

    app = build_reader_graph(max_retry=max_retry)
    thread = {"configurable": {"thread_id": "main_session_auto"}}

    initial_state: ReaderState = {
        "raw_markdown": raw_md,
        "assets_list": assets,
        "human_directives": normalize_human_feedback(human_directives),
        "previous_structured_paper": previous_structured_paper,
        "feedback_history": [],
        "text_structured_paper": None,
        "text_reader_feedback": "",
        "asset_registry": [],
        "asset_binding_candidates": {},
        "asset_binding_feedback": "",
        "critic_passed": False,
        "retry_count": 0,
        "structured_paper": None,
        "paper_folder_name": paper_folder_name,
        "paper_output_dir": str(output_dir),
    }

    print("[PaperAlchemy-System]正在全自动执行信息提取与 Critic 自查流水线...")
    for _ in app.stream(initial_state, thread):
        pass

    final_state = app.get_state(thread)
    structured_result = final_state.values.get("structured_paper")
    structured_result = _normalize_structured_paper(structured_result)

    if not structured_result or not final_state.values.get("critic_passed"):
        print("\n[PaperAlchemy-System]🤡提取流程最终异常或未完美通过 Critic 校验🤡")
    else:
        print("\n" + "=" * 50)
        print(f"[PaperAlchemy-System]自动化提取大成功：{structured_result.paper_title}")
        print(f"[PaperAlchemy-System]共拆解 {len(structured_result.sections)} 个结构化章节。")
        for idx, sec in enumerate(structured_result.sections, start=1):
            print(f"   {idx}. {sec.section_title[:30]}... (具有图表引用数: {len(sec.asset_bindings)})")
        print("=" * 50 + "\n")

    return structured_result


if __name__ == "__main__":
    run_reader_agent("All You Need is DAG")
