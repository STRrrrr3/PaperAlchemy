from __future__ import annotations

import json
import re
import ast
import queue
import threading
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.benchmark_v1.core import CANDIDATE_KEEP_RANGE, TASK_CATALOG, normalize_token, valid_subcategories
from src.services.artifact_store import get_output_paths
from src.services.llm import get_llm
from src.utils.html_utils import read_text_with_fallback

DIFFICULTY_QUOTA = {
    "easy": 2,
    "medium": 3,
    "hard": 3,
}
DIFFICULTY_BY_CATEGORY = {
    "content_multimodal": ("easy", "hard"),
    "styling_aesthetics": ("easy", "medium"),
    "layout_rhythm": ("medium", "hard"),
    "interactivity_functional": ("medium", "hard"),
}

CATALOG_CAPABILITIES = {
    "asset_replacement": (
        "Use an existing figure, table image, logo, or visual asset target and a provided benchmark asset. "
        "The task should be about exact replacement and preserving local layout constraints."
    ),
    "asset_adjustment": (
        "Adjust sizing, grouping, alignment, or placement of existing visual assets such as figures, tables, or cards."
    ),
    "text_compaction_expansion": (
        "Condense, expand, restructure, or clarify a concrete text region while preserving the paper's meaning."
    ),
    "color_contrast": (
        "Change local color, contrast, background, emphasis, or readability for a concrete page region."
    ),
    "typography": (
        "Tune typeface, size, weight, hierarchy, line height, or heading/body text treatment for a concrete text group."
    ),
    "visual_focus": (
        "Add or revise emphasis around an existing region so the most important paper content is easier to notice."
    ),
    "structural_transform": (
        "Convert an existing content region into a clearer structure such as columns, cards, grouped blocks, or lists."
    ),
    "rhythm_spacing": (
        "Improve spacing, section rhythm, crowding, or whitespace around concrete neighboring elements."
    ),
    "alignment_fix": (
        "Fix alignment or centering problems among related text, figures, tables, formulas, captions, or metadata."
    ),
    "hyperlink_anchor": (
        "Create or repair a link or anchor only when the baseline page facts show a concrete eligible target."
    ),
    "navigation": (
        "Improve an existing navigation or section-jump behavior, or add one only when the page structure supports it."
    ),
    "collapsible": (
        "Turn a concrete long or secondary content region into a collapsed-by-default interaction with a clear summary."
    ),
}


class BenchmarkCandidateCase(BaseModel):
    case_id: str = Field(description="Unique short identifier.")
    task_type: str = Field(description="Benchmark task_type category.")
    subcategory: str = Field(description="Chosen subcategory under task_type.")
    category_label_zh: str = ""
    subcategory_label_zh: str = ""
    instruction: str = Field(description="Actionable revision instruction in English.")
    instruction_zh: str = Field(description="Same instruction in Chinese.")
    target_hint: str = Field(description="Concrete baseline target region in English.")
    target_hint_zh: str = Field(description="Concrete baseline target region in Chinese.")
    expected_observable: str = Field(description="Observable completion signal in English.")
    expected_observable_zh: str = Field(description="Observable completion signal in Chinese.")
    difficulty: str = Field(description="One of easy, medium, or hard.")
    difficulty_reason: str = Field(description="Why this case has the selected difficulty.")
    pdf_evidence: str = Field(description="Concrete PDF-derived paper evidence grounding the case.")
    web_evidence: str = Field(description="Concrete baseline webpage evidence grounding the case.")
    forbidden_changes: list[str] = Field(default_factory=list)
    target_selectors: list[str] = Field(default_factory=list)
    notes: str = Field(description="Feasibility note naming concrete baseline evidence.")


class BenchmarkCandidateResponse(BaseModel):
    cases: list[BenchmarkCandidateCase] = Field(description="Exactly eight Benchmark V1 candidate cases.")


BENCHMARK_CANDIDATE_SYSTEM_PROMPT = """You design page-grounded Benchmark V1 revision cases.
Return only cases that are grounded in both PDF-derived paper evidence and baseline page facts.
Never invent missing controls or ask to add a feature that already exists."""


def _render_catalog_for_prompt() -> str:
    blocks: list[str] = []
    for task_type, config in TASK_CATALOG.items():
        label = str(config.get("label") or "")
        label_zh = str(config.get("label_zh") or "")
        count = int(config.get("candidate_count") or 0)
        difficulties = ", ".join(DIFFICULTY_BY_CATEGORY.get(task_type, ()))
        lines = [
            f"[{task_type}] {label} / {label_zh} -- produce exactly {count} PDF-grounded candidates "
            f"with difficulties: {difficulties}"
        ]
        for entry in config.get("subcategories") or []:
            name = str(entry["name"])
            capability = CATALOG_CAPABILITIES.get(name, "Design a local, verifiable page edit in this subcategory.")
            lines.append(
                f"  - {entry['name']} ({entry.get('label')} / {entry.get('label_zh')}): "
                f"{capability}"
            )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def generate_instruction_candidates(
    *,
    paper_folder_name: str,
    baseline_entry_html: str,
    max_html_chars: int = 22000,
    llm_timeout_seconds: float = 180.0,
) -> list[dict[str, Any]]:
    output_dir, structured_json_path, _, _ = get_output_paths(paper_folder_name)
    structured_summary = "{}"
    if structured_json_path.exists():
        structured_summary = structured_json_path.read_text(encoding="utf-8")[:12000]
    html_text = read_text_with_fallback(Path(baseline_entry_html))
    html_summary = html_text[:max_html_chars]
    page_facts = _build_baseline_page_facts(html_text)

    total_count = sum(int(cfg.get("candidate_count") or 0) for cfg in TASK_CATALOG.values())
    keep_lo, keep_hi = CANDIDATE_KEEP_RANGE
    valid_task_types = ", ".join(TASK_CATALOG.keys())

    prompt = f"""You are designing a revision benchmark for PaperAlchemy Benchmark V1.

Produce a JSON array of candidate revision cases. Follow the Bench Design
catalog below exactly: four task_type categories, three subcategory capability
areas in each category, but produce only two feasible candidates per category,
{total_count} candidates total.
The human evaluator will keep all {keep_lo}-{keep_hi} candidates.

=== Task Catalog ===
{_render_catalog_for_prompt()}

=== Output Schema ===
Return a structured object with a `cases` array. Each case element:
{{
  "case_id": "unique short identifier, e.g. c1_hero_replace",
  "task_type": "one of: {valid_task_types}",
  "subcategory": "one of the subcategory names listed under the chosen task_type",
  "category_label_zh": "Chinese category label from the catalog",
  "subcategory_label_zh": "Chinese subcategory label from the catalog",
  "instruction": "actionable revision instruction in English for the benchmark actor",
  "instruction_zh": "same instruction in Chinese for the human evaluator",
  "target_hint": "human-readable target region in English",
  "target_hint_zh": "target region in Chinese",
  "expected_observable": "observable completion signal in English",
  "expected_observable_zh": "observable completion signal in Chinese",
  "difficulty": "one of: easy, medium, hard",
  "difficulty_reason": "short reason for the selected difficulty",
  "pdf_evidence": "specific paper evidence from the PDF-derived structured excerpt, such as section text, figure/table caption, key claim, theorem, or result",
  "web_evidence": "specific current baseline webpage evidence, such as visible section, figure, caption, card, link, or control",
  "forbidden_changes": ["concrete changes the actor must avoid, e.g. do not rewrite unrelated sections"],
  "target_selectors": ["optional anonymous class selectors from the baseline, e.g. .paexp-kXXXXXXXXXX"],
  "notes": "optional notes"
}}

=== Rules ===
- Each candidate must be independently executable from the ORIGINAL baseline page. No chaining to earlier edits.
- Revisions must be local, verifiable, and non-destructive.
- Start from PDF-derived paper evidence and baseline page evidence together, then map the discovered opportunity to the most fitting category, subcategory, and difficulty.
- Do not start from catalog examples or reuse catalog wording patterns; the catalog defines capability areas, not task templates.
- Each task must be specific to this paper webpage. A task that could apply unchanged to most papers is too generic.
- Select the two most feasible subcategories in each category based on the actual baseline page.
- Difficulty quota is mandatory across the eight cases: exactly 2 easy, 3 medium, and 3 hard.
- Category-difficulty allocation is mandatory: content_multimodal has one easy and one hard; styling_aesthetics has one easy and one medium; layout_rhythm has one medium and one hard; interactivity_functional has one medium and one hard.
- Easy means one local target with a screenshot-observable change and no cross-section dependency.
- Medium means one target plus explicit constraints, requiring PDF-derived semantic evidence or local CSS/JS while preserving nearby context.
- Hard means PDF-grounded semantic understanding, cross-region consistency, asset/caption/reference/anchor consistency, or real interaction.
- Do not create a task for a missing target. For example, do not ask to edit a PDF Download button unless the baseline page actually has one.
- Do not ask to add a feature that already exists. If the page already has a TOC/navigation aid, either improve a specific existing navigation behavior or choose another feasible interaction task.
- Prefer tasks that point to concrete existing text, figures, tables, cards, links, or controls visible in the baseline facts.
- Prefer diverse task shapes across papers: vary targets, operations, and evidence rather than repeatedly asking for the same abstract/footer/heading/TOC-style edits.
- Prefer target_selectors referencing opaque classes (paexp-c... or paexp-k...) observable in the baseline HTML.
- Do not produce duplicate case_ids.
- Do not group multiple candidates under one object. Every array item must be one executable case.
- Do not put multiple numbered or bulleted instructions inside a single "instruction" field.
- The final JSON array must contain exactly {total_count} items.
- Every task_type category must appear exactly two times.
- Within the same task_type, do not reuse the same subcategory unless the page offers two clearly different concrete targets and the notes explain why.
- Every case must include non-empty pdf_evidence and web_evidence.
- Every case should include forbidden_changes that constrain unrelated edits, semantic drift, broken anchors, external dependencies, or style pollution.
- Always provide Chinese fields (`instruction_zh`, `target_hint_zh`, `expected_observable_zh`) so the evaluator can read the task directly.
- Each notes field must include a short feasibility note naming the concrete baseline evidence used to create the task.
- Each notes field must also explain why this task is tailored to this particular paper and baseline page, not merely copied from the catalog.

=== Paper Context ===
Paper folder: {paper_folder_name}
Output dir: {output_dir}

Structured paper excerpt (truncated):
{structured_summary}

Baseline page facts:
{json.dumps(page_facts, indent=2, ensure_ascii=False)}

Baseline HTML excerpt (truncated):
{html_summary}
"""
    payload = _invoke_candidate_llm_structured_with_hard_timeout(prompt, timeout_seconds=llm_timeout_seconds)
    cases = normalize_candidate_cases(payload)
    if _has_required_candidate_coverage(cases):
        return cases
    expected_by_category = {task_type: int(config.get("candidate_count") or 0) for task_type, config in TASK_CATALOG.items()}
    observed_by_category: dict[str, int] = {task_type: 0 for task_type in TASK_CATALOG}
    observed_by_difficulty: dict[str, int] = {difficulty: 0 for difficulty in DIFFICULTY_QUOTA}
    for case in cases:
        task_type = str(case.get("task_type") or "")
        if task_type in observed_by_category:
            observed_by_category[task_type] += 1
        difficulty = str(case.get("difficulty") or "")
        if difficulty in observed_by_difficulty:
            observed_by_difficulty[difficulty] += 1
    raise ValueError(
        f"Instruction generator returned {len(cases)} usable case(s), "
        f"but Benchmark V1 requires {total_count} total with category counts "
        f"{expected_by_category} and difficulty counts {DIFFICULTY_QUOTA}. "
        f"Observed category counts: {observed_by_category}. "
        f"Observed difficulty counts: {observed_by_difficulty}."
    )


def _build_baseline_page_facts(html_text: str) -> dict[str, Any]:
    soup = BeautifulSoup(str(html_text or ""), "html.parser")
    headings = []
    for heading in soup.find_all(["h1", "h2", "h3", "h4"]):
        text = heading.get_text(" ", strip=True)
        if not text:
            continue
        headings.append(
            {
                "tag": heading.name,
                "text": text[:160],
                "classes": heading.get("class") or [],
                "id": heading.get("id") or "",
            }
        )
        if len(headings) >= 40:
            break

    links = []
    pdf_or_paper_links = []
    anchor_links = []
    for link in soup.find_all("a"):
        text = link.get_text(" ", strip=True)
        href = str(link.get("href") or "").strip()
        if not text and not href:
            continue
        item = {"text": text[:120], "href": href[:240], "classes": link.get("class") or []}
        if len(links) < 30:
            links.append(item)
        link_blob = f"{text} {href}".lower()
        if any(token in link_blob for token in ("pdf", "arxiv", "doi", "paper", "download")):
            pdf_or_paper_links.append(item)
        if href.startswith("#"):
            anchor_links.append(item)

    nav_like = []
    for node in soup.find_all(["nav", "aside"]):
        text = node.get_text(" ", strip=True)
        nav_like.append({"tag": node.name, "text": text[:200], "classes": node.get("class") or [], "id": node.get("id") or ""})
    for node in soup.find_all(attrs={"class": True}):
        classes = " ".join(str(item) for item in node.get("class") or [])
        node_id = str(node.get("id") or "")
        blob = f"{classes} {node_id}".lower()
        if not any(token in blob for token in ("toc", "nav", "menu", "sidebar")):
            continue
        text = node.get_text(" ", strip=True)
        if text:
            nav_like.append({"tag": node.name, "text": text[:200], "classes": node.get("class") or [], "id": node_id})
        if len(nav_like) >= 12:
            break

    media = []
    for node in soup.find_all(["figure", "img", "table"]):
        item: dict[str, Any] = {"tag": node.name}
        if node.name == "img":
            item["alt"] = str(node.get("alt") or "")[:160]
            item["src"] = str(node.get("src") or "")[:240]
        else:
            item["text"] = node.get_text(" ", strip=True)[:240]
        item["classes"] = node.get("class") or []
        media.append(item)
        if len(media) >= 24:
            break

    interactive = []
    for node in soup.find_all(["button", "details", "summary", "input", "select"]):
        interactive.append(
            {
                "tag": node.name,
                "text": node.get_text(" ", strip=True)[:160],
                "classes": node.get("class") or [],
                "id": node.get("id") or "",
            }
        )
        if len(interactive) >= 24:
            break

    sections = []
    for heading in soup.find_all(["h2", "h3"]):
        title = heading.get_text(" ", strip=True)
        if not title:
            continue
        texts: list[str] = []
        for sibling in heading.find_all_next():
            if sibling.name in {"h1", "h2"}:
                break
            if sibling.name in {"p", "li", "figcaption"}:
                snippet = sibling.get_text(" ", strip=True)
                if snippet:
                    texts.append(snippet)
            if len(" ".join(texts)) > 500:
                break
        sections.append({"heading": title[:160], "text_excerpt": " ".join(texts)[:600]})
        if len(sections) >= 12:
            break

    return {
        "headings": headings,
        "links_sample": links,
        "pdf_or_paper_links": pdf_or_paper_links[:12],
        "anchor_link_count": len(anchor_links),
        "has_pdf_download_control": any(
            "pdf" in f"{item.get('text', '')} {item.get('href', '')}".lower()
            and "download" in f"{item.get('text', '')} {item.get('href', '')}".lower()
            for item in links
        ),
        "has_toc_or_navigation": bool(nav_like or len(anchor_links) >= 4),
        "nav_like_elements": nav_like[:12],
        "media_sample": media,
        "counts": {
            "images": len(soup.find_all("img")),
            "figures": len(soup.find_all("figure")),
            "tables": len(soup.find_all("table")),
            "buttons": len(soup.find_all("button")),
            "details": len(soup.find_all("details")),
            "anchor_links": len(anchor_links),
        },
        "interactive_elements": interactive,
        "section_text_samples": sections,
        "opaque_selectors": _extract_opaque_selectors(str(html_text or ""), limit=30),
    }


def _invoke_candidate_llm_structured_with_hard_timeout(
    prompt: str,
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    timeout = max(0.1, float(timeout_seconds or 0.1))
    result_queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)

    def worker() -> None:
        try:
            llm = get_llm(
                temperature=1.0,
                use_smart_model=True,
                request_timeout=timeout,
                retries=0,
                streaming=False,
            )
            structured_llm = llm.with_structured_output(BenchmarkCandidateResponse)
            response = structured_llm.invoke(
                [
                    SystemMessage(content=BENCHMARK_CANDIDATE_SYSTEM_PROMPT),
                    HumanMessage(content=prompt),
                ]
            )
            result = BenchmarkCandidateResponse.model_validate(response)
            result_queue.put(("ok", result.model_dump()))
        except BaseException as exc:
            result_queue.put(("error", exc))

    thread = threading.Thread(target=worker, name="benchmark-v1-candidate-structured-llm", daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise TimeoutError(f"LLM candidate generation exceeded {timeout:g} seconds.")
    try:
        status, payload = result_queue.get_nowait()
    except queue.Empty as exc:
        raise TimeoutError("LLM candidate generation ended without returning a response.") from exc
    if status == "error":
        raise payload
    return dict(payload or {})


def _invoke_candidate_llm_with_hard_timeout(prompt: str, *, timeout_seconds: float) -> str:
    timeout = max(0.1, float(timeout_seconds or 0.1))
    result_queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)

    def worker() -> None:
        try:
            llm = get_llm(
                temperature=0.2,
                use_smart_model=True,
                request_timeout=timeout,
                retries=0,
                streaming=False,
            )
            response = llm.invoke(prompt)
            result_queue.put(("ok", str(getattr(response, "content", response) or "")))
        except BaseException as exc:
            result_queue.put(("error", exc))

    thread = threading.Thread(target=worker, name="benchmark-v1-candidate-llm", daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise TimeoutError(f"LLM candidate generation exceeded {timeout:g} seconds.")
    try:
        status, payload = result_queue.get_nowait()
    except queue.Empty as exc:
        raise TimeoutError("LLM candidate generation ended without returning a response.") from exc
    if status == "error":
        raise payload
    return str(payload or "")


def parse_candidate_payload(text: str) -> Any:
    clean = str(text or "").strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```(?:json|python)?", "", clean, flags=re.IGNORECASE).strip()
        clean = re.sub(r"```$", "", clean).strip()
    errors: list[str] = []
    candidates = [clean]
    match = re.search(r"(\[[\s\S]*\])", clean)
    if match:
        candidates.append(match.group(1).strip())
    object_match = re.search(r"(\{[\s\S]*\})", clean)
    if object_match:
        candidates.append(object_match.group(1).strip())
    candidates.extend(_quote_unquoted_object_keys(candidate) for candidate in list(candidates))

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            errors.append(str(exc))
        try:
            return ast.literal_eval(candidate)
        except (SyntaxError, ValueError) as exc:
            errors.append(str(exc))

    raise ValueError(
        "Candidate / Cases JSON must be a JSON array. Use double-quoted JSON, "
        'for example [{"case_id": "c1", "task_type": "layout_rhythm", ...}]. '
        f"Parser errors: {' | '.join(errors[:3])}"
    )


def normalize_candidate_cases(payload: Any) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []
    _collect_candidate_cases(payload, {}, collected)
    normalized: list[dict[str, Any]] = []
    for index, raw_case in enumerate(collected, start=1):
        case = _normalize_case(raw_case, index)
        if case:
            normalized.append(case)
    return _dedupe_case_ids(normalized)


def _quote_unquoted_object_keys(text: str) -> str:
    return re.sub(
        r'([{,]\s*)([A-Za-z_][A-Za-z0-9_-]*)(\s*:)',
        r'\1"\2"\3',
        str(text or ""),
    )


def _extract_opaque_selectors(html_text: str, limit: int = 8) -> list[str]:
    selectors: list[str] = []
    seen: set[str] = set()
    for class_name in re.findall(r"\bpaexp-[ck][0-9a-f]{10}\b", html_text):
        if class_name in seen:
            continue
        seen.add(class_name)
        selectors.append(f".{class_name}")
        if len(selectors) >= limit:
            break
    return selectors


def _collect_candidate_cases(value: Any, context: dict[str, Any], collected: list[dict[str, Any]]) -> None:
    if isinstance(value, list):
        for item in value:
            _collect_candidate_cases(item, context, collected)
        return
    if isinstance(value, str):
        for instruction in _split_instruction_text(value):
            collected.append({**context, "instruction": instruction})
        return
    if not isinstance(value, dict):
        return

    local_context = _candidate_context(value, context)
    nested_keys = (
        "cases",
        "candidate_cases",
        "candidates",
        "items",
        "results",
        "revisions",
        "revision_cases",
    )
    nested_found = False
    for key in nested_keys:
        if key in value:
            nested_found = True
            _collect_candidate_cases(value.get(key), local_context, collected)

    for task_type in TASK_CATALOG:
        if task_type in value:
            nested_found = True
            _collect_candidate_cases(value.get(task_type), {**local_context, "task_type": task_type}, collected)

    instruction_value = _instruction_value(value)
    if instruction_value is not None:
        nested_found = True
        _collect_instruction_value(value, instruction_value, local_context, collected)

    if nested_found:
        return

    for key, child in value.items():
        if isinstance(child, (list, dict)):
            child_context = local_context
            if str(key) in TASK_CATALOG:
                child_context = {**local_context, "task_type": str(key)}
            _collect_candidate_cases(child, child_context, collected)


def _collect_instruction_value(
    source: dict[str, Any],
    instruction_value: Any,
    context: dict[str, Any],
    collected: list[dict[str, Any]],
) -> None:
    base = {
        **context,
        **{
            key: value
            for key, value in source.items()
            if key
            not in {
                "instruction",
                "instructions",
                "revision_instruction",
                "request",
                "prompt",
                "description",
            }
        },
    }
    if isinstance(instruction_value, list):
        for item in instruction_value:
            if isinstance(item, dict):
                _collect_candidate_cases(item, base, collected)
            else:
                for instruction in _split_instruction_text(str(item or "")):
                    collected.append({**base, "instruction": instruction})
        return
    if isinstance(instruction_value, dict):
        _collect_candidate_cases(instruction_value, base, collected)
        return
    for instruction in _split_instruction_text(str(instruction_value or "")):
        collected.append({**base, "instruction": instruction})


def _candidate_context(value: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    result = dict(context)
    for key in (
        "task_type",
        "subcategory",
        "category_label_zh",
        "subcategory_label_zh",
        "target_hint",
        "target_hint_zh",
        "expected_observable",
        "expected_observable_zh",
        "difficulty",
        "difficulty_reason",
        "pdf_evidence",
        "web_evidence",
        "forbidden_changes",
        "target_selectors",
        "notes",
    ):
        if key in value and value.get(key) not in (None, ""):
            result[key] = value.get(key)
    return result


def _instruction_value(value: dict[str, Any]) -> Any | None:
    for key in ("instruction", "instructions", "revision_instruction", "request", "prompt", "description"):
        if key in value and value.get(key) not in (None, ""):
            return value.get(key)
    return None


def _split_instruction_text(text: str) -> list[str]:
    clean = str(text or "").strip()
    if not clean:
        return []
    marker = re.compile(r"^\s*(?:[-*•]\s+|\d+[\).、]\s+|case\s*\d+[\).:：]\s*)", re.IGNORECASE)
    parts: list[str] = []
    current: list[str] = []
    for line in clean.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = marker.match(stripped)
        if match:
            if current:
                parts.append(" ".join(current).strip())
            current = [stripped[match.end() :].strip()]
        elif current:
            current.append(stripped)
    if current:
        parts.append(" ".join(current).strip())
    return [part for part in parts if part] if len(parts) > 1 else [clean]


def _normalize_case(raw_case: dict[str, Any], index: int) -> dict[str, Any]:
    case = dict(raw_case)
    instruction = str(case.get("instruction") or "").strip()
    if not instruction:
        return {}
    task_type = _normalize_task_type(case.get("task_type"), case.get("subcategory"))
    subcategory = _normalize_subcategory(task_type, case.get("subcategory"))
    category_label_zh, subcategory_label_zh = _catalog_labels(task_type, subcategory)
    case["case_id"] = normalize_token(str(case.get("case_id") or ""), fallback=f"c{index}_{task_type}")[:80]
    case["task_type"] = task_type
    case["subcategory"] = subcategory
    case["category_label_zh"] = str(case.get("category_label_zh") or category_label_zh).strip()
    case["subcategory_label_zh"] = str(case.get("subcategory_label_zh") or subcategory_label_zh).strip()
    case["instruction"] = instruction
    case["instruction_zh"] = str(case.get("instruction_zh") or instruction).strip()
    case["target_hint"] = str(case.get("target_hint") or "Generated from the candidate instruction.").strip()
    case["target_hint_zh"] = str(case.get("target_hint_zh") or case["target_hint"]).strip()
    case["expected_observable"] = str(
        case.get("expected_observable") or "The requested revision is visibly reflected in the final page."
    ).strip()
    case["expected_observable_zh"] = str(case.get("expected_observable_zh") or case["expected_observable"]).strip()
    case["difficulty"] = _normalize_difficulty(case.get("difficulty"))
    case["difficulty_reason"] = str(case.get("difficulty_reason") or "").strip()
    case["pdf_evidence"] = str(case.get("pdf_evidence") or "").strip()
    case["web_evidence"] = str(case.get("web_evidence") or "").strip()
    forbidden_changes = case.get("forbidden_changes")
    if isinstance(forbidden_changes, list):
        case["forbidden_changes"] = [str(item).strip() for item in forbidden_changes if str(item or "").strip()]
    elif str(forbidden_changes or "").strip():
        case["forbidden_changes"] = [str(forbidden_changes).strip()]
    else:
        case["forbidden_changes"] = []
    selectors = case.get("target_selectors")
    case["target_selectors"] = selectors if isinstance(selectors, list) else []
    return case


def _normalize_difficulty(value: Any) -> str:
    normalized = normalize_token(str(value or "").strip().lower(), fallback="")
    return normalized if normalized in DIFFICULTY_QUOTA else ""


def _normalize_task_type(value: Any, subcategory: Any) -> str:
    raw = str(value or "").strip()
    if raw in TASK_CATALOG:
        return raw
    normalized = normalize_token(raw.lower().replace(" ", "_"), fallback="")
    if normalized in TASK_CATALOG:
        return normalized
    raw_subcategory = str(subcategory or "").strip()
    for task_type in TASK_CATALOG:
        if raw_subcategory in valid_subcategories(task_type):
            return task_type
    return "layout_rhythm"


def _normalize_subcategory(task_type: str, value: Any) -> str:
    raw = str(value or "").strip()
    if raw in valid_subcategories(task_type):
        return raw
    return ""


def _catalog_labels(task_type: str, subcategory: str) -> tuple[str, str]:
    task_config = TASK_CATALOG.get(task_type) or {}
    category_label_zh = str(task_config.get("label_zh") or task_config.get("label") or task_type)
    subcategory_label_zh = str(subcategory or "")
    for entry in task_config.get("subcategories") or []:
        if str(entry.get("name") or "") == subcategory:
            subcategory_label_zh = str(entry.get("label_zh") or entry.get("label") or subcategory)
            break
    return category_label_zh, subcategory_label_zh


def _has_required_candidate_coverage(cases: list[dict[str, Any]]) -> bool:
    expected_total = sum(int(config.get("candidate_count") or 0) for config in TASK_CATALOG.values())
    if len(cases) != expected_total:
        return False

    counts: dict[str, int] = {task_type: 0 for task_type in TASK_CATALOG}
    difficulty_counts: dict[str, int] = {difficulty: 0 for difficulty in DIFFICULTY_QUOTA}
    category_difficulties: dict[str, list[str]] = {task_type: [] for task_type in TASK_CATALOG}
    seen_pairs: set[tuple[str, str]] = set()
    for case in cases:
        task_type = str(case.get("task_type") or "")
        subcategory = str(case.get("subcategory") or "")
        difficulty = str(case.get("difficulty") or "")
        if task_type not in TASK_CATALOG:
            return False
        if subcategory not in valid_subcategories(task_type):
            return False
        if difficulty not in DIFFICULTY_QUOTA:
            return False
        if not str(case.get("pdf_evidence") or "").strip():
            return False
        if not str(case.get("web_evidence") or "").strip():
            return False
        pair = (task_type, subcategory)
        if pair in seen_pairs:
            return False
        seen_pairs.add(pair)
        counts[task_type] += 1
        difficulty_counts[difficulty] += 1
        category_difficulties[task_type].append(difficulty)
    if not all(counts[task_type] == int(config.get("candidate_count") or 0) for task_type, config in TASK_CATALOG.items()):
        return False
    if difficulty_counts != DIFFICULTY_QUOTA:
        return False
    for task_type, expected_difficulties in DIFFICULTY_BY_CATEGORY.items():
        if sorted(category_difficulties.get(task_type) or []) != sorted(expected_difficulties):
            return False
    return True


def _dedupe_case_ids(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        base = normalize_token(str(case.get("case_id") or ""), fallback=f"c{index}_case")[:80]
        candidate = base
        suffix = 2
        while candidate in seen:
            candidate = f"{base}_{suffix}"
            suffix += 1
        seen.add(candidate)
        case["case_id"] = candidate
        result.append(case)
    return result
