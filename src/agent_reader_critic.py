import json
import re
from collections.abc import Callable
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.json_utils import to_pretty_json
from src.llm import get_llm
from src.prompts import CRITIC_SYSTEM_PROMPT
from src.schemas import CriticReport, StructuredPaper
from src.state import ReaderState

MAX_RETRY_DEFAULT = 3

LOW_SIGNAL_SECTION_KEYWORDS = (
    "acknowledgement",
    "acknowledgment",
    "references",
    "reference",
    "appendix",
    "supplementary",
)

METHOD_SECTION_KEYWORDS = (
    "method",
    "design",
    "approach",
    "algorithm",
    "architecture",
    "protocol",
    "model",
)

EVAL_SECTION_KEYWORDS = (
    "evaluation",
    "experiment",
    "results",
    "analysis",
    "ablation",
    "benchmark",
)

RELATED_WORK_SECTION_KEYWORDS = ("related work", "related")
CONCLUSION_SECTION_KEYWORDS = ("conclusion", "discussion", "future work")


def _text_len(text: str | None) -> int:
    return len((text or "").strip())


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(k in lowered for k in keywords)


def _section_density_target(section_title: str) -> tuple[int, int, str]:
    """Return (min_key_details, min_summary_chars, section_type)."""
    normalized_title = section_title.strip().lower()

    if "system model" in normalized_title or "threat model" in normalized_title or "models and goals" in normalized_title:
        return 4, 160, "context"
    if _contains_any(normalized_title, LOW_SIGNAL_SECTION_KEYWORDS):
        return 1, 60, "low_signal"
    if _contains_any(normalized_title, RELATED_WORK_SECTION_KEYWORDS):
        return 3, 120, "related_work"
    if _contains_any(normalized_title, CONCLUSION_SECTION_KEYWORDS):
        return 3, 120, "conclusion"
    if _contains_any(normalized_title, EVAL_SECTION_KEYWORDS):
        return 6, 220, "evaluation"
    if _contains_any(normalized_title, METHOD_SECTION_KEYWORDS):
        return 6, 220, "method"
    if "abstract" in normalized_title:
        return 4, 180, "abstract"
    if "introduction" in normalized_title or "background" in normalized_title:
        return 5, 180, "context"

    return 4, 150, "general"


def _run_density_checks(structured_paper: StructuredPaper) -> list[str]:
    critiques: list[str] = []

    if len(structured_paper.sections) < 6:
        critiques.append(
            f"Insufficient section coverage: only {len(structured_paper.sections)} sections extracted (recommended >= 6)."
        )

    overall_len = _text_len(structured_paper.overall_summary)
    if overall_len < 420:
        critiques.append(
            "overall_summary is too short "
            f"({overall_len} chars, recommended >= 420). Add method novelty, key results, and limitations."
        )

    has_method_like_section = False
    has_eval_like_section = False

    for sec in structured_paper.sections:
        min_details, min_summary_chars, section_type = _section_density_target(sec.section_title)
        summary_len = _text_len(sec.content_summary)
        details_count = len(sec.key_details)

        if section_type == "method":
            has_method_like_section = True
        if section_type == "evaluation":
            has_eval_like_section = True

        if summary_len < min_summary_chars:
            critiques.append(
                f"Section '{sec.section_title}' summary is too short "
                f"({summary_len} < {min_summary_chars}). Add concrete mechanism/process/experiment details."
            )

        if details_count < min_details:
            critiques.append(
                f"Section '{sec.section_title}' key_details are too few "
                f"({details_count} < {min_details}). Add more implementation-useful points."
            )

        if section_type in {"method", "evaluation"}:
            dense_points = [d for d in sec.key_details if _text_len(d) >= 24]
            if len(dense_points) < max(4, min_details - 1):
                critiques.append(
                    f"Section '{sec.section_title}' key_details are too terse. "
                    "Use more specific full-sentence details about steps, constraints, or quantitative outcomes."
                )

        if section_type == "evaluation":
            eval_text = (sec.content_summary or "") + " " + " ".join(sec.key_details or [])
            if not re.search(r"\d", eval_text):
                critiques.append(
                    f"Section '{sec.section_title}' lacks quantitative evidence. Add metrics, gains, or overhead numbers."
                )

    if not has_method_like_section:
        critiques.append("Missing a clear method/design section. Extract Method/Design/Algorithm as a standalone section.")
    if not has_eval_like_section:
        critiques.append("Missing a clear evaluation/results section. Extract Evaluation/Results as a standalone section.")

    return critiques


def _normalize_structured_paper(paper: Any) -> StructuredPaper | None:
    if paper is None:
        return None
    if isinstance(paper, StructuredPaper):
        return paper

    try:
        return StructuredPaper.model_validate(paper)
    except Exception:
        return None


def run_semantic_critic(raw_markdown: str, structured_json: str, assets_list: list[dict]) -> CriticReport:
    """Run Gemini-Flash semantic audit for Reader output."""
    print("[PaperAlchemy-Critic] Running Gemini-Flash semantic audit...")

    llm = get_llm(temperature=0, use_smart_model=False)
    structured_llm = llm.with_structured_output(CriticReport)

    user_msg = f"""
    ### SOURCE DATA (Original Markdown):
    {raw_markdown}

    ### ASSETS LIST:
    {json.dumps(assets_list, indent=2, ensure_ascii=False)}

    ### CANDIDATE OUTPUT (Reader Agent's JSON):
    {structured_json}
    """

    try:
        report = structured_llm.invoke(
            [
                SystemMessage(content=CRITIC_SYSTEM_PROMPT),
                HumanMessage(content=user_msg),
            ]
        )
        return report
    except Exception as e:
        print(f"[PaperAlchemy-Critic] Semantic audit error: {e}")
        return CriticReport(
            is_extraction_valid=False,
            extraction_feedback=f"Semantic audit failed with exception: {e}. Please retry Reader generation.",
        )


def run_code_critic(structured_paper: StructuredPaper | None, assets_list: list[dict]) -> list[str]:
    """Run deterministic checks (no LLM)."""
    critiques: list[str] = []

    if not structured_paper:
        critiques.append("Reader failed: structured_paper is empty. Check source parsing or LLM truncation.")
        return critiques

    valid_asset_paths = {
        str(asset.get("image_path"))
        for asset in assets_list
        if isinstance(asset, dict) and asset.get("image_path")
    }

    for sec in structured_paper.sections:
        for fig in sec.related_figures:
            if fig.image_path not in valid_asset_paths:
                critiques.append(
                    f"Hallucination warning: section '{sec.section_title}' references missing asset path '{fig.image_path}'."
                )

    critiques.extend(_run_density_checks(structured_paper))
    return critiques


def critic_node(state: ReaderState) -> dict[str, Any]:
    """LangGraph node: audit Reader output and return pass/fail with feedback."""
    print("[PaperAlchemy-Critic] Running audit...")
    raw_markdown = str(state.get("raw_markdown") or "")
    assets_list = state.get("assets_list")
    if not isinstance(assets_list, list):
        assets_list = []

    structured_paper = _normalize_structured_paper(state.get("structured_paper"))
    critiques = run_code_critic(structured_paper, assets_list)

    if structured_paper:
        paper_json_str = to_pretty_json(structured_paper)
        report = run_semantic_critic(
            raw_markdown=raw_markdown,
            structured_json=paper_json_str,
            assets_list=assets_list,
        )
        if not report.is_extraction_valid:
            critiques.append(f"Semantic audit failed: {report.extraction_feedback}")

    if critiques:
        feedback_str = "\n".join(critiques)
        print(f"[PaperAlchemy-Critic] Audit failed, retrying:\n{feedback_str}")
        return {
            "critic_passed": False,
            "feedback_history": [feedback_str],
            "retry_count": int(state.get("retry_count", 0)) + 1,
        }

    print("[PaperAlchemy-Critic] All checks passed")
    return {"critic_passed": True}


def build_critic_router(max_retry: int = MAX_RETRY_DEFAULT) -> Callable[[ReaderState], str]:
    """Return LangGraph routing function."""

    def _router(state: ReaderState) -> str:
        if state.get("critic_passed"):
            return "end"

        if int(state.get("retry_count", 0)) >= max_retry:
            print(f"[PaperAlchemy-Critic] Reached max retry limit ({max_retry}), stop retry loop.")
            return "end"

        return "retry"

    return _router
