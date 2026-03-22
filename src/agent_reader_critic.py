import json
import re
from collections.abc import Callable
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.human_feedback import extract_human_feedback_text
from src.json_utils import to_pretty_json
from src.llm import get_llm
from src.prompts import CRITIC_SYSTEM_PROMPT, READER_CRITIC_USER_PROMPT_TEMPLATE
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


def _normalize_directives_text(human_directives: str | None) -> str:
    return str(human_directives or "").strip().lower()


def _has_section_shape_override(human_directives: str | None) -> bool:
    directive_text = _normalize_directives_text(human_directives)
    if not directive_text:
        return False

    section_terms = (
        "section",
        "sections",
        "chapter",
        "chapters",
        "part",
        "parts",
        "\u7ae0\u8282",
        "\u90e8\u5206",
        "\u5c0f\u8282",
    )
    override_terms = (
        "add",
        "added",
        "append",
        "include",
        "insert",
        "extra",
        "more",
        "fewer",
        "less",
        "remove",
        "delete",
        "drop",
        "omit",
        "skip",
        "merge",
        "combine",
        "split",
        "separate",
        "only",
        "just",
        "focus",
        "\u4fdd\u7559",
        "\u589e\u52a0",
        "\u6dfb\u52a0",
        "\u5220\u53bb",
        "\u5220\u9664",
        "\u53bb\u6389",
        "\u7701\u7565",
        "\u8df3\u8fc7",
        "\u5408\u5e76",
        "\u62c6\u5206",
        "\u53ea\u8981",
        "\u4ec5\u4fdd\u7559",
    )
    return any(term in directive_text for term in section_terms) and any(
        term in directive_text for term in override_terms
    )


def _section_density_target(section_title: str) -> tuple[int, int, str]:
    """Return (min_words, min_chars, section_type) for rich section content."""
    normalized_title = section_title.strip().lower()

    if (
        "system model" in normalized_title
        or "threat model" in normalized_title
        or "models and goals" in normalized_title
    ):
        return 90, 500, "context"
    if _contains_any(normalized_title, LOW_SIGNAL_SECTION_KEYWORDS):
        return 40, 220, "low_signal"
    if _contains_any(normalized_title, RELATED_WORK_SECTION_KEYWORDS):
        return 80, 450, "related_work"
    if _contains_any(normalized_title, CONCLUSION_SECTION_KEYWORDS):
        return 80, 420, "conclusion"
    if _contains_any(normalized_title, EVAL_SECTION_KEYWORDS):
        return 180, 1100, "evaluation"
    if _contains_any(normalized_title, METHOD_SECTION_KEYWORDS):
        return 180, 1100, "method"
    if "abstract" in normalized_title:
        return 110, 650, "abstract"
    if "introduction" in normalized_title or "background" in normalized_title:
        return 120, 750, "context"

    return 100, 600, "general"


def _word_count(text: str | None) -> int:
    return len(re.findall(r"\b\w+\b", text or ""))


def _nonempty_lines(text: str | None) -> list[str]:
    return [line.strip() for line in (text or "").splitlines() if line.strip()]


def _paragraph_count(text: str | None) -> int:
    blocks = [block.strip() for block in re.split(r"\n\s*\n", text or "") if block.strip()]
    return len(blocks)


def _is_bullet_line(line: str) -> bool:
    return bool(re.match(r"^([-*+]\s+|\d+\.\s+)", line.strip()))


def _run_density_checks(
    structured_paper: StructuredPaper,
    human_directives: str | None = None,
) -> list[str]:
    critiques: list[str] = []

    min_section_count = 3 if _has_section_shape_override(human_directives) else 5
    if len(structured_paper.sections) < min_section_count:
        critiques.append(
            "Insufficient section coverage: only "
            f"{len(structured_paper.sections)} sections extracted "
            f"(recommended >= {min_section_count})."
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
        min_words, min_chars, section_type = _section_density_target(sec.section_title)
        rich_text = sec.rich_web_content or ""
        rich_len = _text_len(rich_text)
        rich_words = _word_count(rich_text)
        nonempty_lines = _nonempty_lines(rich_text)
        paragraph_count = _paragraph_count(rich_text)
        bullet_lines = sum(1 for line in nonempty_lines if _is_bullet_line(line))

        if section_type == "method":
            has_method_like_section = True
        if section_type == "evaluation":
            has_eval_like_section = True

        if rich_len < min_chars or rich_words < min_words:
            critiques.append(
                f"Section '{sec.section_title}' rich_web_content is too short "
                f"({rich_words} words / {rich_len} chars; recommended >= {min_words} words and >= {min_chars} chars). "
                "Expand it into a fuller technical narrative with mechanism, setup, and results details."
            )

        if paragraph_count < (3 if section_type in {"method", "evaluation"} else 2):
            critiques.append(
                f"Section '{sec.section_title}' rich_web_content is not structured as a developed Markdown narrative. "
                "Add clearer paragraph breaks and richer exposition."
            )

        if nonempty_lines and bullet_lines / max(len(nonempty_lines), 1) > 0.6 and paragraph_count <= 2:
            critiques.append(
                f"Section '{sec.section_title}' rich_web_content reads too much like a bullet list. "
                "Rewrite it as cohesive academic prose with Markdown subheadings and paragraphs."
            )

        if section_type in {"method", "evaluation"} and paragraph_count < 4:
            critiques.append(
                f"Section '{sec.section_title}' needs deeper rich_web_content coverage. "
                "Method and evaluation sections should include multiple dense paragraphs, not a compressed synopsis."
            )

        if section_type in {"method", "evaluation"}:
            technical_markers = ("###", "**", "`", "equation", "algorithm", "dataset", "metric", "baseline", "ablation")
            if not any(marker in rich_text.lower() for marker in technical_markers):
                critiques.append(
                    f"Section '{sec.section_title}' rich_web_content lacks clear technical density markers. "
                    "Preserve subheadings, emphasized concepts, inline code/math, datasets, baselines, or algorithm detail in the Markdown."
                )

        if section_type == "evaluation":
            eval_text = rich_text
            if not re.search(r"\d", eval_text):
                critiques.append(
                    f"Section '{sec.section_title}' lacks quantitative evidence. "
                    "Add metrics, gains, or overhead numbers."
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


def run_semantic_critic(
    raw_markdown: str,
    structured_json: str,
    assets_list: list[dict],
    human_directives: str = "",
) -> CriticReport:
    """Run Gemini-Flash semantic audit for Reader output."""
    print("[PaperAlchemy-Critic] Running Gemini-Flash semantic audit...")

    llm = get_llm(temperature=0, use_smart_model=False)
    structured_llm = llm.with_structured_output(CriticReport)

    user_msg = READER_CRITIC_USER_PROMPT_TEMPLATE.format(
        raw_markdown=raw_markdown,
        human_directives=human_directives or "(none)",
        assets_list_json=json.dumps(assets_list, indent=2, ensure_ascii=False),
        structured_json=structured_json,
    )

    try:
        report = structured_llm.invoke(
            [
                SystemMessage(content=CRITIC_SYSTEM_PROMPT),
                HumanMessage(content=user_msg),
            ]
        )
        return report
    except Exception as exc:
        print(f"[PaperAlchemy-Critic] Semantic audit error: {exc}")
        return CriticReport(
            is_extraction_valid=False,
            extraction_feedback=f"Semantic audit failed with exception: {exc}. Please retry Reader generation.",
        )


def run_code_critic(
    structured_paper: StructuredPaper | None,
    assets_list: list[dict],
    human_directives: str = "",
) -> list[str]:
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

    critiques.extend(_run_density_checks(structured_paper, human_directives=human_directives))
    return critiques


def critic_node(state: ReaderState) -> dict[str, Any]:
    """LangGraph node: audit Reader output and return pass/fail with feedback."""
    print("[PaperAlchemy-Critic] Running audit...")
    raw_markdown = str(state.get("raw_markdown") or "")
    human_directives = extract_human_feedback_text(state.get("human_directives"))
    assets_list = state.get("assets_list")
    if not isinstance(assets_list, list):
        assets_list = []

    structured_paper = _normalize_structured_paper(state.get("structured_paper"))
    critiques = run_code_critic(
        structured_paper,
        assets_list,
        human_directives=human_directives,
    )

    if structured_paper:
        paper_json_str = to_pretty_json(structured_paper)
        report = run_semantic_critic(
            raw_markdown=raw_markdown,
            structured_json=paper_json_str,
            assets_list=assets_list,
            human_directives=human_directives,
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
