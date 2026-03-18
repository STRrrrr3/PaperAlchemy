import json
from collections.abc import Callable
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.llm import get_llm
from src.prompts import PLANNER_CRITIC_SYSTEM_PROMPT
from src.schemas import PagePlan, PlannerCriticReport, StructuredPaper, TemplateCandidate
from src.state import PlannerState

MAX_PLANNER_RETRY_DEFAULT = 2


def _normalize_page_plan(plan: Any) -> PagePlan | None:
    if plan is None:
        return None
    if isinstance(plan, PagePlan):
        return plan
    try:
        return PagePlan.model_validate(plan)
    except Exception:
        return None


def _normalize_structured_paper(paper: Any) -> StructuredPaper | None:
    if paper is None:
        return None
    if isinstance(paper, StructuredPaper):
        return paper
    try:
        return StructuredPaper.model_validate(paper)
    except Exception:
        return None


def _normalize_template_candidate(candidate: Any) -> TemplateCandidate | None:
    if candidate is None:
        return None
    if isinstance(candidate, TemplateCandidate):
        return candidate
    try:
        return TemplateCandidate.model_validate(candidate)
    except Exception:
        return None


def _catalog_to_lookup(template_catalog: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for item in template_catalog:
        if not isinstance(item, dict):
            continue
        template_id = str(item.get("template_id") or "").strip()
        if not template_id:
            continue
        lookup[template_id] = item
    return lookup


def run_planner_code_critic(
    page_plan: PagePlan | None,
    structured_paper: StructuredPaper | None,
    template_catalog: list[dict[str, Any]],
    template_candidates: list[TemplateCandidate],
) -> list[str]:
    critiques: list[str] = []

    if not page_plan:
        critiques.append("Planner output is empty or failed schema validation.")
        return critiques

    if not structured_paper:
        critiques.append("Structured paper is missing, so source grounding cannot be verified.")
        return critiques

    catalog_lookup = _catalog_to_lookup(template_catalog)
    selected_id = page_plan.template_selection.selected_template_id
    selected_entry = page_plan.template_selection.selected_entry_html

    selected_template = catalog_lookup.get(selected_id)
    if not selected_template:
        critiques.append(f"Selected template '{selected_id}' does not exist in template catalog.")
    else:
        entry_candidates = selected_template.get("entry_html_candidates") or []
        if selected_entry not in entry_candidates:
            critiques.append(
                f"selected_entry_html '{selected_entry}' is not in template '{selected_id}' entry candidates."
            )

    if template_candidates:
        candidate_ids = {candidate.template_id for candidate in template_candidates}
        if selected_id not in candidate_ids:
            critiques.append(
                f"Selected template '{selected_id}' is not in selector candidates {sorted(candidate_ids)}."
            )

    valid_sections = {sec.section_title for sec in structured_paper.sections}
    valid_assets = {
        fig.image_path
        for sec in structured_paper.sections
        for fig in sec.related_figures
        if fig.image_path
    }

    outline_block_ids = {item.block_id for item in page_plan.page_outline}
    for item in page_plan.page_outline:
        for sec_title in item.source_sections:
            if sec_title not in valid_sections:
                critiques.append(
                    f"page_outline block '{item.block_id}' references unknown source section '{sec_title}'."
                )

    for block in page_plan.blocks:
        if block.block_id not in outline_block_ids:
            critiques.append(
                f"blocks item '{block.block_id}' does not exist in page_outline."
            )
        for asset_path in block.asset_binding.figure_paths:
            if asset_path not in valid_assets:
                critiques.append(
                    f"block '{block.block_id}' references unknown figure path '{asset_path}'."
                )

    selected_root = page_plan.template_selection.selected_root_dir.rstrip("/")
    for touch in page_plan.coder_handoff.file_touch_plan:
        normalized = touch.path.replace("\\", "/")
        if normalized.startswith("templates/"):
            if not normalized.startswith(selected_root):
                critiques.append(
                    f"file_touch_plan path '{touch.path}' is outside selected template root '{selected_root}'."
                )

    return critiques


def run_planner_semantic_critic(
    structured_paper: StructuredPaper,
    template_catalog: list[dict[str, Any]],
    page_plan: PagePlan,
) -> PlannerCriticReport:
    print("[PaperAlchemy-PlannerCritic] using Gemini-Flash for semantic planning audit...")
    llm = get_llm(temperature=0, use_smart_model=False)
    structured_llm = llm.with_structured_output(PlannerCriticReport)

    user_msg = (
        "### STRUCTURED_PAPER_JSON\n"
        f"{structured_paper.model_dump_json(indent=2, ensure_ascii=False)}\n\n"
        "### TEMPLATE_CATALOG_JSON\n"
        f"{json.dumps(template_catalog, indent=2, ensure_ascii=False)}\n\n"
        "### CANDIDATE_PAGE_PLAN_JSON\n"
        f"{page_plan.model_dump_json(indent=2, ensure_ascii=False)}\n"
    )

    try:
        report = structured_llm.invoke(
            [
                SystemMessage(content=PLANNER_CRITIC_SYSTEM_PROMPT),
                HumanMessage(content=user_msg),
            ]
        )
        return report
    except Exception as exc:
        print(f"[PaperAlchemy-PlannerCritic] semantic critic exception: {exc}")
        return PlannerCriticReport(
            is_plan_valid=False,
            plan_feedback=f"Semantic planning critic failed unexpectedly: {exc}",
        )


def planner_critic_node(state: PlannerState) -> dict[str, Any]:
    print("[PaperAlchemy-PlannerCritic] running plan audit...")
    page_plan = _normalize_page_plan(state.get("page_plan"))
    structured_paper = _normalize_structured_paper(state.get("structured_paper"))
    template_catalog = state.get("template_catalog")
    if not isinstance(template_catalog, list):
        template_catalog = []
    raw_candidates = state.get("template_candidates") or []
    template_candidates = [
        item
        for item in (_normalize_template_candidate(candidate) for candidate in raw_candidates)
        if item is not None
    ]

    critiques = run_planner_code_critic(
        page_plan=page_plan,
        structured_paper=structured_paper,
        template_catalog=template_catalog,
        template_candidates=template_candidates,
    )

    if page_plan and structured_paper and not critiques:
        report = run_planner_semantic_critic(
            structured_paper=structured_paper,
            template_catalog=template_catalog,
            page_plan=page_plan,
        )
        if not report.is_plan_valid:
            critiques.append(f"Semantic planning audit failed: {report.plan_feedback}")

    if critiques:
        feedback = "\n".join(critiques)
        print(f"[PaperAlchemy-PlannerCritic] plan rejected:\n{feedback}")
        return {
            "planner_critic_passed": False,
            "planner_feedback_history": [feedback],
            "planner_retry_count": int(state.get("planner_retry_count", 0)) + 1,
        }

    print("[PaperAlchemy-PlannerCritic] all plan checks passed.")
    return {"planner_critic_passed": True}


def build_planner_critic_router(max_retry: int = MAX_PLANNER_RETRY_DEFAULT) -> Callable[[PlannerState], str]:
    def _router(state: PlannerState) -> str:
        if state.get("planner_critic_passed"):
            return "end"
        if int(state.get("planner_retry_count", 0)) >= max_retry:
            print(
                f"[PaperAlchemy-PlannerCritic] reached max retry limit ({max_retry}), stop retry loop."
            )
            return "end"
        return "retry"

    return _router
