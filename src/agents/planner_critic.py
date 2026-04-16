import json
import re
from collections.abc import Callable
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.utils.json_utils import to_pretty_json
from src.services.llm import get_llm
from src.prompts import PLANNER_CRITIC_SYSTEM_PROMPT, PLANNER_CRITIC_USER_PROMPT_TEMPLATE
from src.contracts.schemas import (
    PagePlan,
    PlannerCriticReport,
    SemanticPagePlan,
    StructuredPaper,
    TemplateCandidate,
    TemplateProfile,
)
from src.contracts.state import PlannerState
from src.template.template_ir import canonical_shell_nodes

MAX_PLANNER_RETRY_DEFAULT = 2
_STABLE_BLOCK_ID_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
_UNSTABLE_BLOCK_ID_PATTERN = re.compile(
    r"^(?:block|section|item|content|slot|module|component|region)_[0-9]+$"
)
_DISALLOWED_BLOCK_ID_TOKENS = {"template", "placeholder", "todo", "tbd", "temp", "dummy"}


def _normalize_page_plan(plan: Any) -> PagePlan | None:
    if plan is None:
        return None
    if isinstance(plan, PagePlan):
        return plan
    try:
        return PagePlan.model_validate(plan)
    except Exception:
        return None


def _normalize_semantic_page_plan(plan: Any) -> SemanticPagePlan | None:
    if plan is None:
        return None
    if isinstance(plan, SemanticPagePlan):
        return plan
    try:
        return SemanticPagePlan.model_validate(plan)
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


def _normalize_template_profile(profile: Any) -> TemplateProfile | None:
    if profile is None:
        return None
    if isinstance(profile, TemplateProfile):
        return profile
    try:
        return TemplateProfile.model_validate(profile)
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


def _selected_template_tokens(template_id: str) -> set[str]:
    return {
        token
        for token in re.split(r"[^a-z0-9]+", str(template_id or "").lower())
        if token and token not in {"github", "io", "www", "com"}
    }


def _validate_block_id(block_id: str, template_tokens: set[str]) -> list[str]:
    critiques: list[str] = []
    clean = str(block_id or "").strip()
    lowered = clean.lower()

    if not clean:
        critiques.append("Encountered an empty block_id in semantic plan.")
        return critiques
    if not _STABLE_BLOCK_ID_PATTERN.match(clean):
        critiques.append(f"block_id '{clean}' must be stable snake_case.")
    if _UNSTABLE_BLOCK_ID_PATTERN.match(lowered):
        critiques.append(f"block_id '{clean}' uses unstable positional naming.")
    if any(token in lowered.split("_") for token in _DISALLOWED_BLOCK_ID_TOKENS):
        critiques.append(f"block_id '{clean}' contains placeholder or template-derived wording.")
    if template_tokens and any(token in lowered.split("_") for token in template_tokens):
        critiques.append(f"block_id '{clean}' should not depend on selected template naming.")

    return critiques


def run_semantic_plan_validation(
    semantic_page_plan: SemanticPagePlan | None,
    structured_paper: StructuredPaper | None,
    template_catalog: list[dict[str, Any]],
    template_candidates: list[TemplateCandidate],
    selected_template: TemplateCandidate | None,
) -> list[str]:
    critiques: list[str] = []

    if semantic_page_plan is None:
        critiques.append("Planner output is empty or failed semantic schema validation.")
        return critiques

    if structured_paper is None:
        critiques.append("Structured paper is missing, so semantic grounding cannot be verified.")
        return critiques

    if selected_template is None:
        critiques.append("Selected template context is missing for semantic planning review.")
        return critiques

    catalog_lookup = _catalog_to_lookup(template_catalog)
    selected_id = str(selected_template.template_id or "").strip()
    selected_entry = str(selected_template.chosen_entry_html or "").strip()
    template_tokens = _selected_template_tokens(selected_id)

    selected_catalog_entry = catalog_lookup.get(selected_id)
    if not selected_catalog_entry:
        critiques.append(f"Selected template '{selected_id}' does not exist in template catalog.")
    else:
        entry_candidates = selected_catalog_entry.get("entry_html_candidates") or []
        if selected_entry not in entry_candidates:
            critiques.append(
                f"selected_template entry_html '{selected_entry}' is not in template '{selected_id}' entry candidates."
            )

    if template_candidates:
        candidate_ids = {candidate.template_id for candidate in template_candidates}
        if selected_id not in candidate_ids:
            critiques.append(
                f"Selected template '{selected_id}' is not in selector candidates {sorted(candidate_ids)}."
            )

        fallback_template_id = str(
            semantic_page_plan.template_selection.fallback_template_id or ""
        ).strip()
        if fallback_template_id and fallback_template_id not in candidate_ids:
            critiques.append(
                f"fallback_template_id '{fallback_template_id}' is not in selector candidates {sorted(candidate_ids)}."
            )

    valid_sections = {sec.section_title for sec in structured_paper.sections}
    valid_assets = {
        fig.image_path
        for sec in structured_paper.sections
        for fig in sec.related_figures
        if fig.image_path
    }

    outline_block_ids: set[str] = set()
    outline_orders: set[int] = set()
    for item in semantic_page_plan.page_outline:
        if item.block_id in outline_block_ids:
            critiques.append(f"Duplicate page_outline block_id '{item.block_id}'.")
        outline_block_ids.add(item.block_id)
        if item.order in outline_orders:
            critiques.append(f"Duplicate page_outline order '{item.order}' detected.")
        outline_orders.add(item.order)
        critiques.extend(_validate_block_id(item.block_id, template_tokens))
        if not item.source_sections:
            critiques.append(
                f"page_outline block '{item.block_id}' must reference at least one source section."
            )
        for sec_title in item.source_sections:
            if sec_title not in valid_sections:
                critiques.append(
                    f"page_outline block '{item.block_id}' references unknown source section '{sec_title}'."
                )

    semantic_block_ids: set[str] = set()
    for block in semantic_page_plan.semantic_blocks:
        if block.block_id in semantic_block_ids:
            critiques.append(f"Duplicate semantic_blocks item block_id '{block.block_id}'.")
        semantic_block_ids.add(block.block_id)
        critiques.extend(_validate_block_id(block.block_id, template_tokens))
        if block.block_id not in outline_block_ids:
            critiques.append(
                f"semantic_blocks item '{block.block_id}' does not exist in page_outline."
            )
        for asset_path in block.asset_binding.figure_paths:
            if asset_path not in valid_assets:
                critiques.append(
                    f"semantic block '{block.block_id}' references unknown figure path '{asset_path}'."
                )

    if outline_block_ids != semantic_block_ids:
        critiques.append(
            "page_outline block_ids and semantic_blocks block_ids must match exactly for stable revision targeting."
        )

    return critiques


def run_bound_plan_validation(
    page_plan: PagePlan | None,
    template_profile: TemplateProfile | None,
) -> list[str]:
    critiques: list[str] = []

    if page_plan is None:
        critiques.append("Bound PagePlan is empty or failed schema validation.")
        return critiques

    if template_profile is None:
        critiques.append("TemplateProfile is missing, so bound plan validation cannot run.")
        return critiques

    selected_root = str(page_plan.template_selection.selected_root_dir or "").strip().rstrip("/")
    if not selected_root:
        critiques.append("template_selection.selected_root_dir is missing in bound PagePlan.")
    if not str(page_plan.template_selection.selected_entry_html or "").strip():
        critiques.append("template_selection.selected_entry_html is missing in bound PagePlan.")

    for touch in page_plan.coder_handoff.file_touch_plan:
        normalized = touch.path.replace("\\", "/")
        if normalized.startswith("templates/") and not normalized.startswith(selected_root):
            critiques.append(
                f"file_touch_plan path '{touch.path}' is outside selected template root '{selected_root}'."
            )

    allowed_shells_by_id = {
        str(candidate.shell_id or "").strip(): candidate
        for candidate in canonical_shell_nodes(template_profile, bindable_only=True)
        if str(candidate.shell_id or "").strip()
    }
    allowed_global_selectors = {
        str(selector or "").strip()
        for selector in template_profile.global_preserve_selectors
        if str(selector or "").strip()
    }
    for block in page_plan.blocks:
        shell_id = str(block.target_template_region.shell_id or "").strip()
        selector_hint = str(block.target_template_region.selector_hint or "").strip()
        if not shell_id:
            critiques.append(
                f"block '{block.block_id}' must bind to a canonical TemplateIR shell_id."
            )
            continue
        shell_node = allowed_shells_by_id.get(shell_id)
        if shell_node is None:
            critiques.append(
                f"block '{block.block_id}' shell_id '{shell_id}' is not present in TemplateProfile.template_ir.shell_nodes."
            )
            continue
        if selector_hint != str(shell_node.selector or "").strip():
            critiques.append(
                f"block '{block.block_id}' selector_hint '{selector_hint}' does not match canonical shell selector '{shell_node.selector}'."
            )

    unexpected_dom_mapping = sorted(set(page_plan.dom_mapping) - allowed_global_selectors)
    if unexpected_dom_mapping:
        critiques.append(
            "dom_mapping should only contain TemplateProfile.global_preserve_selectors; unexpected keys: "
            + ", ".join(unexpected_dom_mapping[:6])
        )

    render_strategy = str(page_plan.plan_meta.render_strategy or "").strip()
    if render_strategy not in {"compiled_block_assembly", "legacy_fullpage"}:
        critiques.append("plan_meta.render_strategy must be a supported planner output.")

    return critiques


def run_planner_semantic_critic(
    structured_paper: StructuredPaper,
    template_catalog: list[dict[str, Any]],
    semantic_page_plan: SemanticPagePlan,
    selected_template: TemplateCandidate | None = None,
) -> PlannerCriticReport:
    print("[PaperAlchemy-PlannerCritic] using Gemini-Flash for semantic planning audit...")
    llm = get_llm(temperature=0, use_smart_model=False)
    structured_llm = llm.with_structured_output(PlannerCriticReport)

    user_msg = PLANNER_CRITIC_USER_PROMPT_TEMPLATE.format(
        structured_paper_json=to_pretty_json(structured_paper),
        template_catalog_json=json.dumps(template_catalog, indent=2, ensure_ascii=False),
        selected_template_json=(
            json.dumps(selected_template.model_dump(), indent=2, ensure_ascii=False)
            if selected_template is not None
            else "null"
        ),
        candidate_semantic_plan_json=to_pretty_json(semantic_page_plan),
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
    print("[PaperAlchemy-PlannerCritic] running semantic plan audit...")
    semantic_page_plan = _normalize_semantic_page_plan(state.get("semantic_page_plan"))
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
    selected_template = _normalize_template_candidate(state.get("selected_template"))

    critiques = run_semantic_plan_validation(
        semantic_page_plan=semantic_page_plan,
        structured_paper=structured_paper,
        template_catalog=template_catalog,
        template_candidates=template_candidates,
        selected_template=selected_template,
    )

    if semantic_page_plan and structured_paper and not critiques:
        report = run_planner_semantic_critic(
            structured_paper=structured_paper,
            template_catalog=template_catalog,
            semantic_page_plan=semantic_page_plan,
            selected_template=selected_template,
        )
        if not report.is_plan_valid:
            critiques.append(f"Semantic planning audit failed: {report.plan_feedback}")

    if critiques:
        feedback = "\n".join(critiques)
        print(f"[PaperAlchemy-PlannerCritic] semantic plan rejected:\n{feedback}")
        return {
            "planner_critic_passed": False,
            "planner_feedback_history": [feedback],
            "planner_retry_count": int(state.get("planner_retry_count", 0)) + 1,
        }

    print("[PaperAlchemy-PlannerCritic] all semantic plan checks passed.")
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
