import json
import re
from typing import Any

from src.utils.html_utils import read_current_page_html
from langchain_core.messages import HumanMessage, SystemMessage

from src.services.human_feedback import (
    build_multimodal_message_content,
    extract_human_feedback_images,
    extract_human_feedback_text,
    has_human_feedback,
)
from src.services.llm import get_llm
from src.validators.page_manifest import build_page_manifest_path, load_page_manifest
from src.prompts import TRANSLATOR_SYSTEM_PROMPT, TRANSLATOR_USER_PROMPT_TEMPLATE
from src.contracts.schemas import CoderArtifact, RevisionPlan
from src.contracts.state import WorkflowState

NON_PATCH_FEEDBACK_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"\b(?:whole|entire|overall|global|page-wide)\s+"
            r"(?:page|webpage|site|layout|style|theme|density|rhythm|spacing|structure)\b",
            flags=re.IGNORECASE,
        ),
        "feedback requests page-level or global adjustments beyond anchored patch scope",
    ),
    (
        re.compile(
            r"\b(?:section|page)\s+(?:redesign|rebuild|rework|revamp|restructure|redo)\b",
            flags=re.IGNORECASE,
        ),
        "feedback requests section/page redesign instead of a local anchored patch",
    ),
    (
        re.compile(
            r"\b(?:switch|replace|change)\s+template\b|\btemplate\s+(?:swap|replacement)\b",
            flags=re.IGNORECASE,
        ),
        "feedback requests template replacement which is outside the patch path",
    ),
    (
        re.compile(
            r"\b(?:global|overall|page-wide)\s+(?:style|styling|theme|density|rhythm|spacing|tone)\b",
            flags=re.IGNORECASE,
        ),
        "feedback requests global style retuning instead of a local anchored edit",
    ),
    (
        re.compile(r"\b(?:replan|rebind)\b", flags=re.IGNORECASE),
        "feedback requests structural replanning or shell rebinding",
    ),
)


def _normalize_coder_artifact(artifact: Any) -> CoderArtifact | None:
    if artifact is None:
        return None
    if isinstance(artifact, CoderArtifact):
        return artifact
    try:
        return CoderArtifact.model_validate(artifact)
    except Exception:
        return None


def _normalize_revision_plan(plan: Any) -> RevisionPlan | None:
    if isinstance(plan, RevisionPlan):
        return plan
    if plan is None:
        return None
    try:
        return RevisionPlan.model_validate(plan)
    except Exception:
        return None


def _read_current_html(artifact: CoderArtifact | None) -> str:
    return read_current_page_html(artifact, missing_value="(none)")


def _read_current_page_manifest(artifact: CoderArtifact | None) -> str:
    if not artifact:
        return "{}"

    manifest = load_page_manifest(build_page_manifest_path(artifact.entry_html))
    if not manifest:
        return "{}"
    return json.dumps(manifest.model_dump(), indent=2, ensure_ascii=False)


def _read_style_context(artifact: CoderArtifact | None) -> str:
    if not artifact:
        return "{}"
    from pathlib import Path
    from src.services.preview_service import load_style_context_json
    return load_style_context_json(Path(artifact.entry_html).resolve())


def _classify_edit_intent(feedback: Any, revision_plan: RevisionPlan | None) -> tuple[str, str]:
    feedback_text = str(extract_human_feedback_text(feedback) or "").strip()
    lowered_feedback = feedback_text.lower()

    for pattern, reason in NON_PATCH_FEEDBACK_PATTERNS:
        if pattern.search(lowered_feedback):
            return "non_patch", reason

    if revision_plan is None or not revision_plan.edits:
        return "non_patch", "current feedback was not translated into an actionable anchored patch"

    return "patch", "feedback maps to local anchored edits that the patch path can handle"


def translator_node(state: WorkflowState) -> dict[str, Any]:
    artifact = _normalize_coder_artifact(state.get("coder_artifact"))
    if not artifact:
        print("[Translator] missing coder artifact, skipping translation.")
        return {"revision_plan": RevisionPlan()}

    feedback = state.get("human_directives")
    if not has_human_feedback(feedback):
        print("[Translator] no human feedback provided, skipping translation.")
        return {"revision_plan": RevisionPlan()}

    human_feedback = extract_human_feedback_text(feedback) or "(no text feedback provided)"
    human_feedback_images = extract_human_feedback_images(feedback)
    current_html = _read_current_html(artifact)
    current_page_manifest_json = _read_current_page_manifest(artifact)
    style_context_json = _read_style_context(artifact)

    user_prompt = TRANSLATOR_USER_PROMPT_TEMPLATE.format(
        human_feedback=human_feedback,
        current_entry_html_path=artifact.entry_html,
        current_template_id=artifact.selected_template_id,
        current_page_manifest_json=current_page_manifest_json,
        current_html=current_html,
        template_style_context_json=style_context_json,
    )

    print("[Translator] Translating multimodal feedback into a structured revision plan...")
    try:
        llm = get_llm(temperature=0.1, use_smart_model=True)
        structured_llm = llm.with_structured_output(RevisionPlan)
        response = structured_llm.invoke(
            [
                SystemMessage(content=TRANSLATOR_SYSTEM_PROMPT),
                HumanMessage(
                    content=build_multimodal_message_content(
                        text=user_prompt,
                        images=human_feedback_images,
                    )
                ),
            ]
        )
    except Exception as exc:
        print(f"[Translator] translation failed: {exc}")
        return {"revision_plan": RevisionPlan()}

    revision_plan = _normalize_revision_plan(response) or RevisionPlan()
    return {"revision_plan": revision_plan}


def edit_intent_router_node(state: WorkflowState) -> dict[str, Any]:
    revision_plan = _normalize_revision_plan(state.get("revision_plan")) or RevisionPlan()
    intent, reason = _classify_edit_intent(state.get("human_directives"), revision_plan)
    print(f"[EditIntentRouter] Routed webpage revision to {intent}: {reason}")
    return {
        "edit_intent": intent,
        "edit_intent_reason": reason,
    }

