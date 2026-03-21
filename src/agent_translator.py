import json
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.human_feedback import (
    build_multimodal_message_content,
    extract_human_feedback_images,
    extract_human_feedback_text,
    has_human_feedback,
)
from src.llm import get_llm
from src.page_manifest import build_page_manifest_path, load_page_manifest
from src.prompts import TRANSLATOR_SYSTEM_PROMPT, TRANSLATOR_USER_PROMPT_TEMPLATE
from src.schemas import CoderArtifact, RevisionPlan
from src.state import WorkflowState


def _normalize_coder_artifact(artifact: Any) -> CoderArtifact | None:
    if artifact is None:
        return None
    if isinstance(artifact, CoderArtifact):
        return artifact
    try:
        return CoderArtifact.model_validate(artifact)
    except Exception:
        return None


def _message_content_to_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str) and item.strip():
                parts.append(item.strip())
                continue
            if isinstance(item, dict):
                text = str(item.get("text") or "").strip()
                if text:
                    parts.append(text)
        return "\n".join(parts).strip()

    return str(content or "").strip()


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
    if not artifact:
        return "(none)"

    entry_path = Path(artifact.entry_html)
    if not entry_path.exists():
        return "(none)"

    try:
        return entry_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return entry_path.read_text(encoding="latin-1")
    except Exception:
        return "(none)"


def _read_current_page_manifest(artifact: CoderArtifact | None) -> str:
    if not artifact:
        return "{}"

    manifest = load_page_manifest(build_page_manifest_path(artifact.entry_html))
    if not manifest:
        return "{}"
    return json.dumps(manifest.model_dump(), indent=2, ensure_ascii=False)


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

    user_prompt = TRANSLATOR_USER_PROMPT_TEMPLATE.format(
        human_feedback=human_feedback,
        current_entry_html_path=artifact.entry_html,
        current_template_id=artifact.selected_template_id,
        current_page_manifest_json=current_page_manifest_json,
        current_html=current_html,
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
