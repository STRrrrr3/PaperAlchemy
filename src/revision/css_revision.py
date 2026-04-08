from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.coder_critic import _encode_image_to_data_url
from src.contracts.schemas import ContentReplacement, CssRevisionPlan, CoderArtifact, PageManifest, PagePlan
from src.contracts.state import WorkflowState
from src.patching.patch_pipeline import (
    LEGACY_PAGE_ERROR,
    _apply_block_replacement,
    _apply_global_replacement,
    _apply_slot_replacement,
    _available_asset_manifest_from_artifact,
    _build_patch_target_paths,
    _ensure_override_style_tag,
    _merge_unique_strings,
    _normalize_coder_artifact,
    _normalize_page_plan,
    _render_override_rule,
    _to_site_relative_paths,
    _write_files_transaction,
)
from src.prompts import CSS_REVISION_AGENT_SYSTEM_PROMPT, CSS_REVISION_AGENT_USER_PROMPT_TEMPLATE
from src.services.artifact_store import get_output_paths, load_coder_artifact, load_page_plan
from src.services.human_feedback import (
    build_multimodal_message_content,
    extract_human_feedback_images,
    extract_human_feedback_text,
    has_human_feedback,
)
from src.services.llm import get_llm
from src.services.preview_service import (
    build_visual_critic_screenshot_path,
    load_style_context_json,
    take_local_screenshot,
)
from src.utils.html_utils import read_current_page_html, read_text_with_fallback
from src.validators.page_manifest import (
    build_page_manifest_path,
    extract_page_manifest,
    load_page_manifest,
)
from src.validators.page_validation import (
    collect_allowed_asset_web_paths,
    collect_local_image_sources,
    validate_fragment_local_image_sources,
    validate_local_image_references,
)


def _normalize_css_revision_plan(plan: Any) -> CssRevisionPlan | None:
    if isinstance(plan, CssRevisionPlan):
        return plan
    if plan is None:
        return None
    try:
        return CssRevisionPlan.model_validate(plan)
    except Exception:
        return None


def _load_workflow_coder_artifact(state: WorkflowState) -> CoderArtifact | None:
    artifact = _normalize_coder_artifact(state.get("coder_artifact"))
    if artifact is not None:
        return artifact
    paper_folder_name = str(state.get("paper_folder_name") or "").strip()
    if not paper_folder_name:
        return None
    _, _, _, coder_json_path = get_output_paths(paper_folder_name)
    return load_coder_artifact(coder_json_path)


def _load_workflow_page_plan(state: WorkflowState) -> PagePlan | None:
    for value in (state.get("page_plan"), state.get("approved_page_plan")):
        page_plan = _normalize_page_plan(value)
        if page_plan is not None:
            return page_plan
    paper_folder_name = str(state.get("paper_folder_name") or "").strip()
    if not paper_folder_name:
        return None
    _, _, planner_json_path, _ = get_output_paths(paper_folder_name)
    return load_page_plan(planner_json_path)


def _available_assets_json(artifact: CoderArtifact | None) -> str:
    if artifact is None:
        return "[]"
    return json.dumps(
        _available_asset_manifest_from_artifact(artifact),
        indent=2,
        ensure_ascii=False,
    )


def _read_current_page_manifest(artifact: CoderArtifact | None) -> PageManifest | None:
    if artifact is None:
        return None
    return load_page_manifest(build_page_manifest_path(artifact.entry_html))


def _build_page_screenshot_payload(artifact: CoderArtifact | None) -> dict[str, str] | None:
    if artifact is None:
        return None
    entry_html_path = Path(artifact.entry_html).resolve()
    screenshot_path = take_local_screenshot(
        str(entry_html_path),
        str(build_visual_critic_screenshot_path(entry_html_path)),
    )
    if not screenshot_path:
        return None
    data_url = _encode_image_to_data_url(Path(screenshot_path))
    if not data_url:
        return None
    return {
        "name": Path(screenshot_path).name,
        "path": screenshot_path,
        "mime_type": "image/png",
        "data_url": data_url,
    }


def _validate_content_replacement_targets(
    replacements: list[ContentReplacement],
    manifest: PageManifest,
) -> list[str]:
    block_lookup = {block.block_id: block for block in manifest.blocks}
    global_ids = {item.global_id for item in manifest.globals}
    errors: list[str] = []

    for replacement in replacements:
        if replacement.scope == "global":
            if replacement.global_id not in global_ids:
                errors.append(f"unknown global_id '{replacement.global_id}'")
            continue

        block = block_lookup.get(str(replacement.block_id or ""))
        if block is None:
            errors.append(f"unknown block_id '{replacement.block_id}'")
            continue
        if replacement.scope == "slot":
            slot_ids = {slot.slot_id for slot in block.slots}
            if replacement.slot_id not in slot_ids:
                errors.append(
                    f"block '{replacement.block_id}' does not expose requested slot '{replacement.slot_id}'"
                )

    return errors


def _update_css_revision_notes(
    existing_notes: str,
    css_rule_count: int,
    content_replacement_count: int,
) -> str:
    base = str(existing_notes or "").strip()
    summary = (
        "v8-css-injection-revision: "
        f"applied {css_rule_count} css rule(s) and {content_replacement_count} content replacement(s)."
    )
    if not base:
        return summary

    parts = [part.strip() for part in base.split("|") if part.strip()]
    filtered = [part for part in parts if not part.startswith("v8-css-injection-revision:")]
    filtered.append(summary)
    return " | ".join(filtered)


def css_revision_agent_node(state: WorkflowState) -> dict[str, Any]:
    artifact = _load_workflow_coder_artifact(state)
    page_plan = _load_workflow_page_plan(state)
    manifest = _read_current_page_manifest(artifact)
    feedback = state.get("human_directives")
    current_html = read_current_page_html(artifact, missing_value="")

    if artifact is None or page_plan is None or not current_html:
        message = "CSS Revision Agent could not run because coder_artifact, page_plan, or current HTML is missing."
        print(f"[CSSRevisionAgent] {message}")
        return {"css_revision_plan": None, "css_revision_summary": "", "patch_error": message}
    if manifest is None:
        print(f"[CSSRevisionAgent] {LEGACY_PAGE_ERROR}")
        return {"css_revision_plan": None, "css_revision_summary": "", "patch_error": LEGACY_PAGE_ERROR}
    if not has_human_feedback(feedback):
        message = "CSS Revision Agent requires human feedback text or images."
        print(f"[CSSRevisionAgent] {message}")
        return {"css_revision_plan": None, "css_revision_summary": "", "patch_error": message}

    page_screenshot = _build_page_screenshot_payload(artifact)
    if page_screenshot is None:
        message = "CSS Revision Agent could not capture a fresh screenshot of the current page."
        print(f"[CSSRevisionAgent] {message}")
        return {"css_revision_plan": None, "css_revision_summary": "", "patch_error": message}

    human_feedback = extract_human_feedback_text(feedback) or "(no text feedback provided)"
    style_context_json = load_style_context_json(Path(artifact.entry_html).resolve())
    current_page_manifest_json = json.dumps(manifest.model_dump(), indent=2, ensure_ascii=False)
    multimodal_images = [page_screenshot, *extract_human_feedback_images(feedback)]
    user_prompt = CSS_REVISION_AGENT_USER_PROMPT_TEMPLATE.format(
        human_feedback=human_feedback,
        current_entry_html_path=artifact.entry_html,
        current_template_id=artifact.selected_template_id,
        current_page_manifest_json=current_page_manifest_json,
        current_html=current_html,
        template_style_context_json=style_context_json,
        available_paper_assets_json=_available_assets_json(artifact),
    )

    print("[CSSRevisionAgent] Translating multimodal feedback into a CssRevisionPlan...")
    try:
        llm = get_llm(temperature=0.1, use_smart_model=True)
        structured_llm = llm.with_structured_output(CssRevisionPlan)
        response = structured_llm.invoke(
            [
                SystemMessage(content=CSS_REVISION_AGENT_SYSTEM_PROMPT),
                HumanMessage(content=build_multimodal_message_content(text=user_prompt, images=multimodal_images)),
            ]
        )
    except Exception as exc:
        message = f"CSS Revision Agent failed generating a revision plan: {exc}"
        print(f"[CSSRevisionAgent] {message}")
        return {"css_revision_plan": None, "css_revision_summary": "", "patch_error": message}

    revision_plan = _normalize_css_revision_plan(response)
    if revision_plan is None:
        message = "CSS Revision Agent returned invalid CssRevisionPlan output."
        print(f"[CSSRevisionAgent] {message}")
        return {"css_revision_plan": None, "css_revision_summary": "", "patch_error": message}

    summary = str(revision_plan.revision_summary or "").strip()
    if str(revision_plan.not_possible_explanation or "").strip():
        explanation = str(revision_plan.not_possible_explanation).strip()
        print(f"[CSSRevisionAgent] {explanation}")
        return {"css_revision_plan": None, "css_revision_summary": summary, "patch_error": explanation}

    target_errors = _validate_content_replacement_targets(revision_plan.content_replacements, manifest)
    if target_errors:
        message = "CSS Revision Agent referenced unavailable manifest targets: " + "; ".join(target_errors)
        print(f"[CSSRevisionAgent] {message}")
        return {"css_revision_plan": None, "css_revision_summary": summary, "patch_error": message}

    if not revision_plan.css_rules and not revision_plan.content_replacements:
        message = "CSS Revision Agent returned an empty revision plan."
        print(f"[CSSRevisionAgent] {message}")
        return {"css_revision_plan": None, "css_revision_summary": summary, "patch_error": message}

    return {
        "css_revision_plan": revision_plan,
        "css_revision_summary": summary,
        "patch_error": "",
    }


def css_revision_executor_node(state: WorkflowState) -> dict[str, Any]:
    existing_error = str(state.get("patch_error") or "").strip()
    if existing_error:
        print(f"[CSSRevisionExecutor] upstream safe fail: {existing_error}")
        return {"patch_error": existing_error, "css_revision_summary": str(state.get("css_revision_summary") or "")}

    artifact = _load_workflow_coder_artifact(state)
    page_plan = _load_workflow_page_plan(state)
    revision_plan = _normalize_css_revision_plan(state.get("css_revision_plan"))
    if artifact is None:
        message = "CSS Revision Executor could not run because coder_artifact is missing."
        print(f"[CSSRevisionExecutor] {message}")
        return {"patch_error": message}
    if page_plan is None:
        message = "CSS Revision Executor could not run because page_plan is missing."
        print(f"[CSSRevisionExecutor] {message}")
        return {"patch_error": message}
    if revision_plan is None:
        message = "CSS Revision Executor could not run because css_revision_plan is missing."
        print(f"[CSSRevisionExecutor] {message}")
        return {"patch_error": message}

    entry_html_path = Path(artifact.entry_html).resolve()
    manifest_path = build_page_manifest_path(entry_html_path)
    manifest = load_page_manifest(manifest_path)
    if manifest is None:
        print(f"[CSSRevisionExecutor] {LEGACY_PAGE_ERROR}")
        return {"patch_error": LEGACY_PAGE_ERROR}

    try:
        current_html = read_text_with_fallback(entry_html_path)
    except Exception as exc:
        message = f"CSS Revision Executor failed reading current HTML: {exc}"
        print(f"[CSSRevisionExecutor] {message}")
        return {"patch_error": message}

    asset_manifest = _available_asset_manifest_from_artifact(artifact)
    allowed_asset_web_paths = collect_allowed_asset_web_paths(asset_manifest)
    allowed_existing_local_sources = set(collect_local_image_sources(current_html))
    soup = BeautifulSoup(current_html, "html.parser")

    applied_replacements = 0
    for replacement in revision_plan.content_replacements:
        fragment_critiques = validate_fragment_local_image_sources(
            html_text=replacement.html,
            allowed_asset_web_paths=allowed_asset_web_paths,
            allowed_existing_local_sources=allowed_existing_local_sources,
        )
        if fragment_critiques:
            message = "CSS Revision Executor generated invalid local image references: " + "; ".join(fragment_critiques)
            print(f"[CSSRevisionExecutor] {message}")
            return {"patch_error": message}

        try:
            if replacement.scope == "slot":
                _apply_slot_replacement(
                    soup=soup,
                    block_id=str(replacement.block_id or ""),
                    slot_id=str(replacement.slot_id or ""),
                    html_fragment=replacement.html,
                )
            elif replacement.scope == "global":
                _apply_global_replacement(
                    soup=soup,
                    global_id=str(replacement.global_id or ""),
                    html_fragment=replacement.html,
                )
            else:
                _apply_block_replacement(
                    soup=soup,
                    block_id=str(replacement.block_id or ""),
                    html_fragment=replacement.html,
                    page_plan=page_plan,
                )
            applied_replacements += 1
        except Exception as exc:
            message = f"CSS Revision Executor failed applying content replacement: {exc}"
            print(f"[CSSRevisionExecutor] {message}")
            return {"patch_error": message}

    if revision_plan.css_rules:
        style_tag = _ensure_override_style_tag(soup)
        existing_text = style_tag.string or style_tag.get_text() or ""
        rendered_rules = [
            _render_override_rule(rule.selector, {str(prop): value for prop, value in rule.declarations.items()})
            for rule in revision_plan.css_rules
        ]
        combined = "\n\n".join([part for part in [existing_text.strip(), *rendered_rules] if part])
        style_tag.string = combined

    updated_html = str(soup)
    strict_validation = str(manifest.schema_version or "").strip() != "1.0"
    if strict_validation:
        asset_critiques = validate_local_image_references(
            html_text=updated_html,
            entry_html_path=entry_html_path,
            site_dir=Path(artifact.site_dir),
            allowed_asset_web_paths=allowed_asset_web_paths,
            enforce_paper_asset_whitelist=True,
        )
        if asset_critiques:
            message = "Post-revision asset validation failed: " + "; ".join(asset_critiques)
            print(f"[CSSRevisionExecutor] {message}")
            return {"patch_error": message}

    try:
        updated_manifest = extract_page_manifest(
            html_text=updated_html,
            entry_html=entry_html_path,
            selected_template_id=artifact.selected_template_id,
            page_plan=page_plan,
            require_expected_globals=strict_validation,
        )
    except Exception as exc:
        message = f"CSS revision validation failed after applying updates: {exc}"
        print(f"[CSSRevisionExecutor] {message}")
        return {"patch_error": message}

    updated_artifact = artifact.model_copy(deep=True)
    patch_target_paths = _build_patch_target_paths(updated_artifact, page_plan)
    site_dir = Path(updated_artifact.site_dir)
    edited_html_files = _to_site_relative_paths(patch_target_paths, site_dir)
    updated_artifact.edited_files = _merge_unique_strings(updated_artifact.edited_files, edited_html_files)
    updated_artifact.notes = _update_css_revision_notes(
        updated_artifact.notes,
        css_rule_count=len(revision_plan.css_rules),
        content_replacement_count=applied_replacements,
    )

    artifact_json_path = entry_html_path.parent.parent / "coder_artifact.json"
    file_contents: dict[Path, str] = {path: updated_html for path in patch_target_paths}
    file_contents[manifest_path] = json.dumps(updated_manifest.model_dump(), indent=2, ensure_ascii=False)
    file_contents[artifact_json_path] = json.dumps(updated_artifact.model_dump(), indent=2, ensure_ascii=False)

    try:
        _write_files_transaction(file_contents)
    except Exception as exc:
        message = str(exc)
        print(f"[CSSRevisionExecutor] {message}")
        return {"patch_error": message}

    summary = str(revision_plan.revision_summary or "").strip() or (
        f"applied {len(revision_plan.css_rules)} css rule(s) and {applied_replacements} content replacement(s)."
    )
    print(
        "[CSSRevisionExecutor] applied CSS revision: "
        f"{len(revision_plan.css_rules)} css rule(s), "
        f"{applied_replacements} content replacement(s)."
    )
    return {
        "coder_artifact": updated_artifact,
        "patch_error": "",
        "css_revision_summary": summary,
    }
