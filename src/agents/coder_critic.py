import base64
import json
from collections.abc import Callable
from pathlib import Path
import re
from typing import Any

from bs4 import BeautifulSoup
from langchain_core.messages import HumanMessage, SystemMessage

from src.utils.html_utils import message_content_to_text
from src.services.llm import get_llm
from src.validators.page_manifest import build_page_manifest_path, extract_page_manifest, load_page_manifest
from src.validators.page_validation import collect_allowed_asset_web_paths, validate_local_image_references
from src.services.preview_service import build_visual_critic_screenshot_path, take_local_screenshot
from src.prompts import VISION_CRITIC_SYSTEM_PROMPT, VISION_CRITIC_USER_PROMPT_TEMPLATE
from src.contracts.schemas import CoderArtifact, PagePlan, VisualSmokeReport
from src.contracts.state import CoderState

MAX_CODER_RETRY_DEFAULT = 1
MAX_VISUAL_QA_ITERATIONS_DEFAULT = 2
VISUAL_QA_STYLE_TAG_ID = "paperalchemy-visual-qa-overrides"


def _normalize_coder_artifact(artifact: Any) -> CoderArtifact | None:
    if artifact is None:
        return None
    if isinstance(artifact, CoderArtifact):
        return artifact
    try:
        return CoderArtifact.model_validate(artifact)
    except Exception:
        return None


def _normalize_page_plan(plan: Any) -> PagePlan | None:
    if plan is None:
        return None
    if isinstance(plan, PagePlan):
        return plan
    try:
        return PagePlan.model_validate(plan)
    except Exception:
        return None


def _extract_json_object(text: str) -> dict[str, Any] | None:
    raw_text = str(text or "").strip()
    if not raw_text:
        return None

    fenced_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw_text, flags=re.DOTALL)
    if fenced_match:
        raw_text = fenced_match.group(1).strip()

    try:
        parsed = json.loads(raw_text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass

    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        parsed = json.loads(raw_text[start : end + 1])
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _encode_image_to_data_url(image_path: Path) -> str | None:
    try:
        image_bytes = image_path.read_bytes()
    except Exception as exc:
        print(f"[PaperAlchemy-VisionCritic] failed to read screenshot: {exc}")
        return None

    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _normalize_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        clean = value.strip()
        return [clean] if clean else []

    if not isinstance(value, list):
        return []

    normalized: list[str] = []
    seen: set[str] = set()
    for item in value:
        clean = str(item or "").strip()
        if clean and clean not in seen:
            seen.add(clean)
            normalized.append(clean)
    return normalized


def _normalize_issue_class(value: Any, *, passed: bool) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"none", "cosmetic", "structure"}:
        return normalized
    return "none" if passed else "cosmetic"


def _normalize_suggested_recovery(value: Any, *, passed: bool, issue_class: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"accept", "patch_or_review", "rerun_planner"}:
        return normalized
    if passed or issue_class == "none":
        return "accept"
    if issue_class == "structure":
        return "rerun_planner"
    return "patch_or_review"


def _normalize_visual_feedback_payload(payload: dict[str, Any]) -> dict[str, Any]:
    passed = bool(payload.get("passed"))
    issue_class = _normalize_issue_class(payload.get("issue_class"), passed=passed)
    return {
        "passed": passed,
        "issue_class": issue_class,
        "suggested_recovery": _normalize_suggested_recovery(
            payload.get("suggested_recovery"),
            passed=passed,
            issue_class=issue_class,
        ),
        "issues": _normalize_string_list(payload.get("issues")),
        "selectors_to_remove": _normalize_string_list(payload.get("selectors_to_remove")),
        "css_rules_to_inject": _normalize_string_list(
            payload.get("css_rules_to_inject") or payload.get("css_rules")
        ),
    }


def _build_visual_smoke_report(
    payload: dict[str, Any] | None = None,
    screenshot_path: str = "",
) -> VisualSmokeReport:
    normalized_payload = _normalize_visual_feedback_payload(payload or {})
    return VisualSmokeReport(
        passed=bool(normalized_payload.get("passed", True)),
        issue_class=str(normalized_payload.get("issue_class") or "none"),
        suggested_recovery=str(normalized_payload.get("suggested_recovery") or "accept"),
        issues=normalized_payload.get("issues") or [],
        selectors_to_remove=normalized_payload.get("selectors_to_remove") or [],
        css_rules_to_inject=normalized_payload.get("css_rules_to_inject") or [],
        screenshot_path=str(screenshot_path or "").strip(),
    )


def _available_asset_manifest_from_artifact(artifact: CoderArtifact) -> list[dict[str, str]]:
    site_dir = Path(artifact.site_dir)
    entry_html_parent = Path(artifact.entry_html).parent
    manifest: list[dict[str, str]] = []
    for rel_path in artifact.copied_assets:
        asset_path = site_dir / rel_path
        if not asset_path.exists():
            continue
        try:
            web_path = asset_path.relative_to(entry_html_parent).as_posix()
        except ValueError:
            web_path = Path(
                Path(*asset_path.parts[len(entry_html_parent.parts) :]).as_posix()
            ).as_posix()
        if not web_path.startswith((".", "/")):
            web_path = f"./{web_path}"
        manifest.append({"web_path": web_path})
    return manifest


def run_coder_code_critic(artifact: CoderArtifact | None, page_plan: PagePlan | None) -> list[str]:
    critiques: list[str] = []
    if not artifact:
        critiques.append("Coder output is empty or failed schema validation.")
        return critiques
    if not page_plan:
        critiques.append("Page plan is missing, so anchored revision structure cannot be verified.")
        return critiques

    site_dir = Path(artifact.site_dir)
    entry_html = Path(artifact.entry_html)

    if not site_dir.exists():
        critiques.append(f"Generated site directory does not exist: {site_dir}")
        return critiques

    if not entry_html.exists():
        critiques.append(f"Entry html does not exist: {entry_html}")
        return critiques

    try:
        html_text = entry_html.read_text(encoding="utf-8")
    except Exception as exc:
        critiques.append(f"Cannot read entry html: {exc}")
        return critiques

    try:
        soup = BeautifulSoup(html_text, "html.parser")
    except Exception as exc:
        critiques.append(f"Cannot parse entry html: {exc}")
        return critiques

    body_tag = soup.body
    if body_tag is None:
        critiques.append("Entry html does not contain a <body> element.")

    if "PaperAlchemy Generated Body Start" not in html_text or "PaperAlchemy Generated Body End" not in html_text:
        critiques.append("Generated body markers are missing in entry html.")

    body_start_pattern = re.compile(
        r"<body[^>]*>\s*<!--\s*PaperAlchemy Generated Body Start\s*-->",
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not body_start_pattern.search(html_text):
        critiques.append("Generated body marker is not at body start; template content leakage is likely.")

    manifest_path = build_page_manifest_path(entry_html)
    manifest = load_page_manifest(manifest_path)
    if manifest is None:
        critiques.append(f"Anchored revision manifest is missing or invalid: {manifest_path}")
    else:
        try:
            rebuilt_manifest = extract_page_manifest(
                html_text=html_text,
                entry_html=entry_html,
                selected_template_id=artifact.selected_template_id,
                page_plan=page_plan,
                require_expected_globals=str(manifest.schema_version or "").strip() != "1.0",
            )
            if manifest.model_dump() != rebuilt_manifest.model_dump():
                critiques.append("page_manifest.json is out of sync with current entry html anchors.")
        except Exception as exc:
            critiques.append(f"Anchored revision validation failed: {exc}")

        expected_block_order = [item.block_id for item in sorted(page_plan.page_outline, key=lambda item: item.order)]
        actual_block_order = [item.block_id for item in manifest.blocks]
        if actual_block_order != expected_block_order:
            critiques.append(
                "Generated data-pa-block order does not match approved page_outline order. "
                f"expected={expected_block_order}, actual={actual_block_order}"
            )

    critiques.extend(
        validate_local_image_references(
            html_text=html_text,
            entry_html_path=entry_html,
            site_dir=site_dir,
            allowed_asset_web_paths=collect_allowed_asset_web_paths(
                _available_asset_manifest_from_artifact(artifact)
            ),
            enforce_paper_asset_whitelist=True,
        )
    )

    title_count = len(re.findall(r"<title\b", html_text, flags=re.IGNORECASE))
    if title_count != 1:
        critiques.append(f"Expected exactly one <title> tag, found {title_count}.")

    referenced_values: list[str] = []
    for element in soup.find_all(True):
        for attr_name in ("src", "href", "data-src", "poster"):
            attr_value = element.get(attr_name)
            if isinstance(attr_value, str) and attr_value.strip():
                referenced_values.append(attr_value.strip().replace("\\", "/"))

    for rel_asset in artifact.copied_assets:
        asset_path = site_dir / rel_asset
        if not asset_path.exists():
            critiques.append(f"Copied asset missing: {asset_path}")
        rel_asset_norm = rel_asset.replace("\\", "/")
        if not any(rel_asset_norm in value for value in referenced_values):
            critiques.append(f"Copied asset is not referenced in entry html: {rel_asset}")

    return critiques


def coder_critic_node(state: CoderState) -> dict[str, Any]:
    print("[PaperAlchemy-CoderCritic] running build checks...")
    artifact = _normalize_coder_artifact(state.get("coder_artifact"))
    page_plan = _normalize_page_plan(state.get("page_plan"))
    critiques = run_coder_code_critic(artifact, page_plan)

    if critiques:
        feedback = "\n".join(critiques)
        print(f"[PaperAlchemy-CoderCritic] build rejected:\n{feedback}")
        return {
            "coder_critic_passed": False,
            "coder_feedback_history": [feedback],
            "coder_retry_count": int(state.get("coder_retry_count", 0)) + 1,
        }

    print("[PaperAlchemy-CoderCritic] build checks passed.")
    return {"coder_critic_passed": True}


def take_screenshot_action(state: CoderState) -> dict[str, Any]:
    artifact = _normalize_coder_artifact(state.get("coder_artifact"))
    if not artifact:
        print("[PaperAlchemy-VisionCritic] screenshot step skipped: missing coder artifact.")
        return {"visual_screenshot_path": ""}

    entry_html_path = Path(artifact.entry_html).resolve()
    screenshot_path = take_local_screenshot(
        str(entry_html_path),
        str(build_visual_critic_screenshot_path(entry_html_path)),
    )
    if not screenshot_path:
        print("[PaperAlchemy-VisionCritic] screenshot step skipped: failed to render current html.")
        return {"visual_screenshot_path": ""}

    print(f"[PaperAlchemy-VisionCritic] captured screenshot at {screenshot_path}")
    return {"visual_screenshot_path": screenshot_path}


def vision_critic_node(state: CoderState) -> dict[str, Any]:
    iteration = int(state.get("visual_iterations", 0)) + 1
    screenshot_path_text = str(state.get("visual_screenshot_path") or "").strip()
    if not screenshot_path_text:
        print("[PaperAlchemy-VisionCritic] no screenshot available; skipping visual QA.")
        return {
            "visual_iterations": iteration,
            "is_visually_approved": True,
            "visual_smoke_report": VisualSmokeReport(),
        }

    screenshot_path = Path(screenshot_path_text)
    if not screenshot_path.exists():
        print(f"[PaperAlchemy-VisionCritic] screenshot not found: {screenshot_path}")
        return {
            "visual_iterations": iteration,
            "is_visually_approved": True,
            "visual_smoke_report": VisualSmokeReport(),
        }

    artifact = _normalize_coder_artifact(state.get("coder_artifact"))
    image_data_url = _encode_image_to_data_url(screenshot_path)
    if not image_data_url:
        return {
            "visual_iterations": iteration,
            "is_visually_approved": True,
            "visual_smoke_report": VisualSmokeReport(screenshot_path=screenshot_path_text),
        }

    entry_html_path = str(Path(artifact.entry_html).resolve()) if artifact else "(unknown)"
    selected_template_id = str(artifact.selected_template_id) if artifact else "(unknown)"
    user_prompt = VISION_CRITIC_USER_PROMPT_TEMPLATE.format(
        entry_html_path=entry_html_path,
        selected_template_id=selected_template_id,
    )

    try:
        llm = get_llm(temperature=0, use_smart_model=False)
        response = llm.invoke(
            [
                SystemMessage(content=VISION_CRITIC_SYSTEM_PROMPT),
                HumanMessage(
                    content=[
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": image_data_url},
                    ]
                ),
            ]
        )
        payload = _extract_json_object(message_content_to_text(response))
        if not payload:
            print("[PaperAlchemy-VisionCritic] invalid JSON response; skipping visual retry.")
            return {
                "visual_iterations": iteration,
                "is_visually_approved": True,
                "visual_smoke_report": VisualSmokeReport(screenshot_path=screenshot_path_text),
            }

        normalized_payload = _normalize_visual_feedback_payload(payload)
        report = _build_visual_smoke_report(normalized_payload, screenshot_path_text)
        if normalized_payload["passed"]:
            print("[PaperAlchemy-VisionCritic] visual QA passed.")
            return {
                "visual_iterations": iteration,
                "is_visually_approved": True,
                "visual_smoke_report": report,
            }

        feedback_json = json.dumps(normalized_payload, ensure_ascii=False)
        issues_text = "; ".join(normalized_payload["issues"]) or "visual bugs detected"
        print(f"[PaperAlchemy-VisionCritic] visual QA rejected: {issues_text}")
        return {
            "visual_iterations": iteration,
            "is_visually_approved": False,
            "visual_feedback": [feedback_json],
            "visual_smoke_report": report,
        }
    except Exception as exc:
        print(f"[PaperAlchemy-VisionCritic] vision QA failed unexpectedly: {exc}")
        return {
            "visual_iterations": iteration,
            "is_visually_approved": True,
            "visual_smoke_report": VisualSmokeReport(screenshot_path=screenshot_path_text),
        }


def build_coder_critic_router(max_retry: int = MAX_CODER_RETRY_DEFAULT) -> Callable[[CoderState], str]:
    def _router(state: CoderState) -> str:
        if state.get("coder_critic_passed"):
            return "visual_qa"
        if int(state.get("coder_retry_count", 0)) >= max_retry:
            print(f"[PaperAlchemy-CoderCritic] reached max retry limit ({max_retry}), stop.")
            return "end"
        return "retry"

    return _router


def build_vision_qa_router(
    max_retry: int = MAX_VISUAL_QA_ITERATIONS_DEFAULT,
) -> Callable[[CoderState], str]:
    def _router(state: CoderState) -> str:
        _ = max_retry
        _ = state
        return "end"

    return _router

