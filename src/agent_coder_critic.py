import base64
import json
from collections.abc import Callable
from pathlib import Path
import re
from typing import Any

from bs4 import BeautifulSoup
from langchain_core.messages import HumanMessage, SystemMessage

from src.llm import get_llm
from src.preview_utils import build_visual_critic_screenshot_path, take_local_screenshot
from src.schemas import CoderArtifact, CoderCriticReport
from src.state import CoderState

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


def _normalize_visual_feedback_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "passed": bool(payload.get("passed")),
        "issues": _normalize_string_list(payload.get("issues")),
        "selectors_to_remove": _normalize_string_list(payload.get("selectors_to_remove")),
        "css_rules_to_inject": _normalize_string_list(
            payload.get("css_rules_to_inject") or payload.get("css_rules")
        ),
    }


def run_coder_code_critic(artifact: CoderArtifact | None) -> list[str]:
    critiques: list[str] = []
    if not artifact:
        critiques.append("Coder output is empty or failed schema validation.")
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

    is_dom_injection_build = "v2-dom-injection" in (artifact.notes or "")
    if not is_dom_injection_build:
        if "PaperAlchemy Generated Body Start" not in html_text or "PaperAlchemy Generated Body End" not in html_text:
            critiques.append("Generated body markers are missing in entry html.")

        body_start_pattern = re.compile(
            r"<body[^>]*>\s*<!--\s*PaperAlchemy Generated Body Start\s*-->",
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not body_start_pattern.search(html_text):
            critiques.append("Generated body marker is not at body start; template content leakage is likely.")
    elif body_tag is not None:
        visible_text = " ".join(body_tag.get_text(" ", strip=True).split())
        has_media = body_tag.find(["img", "video", "iframe", "svg", "canvas"]) is not None
        if not visible_text and not has_media:
            critiques.append("DOM-injection build produced an empty body.")

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
    critiques = run_coder_code_critic(artifact)

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
        return {"visual_iterations": iteration, "is_visually_approved": True}

    screenshot_path = Path(screenshot_path_text)
    if not screenshot_path.exists():
        print(f"[PaperAlchemy-VisionCritic] screenshot not found: {screenshot_path}")
        return {"visual_iterations": iteration, "is_visually_approved": True}

    artifact = _normalize_coder_artifact(state.get("coder_artifact"))
    image_data_url = _encode_image_to_data_url(screenshot_path)
    if not image_data_url:
        return {"visual_iterations": iteration, "is_visually_approved": True}

    system_prompt = """
You are an expert Frontend QA Engineer.
Analyze this screenshot of an academic project page and return strict JSON only.

Look for critical visual bugs:
1. Dummy text such as Lorem Ipsum or placeholder copy.
2. Irrelevant template leftovers such as unrelated university names, stale copyright footers, template leaderboards, or foreign-brand sections that do not belong to the paper.
3. Severe overlap, clipping, unreadable stacking, broken hero areas, or obviously broken images.

Return exactly:
{
  "passed": true | false,
  "issues": ["string"],
  "selectors_to_remove": ["string"],
  "css_rules_to_inject": ["string"]
}

Rules:
- If the page looks visually clean, set passed=true and leave the lists empty.
- If exact selectors are uncertain, keep selectors_to_remove empty rather than inventing unsafe selectors.
- Prefer small, concrete CSS fixes in css_rules_to_inject.
- Do not return markdown.
""".strip()

    entry_html_path = str(Path(artifact.entry_html).resolve()) if artifact else "(unknown)"
    selected_template_id = str(artifact.selected_template_id) if artifact else "(unknown)"
    user_prompt = (
        "Review this rendered page screenshot.\n"
        f"Current entry html: {entry_html_path}\n"
        f"Template id: {selected_template_id}\n"
        "Return strict JSON with actionable selectors_to_remove and css_rules_to_inject.\n"
    )

    try:
        llm = get_llm(temperature=0, use_smart_model=False)
        response = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(
                    content=[
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": image_data_url},
                    ]
                ),
            ]
        )
        payload = _extract_json_object(_message_content_to_text(response))
        if not payload:
            print("[PaperAlchemy-VisionCritic] invalid JSON response; skipping visual retry.")
            return {"visual_iterations": iteration, "is_visually_approved": True}

        normalized_payload = _normalize_visual_feedback_payload(payload)
        if normalized_payload["passed"]:
            print("[PaperAlchemy-VisionCritic] visual QA passed.")
            return {"visual_iterations": iteration, "is_visually_approved": True}

        feedback_json = json.dumps(normalized_payload, ensure_ascii=False)
        issues_text = "; ".join(normalized_payload["issues"]) or "visual bugs detected"
        print(f"[PaperAlchemy-VisionCritic] visual QA rejected: {issues_text}")
        return {
            "visual_iterations": iteration,
            "is_visually_approved": False,
            "visual_feedback": [feedback_json],
        }
    except Exception as exc:
        print(f"[PaperAlchemy-VisionCritic] vision QA failed unexpectedly: {exc}")
        return {"visual_iterations": iteration, "is_visually_approved": True}


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
        if state.get("is_visually_approved"):
            return "end"
        if int(state.get("visual_iterations", 0)) >= max_retry:
            print(f"[PaperAlchemy-VisionCritic] reached max visual retry limit ({max_retry}), stop.")
            return "end"
        return "retry"

    return _router
