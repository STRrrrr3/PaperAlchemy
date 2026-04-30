from __future__ import annotations

import re
from typing import Iterable

from src.contracts.schemas import ArbiterReport, ReviewItem, RevisionRouteDecision
from src.contracts.state import WorkflowState
from src.services.human_feedback import extract_human_feedback_images, extract_human_feedback_text

_PATCH_KEYWORDS = {
    "asset",
    "author",
    "caption",
    "change",
    "copy",
    "figure",
    "heading",
    "image",
    "label",
    "logo",
    "media",
    "rename",
    "replace",
    "retitle",
    "text",
    "title",
    "wording",
    "换图",
    "图片",
    "图表",
    "标题",
    "文案",
    "文字",
    "作者",
}

_CSS_KEYWORDS = {
    "align",
    "alignment",
    "bottom",
    "box",
    "center",
    "color",
    "fixed",
    "font",
    "footer",
    "gap",
    "height",
    "layout",
    "margin",
    "max-width",
    "middle",
    "overflow",
    "padding",
    "position",
    "readability",
    "spacing",
    "style",
    "typography",
    "width",
    "遮挡",
    "布局",
    "间距",
    "宽度",
    "位置",
    "样式",
    "字体",
}

_STRUCTURAL_KEYWORDS = {
    "rebind",
    "replan",
    "restructure",
    "shell",
    "template",
    "模板",
    "重排",
    "重新规划",
}

_PATCH_PATTERNS = (
    re.compile(r"\b(?:wrong|incorrect)\s+(?:image|figure|asset)\b", re.IGNORECASE),
    re.compile(r"\b(?:asset|image|figure)\s+(?:rebind|replacement|swap)\b", re.IGNORECASE),
    re.compile(r"\b(?:change|rename|retitle|replace)\b.+\b(?:title|heading|caption|text|wording|image|figure|asset)\b", re.IGNORECASE),
)

_CSS_PATTERNS = (
    re.compile(r"\b(?:move|place)\b.+\b(?:footer|bottom|document flow)\b", re.IGNORECASE),
    re.compile(r"\b(?:max-width|margin|padding|spacing|font-size|line-height|position|fixed|overflow)\b", re.IGNORECASE),
)


def _normalize_review_items(value: object) -> list[ReviewItem]:
    if value is None:
        return []
    try:
        report = value if isinstance(value, ArbiterReport) else ArbiterReport.model_validate(value)
    except Exception:
        return []
    return list(report.items)


def _clean(value: object) -> str:
    return " ".join(str(value or "").split()).strip()


def _item_text(item: ReviewItem) -> str:
    return f"[{item.severity}] {item.target}: {item.advice}"


def _contains_any(text: str, keywords: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def _classify_text(text: str, *, preferred_route: str | None = None, has_images: bool = False) -> str:
    route_hint = str(preferred_route or "").strip().lower()
    if route_hint in {"css", "patch"}:
        return route_hint

    clean = _clean(text)
    if not clean and has_images:
        return "css"
    if not clean:
        return "none"

    patch_signal, css_signal = _detect_signals(clean)
    lowered = clean.lower()
    if _contains_any(lowered, _STRUCTURAL_KEYWORDS) and not patch_signal:
        return "none"

    if patch_signal and not css_signal:
        return "patch"
    if css_signal and not patch_signal:
        return "css"
    if patch_signal and css_signal:
        if any(pattern.search(clean) for pattern in _PATCH_PATTERNS):
            return "patch"
        return "css"
    return "css"


def _detect_signals(text: str) -> tuple[bool, bool]:
    clean = _clean(text)
    lowered = clean.lower()
    patch_signal = _contains_any(lowered, _PATCH_KEYWORDS) or any(pattern.search(clean) for pattern in _PATCH_PATTERNS)
    css_signal = _contains_any(lowered, _CSS_KEYWORDS) or any(pattern.search(clean) for pattern in _CSS_PATTERNS)
    return patch_signal, css_signal


def _split_user_feedback(text: str) -> list[str]:
    clean = _clean(text)
    if not clean:
        return []
    parts = re.split(r"(?:\n+|(?<=[.!?。！？])\s+|;\s+|\|\s+)", clean)
    return [part.strip() for part in parts if part.strip()]


def classify_revision_feedback(state: WorkflowState) -> RevisionRouteDecision:
    feedback = state.get("human_directives")
    user_text = _clean(extract_human_feedback_text(feedback))
    feedback_images = extract_human_feedback_images(feedback)
    if user_text or feedback_images:
        patch_items: list[str] = []
        css_items: list[str] = []
        none_items: list[str] = []
        segments = _split_user_feedback(user_text) or ([user_text] if user_text else [])
        if not segments and feedback_images:
            css_items.append("(screenshot-only visual revision request)")
        for segment in segments:
            preferred_route = _classify_text(segment, has_images=bool(feedback_images))
            patch_signal, css_signal = _detect_signals(segment)
            if patch_signal and css_signal:
                patch_items.append(segment)
                css_items.append(segment)
            elif preferred_route == "patch":
                patch_items.append(segment)
            elif preferred_route == "css":
                css_items.append(segment)
            else:
                none_items.append(segment)

        if patch_items and css_items:
            route = "mixed"
        elif patch_items:
            route = "patch"
        elif css_items:
            route = "css"
        else:
            route = "none"
        reason = "classified user feedback"
        if none_items and route == "none":
            reason = "user feedback appears structural or unsupported in v1"
        return RevisionRouteDecision(
            route=route,
            source="user",
            patch_text="\n".join(patch_items),
            css_text="\n".join(css_items),
            reason=reason,
            confidence=0.85 if route != "none" else 0.4,
        )

    items = _normalize_review_items(state.get("arbiter_review"))
    if not items:
        return RevisionRouteDecision(route="none", source="none", reason="no feedback to route", confidence=1.0)

    patch_items = []
    css_items = []
    unsupported_items = []
    for item in items:
        text = _item_text(item)
        route = _classify_text(text, preferred_route=item.preferred_route)
        if route == "patch":
            patch_items.append(text)
        elif route == "css":
            css_items.append(text)
        else:
            unsupported_items.append(text)

    if patch_items and css_items:
        route = "mixed"
    elif patch_items:
        route = "patch"
    elif css_items:
        route = "css"
    else:
        route = "none"

    reason_parts = [f"arbiter items classified as {route}"]
    if unsupported_items:
        reason_parts.append(f"{len(unsupported_items)} unsupported structural item(s)")
    return RevisionRouteDecision(
        route=route,
        source="arbiter",
        patch_text="\n".join(patch_items),
        css_text="\n".join(css_items),
        reason="; ".join(reason_parts),
        confidence=0.9 if route != "none" else 0.45,
    )


def revision_classifier_node(state: WorkflowState) -> dict[str, object]:
    decision = classify_revision_feedback(state)
    print(
        "[RevisionClassifier] "
        f"source={decision.source} route={decision.route} confidence={decision.confidence:.2f}: {decision.reason}"
    )
    patch_error = ""
    if decision.route == "none" and decision.source != "none":
        patch_error = decision.reason or "Revision request is not automatically supported in v1."
    return {
        "revision_route_decision": decision,
        "revision_plan": None,
        "targeted_replacement_plan": None,
        "css_revision_plan": None,
        "css_revision_summary": "",
        "patch_agent_output": "",
        "patch_applied_summary": "",
        "patch_error": patch_error,
    }
