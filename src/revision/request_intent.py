from __future__ import annotations

import re
from typing import Literal

_CONTENT_KEYWORDS = {
    "title",
    "heading",
    "label",
    "text",
    "wording",
    "copy",
    "caption",
    "rename",
    "retitle",
    "author",
    "authors",
    "affiliation",
    "affiliations",
    "abstract",
    "subtitle",
    "image",
    "figure",
    "logo",
    "section name",
    "标题",
    "文案",
    "正文",
    "摘要",
    "作者",
    "图片",
    "图像",
    "配图",
    "图表",
    "插图",
    "架构图",
    "流程图",
    "曲线图",
    "换图",
    "替换图片",
    "替换配图",
    "更换图片",
    "更换配图",
}
_VISUAL_KEYWORDS = {
    "align",
    "alignment",
    "background",
    "banner",
    "border",
    "box",
    "button",
    "card",
    "center",
    "color",
    "font",
    "header",
    "height",
    "layout",
    "margin",
    "nav",
    "navigation",
    "padding",
    "position",
    "radius",
    "shadow",
    "size",
    "spacing",
    "typography",
    "underline",
    "visible",
    "visibility",
    "width",
    "wrapper",
    "布局",
    "排版",
    "样式",
    "间距",
    "边距",
    "内边距",
    "对齐",
    "居中",
    "颜色",
    "字体",
    "边框",
    "宽度",
    "高度",
    "背景",
}
_EXPLICIT_CONTENT_PATTERNS = (
    r"\brename\b",
    r"\bretitle\b",
    r"\bchange\b.+\bto\b",
    r"\breplace\b.+\bwith\b",
    r"\bupdate\b.+\btext\b",
    r"\bmake\b.+\b(read|say|show)\b",
    r"替换.+为",
    r"更换.+为",
    r"把.+换成",
    r"把.+替换成",
    r"改成.+图",
)


def has_explicit_content_change(text: str) -> bool:
    lowered = str(text or "").strip().lower()
    return any(re.search(pattern, lowered) for pattern in _EXPLICIT_CONTENT_PATTERNS)


def classify_revision_request(text: str, has_images: bool) -> Literal["content", "visual", "mixed"]:
    lowered = str(text or "").strip().lower()
    if not lowered:
        return "visual" if has_images else "content"

    has_content_signal = any(keyword in lowered for keyword in _CONTENT_KEYWORDS) or has_explicit_content_change(lowered)
    has_visual_signal = any(keyword in lowered for keyword in _VISUAL_KEYWORDS)
    if has_content_signal and has_visual_signal:
        return "mixed"
    if has_content_signal:
        return "content"
    return "visual"
