from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from src.schemas import CoderArtifact, PagePlan


def _normalize_coder_artifact(artifact: Any) -> CoderArtifact | None:
    if artifact is None:
        return None
    if isinstance(artifact, CoderArtifact):
        return artifact
    try:
        return CoderArtifact.model_validate(artifact)
    except Exception:
        return None


def _normalize_page_plan(page_plan: Any) -> PagePlan | None:
    if page_plan is None:
        return None
    if isinstance(page_plan, PagePlan):
        return page_plan
    try:
        return PagePlan.model_validate(page_plan)
    except Exception:
        return None


def read_text_with_fallback(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def message_content_to_text(message: Any) -> str:
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


def resolve_template_entry_html_path(
    page_plan: PagePlan | Any | None,
    *,
    project_root: Path | None = None,
) -> Path | None:
    normalized_page_plan = _normalize_page_plan(page_plan)
    if normalized_page_plan is None:
        return None

    selected_root = str(normalized_page_plan.template_selection.selected_root_dir or "").strip()
    selected_entry = str(normalized_page_plan.template_selection.selected_entry_html or "").strip()
    if not selected_root or not selected_entry:
        return None

    effective_project_root = project_root or Path(__file__).resolve().parent.parent
    return effective_project_root / selected_root / selected_entry


def read_template_reference_html(
    page_plan: PagePlan | Any | None,
    *,
    project_root: Path | None = None,
    missing_value: str = "",
) -> str:
    template_entry_path = resolve_template_entry_html_path(
        page_plan,
        project_root=project_root,
    )
    if template_entry_path is None or not template_entry_path.exists():
        return missing_value

    try:
        return read_text_with_fallback(template_entry_path)
    except Exception:
        return missing_value


def read_current_page_html(
    artifact: CoderArtifact | Any | None,
    *,
    missing_value: str = "",
) -> str:
    normalized_artifact = _normalize_coder_artifact(artifact)
    if normalized_artifact is None:
        return missing_value

    entry_path = Path(normalized_artifact.entry_html)
    if not entry_path.exists():
        return missing_value

    try:
        return read_text_with_fallback(entry_path)
    except Exception:
        return missing_value


def extract_html_document(text: str) -> str:
    raw_text = str(text or "").strip()
    if not raw_text:
        return ""

    fenced_match = re.search(r"```(?:html)?\s*(.*?)```", raw_text, flags=re.IGNORECASE | re.DOTALL)
    if fenced_match:
        raw_text = fenced_match.group(1).strip()

    doctype_match = re.search(r"<!DOCTYPE\s+html[^>]*>", raw_text, flags=re.IGNORECASE)
    html_match = re.search(r"<html\b.*?</html>", raw_text, flags=re.IGNORECASE | re.DOTALL)

    if html_match:
        start = html_match.start()
        end = html_match.end()
        if doctype_match and doctype_match.start() <= start:
            raw_text = raw_text[doctype_match.start() : end]
        else:
            raw_text = raw_text[start:end]
    elif raw_text.lower().startswith("<!doctype") or raw_text.lower().startswith("<html"):
        pass
    else:
        return ""

    if "<html" not in raw_text.lower():
        return ""

    if "<!doctype" not in raw_text.lower():
        raw_text = "<!DOCTYPE html>\n" + raw_text

    return raw_text.strip()


def extract_html_fragment(text: str) -> str:
    raw_text = str(text or "").strip()
    if not raw_text:
        return ""

    fenced_match = re.search(r"```(?:html)?\s*(.*?)```", raw_text, flags=re.IGNORECASE | re.DOTALL)
    if fenced_match:
        raw_text = fenced_match.group(1).strip()

    return raw_text.strip()


def normalize_html_document_whitespace(html_text: str) -> str:
    return str(html_text or "").replace("\r\n", "\n").strip() + "\n"
