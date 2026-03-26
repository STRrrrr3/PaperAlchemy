from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

from bs4 import BeautifulSoup, Tag
from langchain_core.messages import HumanMessage, SystemMessage

from src.deterministic_template_selector import score_and_select_template
from src.llm import get_llm
from src.page_manifest import _capture_wrapper_chain, _tag_classes, _tag_ids, _tag_tokens
from src.planner_template_catalog import find_entry_html_candidates
from src.schemas import (
    TemplateCandidate,
    TemplateProfile,
    TemplateShellCandidate,
    TemplateWidget,
)
from src.template_resources import SyncedTemplateAssets, ensure_autopage_template_assets

_CACHE_DIR_NAME = ".paperalchemy"
_CACHE_PROFILE_DIR_NAME = "template_compile_cache"
_MAX_SHELL_CANDIDATES = 24
_MAX_STYLE_FILES = 8
_MAX_SCRIPT_FILES = 8
_MAX_CACHE_FILES_PER_TEMPLATE = 6
_LLM_ENRICHMENT_SYSTEM_PROMPT = """You refine a deterministic template compilation summary for PaperAlchemy.

Return strict JSON only with:
{
  "archetype": "string",
  "risk_flags": ["string"],
  "notes": ["string"]
}

Rules:
- Do not invent selectors or widgets.
- Use concise archetype names such as hero_bulma, bootstrap_navbar, single_column_article, chart_fetch_dashboard, generic_multi_section.
- Only describe risks that are directly supported by the provided summary.
"""
_LLM_ENRICHMENT_USER_TEMPLATE = """Refine this template summary.

### TEMPLATE_SUMMARY_JSON
{summary_json}
"""


def _to_project_relative_path(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def _normalize_template_candidate(candidate: Any) -> TemplateCandidate | None:
    if isinstance(candidate, TemplateCandidate):
        return candidate
    if candidate is None:
        return None
    try:
        return TemplateCandidate.model_validate(candidate)
    except Exception:
        return None


def _candidate_from_ranked_match(
    ranked_match: dict[str, Any],
    project_root: Path,
) -> TemplateCandidate | None:
    template_id = str(ranked_match.get("template_id") or "").strip()
    template_path = str(ranked_match.get("template_path") or "").strip()
    entry_html = str(ranked_match.get("entry_html") or "").strip()
    if not template_id or not template_path or not entry_html:
        return None

    reasons = [
        str(item).strip()
        for item in (ranked_match.get("reasons") or [])
        if str(item).strip()
    ]
    return TemplateCandidate(
        template_id=template_id,
        root_dir=_to_project_relative_path(Path(template_path), project_root),
        chosen_entry_html=entry_html,
        score=float(ranked_match.get("score") or 0.0),
        reasons=reasons[:8],
    )


def _select_designated_template(
    constraints: dict[str, Any],
    project_root: Path,
) -> tuple[list[TemplateCandidate], TemplateCandidate | None]:
    raw_candidates = constraints.get("ui_ranked_candidates") or []
    if not isinstance(raw_candidates, list):
        raw_candidates = []

    top_k = max(1, int(constraints.get("template_candidate_top_k", 3)))
    candidates = [
        candidate
        for candidate in (
            _candidate_from_ranked_match(item, project_root)
            for item in raw_candidates[:top_k]
            if isinstance(item, dict)
        )
        if candidate is not None
    ]

    designated_template_id = str(constraints.get("designated_template_id") or "").strip()
    designated_template_path = str(constraints.get("designated_template_path") or "").strip()
    designated_entry_html = str(constraints.get("designated_entry_html") or "").strip()
    if not (designated_template_id and designated_template_path and designated_entry_html):
        return candidates, None

    selected_match = next(
        (
            item
            for item in raw_candidates
            if isinstance(item, dict)
            and str(item.get("template_id") or "").strip() == designated_template_id
            and str(item.get("entry_html") or "").strip() == designated_entry_html
        ),
        None,
    )
    if selected_match is None:
        selected_match = {
            "template_id": designated_template_id,
            "template_name": designated_template_id,
            "template_path": designated_template_path,
            "entry_html": designated_entry_html,
            "entry_html_path": str((Path(designated_template_path) / designated_entry_html).resolve()),
            "score": float(constraints.get("designated_template_score") or 0.0),
            "reasons": ["manual_ui_selection"],
        }

    selected = _candidate_from_ranked_match(selected_match, project_root)
    if selected is None:
        return candidates, None

    if selected.template_id not in {item.template_id for item in candidates}:
        candidates = [selected, *candidates]
    else:
        candidates = [selected, *[item for item in candidates if item.template_id != selected.template_id]]
    return candidates, selected


def select_template_candidates(
    *,
    project_root: Path,
    generation_constraints: dict[str, Any] | None,
    user_constraints: dict[str, Any] | None,
    synced_assets: SyncedTemplateAssets | None = None,
) -> tuple[list[TemplateCandidate], TemplateCandidate, SyncedTemplateAssets]:
    constraints = dict(generation_constraints or {})
    assets = synced_assets or ensure_autopage_template_assets(
        project_root=project_root,
        force=bool(constraints.get("force_template_sync")),
    )
    constraints.setdefault("template_tags_json_path", str(assets.tags_json_path))

    designated_candidates, designated_selected = _select_designated_template(
        constraints=constraints,
        project_root=project_root,
    )
    if designated_selected is not None:
        return designated_candidates or [designated_selected], designated_selected, assets

    selection_result = score_and_select_template(
        user_constraints=user_constraints or {},
        tags_json_path=constraints["template_tags_json_path"],
    )
    top_k = max(1, int(constraints.get("template_candidate_top_k", 3)))
    ranked_matches = list(selection_result.get("ranking") or [])
    candidates = [
        candidate
        for candidate in (
            _candidate_from_ranked_match(item, project_root)
            for item in ranked_matches[:top_k]
        )
        if candidate is not None
    ]
    selected = _candidate_from_ranked_match(selection_result, project_root)
    if selected is None:
        raise ValueError("Template selection returned an invalid selected candidate.")
    if not candidates:
        candidates = [selected]
    return candidates, selected, assets


def _resolve_template_root(project_root: Path, candidate: TemplateCandidate) -> Path:
    root_dir = Path(str(candidate.root_dir or "").strip())
    if root_dir.is_absolute():
        return root_dir
    return (project_root / root_dir).resolve()


def _resolve_entry_html_path(project_root: Path, candidate: TemplateCandidate) -> Path:
    return (_resolve_template_root(project_root, candidate) / candidate.chosen_entry_html).resolve()


def _iter_template_files(root: Path, suffixes: set[str], max_depth: int = 4) -> Iterable[Path]:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in suffixes:
            continue
        try:
            depth = len(path.relative_to(root).parts)
        except Exception:
            depth = max_depth + 1
        if depth <= max_depth:
            yield path


def _read_text_safely(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")
    except Exception:
        return ""


def _build_source_fingerprint(template_root: Path, entry_html_path: Path) -> str:
    hasher = hashlib.sha256()
    watched_files: list[Path] = [entry_html_path]
    watched_files.extend(sorted(_iter_template_files(template_root, {".css"}, max_depth=3), key=lambda item: str(item))[:_MAX_STYLE_FILES])
    watched_files.extend(sorted(_iter_template_files(template_root, {".js", ".mjs", ".ts"}, max_depth=3), key=lambda item: str(item))[:_MAX_SCRIPT_FILES])

    for path in watched_files:
        text = _read_text_safely(path)
        hasher.update(str(path.relative_to(template_root)).encode("utf-8", errors="ignore"))
        hasher.update(text.encode("utf-8", errors="ignore"))
    return hasher.hexdigest()


def _selector_segment(tag: Tag) -> str:
    tag_name = str(tag.name or "div")
    tag_id = str(tag.get("id") or "").strip()
    if tag_id:
        return f"{tag_name}#{tag_id}"

    classes = [name for name in _tag_classes(tag)[:2] if name]
    if classes:
        return tag_name + "".join(f".{name}" for name in classes)

    siblings = [sibling for sibling in tag.find_previous_siblings(tag_name) if isinstance(sibling, Tag)]
    if siblings:
        return f"{tag_name}:nth-of-type({len(siblings) + 1})"
    return tag_name


def build_unique_selector(tag: Tag, soup: BeautifulSoup) -> str:
    segments: list[str] = []
    current: Tag | None = tag
    while current is not None and isinstance(current, Tag):
        if str(current.name or "") in {"html", "body"}:
            break
        segments.insert(0, _selector_segment(current))
        selector = " > ".join(segments)
        try:
            matches = [match for match in soup.select(selector) if isinstance(match, Tag)]
        except Exception:
            matches = []
        if len(matches) == 1 and matches[0] is tag:
            return selector
        parent = current.parent
        current = parent if isinstance(parent, Tag) else None
    return " > ".join(segments) or str(tag.name or "div")


def _is_candidate_root(tag: Tag) -> bool:
    tag_name = str(tag.name or "")
    if tag_name not in {"section", "div", "article", "header", "main", "aside", "nav", "footer"}:
        return False
    if tag.find_parent(["script", "style", "noscript"]) is not None:
        return False
    tokens = _tag_tokens(tag)
    if tokens & {"modal", "toast", "tooltip", "dropdown"}:
        return False
    if tag_name in {"header", "nav", "footer", "main", "article"}:
        return True
    if tag.get("id") or _tag_classes(tag):
        return True
    return tag.find(["h1", "h2", "h3", "img", "figure", "table", "canvas", "svg"]) is not None


def _infer_shell_role(tag: Tag) -> tuple[str, list[str]]:
    tag_name = str(tag.name or "")
    tokens = _tag_tokens(tag)
    signals: list[str] = []
    role = "section"
    if tag_name == "footer" or "footer" in tokens:
        role = "footer"
        signals.append("footer_token")
    elif tag_name == "nav" or tokens & {"nav", "navbar", "menu"}:
        role = "nav"
        signals.append("nav_token")
    elif tag_name == "header" or tokens & {"hero", "lead", "intro", "banner", "masthead", "jumbotron"}:
        role = "hero"
        signals.append("hero_token")
    elif tag.find("table") is not None or tokens & {"table", "metric", "metrics", "results", "benchmark"}:
        role = "table"
        signals.append("table_token")
    elif tag.find(["img", "video", "figure", "canvas", "svg"]) is not None or tokens & {
        "gallery",
        "carousel",
        "media",
        "video",
        "image",
        "figure",
        "chart",
    }:
        role = "gallery"
        signals.append("media_token")
    return role, signals


def _dom_index_for_tag(template_soup: BeautifulSoup, target: Tag) -> int:
    root = template_soup.body or template_soup
    for dom_index, tag in enumerate(root.find_all(True)):
        if tag is target:
            return dom_index
    return 10_000


def discover_template_shell_candidates(template_html: str) -> list[TemplateShellCandidate]:
    template_soup = BeautifulSoup(str(template_html or ""), "html.parser")
    root = template_soup.body or template_soup
    candidates: list[TemplateShellCandidate] = []
    seen: set[str] = set()

    for tag in root.find_all(True):
        if not _is_candidate_root(tag):
            continue
        selector = build_unique_selector(tag, template_soup)
        if not selector or selector in seen:
            continue
        seen.add(selector)
        role, signals = _infer_shell_role(tag)
        confidence = 0.45
        if "#" in selector:
            confidence += 0.2
            signals.append("id_selector")
        elif "." in selector:
            confidence += 0.12
            signals.append("class_selector")
        if tag.find(["h1", "h2", "h3", "h4"]) is not None:
            confidence += 0.08
            signals.append("has_heading")
        if tag.find(["img", "video", "figure", "canvas", "svg", "table"]) is not None:
            confidence += 0.08
            signals.append("has_media")
        if str(tag.name or "") in {"header", "main", "article", "section", "footer", "nav"}:
            confidence += 0.08
            signals.append("semantic_tag")
        if ":nth-of-type" in selector:
            confidence -= 0.1
            signals.append("positional_selector")
        dom_index = _dom_index_for_tag(template_soup, tag)
        candidates.append(
            TemplateShellCandidate(
                selector=selector,
                role=role,  # type: ignore[arg-type]
                root_tag=str(tag.name or "div"),
                required_classes=_tag_classes(tag),
                preserve_ids=_tag_ids(tag),
                wrapper_chain=_capture_wrapper_chain(tag),
                dom_index=dom_index,
                confidence=round(max(0.05, min(confidence, 0.99)), 3),
                signals=sorted(set(signals)),
            )
        )

    candidates.sort(key=lambda item: (item.dom_index, -item.confidence, item.selector))
    return candidates[:_MAX_SHELL_CANDIDATES]


def _collect_global_preserve_selectors(template_html: str) -> list[str]:
    soup = BeautifulSoup(str(template_html or ""), "html.parser")
    root = soup.body or soup
    selectors: list[str] = []
    seen: set[str] = set()
    sticky_pattern = re.compile(r"\b(sticky|affix|toc|table-of-contents|sidebar)\b", flags=re.IGNORECASE)

    for tag in root.find_all(True):
        if not isinstance(tag, Tag):
            continue
        tokens = _tag_tokens(tag)
        keep = False
        if str(tag.name or "") in {"header", "nav", "footer"}:
            keep = True
        elif tokens & {"header", "navbar", "nav", "footer", "brand"}:
            keep = True
        elif any(sticky_pattern.search(token) for token in tokens):
            keep = True
        if not keep:
            continue
        selector = build_unique_selector(tag, soup)
        if selector and selector not in seen:
            seen.add(selector)
            selectors.append(selector)
    return selectors[:10]


def _collect_demo_selectors(template_html: str) -> list[str]:
    soup = BeautifulSoup(str(template_html or ""), "html.parser")
    demo_text_pattern = re.compile(
        r"(lorem ipsum|placeholder|demo content|your text here|sample text|dummy data)",
        flags=re.IGNORECASE,
    )
    selectors: list[str] = []
    seen: set[str] = set()
    for tag in soup.find_all(True):
        if tag.name in {"script", "style", "noscript"}:
            continue
        text_preview = " ".join(tag.get_text(" ", strip=True).split())
        if not text_preview or not demo_text_pattern.search(text_preview):
            continue
        selector = build_unique_selector(tag, soup)
        if selector and selector not in seen:
            seen.add(selector)
            selectors.append(selector)
    return selectors[:12]


def _collect_script_signals(template_root: Path) -> tuple[list[str], list[str]]:
    script_paths = sorted(_iter_template_files(template_root, {".js", ".mjs", ".ts"}, max_depth=3), key=lambda item: str(item))
    dependencies: list[str] = []
    contents: list[str] = []
    for path in script_paths[:_MAX_SCRIPT_FILES]:
        rel_path = str(path.relative_to(template_root)).replace("\\", "/")
        dependencies.append(rel_path)
        contents.append(_read_text_safely(path))
    return dependencies, contents


def _widget(selector: str, widget_type: str, scripts: list[str], risk_flags: list[str], optional: bool = True) -> TemplateWidget:
    return TemplateWidget(
        selector=selector,
        widget_type=widget_type,
        required_selectors=[selector],
        script_dependencies=scripts,
        risk_flags=risk_flags,
        optional=optional,
    )


def _collect_widgets_and_risks(template_root: Path, template_html: str) -> tuple[list[TemplateWidget], list[str], list[str]]:
    soup = BeautifulSoup(str(template_html or ""), "html.parser")
    script_dependencies, script_contents = _collect_script_signals(template_root)
    script_blob = "\n".join(script_contents)

    widgets: list[TemplateWidget] = []
    unsafe_selectors: list[str] = []
    risk_flags: list[str] = []
    seen_widget_selectors: set[str] = set()

    def _append_widget(tag: Tag, widget_type: str, widget_risks: list[str], optional: bool = True) -> None:
        selector = build_unique_selector(tag, soup)
        if not selector or selector in seen_widget_selectors:
            return
        seen_widget_selectors.add(selector)
        widgets.append(_widget(selector, widget_type, script_dependencies, widget_risks, optional=optional))
        if widget_risks:
            unsafe_selectors.append(selector)
            risk_flags.extend(widget_risks)

    for tag in soup.find_all(True):
        if not isinstance(tag, Tag):
            continue
        tokens = _tag_tokens(tag)
        if tag.name == "canvas" or tokens & {"chart", "plot", "echart", "apexchart"}:
            _append_widget(tag, "chart", ["chart_runtime_dependency"])
        if tokens & {"carousel", "slider", "swiper", "splide"}:
            _append_widget(tag, "carousel", ["carousel_runtime_dependency"])
        if tokens & {"toc", "table", "contents"} and "content" not in tokens:
            _append_widget(tag, "toc", [], optional=False)
        if tokens & {"copy", "clipboard"} and tag.name in {"button", "a", "div"}:
            _append_widget(tag, "copy_button", ["copy_button_behavior"])
        if tokens & {"citation", "bibtex", "reference"}:
            _append_widget(tag, "bibtex", ["citation_widget"])
        if tokens & {"sticky", "affix"}:
            _append_widget(tag, "sticky", [], optional=False)

    if re.search(r"\b(fetch|axios|XMLHttpRequest)\b", script_blob, flags=re.IGNORECASE):
        risk_flags.append("fetch_runtime_dependency")
        for tag in soup.find_all(attrs={"data-url": True}):
            _append_widget(tag, "fetch", ["fetch_runtime_dependency"])

    if re.search(r"\b(MathJax|katex)\b", script_blob, flags=re.IGNORECASE):
        risk_flags.append("math_runtime_dependency")
        math_tag = soup.find(class_=re.compile(r"(math|equation|katex|MathJax)", re.IGNORECASE))
        if isinstance(math_tag, Tag):
            _append_widget(math_tag, "math", ["math_runtime_dependency"])

    return widgets, sorted(set(unsafe_selectors)), sorted(set(risk_flags))


def _infer_archetype(template_root: Path, template_html: str, widgets: list[TemplateWidget]) -> str:
    soup = BeautifulSoup(str(template_html or ""), "html.parser")
    root_tokens = {
        token
        for tag in soup.find_all(["body", "main", "header", "nav", "section", "article", "div"], limit=30)
        for token in _tag_tokens(tag)
    }
    linked_assets = "\n".join(
        _read_text_safely(path)
        for path in sorted(_iter_template_files(template_root, {".css"}, max_depth=2), key=lambda item: str(item))[:4]
    )
    widget_types = {item.widget_type for item in widgets}

    if ("hero" in root_tokens or "bulma" in linked_assets.lower()) and ("bulma" in linked_assets.lower() or "hero" in root_tokens):
        return "hero_bulma"
    if ("navbar" in root_tokens or "bootstrap" in linked_assets.lower()) and ("bootstrap" in linked_assets.lower() or "nav" in root_tokens):
        return "bootstrap_navbar"
    if {"chart", "fetch"} & widget_types or ("dashboard" in root_tokens or "metrics" in root_tokens):
        return "chart_fetch_dashboard"
    if ("article" in root_tokens or "prose" in root_tokens or "content" in root_tokens) and len(widget_types) <= 1:
        return "single_column_article"
    return "generic_multi_section"


def _calculate_compile_confidence(
    shell_candidates: list[TemplateShellCandidate],
    global_preserve_selectors: list[str],
    risk_flags: list[str],
    archetype: str,
) -> float:
    confidence = 0.42
    confidence += min(0.22, len(shell_candidates) * 0.018)
    confidence += min(0.12, len(global_preserve_selectors) * 0.03)
    confidence += 0.08 if any(candidate.role == "hero" for candidate in shell_candidates) else 0.0
    confidence += 0.05 if any(candidate.role == "section" for candidate in shell_candidates) else 0.0
    if archetype in {"hero_bulma", "bootstrap_navbar", "single_column_article"}:
        confidence += 0.06
    confidence -= min(0.28, len(risk_flags) * 0.06)
    return round(max(0.05, min(confidence, 0.99)), 3)


def _llm_enrich_profile(summary: dict[str, Any]) -> tuple[list[str], list[str], str | None]:
    try:
        llm = get_llm(temperature=0, use_smart_model=False)
        response = llm.invoke(
            [
                SystemMessage(content=_LLM_ENRICHMENT_SYSTEM_PROMPT),
                HumanMessage(
                    content=_LLM_ENRICHMENT_USER_TEMPLATE.format(
                        summary_json=json.dumps(summary, indent=2, ensure_ascii=False)
                    )
                ),
            ]
        )
        text = getattr(response, "content", response)
        if isinstance(text, list):
            text = "\n".join(str(item) for item in text)
        parsed = json.loads(str(text).strip())
        if not isinstance(parsed, dict):
            return [], [], None
        notes = [str(item).strip() for item in (parsed.get("notes") or []) if str(item).strip()]
        risk_flags = [str(item).strip() for item in (parsed.get("risk_flags") or []) if str(item).strip()]
        archetype = str(parsed.get("archetype") or "").strip() or None
        return notes[:6], risk_flags[:8], archetype
    except Exception:
        return [], [], None


def _cleanup_old_cache_files(cache_dir: Path) -> None:
    if not cache_dir.exists():
        return
    cache_files = sorted(
        [path for path in cache_dir.glob("*.json") if path.is_file()],
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    for stale_path in cache_files[_MAX_CACHE_FILES_PER_TEMPLATE:]:
        stale_path.unlink(missing_ok=True)


def build_template_profile_cache_path(template_root: Path, template_id: str, source_fingerprint: str) -> Path:
    cache_dir = template_root / _CACHE_DIR_NAME / _CACHE_PROFILE_DIR_NAME
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{template_id}-{source_fingerprint[:16]}.json"


def _load_cached_profile(cache_path: Path) -> TemplateProfile | None:
    if not cache_path.exists():
        return None
    try:
        return TemplateProfile.model_validate_json(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def compile_template_profile(
    *,
    project_root: Path,
    candidate: TemplateCandidate,
    allow_llm: bool = True,
    force_recompile: bool = False,
) -> tuple[TemplateProfile, Path, bool]:
    normalized_candidate = _normalize_template_candidate(candidate)
    if normalized_candidate is None:
        raise ValueError("compile_template_profile received an invalid template candidate.")

    template_root = _resolve_template_root(project_root, normalized_candidate)
    entry_html_path = _resolve_entry_html_path(project_root, normalized_candidate)
    if not template_root.exists():
        raise FileNotFoundError(f"Template root not found: {template_root}")
    if not entry_html_path.exists():
        raise FileNotFoundError(f"Template entry html not found: {entry_html_path}")

    entry_html_text = _read_text_safely(entry_html_path)
    source_fingerprint = _build_source_fingerprint(template_root, entry_html_path)
    cache_path = build_template_profile_cache_path(
        template_root=template_root,
        template_id=normalized_candidate.template_id,
        source_fingerprint=source_fingerprint,
    )
    if not force_recompile:
        cached_profile = _load_cached_profile(cache_path)
        if cached_profile is not None:
            return cached_profile, cache_path, True

    shell_candidates = discover_template_shell_candidates(entry_html_text)
    global_preserve_selectors = _collect_global_preserve_selectors(entry_html_text)
    removable_demo_selectors = _collect_demo_selectors(entry_html_text)
    widgets, unsafe_selectors, widget_risks = _collect_widgets_and_risks(template_root, entry_html_text)
    archetype = _infer_archetype(template_root, entry_html_text, widgets)
    notes = [
        f"shell_candidates={len(shell_candidates)}",
        f"globals={len(global_preserve_selectors)}",
        f"widgets={len(widgets)}",
        f"archetype={archetype}",
    ]
    risk_flags = list(widget_risks)

    if any(bool(candidate.preserve_ids) for candidate in shell_candidates):
        notes.append("stable_ids_detected")
    if removable_demo_selectors:
        notes.append("demo_content_detected")

    deterministic_summary = {
        "template_id": normalized_candidate.template_id,
        "entry_html": normalized_candidate.chosen_entry_html,
        "archetype": archetype,
        "global_preserve_selectors": global_preserve_selectors,
        "shell_candidates": [candidate.model_dump() for candidate in shell_candidates[:10]],
        "optional_widgets": [widget.model_dump() for widget in widgets[:10]],
        "risk_flags": risk_flags,
        "notes": notes,
    }
    if allow_llm:
        llm_notes, llm_risks, llm_archetype = _llm_enrich_profile(deterministic_summary)
        if llm_notes:
            notes.extend(llm_notes)
        if llm_risks:
            risk_flags.extend(llm_risks)
        if llm_archetype:
            archetype = llm_archetype

    compile_confidence = _calculate_compile_confidence(
        shell_candidates=shell_candidates,
        global_preserve_selectors=global_preserve_selectors,
        risk_flags=sorted(set(risk_flags)),
        archetype=archetype,
    )
    profile = TemplateProfile(
        template_id=normalized_candidate.template_id,
        template_root_dir=_to_project_relative_path(template_root, project_root),
        entry_html=normalized_candidate.chosen_entry_html,
        archetype=archetype,
        global_preserve_selectors=global_preserve_selectors,
        shell_candidates=shell_candidates,
        optional_widgets=widgets,
        removable_demo_selectors=removable_demo_selectors,
        unsafe_selectors=unsafe_selectors,
        compile_confidence=compile_confidence,
        risk_flags=sorted(set(risk_flags)),
        notes=sorted(set(notes)),
        source_fingerprint=source_fingerprint,
    )
    cache_path.write_text(
        json.dumps(profile.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _cleanup_old_cache_files(cache_path.parent)
    return profile, cache_path, False


def prepare_template_compile_bundle(
    *,
    project_root: Path,
    generation_constraints: dict[str, Any] | None,
    user_constraints: dict[str, Any] | None,
    synced_assets: SyncedTemplateAssets | None = None,
    allow_llm: bool = True,
    force_recompile: bool = False,
) -> tuple[list[TemplateCandidate], TemplateCandidate, TemplateProfile, Path, bool, SyncedTemplateAssets]:
    candidates, selected, assets = select_template_candidates(
        project_root=project_root,
        generation_constraints=generation_constraints,
        user_constraints=user_constraints,
        synced_assets=synced_assets,
    )
    profile, cache_path, cache_hit = compile_template_profile(
        project_root=project_root,
        candidate=selected,
        allow_llm=allow_llm,
        force_recompile=force_recompile,
    )
    return candidates, selected, profile, cache_path, cache_hit, assets


def hydrate_template_candidate_from_root(
    *,
    project_root: Path,
    template_root: Path,
    template_id: str | None = None,
    score: float = 0.0,
    reasons: list[str] | None = None,
) -> TemplateCandidate:
    entry_candidates = find_entry_html_candidates(template_root, max_candidates=1)
    if not entry_candidates:
        raise ValueError(f"No entry html candidate found for template root: {template_root}")
    return TemplateCandidate(
        template_id=str(template_id or template_root.name).strip() or template_root.name,
        root_dir=_to_project_relative_path(template_root, project_root),
        chosen_entry_html=entry_candidates[0],
        score=float(score),
        reasons=list(reasons or []),
    )
