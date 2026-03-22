from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from bs4 import BeautifulSoup

REVISION_OVERRIDE_STYLE_TAG_ID = "paperalchemy-revision-overrides"


def _normalize_web_path(value: str) -> str:
    parsed = urlparse(str(value or "").strip())
    path = str(parsed.path or "").strip().replace("\\", "/")
    if not path:
        return ""
    if path.startswith("./"):
        return path
    if path.startswith("/"):
        return path
    return f"./{path}"


def is_external_url(value: str) -> bool:
    lowered = str(value or "").strip().lower()
    return lowered.startswith(("http://", "https://", "//", "data:", "mailto:", "tel:", "javascript:", "#"))


def resolve_local_web_path(src: str, entry_html_path: Path, site_dir: Path) -> Path | None:
    parsed = urlparse(str(src or "").strip())
    if parsed.scheme or parsed.netloc:
        return None
    clean_src = str(parsed.path or "").strip()
    if not clean_src or is_external_url(clean_src):
        return None

    candidate = (entry_html_path.parent / clean_src).resolve()
    try:
        candidate.relative_to(site_dir.resolve())
    except ValueError:
        return None
    return candidate


def collect_local_image_sources(html_text: str) -> list[str]:
    soup = BeautifulSoup(str(html_text or ""), "html.parser")
    results: list[str] = []
    seen: set[str] = set()
    for image in soup.find_all("img"):
        src = str(image.get("src") or "").strip()
        if not src or is_external_url(src):
            continue
        normalized = _normalize_web_path(src)
        if normalized and normalized not in seen:
            seen.add(normalized)
            results.append(normalized)
    return results


def collect_allowed_asset_web_paths(available_assets: list[dict[str, str]]) -> set[str]:
    allowed: set[str] = set()
    for item in available_assets:
        normalized = _normalize_web_path(str(item.get("web_path") or ""))
        if normalized:
            allowed.add(normalized)
    return allowed


def validate_local_image_references(
    html_text: str,
    entry_html_path: Path,
    site_dir: Path,
    allowed_asset_web_paths: set[str] | None = None,
    enforce_paper_asset_whitelist: bool = False,
) -> list[str]:
    soup = BeautifulSoup(str(html_text or ""), "html.parser")
    critiques: list[str] = []
    allowed_asset_web_paths = allowed_asset_web_paths or set()

    for image in soup.find_all("img"):
        src = str(image.get("src") or "").strip()
        if not src or is_external_url(src):
            continue

        normalized = _normalize_web_path(src)
        resolved = resolve_local_web_path(src, entry_html_path=entry_html_path, site_dir=site_dir)
        if resolved is None:
            critiques.append(f"Local image path escapes the generated site directory: {src}")
            continue
        if not resolved.exists():
            critiques.append(f"Local image path does not exist in the generated site: {src}")
            continue

        if not enforce_paper_asset_whitelist:
            continue

        try:
            relative_to_assets = resolved.relative_to((site_dir / "assets" / "paper").resolve())
        except ValueError:
            relative_to_assets = None
        if relative_to_assets is not None and normalized not in allowed_asset_web_paths:
            critiques.append(
                "Paper image path is not in the copied asset manifest: "
                f"{src}"
            )

    return critiques


def validate_fragment_local_image_sources(
    html_text: str,
    allowed_asset_web_paths: set[str],
    allowed_existing_local_sources: set[str] | None = None,
) -> list[str]:
    allowed_existing_local_sources = allowed_existing_local_sources or set()
    critiques: list[str] = []

    for src in collect_local_image_sources(html_text):
        if src not in allowed_asset_web_paths and src not in allowed_existing_local_sources:
            critiques.append(
                "Patch output referenced a local image path that is not allowed in the current page context: "
                f"{src}"
            )

    return critiques


def is_safe_anchored_selector(selector: str, allowed_anchor_selectors: set[str]) -> bool:
    clean = str(selector or "").strip()
    if not clean:
        return False
    if any(token in clean for token in (",", "{", "}", "@", "\n", "\r", ";")):
        return False

    for anchor in allowed_anchor_selectors:
        if clean == anchor:
            return True
        if not clean.startswith(anchor):
            continue
        suffix = clean[len(anchor) :]
        if not suffix:
            return True
        if suffix[0] in (" ", ">", "+", "~", ":", "[", ".", "#"):
            return True
    return False
