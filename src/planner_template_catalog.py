import json
from pathlib import Path
from typing import Any


SKIP_DIR_NAMES = {
    ".git",
    "node_modules",
    "dist",
    "build",
    ".next",
    ".nuxt",
    "coverage",
    "vendor",
    "tmp",
    "temp",
    "__pycache__",
}


def _is_within_depth(base: Path, candidate: Path, max_depth: int) -> bool:
    rel = candidate.relative_to(base)
    return len(rel.parts) <= max_depth


def _iter_files_limited(root: Path, max_depth: int) -> list[Path]:
    files: list[Path] = []
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            children = list(current.iterdir())
        except Exception:
            continue

        for child in children:
            if child.is_dir():
                if child.name in SKIP_DIR_NAMES:
                    continue
                if _is_within_depth(root, child, max_depth):
                    stack.append(child)
                continue
            if _is_within_depth(root, child, max_depth):
                files.append(child)
    return files


def find_entry_html_candidates(template_root: Path, max_candidates: int = 3) -> list[str]:
    candidates: list[Path] = []
    root_index = template_root / "index.html"
    if root_index.exists():
        candidates.append(root_index)

    files = _iter_files_limited(template_root, max_depth=4)
    html_files = [p for p in files if p.suffix.lower() in {".html", ".htm"}]

    def _sort_key(path: Path) -> tuple[int, int, str]:
        rel = path.relative_to(template_root)
        depth = len(rel.parts)
        score = 0
        if rel.name.lower() == "index.html":
            score -= 10
        return score, depth, str(rel).lower()

    for html in sorted(html_files, key=_sort_key):
        if html not in candidates:
            candidates.append(html)
        if len(candidates) >= max_candidates:
            break

    return [str(p.relative_to(template_root)).replace("\\", "/") for p in candidates[:max_candidates]]


def _find_entry_html_candidates(template_root: Path, max_candidates: int = 3) -> list[str]:
    return find_entry_html_candidates(template_root, max_candidates=max_candidates)


def _sample_files(template_root: Path, suffixes: set[str], max_files: int = 5) -> list[str]:
    files = _iter_files_limited(template_root, max_depth=4)
    filtered = [p for p in files if p.suffix.lower() in suffixes]
    filtered.sort(key=lambda p: (len(p.relative_to(template_root).parts), str(p).lower()))
    return [
        str(path.relative_to(template_root)).replace("\\", "/")
        for path in filtered[:max_files]
    ]


def build_template_catalog(
    templates_dir: Path,
    project_root: Path,
    max_templates: int = 150,
    max_entry_candidates: int = 3,
) -> list[dict[str, Any]]:
    if not templates_dir.exists():
        return []

    template_dirs = sorted([d for d in templates_dir.iterdir() if d.is_dir()], key=lambda p: p.name.lower())
    if max_templates > 0:
        template_dirs = template_dirs[:max_templates]

    catalog: list[dict[str, Any]] = []
    for template_dir in template_dirs:
        entry_candidates = find_entry_html_candidates(template_dir, max_candidates=max_entry_candidates)
        if not entry_candidates:
            continue

        rel_root = str(template_dir.relative_to(project_root)).replace("\\", "/")
        styles = _sample_files(template_dir, suffixes={".css", ".scss", ".sass", ".less"}, max_files=5)
        scripts = _sample_files(template_dir, suffixes={".js", ".mjs", ".ts"}, max_files=5)

        catalog.append(
            {
                "template_id": template_dir.name,
                "root_dir": rel_root,
                "entry_html_candidates": entry_candidates,
                "style_files": styles,
                "script_files": scripts,
                "has_image_info": (template_dir / "image_info.json").exists(),
                "has_video_info": (template_dir / "video_info.json").exists(),
            }
        )

    return catalog


def load_template_link_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    return {str(k): str(v) for k, v in payload.items()}


def load_module_index(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload
