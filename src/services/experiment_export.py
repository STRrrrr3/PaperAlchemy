from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
import uuid
import zipfile

from bs4 import BeautifulSoup, Comment, Tag

from src.contracts.schemas import CoderArtifact
from src.services.artifact_store import get_output_paths, load_cached_structured_data, load_coder_artifact
from src.services.preview_service import take_local_screenshot
from src.utils.html_utils import read_text_with_fallback

EXPERIMENT_EXPORTS_DIRNAME = "experiments"
EXPORT_METADATA_FILENAME = "export_metadata.json"
EXPORT_SCREENSHOT_FILENAME = "clean_page.png"
_EXPORT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
_INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')
_BODY_MARKERS = {
    "PaperAlchemy Generated Body Start",
    "PaperAlchemy Generated Body End",
}
_TEXT_SCAN_SUFFIXES = {".css", ".html", ".js", ".json", ".md", ".txt"}
_ANCHOR_ATTR_CONFIG = {
    "data-pa-block": {"kind": "block", "base_class": "paexp-a1", "offset": 1},
    "data-pa-slot": {"kind": "slot", "base_class": "paexp-a2", "offset": 101},
    "data-pa-global": {"kind": "global", "base_class": "paexp-a3", "offset": 201},
}
_AUXILIARY_FILENAMES = {
    "coder_artifact.json",
    "page_manifest.json",
    "style_context.json",
}


def build_experiment_exports_dir(paper_folder_name: str) -> Path:
    output_dir, _, _, _ = get_output_paths(str(paper_folder_name or "").strip())
    return output_dir / EXPERIMENT_EXPORTS_DIRNAME


def build_experiment_export_dir(paper_folder_name: str, export_name: str) -> Path:
    return build_experiment_exports_dir(paper_folder_name) / _validate_export_name(export_name)


def export_live_experiment_snapshot(paper_folder_name: str, export_name: str) -> dict[str, object]:
    clean_paper_folder_name = str(paper_folder_name or "").strip()
    if not clean_paper_folder_name:
        raise ValueError("paper_folder_name is required.")
    clean_export_name = _validate_export_name(export_name)

    output_dir, structured_json_path, _, coder_json_path = get_output_paths(clean_paper_folder_name)
    artifact = load_coder_artifact(coder_json_path)
    if artifact is None:
        raise FileNotFoundError(f"Live coder_artifact.json is missing or invalid: {coder_json_path}")
    site_zip_filename = _build_site_zip_filename(
        paper_folder_name=clean_paper_folder_name,
        structured_json_path=structured_json_path,
    )

    live_site_dir = _resolve_artifact_path(str(artifact.site_dir or ""), output_dir=output_dir)
    live_entry_html = _resolve_artifact_path(str(artifact.entry_html or ""), output_dir=output_dir)
    if not live_site_dir.exists() or not live_site_dir.is_dir():
        raise FileNotFoundError(f"Live site_dir does not exist: {live_site_dir}")
    if not live_entry_html.exists() or not live_entry_html.is_file():
        raise FileNotFoundError(f"Live entry_html does not exist: {live_entry_html}")
    try:
        entry_relative_path = live_entry_html.resolve().relative_to(live_site_dir.resolve())
    except ValueError as exc:
        raise ValueError("Live entry_html must be located under the live site_dir.") from exc

    experiments_dir = build_experiment_exports_dir(clean_paper_folder_name)
    experiments_dir.mkdir(parents=True, exist_ok=True)
    final_export_dir = experiments_dir / clean_export_name
    if final_export_dir.exists():
        raise FileExistsError(f"Experiment export already exists: {final_export_dir}")

    temp_export_dir = _create_temp_export_dir(
        experiments_dir=experiments_dir,
        export_name=clean_export_name,
    )
    committed_export_dir: Path | None = None
    try:
        export_site_dir = temp_export_dir / "site"
        shutil.copytree(live_site_dir, export_site_dir)
        _remove_auxiliary_artifacts(export_site_dir)

        target_html_relative_paths = _resolve_target_html_relative_paths(
            artifact=artifact,
            live_site_dir=live_site_dir,
            live_entry_html=live_entry_html,
        )
        specific_class_maps = _build_specific_class_maps(
            export_site_dir=export_site_dir,
            target_html_relative_paths=target_html_relative_paths,
        )
        sanitized_html_relative_paths = _sanitize_html_files(
            export_site_dir=export_site_dir,
            target_html_relative_paths=target_html_relative_paths,
            specific_class_maps=specific_class_maps,
        )
        rewritten_css_relative_paths = _rewrite_local_css_files(
            export_site_dir=export_site_dir,
            specific_class_maps=specific_class_maps,
        )

        exported_entry_html = export_site_dir / entry_relative_path
        screenshot_path = temp_export_dir / EXPORT_SCREENSHOT_FILENAME
        screenshot_result = take_local_screenshot(str(exported_entry_html), str(screenshot_path))
        if not screenshot_result or not screenshot_path.exists():
            raise ValueError("Experiment export screenshot failed for the clean entry HTML.")

        remaining_anchor_files = _scan_for_anchor_markers(temp_export_dir)
        if remaining_anchor_files:
            raise ValueError(
                "Experiment export still contains data-pa markers in: "
                + ", ".join(remaining_anchor_files)
            )

        _build_site_zip(
            export_root_dir=temp_export_dir,
            export_site_dir=export_site_dir,
            zip_filename=site_zip_filename,
        )

        temp_export_dir.replace(final_export_dir)
        committed_export_dir = final_export_dir

        metadata = {
            "paper_folder_name": clean_paper_folder_name,
            "export_name": clean_export_name,
            "selected_template_id": str(artifact.selected_template_id or "").strip(),
            "source_entry_html": str(live_entry_html.resolve()),
            "exported_entry_html": Path("site") / entry_relative_path,
            "screenshot_path": Path(EXPORT_SCREENSHOT_FILENAME),
            "site_zip_path": Path(site_zip_filename),
            "sanitized_html_files": [Path("site") / Path(item) for item in sanitized_html_relative_paths],
            "rewritten_css_files": [Path("site") / Path(item) for item in rewritten_css_relative_paths],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        metadata_path = committed_export_dir / EXPORT_METADATA_FILENAME
        metadata_path.write_text(
            json.dumps(_stringify_metadata_paths(metadata), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return _stringify_metadata_paths(metadata)
    except Exception:
        if committed_export_dir is not None:
            shutil.rmtree(committed_export_dir, ignore_errors=True)
        else:
            shutil.rmtree(temp_export_dir, ignore_errors=True)
        raise


def _validate_export_name(export_name: str) -> str:
    clean_export_name = str(export_name or "").strip()
    if not clean_export_name:
        raise ValueError("export_name is required.")
    if not _EXPORT_NAME_PATTERN.fullmatch(clean_export_name):
        raise ValueError(
            "export_name may only contain letters, numbers, '.', '_' or '-'."
        )
    return clean_export_name


def _create_temp_export_dir(*, experiments_dir: Path, export_name: str) -> Path:
    for _ in range(8):
        candidate = experiments_dir / f"_tmp_{export_name}_{uuid.uuid4().hex[:8]}"
        try:
            candidate.mkdir(parents=False, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise FileExistsError(
        f"Could not allocate a temporary export directory under {experiments_dir}."
    )


def _build_site_zip(*, export_root_dir: Path, export_site_dir: Path, zip_filename: str) -> Path:
    zip_path = export_root_dir / zip_filename
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(export_site_dir.rglob("*")):
            if not path.is_file():
                continue
            archive.write(path, arcname=path.resolve().relative_to(export_root_dir.resolve()).as_posix())
    return zip_path


def _build_site_zip_filename(*, paper_folder_name: str, structured_json_path: Path) -> str:
    paper_title = ""
    structured_paper = load_cached_structured_data(structured_json_path)
    if structured_paper is not None:
        paper_title = str(structured_paper.paper_title or "").strip()
    filename_stem = _sanitize_export_filename_stem(paper_title or paper_folder_name)
    return f"{filename_stem}.zip"


def _sanitize_export_filename_stem(value: str) -> str:
    cleaned = _INVALID_FILENAME_CHARS.sub("_", str(value or "").strip())
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    return cleaned or "site"


def _resolve_artifact_path(raw_path: str, *, output_dir: Path) -> Path:
    clean_path = str(raw_path or "").strip()
    if not clean_path:
        raise ValueError("Artifact path is empty.")
    candidate = Path(clean_path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (output_dir / candidate).resolve()


def _resolve_target_html_relative_paths(
    *,
    artifact: CoderArtifact,
    live_site_dir: Path,
    live_entry_html: Path,
) -> list[str]:
    relative_paths: list[str] = []
    seen: set[str] = set()
    site_root = live_site_dir.resolve()

    for raw_rel_path in artifact.edited_files:
        clean_rel_path = str(raw_rel_path or "").strip().replace("\\", "/")
        if not clean_rel_path or not clean_rel_path.lower().endswith(".html"):
            continue
        candidate = (site_root / clean_rel_path).resolve()
        try:
            relative = candidate.relative_to(site_root).as_posix()
        except ValueError as exc:
            raise ValueError(
                f"Edited HTML path escapes site_dir: {clean_rel_path}"
            ) from exc
        if not candidate.exists():
            raise FileNotFoundError(f"Edited HTML file is missing: {candidate}")
        if relative not in seen:
            seen.add(relative)
            relative_paths.append(relative)

    if relative_paths:
        return relative_paths

    fallback_relative = live_entry_html.resolve().relative_to(site_root).as_posix()
    return [fallback_relative]


def _build_specific_class_maps(
    *,
    export_site_dir: Path,
    target_html_relative_paths: list[str],
) -> dict[str, dict[str, str]]:
    collected_values = {"block": set(), "slot": set(), "global": set()}
    for relative_path in target_html_relative_paths:
        html_path = export_site_dir / relative_path
        soup = BeautifulSoup(read_text_with_fallback(html_path), "html.parser")
        for attr_name, config in _ANCHOR_ATTR_CONFIG.items():
            kind = str(config["kind"])
            for tag in soup.select(f"[{attr_name}]"):
                if not isinstance(tag, Tag):
                    continue
                value = str(tag.get(attr_name) or "").strip()
                if value:
                    collected_values[kind].add(value)

    specific_maps: dict[str, dict[str, str]] = {"block": {}, "slot": {}, "global": {}}
    for attr_name, config in _ANCHOR_ATTR_CONFIG.items():
        kind = str(config["kind"])
        offset = int(config["offset"])
        for index, value in enumerate(sorted(collected_values[kind]), start=offset):
            specific_maps[kind][value] = f"paexp-u{index:03d}"
    return specific_maps


def _sanitize_html_files(
    *,
    export_site_dir: Path,
    target_html_relative_paths: list[str],
    specific_class_maps: dict[str, dict[str, str]],
) -> list[str]:
    sanitized_paths: list[str] = []
    for relative_path in target_html_relative_paths:
        html_path = export_site_dir / relative_path
        html_text = read_text_with_fallback(html_path)
        soup = BeautifulSoup(html_text, "html.parser")

        for comment in soup.find_all(string=lambda item: isinstance(item, Comment)):
            if str(comment).strip() in _BODY_MARKERS:
                comment.extract()

        for attr_name, config in _ANCHOR_ATTR_CONFIG.items():
            kind = str(config["kind"])
            base_class = str(config["base_class"])
            specific_map = specific_class_maps[kind]
            for tag in soup.select(f"[{attr_name}]"):
                if not isinstance(tag, Tag):
                    continue
                clean_value = str(tag.get(attr_name) or "").strip()
                if not clean_value:
                    del tag[attr_name]
                    continue
                specific_class = specific_map.get(clean_value)
                if not specific_class:
                    raise ValueError(
                        f"Missing anonymous class mapping for {attr_name}='{clean_value}' in {relative_path}."
                    )
                existing_classes = [
                    str(item).strip()
                    for item in (tag.get("class") or [])
                    if str(item).strip()
                ]
                for class_name in (base_class, specific_class):
                    if class_name not in existing_classes:
                        existing_classes.append(class_name)
                tag["class"] = existing_classes
                del tag[attr_name]

        for style_tag in soup.find_all("style"):
            original_css = style_tag.string if style_tag.string is not None else style_tag.get_text()
            rewritten_css = _rewrite_css_text(
                str(original_css or ""),
                specific_class_maps=specific_class_maps,
            )
            style_tag.string = rewritten_css

        html_path.write_text(soup.decode(formatter="minimal"), encoding="utf-8")
        sanitized_paths.append(relative_path.replace("\\", "/"))
    return sanitized_paths


def _rewrite_local_css_files(
    *,
    export_site_dir: Path,
    specific_class_maps: dict[str, dict[str, str]],
) -> list[str]:
    rewritten_paths: list[str] = []
    site_root = export_site_dir.resolve()
    for css_path in sorted(export_site_dir.rglob("*.css")):
        original_css = read_text_with_fallback(css_path)
        rewritten_css = _rewrite_css_text(
            original_css,
            specific_class_maps=specific_class_maps,
        )
        if rewritten_css == original_css:
            continue
        css_path.write_text(rewritten_css, encoding="utf-8")
        rewritten_paths.append(css_path.resolve().relative_to(site_root).as_posix())
    return rewritten_paths


def _rewrite_css_text(
    css_text: str,
    *,
    specific_class_maps: dict[str, dict[str, str]],
) -> str:
    rewritten = str(css_text or "")
    for attr_name, config in _ANCHOR_ATTR_CONFIG.items():
        kind = str(config["kind"])
        base_class = str(config["base_class"])
        for value, specific_class in specific_class_maps[kind].items():
            rewritten = _replace_attr_selector(
                rewritten,
                attr_name=attr_name,
                value=value,
                replacement=f".{specific_class}",
            )
        rewritten = re.sub(
            rf"\[\s*{re.escape(attr_name)}\s*\]",
            f".{base_class}",
            rewritten,
        )
    return rewritten


def _replace_attr_selector(
    css_text: str,
    *,
    attr_name: str,
    value: str,
    replacement: str,
) -> str:
    rewritten = str(css_text or "")
    escaped_attr_name = re.escape(attr_name)
    escaped_value = re.escape(value)
    patterns = (
        rf'\[\s*{escaped_attr_name}\s*=\s*"{escaped_value}"\s*\]',
        rf"\[\s*{escaped_attr_name}\s*=\s*'{escaped_value}'\s*\]",
        rf"\[\s*{escaped_attr_name}\s*=\s*{escaped_value}\s*\]",
    )
    for pattern in patterns:
        rewritten = re.sub(pattern, replacement, rewritten)
    return rewritten


def _remove_auxiliary_artifacts(export_site_dir: Path) -> None:
    for path in export_site_dir.rglob("*"):
        if path.is_file() and path.name in _AUXILIARY_FILENAMES:
            path.unlink()


def _scan_for_anchor_markers(root_dir: Path) -> list[str]:
    findings: list[str] = []
    root = root_dir.resolve()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in _TEXT_SCAN_SUFFIXES:
            continue
        try:
            text = read_text_with_fallback(path)
        except Exception:
            continue
        if "data-pa-" in text:
            findings.append(path.relative_to(root).as_posix())
    return findings


def _stringify_metadata_paths(metadata: dict[str, object]) -> dict[str, object]:
    normalized: dict[str, object] = {}
    for key, value in metadata.items():
        if isinstance(value, Path):
            normalized[key] = value.as_posix()
            continue
        if isinstance(value, list):
            normalized[key] = [item.as_posix() if isinstance(item, Path) else item for item in value]
            continue
        normalized[key] = value
    return normalized
