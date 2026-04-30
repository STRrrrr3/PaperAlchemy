from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.benchmark_v1.render import take_benchmark_screenshot
from src.services.artifact_store import get_output_paths, load_cached_structured_data, load_coder_artifact
from src.services.experiment_export import (
    build_experiment_export_dir,
    build_experiment_exports_dir,
    export_live_experiment_snapshot,
    resolve_artifact_path,
)
from src.services.paper_assets import asset_target_filename, resolved_asset_source_path

RUN_SCHEMA_VERSION = "paperalchemy-benchmark-v1"
RUNS_DIRNAME = "v1_runs"
E2E_INPUTS_DIRNAME = "e2e_inputs"
LLM_SITES_DIRNAME = "llm_sites"
E2E_PAPER_PDF_PATH = "benchmark_sources/paper.pdf"
RUN_JSON = "run.json"
CASES_JSON = "cases.json"
CASE_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
OPAQUE_CLASS_PATTERN = re.compile(r"\bpaexp-[ck][0-9a-f]{10}\b")
TEXT_SUFFIXES = {".css", ".html", ".js", ".json", ".md", ".txt"}
DEFAULT_CAPTURE_COOLDOWN_SECONDS = 3.0
E2E_EXCLUDED_FILENAMES = {"final_render.png", "review_current.png"}
E2E_EXCLUDED_DIRNAMES = {".paperalchemy"}
CONTROL_MODEL_IDS = ("gemini", "gpt", "deepseek")
CONTROL_MODEL_LABELS = {
    "gemini": "Gemini",
    "gpt": "GPT",
    "deepseek": "DeepSeek",
}
CANDIDATE_KEEP_RANGE = (8, 8)

# Bench Design 4.1 single-turn revision taxonomy. Each category asks Gemini to
# produce two page-grounded candidates, for eight candidates total.
TASK_CATALOG = {
    "content_multimodal": {
        "label": "Content & Multimodal Editing",
        "label_zh": "内容与多模态资产编辑",
        "candidate_count": 2,
        "subcategories": [
            {
                "name": "asset_replacement",
                "label": "Asset Replacement",
                "label_zh": "资产替换",
                "example": "Replace the current Figure 1 with the provided target asset while preserving the original aspect ratio.",
                "example_zh": "把目前 Figure 1 替换为随 benchmark 提供的目标素材，并保持原始长宽比。",
            },
            {
                "name": "asset_adjustment",
                "label": "Asset Adjustment",
                "label_zh": "资产修改",
                "example": "Adjust the size and layout of images or tables, place two comparison figures side by side, and align them with the corresponding section.",
                "example_zh": "调整图片/表格等的大小与布局，把两张对比图并排，让图和对应的 section 对齐。",
            },
            {
                "name": "text_compaction_expansion",
                "label": "Text Compaction / Expansion",
                "label_zh": "内容精简/扩充",
                "example": "Condense the overly long Related Work section into three core bullet points.",
                "example_zh": "当前 Related Work 部分文本过于冗长，请将其精简提炼为三个核心 bullet points。",
            },
        ],
    },
    "styling_aesthetics": {
        "label": "Styling & Aesthetics Revision",
        "label_zh": "样式与美学修订",
        "candidate_count": 2,
        "subcategories": [
            {
                "name": "color_contrast",
                "label": "Color & Contrast",
                "label_zh": "色彩与对比度",
                "example": "Change the Abstract block background to light gray to distinguish it from the main body.",
                "example_zh": "将 Abstract 区块背景色改为浅灰色，以区分正文。",
            },
            {
                "name": "typography",
                "label": "Typography Tuning",
                "label_zh": "排版微调",
                "example": "Change all H2 headings to a serif typeface to strengthen the academic tone.",
                "example_zh": "将所有二级标题 H2 改为衬线字体，以增强学术严肃感。",
            },
            {
                "name": "visual_focus",
                "label": "Visual Focus",
                "label_zh": "视觉焦点",
                "example": "Add a thin border and subtle shadow around Figure 1 so it stands out more on the page.",
                "example_zh": "给 Figure 1 外部加上一层细边框和浅阴影，使其在页面中更突出。",
            },
        ],
    },
    "layout_rhythm": {
        "label": "Layout & Rhythm Reorganization",
        "label_zh": "布局与空间节律重构",
        "candidate_count": 2,
        "subcategories": [
            {
                "name": "structural_transform",
                "label": "Structural Transform",
                "label_zh": "结构转换",
                "example": "Convert the Contributions section from a single unordered list into a responsive two-column grid-card layout.",
                "example_zh": "将主要贡献 Contributions section 的单行无序列表转换为两列并排的响应式 Grid 卡片布局。",
            },
            {
                "name": "rhythm_spacing",
                "label": "Rhythm & Spacing",
                "label_zh": "节律与留白",
                "example": "Increase the spacing between a table and the following caption to fix the current crowding.",
                "example_zh": "增加表格与下方图注 caption 之间的间距，修复目前的拥挤感。",
            },
            {
                "name": "alignment_fix",
                "label": "Alignment Fix",
                "label_zh": "对齐修复",
                "example": "Strictly center-align formula blocks and their numbers horizontally and vertically.",
                "example_zh": "将所有公式模块与其编号进行严格的水平垂直居中对齐。",
            },
        ],
    },
    "interactivity_functional": {
        "label": "Interactivity & Functional Editing",
        "label_zh": "高级交互与功能性修改",
        "candidate_count": 2,
        "subcategories": [
            {
                "name": "hyperlink_anchor",
                "label": "Hyperlink & Anchor",
                "label_zh": "超链接与锚点",
                "example": "Add a real hyperlink to the top PDF Download shortcut button, pointing to the provided arXiv URL.",
                "example_zh": "给顶部 PDF Download 快捷访问按钮添加真实超链接，指向提供的 arXiv 地址。",
            },
            {
                "name": "navigation",
                "label": "Navigation",
                "label_zh": "导航功能",
                "example": "Generate a floating table-of-contents sidebar on the left; clicking a heading should smoothly scroll to the corresponding section.",
                "example_zh": "在页面左侧生成悬浮目录导航栏，点击标题可平滑滚动到对应章节。",
            },
            {
                "name": "collapsible",
                "label": "Collapsible Interaction",
                "label_zh": "交互折叠",
                "example": "Turn Appendix and Proof Details into collapsed-by-default accordion panels that expand when clicked.",
                "example_zh": "把 Appendix 和 Proof Details 改为默认折叠的 Accordion 面板，用户点击后展开。",
            },
        ],
    },
}
VALID_TASK_TYPES = frozenset(TASK_CATALOG.keys())
VALID_DIFFICULTIES = frozenset({"easy", "medium", "hard"})


def valid_subcategories(task_type: str) -> frozenset[str]:
    config = TASK_CATALOG.get(str(task_type or ""))
    if not config:
        return frozenset()
    return frozenset(str(entry.get("name") or "") for entry in config.get("subcategories") or [])


def make_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{os.urandom(4).hex()}"


def normalize_token(value: str, *, fallback: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
    token = token.strip("._-")
    return token or fallback


def make_baseline_export_name(run_id: str) -> str:
    return f"baseline_{normalize_token(run_id, fallback='run')}"


def make_pa_export_name(run_id: str, case_id: str, revision: int) -> str:
    return (
        f"pa_{normalize_token(run_id, fallback='run')}_"
        f"{normalize_token(case_id, fallback='case')}_r{max(1, int(revision))}"
    )


def build_v1_runs_dir(paper_folder_name: str) -> Path:
    return build_experiment_exports_dir(paper_folder_name) / RUNS_DIRNAME


def build_run_dir(paper_folder_name: str, run_id: str) -> Path:
    return build_v1_runs_dir(paper_folder_name) / normalize_token(run_id, fallback="run")


def list_paper_folder_names() -> list[str]:
    output_root = get_output_paths("")[0]
    if not output_root.exists():
        return []
    candidates: list[tuple[float, str]] = []
    for item in output_root.iterdir():
        if not item.is_dir() or item.name.startswith("_"):
            continue
        has_live_artifact = (item / "coder_artifact.json").exists()
        has_cached_runs = build_v1_runs_dir(item.name).exists()
        if has_live_artifact or has_cached_runs:
            candidates.append((item.stat().st_mtime, item.name))
    return [name for _, name in sorted(candidates, key=lambda entry: (-entry[0], entry[1].lower()))]


def list_run_ids(paper_folder_name: str) -> list[str]:
    runs_dir = build_v1_runs_dir(str(paper_folder_name or "").strip())
    if not runs_dir.exists():
        return []
    candidates: list[tuple[float, str]] = []
    for item in runs_dir.iterdir():
        run_json = item / RUN_JSON
        if item.is_dir() and run_json.exists():
            candidates.append((run_json.stat().st_mtime, item.name))
    return [name for _, name in sorted(candidates, key=lambda entry: (-entry[0], entry[1].lower()))]


def latest_run_id(paper_folder_name: str) -> str:
    run_ids = list_run_ids(paper_folder_name)
    return run_ids[0] if run_ids else ""


def create_run(paper_folder_name: str, run_id: str | None = None) -> dict[str, Any]:
    clean_paper_folder_name = str(paper_folder_name or "").strip()
    if not clean_paper_folder_name:
        raise ValueError("paper_folder_name is required.")
    clean_run_id = normalize_token(run_id or make_run_id(), fallback="run")
    run_dir = build_run_dir(clean_paper_folder_name, clean_run_id)
    if run_dir.exists():
        raise FileExistsError(f"Benchmark V1 run already exists: {run_dir}")

    output_dir, _, _, coder_json_path = get_output_paths(clean_paper_folder_name)
    artifact = load_coder_artifact(coder_json_path)
    if artifact is None:
        raise FileNotFoundError(f"Live coder_artifact.json is missing or invalid: {coder_json_path}")
    live_site_dir = resolve_artifact_path(str(artifact.site_dir or ""), output_dir=output_dir)
    if not live_site_dir.exists() or not live_site_dir.is_dir():
        raise FileNotFoundError(f"Live site_dir does not exist: {live_site_dir}")

    baseline_export_name = make_baseline_export_name(clean_run_id)
    baseline_export_dir = build_experiment_export_dir(clean_paper_folder_name, baseline_export_name)
    if baseline_export_dir.exists():
        raise FileExistsError(f"Baseline export already exists: {baseline_export_dir}")

    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        baseline_live_site_dir = run_dir / "baseline_live_site"
        shutil.copytree(live_site_dir, baseline_live_site_dir)
        baseline_export_metadata = export_live_experiment_snapshot(
            paper_folder_name=clean_paper_folder_name,
            export_name=baseline_export_name,
        )
        baseline_zip_path = baseline_export_dir / str(baseline_export_metadata.get("site_zip_path") or "")
        run = {
            "schema_version": RUN_SCHEMA_VERSION,
            "paper_folder_name": clean_paper_folder_name,
            "run_id": clean_run_id,
            "baseline_export_name": baseline_export_name,
            "baseline_export_dir": str(baseline_export_dir),
            "baseline_zip_path": str(baseline_zip_path),
            "baseline_live_site_dir": str(baseline_live_site_dir),
            "baseline_export_metadata": baseline_export_metadata,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "notes": (
                "V1 intentionally delays hidden eval_spec.json to V2. "
                "Preservation/Minimality use coarse diff metrics plus human override."
            ),
        }
        save_run(run_dir, run)
        save_cases(run_dir, [])
        return run
    except Exception:
        shutil.rmtree(run_dir, ignore_errors=True)
        raise


def load_run(run_dir: Path) -> dict[str, Any]:
    path = Path(run_dir) / RUN_JSON
    payload = json.loads(path.read_text(encoding="utf-8"))
    if str(payload.get("schema_version") or "") != RUN_SCHEMA_VERSION:
        raise ValueError(f"Unsupported Benchmark V1 run schema in {path}")
    return payload


def save_run(run_dir: Path, run: dict[str, Any]) -> None:
    path = Path(run_dir) / RUN_JSON
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8")


def load_cases(run_dir: Path) -> list[dict[str, Any]]:
    path = Path(run_dir) / CASES_JSON
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{CASES_JSON} must contain a JSON list.")
    return [dict(item) for item in payload]


def save_cases(run_dir: Path, cases: list[dict[str, Any]]) -> None:
    validate_cases(cases)
    path = Path(run_dir) / CASES_JSON
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cases, indent=2, ensure_ascii=False), encoding="utf-8")


def validate_cases(cases: list[dict[str, Any]]) -> None:
    seen: set[str] = set()
    for raw_case in cases:
        case = dict(raw_case)
        case_id = str(case.get("case_id") or "").strip()
        if not case_id or not CASE_ID_PATTERN.fullmatch(case_id):
            raise ValueError(f"Invalid case_id: {case_id or '(empty)'}")
        if case_id in seen:
            raise ValueError(f"Duplicate case_id: {case_id}")
        seen.add(case_id)
        for key in ("instruction", "task_type", "target_hint", "expected_observable"):
            if not str(case.get(key) or "").strip():
                raise ValueError(f"Case {case_id} is missing required field: {key}")
        task_type = str(case.get("task_type") or "").strip()
        if task_type not in VALID_TASK_TYPES:
            raise ValueError(
                f"Case {case_id} task_type {task_type!r} must be one of {sorted(VALID_TASK_TYPES)}."
            )
        subcategory = str(case.get("subcategory") or "").strip()
        if subcategory:
            allowed = valid_subcategories(task_type)
            if subcategory not in allowed:
                raise ValueError(
                    f"Case {case_id} subcategory {subcategory!r} must be one of "
                    f"{sorted(allowed)} for task_type {task_type}."
                )
        selectors = case.get("target_selectors")
        if selectors is None:
            case["target_selectors"] = []
        elif not isinstance(selectors, list):
            raise ValueError(f"Case {case_id} target_selectors must be a list.")
        difficulty = str(case.get("difficulty") or "").strip()
        if difficulty and difficulty not in VALID_DIFFICULTIES:
            raise ValueError(
                f"Case {case_id} difficulty {difficulty!r} must be one of {sorted(VALID_DIFFICULTIES)}."
            )
        forbidden_changes = case.get("forbidden_changes")
        if forbidden_changes is not None and not isinstance(forbidden_changes, list):
            raise ValueError(f"Case {case_id} forbidden_changes must be a list.")


def get_case(run_dir: Path, case_id: str) -> dict[str, Any]:
    clean_case_id = str(case_id or "").strip()
    for case in load_cases(run_dir):
        if str(case.get("case_id") or "") == clean_case_id:
            return case
    raise KeyError(f"Unknown case_id: {clean_case_id}")


def update_case(run_dir: Path, case_id: str, updates: dict[str, Any]) -> dict[str, Any]:
    cases = load_cases(run_dir)
    for index, case in enumerate(cases):
        if str(case.get("case_id") or "") == str(case_id or "").strip():
            updated = {**case, **updates}
            cases[index] = updated
            save_cases(run_dir, cases)
            return updated
    raise KeyError(f"Unknown case_id: {case_id}")


def materialize_case_branch_instructions(run_dir: Path, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    run = load_run(run_dir)
    enriched_cases: list[dict[str, Any]] = []
    for raw_case in cases:
        case = dict(raw_case)
        target_assets = resolve_case_target_assets(run, case)
        task_payload = _build_e2e_task_payload(case, target_assets)
        case.update(
            {
                "target_assets": target_assets,
                "pa_instruction": task_payload["paperalchemy_instruction"],
                "pa_instruction_zh": task_payload["paperalchemy_instruction_zh"],
                "e2e_instruction": task_payload["e2e_instruction"],
                "e2e_instruction_zh": task_payload["e2e_instruction_zh"],
            }
        )
        enriched_cases.append(case)
    return enriched_cases


def build_e2e_input_package(run_dir: Path, case_id: str) -> dict[str, Any]:
    run = load_run(run_dir)
    case = get_case(run_dir, case_id)
    clean_case_id = normalize_token(str(case["case_id"]), fallback="case")
    package_root = Path(run_dir) / E2E_INPUTS_DIRNAME
    package_root.mkdir(parents=True, exist_ok=True)
    package_dir = package_root / f"{clean_case_id}_package"
    zip_path = package_root / f"{clean_case_id}_e2e_input.zip"
    _ensure_within(package_dir, package_root)
    _ensure_within(zip_path, package_root)

    baseline_zip = Path(str(run["baseline_zip_path"]))
    if not baseline_zip.exists() or not baseline_zip.is_file():
        raise FileNotFoundError(f"Baseline zip is missing: {baseline_zip}")

    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True, exist_ok=False)
    safe_extract_zip(baseline_zip, package_dir)
    _prune_e2e_package_dir(package_dir)
    paper_pdf_path = _copy_e2e_source_pdf(run, package_dir)

    target_assets = resolve_case_target_assets(run, case)
    copied_target_assets = _copy_e2e_target_assets(package_dir, target_assets)
    llm_site_dirs = _prepare_llm_site_dirs(run_dir, str(case["case_id"]), package_dir)
    task_payload = _build_e2e_task_payload(case, copied_target_assets, paper_pdf_path=paper_pdf_path)

    task_dir = package_dir / "benchmark_case"
    task_dir.mkdir(parents=True, exist_ok=True)
    task_json_path = task_dir / "task.json"
    task_json_path.write_text(json.dumps(task_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    readme_path = package_dir / "README_E2E.md"
    readme_path.write_text(_build_e2e_readme(task_payload), encoding="utf-8")

    _build_directory_zip(package_dir, zip_path)
    updates: dict[str, Any] = {
        "target_assets": copied_target_assets,
        "pa_instruction": task_payload["paperalchemy_instruction"],
        "pa_instruction_zh": task_payload["paperalchemy_instruction_zh"],
        "e2e_instruction": task_payload["e2e_instruction"],
        "e2e_instruction_zh": task_payload["e2e_instruction_zh"],
        "paper_pdf_path": paper_pdf_path,
        "e2e_input_zip_path": str(zip_path),
        "e2e_task_json_path": str(task_json_path),
        "e2e_prompt": task_payload["e2e_prompt"],
        "llm_site_dirs": llm_site_dirs,
    }
    updated_case = update_case(run_dir, str(case["case_id"]), updates)
    return {
        "case": updated_case,
        "zip_path": str(zip_path),
        "package_dir": str(package_dir),
        "task_json_path": str(task_json_path),
        "llm_site_dirs": llm_site_dirs,
        "target_assets": copied_target_assets,
        "paper_pdf_path": paper_pdf_path,
        "e2e_prompt": task_payload["e2e_prompt"],
    }


def _copy_e2e_source_pdf(run: dict[str, Any], package_dir: Path) -> str:
    paper_folder_name = str(run.get("paper_folder_name") or "").strip()
    output_dir, _, _, _ = get_output_paths(paper_folder_name)
    source_pdf = output_dir.parent.parent / "input" / f"{paper_folder_name}.pdf"
    if not source_pdf.exists() or not source_pdf.is_file():
        raise FileNotFoundError(f"Original paper PDF is missing: {source_pdf}")
    destination = (package_dir / E2E_PAPER_PDF_PATH).resolve()
    _ensure_within(destination, package_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_pdf, destination)
    return E2E_PAPER_PDF_PATH


def _prepare_llm_site_dirs(run_dir: Path, case_id: str, package_dir: Path) -> dict[str, str]:
    clean_case_id = normalize_token(str(case_id or ""), fallback="case")
    sites_root = Path(run_dir) / LLM_SITES_DIRNAME / clean_case_id
    _ensure_within(sites_root, Path(run_dir) / LLM_SITES_DIRNAME)
    if sites_root.exists():
        shutil.rmtree(sites_root)
    sites_root.mkdir(parents=True, exist_ok=True)

    site_dirs: dict[str, str] = {}
    for model_id in CONTROL_MODEL_IDS:
        model_site_dir = sites_root / f"{model_id}_site"
        shutil.copytree(package_dir, model_site_dir)
        site_dirs[model_id] = str(model_site_dir)
    return site_dirs


def resolve_case_target_assets(run: dict[str, Any], case: dict[str, Any]) -> list[dict[str, str]]:
    paper_folder_name = str(run["paper_folder_name"])
    output_dir, structured_json_path, _, _ = get_output_paths(paper_folder_name)
    structured_paper = load_cached_structured_data(structured_json_path)
    if structured_paper is None:
        return []

    registry = list(structured_paper.asset_registry)
    lookup = {str(asset.asset_id or "").strip(): asset for asset in registry}
    target_assets = _normalize_requested_target_assets(
        raw_assets=case.get("target_assets"),
        paper_folder_name=paper_folder_name,
        output_dir=output_dir,
        asset_lookup=lookup,
    )
    if target_assets:
        return target_assets

    if str(case.get("subcategory") or "").strip() != "asset_replacement":
        return []

    used_asset_ids = _baseline_used_asset_ids(Path(str(run["baseline_live_site_dir"])), lookup)
    selected = next(
        (asset for asset in registry if str(asset.asset_id or "").strip() not in used_asset_ids),
        registry[0] if registry else None,
    )
    if selected is None:
        return []
    return [
        _target_asset_payload(
            paper_folder_name=paper_folder_name,
            asset=selected,
            source_path=resolved_asset_source_path(Path(__file__).resolve().parents[2], paper_folder_name, selected),
        )
    ]


def _normalize_requested_target_assets(
    *,
    raw_assets: Any,
    paper_folder_name: str,
    output_dir: Path,
    asset_lookup: dict[str, Any],
) -> list[dict[str, str]]:
    if not isinstance(raw_assets, list):
        return []
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    project_root = Path(__file__).resolve().parents[2]
    for index, raw_asset in enumerate(raw_assets, start=1):
        asset_id = ""
        source_path: Path | None = None
        if isinstance(raw_asset, str):
            asset_id = raw_asset.strip()
        elif isinstance(raw_asset, dict):
            asset_id = str(raw_asset.get("asset_id") or "").strip()
            raw_source = str(raw_asset.get("source_path") or raw_asset.get("path") or "").strip()
            if raw_source:
                source_path = _resolve_target_source_path(raw_source, output_dir=output_dir)
        else:
            continue

        asset = asset_lookup.get(asset_id)
        if asset is not None:
            source_path = resolved_asset_source_path(project_root, paper_folder_name, asset)
            asset_id = str(asset.asset_id or "").strip()
            payload = _target_asset_payload(
                paper_folder_name=paper_folder_name,
                asset=asset,
                source_path=source_path,
            )
        elif source_path is not None:
            fallback_id = normalize_token(asset_id or source_path.stem, fallback=f"target_{index}")
            payload = {
                "asset_id": fallback_id,
                "source_path": str(source_path),
                "pa_web_path": "",
                "e2e_path": f"benchmark_targets/{fallback_id}{source_path.suffix or '.png'}",
                "filename": f"{fallback_id}{source_path.suffix or '.png'}",
                "caption": "",
                "type": "target_asset",
                "section_title": "",
                "page_number": "",
            }
        else:
            continue

        key = str(payload.get("asset_id") or payload.get("source_path") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(payload)
    return result


def _resolve_target_source_path(raw_source: str, *, output_dir: Path) -> Path:
    candidate = Path(raw_source)
    if not candidate.is_absolute():
        candidate = output_dir / candidate
    resolved = candidate.resolve()
    _ensure_within(resolved, output_dir)
    if not resolved.exists() or not resolved.is_file():
        raise FileNotFoundError(f"Target asset source file is missing: {resolved}")
    return resolved


def _target_asset_payload(*, paper_folder_name: str, asset: Any, source_path: Path) -> dict[str, str]:
    asset_id = str(asset.asset_id or "").strip()
    filename = asset_target_filename(asset)
    return {
        "asset_id": asset_id,
        "source_path": str(source_path),
        "paperalchemy_source_path": str(asset.image_path or "").strip(),
        "pa_web_path": f"./assets/paper/{filename}",
        "e2e_path": f"benchmark_targets/{filename}",
        "filename": filename,
        "caption": str(asset.caption or "").strip(),
        "type": str(asset.type or "").strip(),
        "section_title": "",
        "page_number": str(asset.page_number or ""),
        "paper_folder_name": str(paper_folder_name),
    }


def _baseline_used_asset_ids(baseline_live_site_dir: Path, asset_lookup: dict[str, Any]) -> set[str]:
    html_text = ""
    for html_path in sorted(Path(baseline_live_site_dir).rglob("*.html")):
        try:
            html_text += "\n" + html_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
    used: set[str] = set()
    for asset_id, asset in asset_lookup.items():
        filename = asset_target_filename(asset)
        if filename and filename in html_text:
            used.add(asset_id)
    return used


def _copy_e2e_target_assets(package_dir: Path, target_assets: list[dict[str, str]]) -> list[dict[str, str]]:
    copied: list[dict[str, str]] = []
    targets_dir = package_dir / "benchmark_targets"
    for index, raw_asset in enumerate(target_assets, start=1):
        source_path = Path(str(raw_asset.get("source_path") or "")).resolve()
        if not source_path.exists() or not source_path.is_file():
            continue
        filename = normalize_token(str(raw_asset.get("filename") or source_path.name), fallback=f"target_{index}")
        suffix = source_path.suffix or Path(filename).suffix or ".png"
        if Path(filename).suffix == "":
            filename = f"{filename}{suffix}"
        e2e_path = f"benchmark_targets/{filename}"
        destination = (package_dir / e2e_path).resolve()
        _ensure_within(destination, targets_dir)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)
        copied.append({**raw_asset, "filename": filename, "e2e_path": e2e_path})
    return copied


def _build_e2e_task_payload(
    case: dict[str, Any],
    target_assets: list[dict[str, str]],
    *,
    paper_pdf_path: str = E2E_PAPER_PDF_PATH,
) -> dict[str, Any]:
    base_instruction = str(case.get("instruction") or "").strip()
    base_instruction_zh = str(case.get("instruction_zh") or base_instruction).strip()
    pa_assets = _format_asset_path_list(target_assets, "pa_web_path")
    e2e_assets = _format_asset_path_list(target_assets, "e2e_path")

    paperalchemy_instruction = base_instruction
    paperalchemy_instruction_zh = base_instruction_zh
    e2e_instruction = base_instruction
    e2e_instruction_zh = base_instruction_zh
    if pa_assets:
        paperalchemy_instruction = f"{base_instruction} Use the PaperAlchemy-accessible target asset path(s): {pa_assets}."
        paperalchemy_instruction_zh = f"{base_instruction_zh} 请使用 PaperAlchemy 可访问的目标素材路径：{pa_assets}。"
    if e2e_assets:
        e2e_instruction = f"{base_instruction} Use the provided target asset file(s): {e2e_assets}."
        e2e_instruction_zh = f"{base_instruction_zh} 请使用随包提供的目标素材文件：{e2e_assets}。"

    payload = {
        "schema_version": "paperalchemy-benchmark-v1-e2e-task",
        "case_id": case.get("case_id"),
        "task_type": case.get("task_type"),
        "subcategory": case.get("subcategory"),
        "category_label_zh": case.get("category_label_zh"),
        "subcategory_label_zh": case.get("subcategory_label_zh"),
        "difficulty": case.get("difficulty"),
        "difficulty_reason": case.get("difficulty_reason"),
        "pdf_evidence": case.get("pdf_evidence"),
        "web_evidence": case.get("web_evidence"),
        "forbidden_changes": case.get("forbidden_changes") or [],
        "instruction": base_instruction,
        "instruction_zh": base_instruction_zh,
        "paperalchemy_instruction": paperalchemy_instruction,
        "paperalchemy_instruction_zh": paperalchemy_instruction_zh,
        "e2e_instruction": e2e_instruction,
        "e2e_instruction_zh": e2e_instruction_zh,
        "target_hint": case.get("target_hint"),
        "target_hint_zh": case.get("target_hint_zh"),
        "expected_observable": case.get("expected_observable"),
        "expected_observable_zh": case.get("expected_observable_zh"),
        "target_selectors": case.get("target_selectors") or [],
        "target_assets": target_assets,
        "paper_pdf_path": paper_pdf_path,
    }
    payload["e2e_prompt"] = _build_e2e_prompt(payload)
    return payload


def _format_asset_path_list(target_assets: list[dict[str, str]], key: str) -> str:
    paths = [str(asset.get(key) or "").strip() for asset in target_assets]
    paths = [path for path in paths if path]
    return ", ".join(paths)


def _build_e2e_prompt(task_payload: dict[str, Any]) -> str:
    target_asset_lines = []
    for asset in task_payload.get("target_assets") or []:
        line = f"- {asset.get('e2e_path')}"
        caption = str(asset.get("caption") or "").strip()
        if caption:
            line += f" ({caption})"
        target_asset_lines.append(line)
    target_assets = "\n".join(target_asset_lines) if target_asset_lines else "(none)"
    forbidden_changes = task_payload.get("forbidden_changes") or []
    forbidden_text = "\n".join(f"- {item}" for item in forbidden_changes if str(item or "").strip()) or "(none)"
    paper_pdf_path = str(task_payload.get("paper_pdf_path") or E2E_PAPER_PDF_PATH).strip()
    return f"""You are given a static website package for a paper webpage.

Modify the website under the `site/` directory according to the instruction below.
The original paper PDF is included at `{paper_pdf_path}`. Inspect it yourself when the task requires paper content, figures, tables, captions, references, terminology, or semantic consistency.
No PaperAlchemy parser, reader, planner, target-selector, asset-manifest, or other intermediate representation is provided.
Preserve unrelated content, layout, anonymous anchor classes, filenames, and existing behavior unless the instruction requires changes.
Use only files already inside this package. If target assets are listed, use those files exactly.
Return only the files you changed for this task.

Output requirements:
- Do not return a zip file, base64 archive, or a full project dump.
- Do not return files that you did not modify.
- For each changed file, provide the complete final file content and its original relative path under `site/`, such as `site/index.html`, `site/css/styles.css`, or `site/js/homeSetup.js`.
- Keep all asset references resolvable from the file you edit. Target assets below are package-root paths; from `site/index.html`, a target asset is usually referenced as `../benchmark_targets/<filename>`.

Instruction:
{task_payload.get("e2e_instruction") or task_payload.get("instruction") or ""}

PDF-derived evidence:
{task_payload.get("pdf_evidence") or ""}

Baseline webpage evidence:
{task_payload.get("web_evidence") or ""}

Forbidden changes:
{forbidden_text}

Paper PDF:
- {paper_pdf_path}

Target assets:
{target_assets}
"""


def _build_e2e_readme(task_payload: dict[str, Any]) -> str:
    return f"""# PaperAlchemy Benchmark V1 E2E Input

Open `benchmark_case/task.json` for machine-readable metadata.

Send the following prompt to the endpoint LLM together with this zip package:

```text
{task_payload["e2e_prompt"].strip()}
```
"""


def _build_directory_zip(source_dir: Path, zip_path: Path) -> Path:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source_dir.rglob("*")):
            if not path.is_file():
                continue
            arcname = path.resolve().relative_to(source_dir.resolve()).as_posix()
            if should_exclude_benchmark_member(arcname):
                continue
            archive.write(path, arcname=arcname)
    return zip_path


def _prune_e2e_package_dir(package_dir: Path) -> None:
    root = Path(package_dir).resolve()
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        try:
            relative = path.resolve().relative_to(root).as_posix()
        except ValueError:
            continue
        if not should_exclude_benchmark_member(relative):
            continue
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.exists():
            path.unlink(missing_ok=True)


def should_exclude_benchmark_member(relative_path: str) -> bool:
    parts = [part.lower() for part in str(relative_path or "").replace("\\", "/").split("/") if part]
    if not parts:
        return False
    if any(part in E2E_EXCLUDED_DIRNAMES for part in parts):
        return True
    return parts[-1] in E2E_EXCLUDED_FILENAMES


def _should_exclude_e2e_member(relative_path: str) -> bool:
    return should_exclude_benchmark_member(relative_path)


def restore_pa_baseline(run_dir: Path) -> dict[str, Any]:
    run = load_run(run_dir)
    paper_folder_name = str(run["paper_folder_name"])
    output_dir, _, _, coder_json_path = get_output_paths(paper_folder_name)
    artifact = load_coder_artifact(coder_json_path)
    if artifact is None:
        raise FileNotFoundError(f"Live coder_artifact.json is missing or invalid: {coder_json_path}")
    live_site_dir = resolve_artifact_path(str(artifact.site_dir or ""), output_dir=output_dir)
    baseline_live_site_dir = Path(str(run["baseline_live_site_dir"]))
    if not baseline_live_site_dir.exists() or not baseline_live_site_dir.is_dir():
        raise FileNotFoundError(f"Baseline live site copy is missing: {baseline_live_site_dir}")
    _ensure_within(live_site_dir, output_dir)

    backups_dir = Path(run_dir) / "backups"
    backups_dir.mkdir(parents=True, exist_ok=True)
    backup_dir = backups_dir / f"restore_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{os.urandom(3).hex()}"
    if live_site_dir.exists():
        shutil.copytree(live_site_dir, backup_dir)
        shutil.rmtree(live_site_dir)
    shutil.copytree(baseline_live_site_dir, live_site_dir)
    return {
        "live_site_dir": str(live_site_dir),
        "baseline_live_site_dir": str(baseline_live_site_dir),
        "backup_dir": str(backup_dir) if backup_dir.exists() else "",
    }


def capture_pa_final(
    run_dir: Path,
    case_id: str,
    *,
    force_noop: bool = False,
    cooldown_seconds: float = DEFAULT_CAPTURE_COOLDOWN_SECONDS,
) -> dict[str, Any]:
    run = load_run(run_dir)
    case = get_case(run_dir, case_id)
    paper_folder_name = str(run["paper_folder_name"])
    output_dir, _, _, coder_json_path = get_output_paths(paper_folder_name)
    artifact = load_coder_artifact(coder_json_path)
    if artifact is None:
        raise FileNotFoundError(f"Live coder_artifact.json is missing or invalid: {coder_json_path}")
    live_site_dir = resolve_artifact_path(str(artifact.site_dir or ""), output_dir=output_dir)
    _ensure_capture_cooldown(live_site_dir, coder_json_path, cooldown_seconds=cooldown_seconds)

    baseline_hash = hash_directory(Path(str(run["baseline_live_site_dir"])))
    live_hash = hash_directory(live_site_dir)
    no_op = baseline_hash == live_hash
    if no_op and not force_noop:
        raise ValueError(
            "Current PA live site still matches the V1 baseline. "
            "Run the PA revision first, or enable forced no-op capture."
        )

    revision = _next_pa_revision(paper_folder_name, str(run["run_id"]), str(case["case_id"]))
    export_name = make_pa_export_name(str(run["run_id"]), str(case["case_id"]), revision)
    metadata = export_live_experiment_snapshot(
        paper_folder_name=paper_folder_name,
        export_name=export_name,
    )
    updated_case = update_case(
        run_dir,
        str(case["case_id"]),
        {
            "pa_export_name": export_name,
            "pa_export_metadata": metadata,
            "capture_forced": bool(force_noop),
            "capture_noop_detected": bool(no_op),
            "captured_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return {
        "case": updated_case,
        "pa_export_name": export_name,
        "metadata": metadata,
        "capture_forced": bool(force_noop),
        "capture_noop_detected": bool(no_op),
    }


def ingest_model_changed_files(
    run_dir: Path,
    case_id: str,
    uploaded_paths: Any,
    relative_paths: str,
    *,
    model_id: str,
    model_label: str | None = None,
) -> dict[str, Any]:
    run = load_run(run_dir)
    case = get_case(run_dir, case_id)
    clean_model_id = normalize_token(str(model_id or "").lower(), fallback="model")
    if clean_model_id not in CONTROL_MODEL_IDS:
        raise ValueError(f"Unsupported control model: {clean_model_id}")
    clean_model_label = str(model_label or CONTROL_MODEL_LABELS.get(clean_model_id) or clean_model_id).strip()
    model_site_dir = _reset_model_working_site(run_dir, case, clean_model_id, clean_model_label)

    uploads = _normalize_uploaded_file_paths(uploaded_paths)
    rel_paths = _parse_changed_file_relative_paths(relative_paths)
    if not uploads:
        raise ValueError(f"Upload at least one changed file for {clean_model_label}.")
    if len(uploads) != len(rel_paths):
        raise ValueError(
            f"{clean_model_label} changed file count ({len(uploads)}) must match the relative path line count "
            f"({len(rel_paths)})."
        )

    pending: list[tuple[Path, str, Path]] = []
    for source_path, raw_rel_path in zip(uploads, rel_paths):
        source = Path(str(source_path or "")).resolve()
        if not source.exists() or not source.is_file():
            raise FileNotFoundError(f"Uploaded changed file does not exist: {source}")
        clean_rel_path, target = _validate_changed_file_target(model_site_dir, raw_rel_path)
        pending.append((source, clean_rel_path, target))

    applied_files: list[dict[str, Any]] = []
    for source, clean_rel_path, target in pending:
        shutil.copy2(source, target)
        applied_files.append(
            {
                "relative_path": clean_rel_path,
                "source_path": str(source),
                "target_path": str(target),
                "size_bytes": target.stat().st_size,
            }
        )

    target_dir = Path(run_dir) / "model_uploads" / str(case["case_id"]) / clean_model_id
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir = target_dir / "extracted"
    shutil.copytree(model_site_dir, extracted_dir)
    entry_html = find_entry_html(extracted_dir)
    stored_zip = target_dir / f"{clean_model_id}_final.zip"
    _build_directory_zip(extracted_dir, stored_zip)
    screenshot_path = take_benchmark_screenshot(str(entry_html), str(target_dir / "final.png"))

    class_validation = _class_validation_from_dir(Path(str(run["baseline_zip_path"])), extracted_dir)
    ingested_at = datetime.now(timezone.utc).isoformat()
    control_model_results = dict(case.get("control_model_results") or {})
    control_model_results[clean_model_id] = {
        "model_id": clean_model_id,
        "model_label": clean_model_label,
        "input_mode": "changed_files",
        "zip_path": str(stored_zip),
        "result_dir": str(extracted_dir),
        "entry_html": str(entry_html),
        "working_site_dir": str(model_site_dir),
        "applied_files": applied_files,
        "screenshot_path": screenshot_path,
        "class_validation": class_validation,
        "ingested_at": ingested_at,
    }
    updated_case = update_case(
        run_dir,
        str(case["case_id"]),
        {
            "control_model_results": control_model_results,
        },
    )
    return {
        "case": updated_case,
        "model_id": clean_model_id,
        "model_label": clean_model_label,
        "applied_files": applied_files,
        "entry_html": str(entry_html),
        "screenshot_path": screenshot_path,
        "warnings": class_validation.get("warnings") or [],
    }


def ensure_blind_model_order(run_dir: Path, case_id: str) -> list[str]:
    run = load_run(run_dir)
    case = get_case(run_dir, case_id)
    existing = [str(item) for item in case.get("blind_model_order") or []]
    if sorted(existing) == sorted(CONTROL_MODEL_IDS):
        return existing
    seed = f"{run.get('run_id')}:{case.get('case_id')}"
    order = sorted(
        CONTROL_MODEL_IDS,
        key=lambda model_id: hashlib.sha256(f"{seed}:{model_id}".encode("utf-8")).hexdigest(),
    )
    update_case(run_dir, str(case["case_id"]), {"blind_model_order": order})
    return order


def record_manual_score(
    run_dir: Path,
    case_id: str,
    system_name: str,
    *,
    completion_score: float,
    visual_score: float,
    evaluator_note: str = "",
) -> dict[str, Any]:
    case = get_case(run_dir, case_id)
    clean_system_name = str(system_name or "").strip().lower()
    if not clean_system_name:
        raise ValueError("system_name is required.")
    manual_scores = dict(case.get("manual_score_inputs") or {})
    manual_scores[clean_system_name] = {
        "completion": float(completion_score),
        "visual": float(visual_score),
        "note": str(evaluator_note or ""),
        "scored_at": datetime.now(timezone.utc).isoformat(),
    }
    return update_case(run_dir, str(case["case_id"]), {"manual_score_inputs": manual_scores})


def record_model_failure(
    run_dir: Path,
    case_id: str,
    model_id: str,
    *,
    failure_reason: str = "",
    model_label: str | None = None,
) -> dict[str, Any]:
    case = get_case(run_dir, case_id)
    clean_model_id = normalize_token(str(model_id or "").lower(), fallback="model")
    if clean_model_id not in CONTROL_MODEL_IDS:
        raise ValueError(f"Unsupported control model: {clean_model_id}")
    clean_model_label = str(model_label or CONTROL_MODEL_LABELS.get(clean_model_id) or clean_model_id).strip()
    reason = str(failure_reason or "").strip() or "Model did not provide a valid changed-file result."
    control_model_results = dict(case.get("control_model_results") or {})
    control_model_results[clean_model_id] = {
        "model_id": clean_model_id,
        "model_label": clean_model_label,
        "input_mode": "failed",
        "failure_reason": reason,
        "zip_path": "",
        "result_dir": "",
        "entry_html": "",
        "working_site_dir": str(_model_working_site_dir(run_dir, case, clean_model_id)),
        "applied_files": [],
        "screenshot_path": "",
        "class_validation": {
            "status": "failed",
            "warnings": [reason],
        },
        "failed_at": datetime.now(timezone.utc).isoformat(),
    }
    return update_case(run_dir, str(case["case_id"]), {"control_model_results": control_model_results})


def ingest_web_llm_result(
    run_dir: Path,
    case_id: str,
    zip_path: str,
    *,
    allow_partial_class_match: bool = False,
) -> dict[str, Any]:
    return ingest_model_result(
        run_dir,
        case_id,
        zip_path,
        model_id="web_llm",
        model_label="Web-LLM",
        allow_partial_class_match=allow_partial_class_match,
    )


def ingest_model_result(
    run_dir: Path,
    case_id: str,
    zip_path: str,
    *,
    model_id: str,
    model_label: str | None = None,
    allow_partial_class_match: bool = False,
) -> dict[str, Any]:
    run = load_run(run_dir)
    case = get_case(run_dir, case_id)
    clean_model_id = normalize_token(str(model_id or "").lower(), fallback="model")
    clean_model_label = str(model_label or CONTROL_MODEL_LABELS.get(clean_model_id) or clean_model_id).strip()
    source_zip = Path(str(zip_path or "")).resolve()
    if not source_zip.exists() or not source_zip.is_file():
        raise FileNotFoundError(f"{clean_model_label} zip does not exist: {source_zip}")

    baseline_zip = Path(str(run["baseline_zip_path"]))
    baseline_classes = extract_opaque_classes_from_zip(baseline_zip)
    uploaded_classes = extract_opaque_classes_from_zip(source_zip)
    overlap = baseline_classes & uploaded_classes
    missing = sorted(baseline_classes - uploaded_classes)
    status = "ok"
    warnings: list[str] = []
    if baseline_classes and not overlap:
        raise ValueError(f"Uploaded {clean_model_label} zip does not share opaque PA classes with the baseline zip.")
    if missing:
        status = "partial"
        warnings.append(
            f"Uploaded {clean_model_label} zip is missing {len(missing)} baseline opaque class(es)."
        )
        if not allow_partial_class_match:
            raise ValueError(
                f"Uploaded {clean_model_label} zip is missing some baseline opaque PA classes. "
                "Confirm the partial class mismatch to ingest it anyway."
            )

    target_dir = Path(run_dir) / "model_uploads" / str(case["case_id"]) / clean_model_id
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    stored_zip = target_dir / f"{clean_model_id}_final.zip"
    shutil.copy2(source_zip, stored_zip)
    extracted_dir = target_dir / "extracted"
    safe_extract_zip(stored_zip, extracted_dir)
    entry_html = find_entry_html(extracted_dir)
    class_validation = {
        "status": status,
        "baseline_class_count": len(baseline_classes),
        "uploaded_class_count": len(uploaded_classes),
        "overlap_count": len(overlap),
        "missing_count": len(missing),
        "partial_class_match_confirmed": bool(missing and allow_partial_class_match),
        "warnings": warnings,
    }
    ingested_at = datetime.now(timezone.utc).isoformat()
    control_model_results = dict(case.get("control_model_results") or {})
    control_model_results[clean_model_id] = {
        "model_id": clean_model_id,
        "model_label": clean_model_label,
        "zip_path": str(stored_zip),
        "result_dir": str(extracted_dir),
        "entry_html": str(entry_html),
        "class_validation": class_validation,
        "ingested_at": ingested_at,
    }
    updates: dict[str, Any] = {
        "control_model_results": control_model_results,
    }
    if clean_model_id == "web_llm":
        updates.update(
            {
                "web_llm_zip_path": str(stored_zip),
                "web_llm_result_dir": str(extracted_dir),
                "web_llm_entry_html": str(entry_html),
                "web_llm_class_validation": class_validation,
                "web_llm_ingested_at": ingested_at,
            }
        )
    updated_case = update_case(
        run_dir,
        str(case["case_id"]),
        updates,
    )
    return {
        "case": updated_case,
        "model_id": clean_model_id,
        "model_label": clean_model_label,
        "warnings": warnings,
        "entry_html": str(entry_html),
    }


def _model_working_site_dir(run_dir: Path, case: dict[str, Any], model_id: str) -> Path:
    llm_site_dirs = dict(case.get("llm_site_dirs") or {})
    stored = str(llm_site_dirs.get(model_id) or "").strip()
    if stored:
        return Path(stored)
    clean_case_id = normalize_token(str(case.get("case_id") or ""), fallback="case")
    return Path(run_dir) / LLM_SITES_DIRNAME / clean_case_id / f"{model_id}_site"


def _reset_model_working_site(
    run_dir: Path,
    case: dict[str, Any],
    model_id: str,
    model_label: str,
) -> Path:
    destination = _model_working_site_dir(run_dir, case, model_id)
    clean_case_id = normalize_token(str(case.get("case_id") or ""), fallback="case")
    sites_root = Path(run_dir) / LLM_SITES_DIRNAME / clean_case_id
    _ensure_within(destination, sites_root)

    task_json_path = Path(str(case.get("e2e_task_json_path") or ""))
    package_dir = task_json_path.parent.parent if task_json_path.name else None
    if package_dir is None or not package_dir.exists() or not package_dir.is_dir():
        if destination.exists() and destination.is_dir():
            return destination
        raise FileNotFoundError(
            f"{model_label} working site is missing. Build the E2E input package for this case first."
        )

    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for child in sorted(package_dir.iterdir(), key=lambda item: item.name.lower()):
        if child.name in {"benchmark_case", "README_E2E.md"}:
            continue
        target = destination / child.name
        if child.is_dir():
            shutil.copytree(child, target)
        elif child.is_file():
            shutil.copy2(child, target)
    return destination


def _normalize_uploaded_file_paths(uploaded_paths: Any) -> list[Path]:
    if uploaded_paths is None:
        return []
    raw_items = uploaded_paths if isinstance(uploaded_paths, (list, tuple)) else [uploaded_paths]
    paths: list[Path] = []
    for item in raw_items:
        if item is None:
            continue
        raw_path = ""
        if isinstance(item, dict):
            raw_path = str(item.get("path") or item.get("name") or "")
        else:
            raw_path = str(getattr(item, "path", "") or item)
        if raw_path.strip():
            paths.append(Path(raw_path))
    return paths


def _parse_changed_file_relative_paths(relative_paths: str) -> list[str]:
    return [line.strip() for line in str(relative_paths or "").splitlines() if line.strip()]


def _validate_changed_file_target(model_site_dir: Path, relative_path: str) -> tuple[str, Path]:
    raw = str(relative_path or "").strip().replace("\\", "/")
    while raw.startswith("./"):
        raw = raw[2:]
    parts = [part for part in raw.split("/") if part]
    clean = "/".join(parts)
    if not clean or Path(raw).is_absolute() or any(part == ".." for part in parts):
        raise ValueError(f"Invalid changed file relative path: {relative_path}")
    if parts[0] != "site":
        raise ValueError(f"Changed files must target files under site/: {relative_path}")
    if should_exclude_benchmark_member(clean):
        raise ValueError(f"Changed file targets an excluded Benchmark artifact: {relative_path}")
    suffix = Path(clean).suffix.lower()
    if suffix not in TEXT_SUFFIXES:
        raise ValueError(f"Changed file must be a text/code file, got {suffix or '(none)'}: {relative_path}")
    target = (Path(model_site_dir) / clean).resolve()
    _ensure_within(target, Path(model_site_dir))
    if not target.exists() or not target.is_file():
        raise FileNotFoundError(f"Changed file target does not exist in the working site: {clean}")
    return clean, target


def _class_validation_from_dir(baseline_zip: Path, final_dir: Path) -> dict[str, Any]:
    baseline_classes = extract_opaque_classes_from_zip(baseline_zip)
    final_classes = extract_opaque_classes_from_dir(final_dir)
    overlap = baseline_classes & final_classes
    missing = sorted(baseline_classes - final_classes)
    status = "ok"
    warnings: list[str] = []
    if baseline_classes and not overlap:
        status = "none"
        warnings.append("Final site does not share opaque PA classes with the baseline.")
    elif missing:
        status = "partial"
        warnings.append(f"Final site is missing {len(missing)} baseline opaque class(es).")
    return {
        "status": status,
        "baseline_class_count": len(baseline_classes),
        "final_class_count": len(final_classes),
        "overlap_count": len(overlap),
        "missing_count": len(missing),
        "warnings": warnings,
    }


def extract_opaque_classes_from_zip(zip_path: Path) -> set[str]:
    classes: set[str] = set()
    with zipfile.ZipFile(zip_path, "r") as archive:
        for member in archive.infolist():
            suffix = Path(member.filename).suffix.lower()
            if suffix not in TEXT_SUFFIXES:
                continue
            try:
                text = archive.read(member).decode("utf-8", errors="ignore")
            except Exception:
                continue
            classes.update(OPAQUE_CLASS_PATTERN.findall(text))
    return classes


def extract_opaque_classes_from_dir(root_dir: Path) -> set[str]:
    classes: set[str] = set()
    root = Path(root_dir)
    if not root.exists():
        return classes
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        classes.update(OPAQUE_CLASS_PATTERN.findall(text))
    return classes


def safe_extract_zip(zip_path: Path, target_dir: Path) -> None:
    root = target_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        for member in archive.infolist():
            member_name = str(member.filename or "").replace("\\", "/")
            if should_exclude_benchmark_member(member_name):
                continue
            destination = (root / member_name).resolve()
            _ensure_within(destination, root)
            if member.is_dir() or member_name.endswith("/"):
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member, "r") as source, destination.open("wb") as output:
                shutil.copyfileobj(source, output)


def find_entry_html(root_dir: Path) -> Path:
    root = Path(root_dir)
    preferred = root / "site" / "index.html"
    if preferred.exists():
        return preferred
    candidates = sorted(root.rglob("index.html"))
    if candidates:
        return candidates[0]
    html_candidates = sorted(root.rglob("*.html"))
    if html_candidates:
        return html_candidates[0]
    raise FileNotFoundError(f"No HTML entry file found under {root}")


def hash_directory(root_dir: Path) -> str:
    root = Path(root_dir).resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {root}")
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.resolve().relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _next_pa_revision(paper_folder_name: str, run_id: str, case_id: str) -> int:
    revision = 1
    while True:
        export_name = make_pa_export_name(run_id, case_id, revision)
        if not build_experiment_export_dir(paper_folder_name, export_name).exists():
            return revision
        revision += 1


def _ensure_capture_cooldown(site_dir: Path, coder_json_path: Path, *, cooldown_seconds: float) -> None:
    if cooldown_seconds <= 0:
        return
    latest_mtime = max(_latest_mtime(site_dir), coder_json_path.stat().st_mtime if coder_json_path.exists() else 0.0)
    age = time.time() - latest_mtime
    if age < cooldown_seconds:
        raise ValueError(
            "PA live artifacts were modified recently. Confirm the main App is idle, "
            f"then retry after {cooldown_seconds - age:.1f}s."
        )


def _latest_mtime(root: Path) -> float:
    if not root.exists():
        return 0.0
    latest = root.stat().st_mtime
    for path in root.rglob("*"):
        try:
            latest = max(latest, path.stat().st_mtime)
        except OSError:
            continue
    return latest


def _ensure_within(path: Path, root: Path) -> None:
    resolved_path = Path(path).resolve()
    resolved_root = Path(root).resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"Path escapes expected root: {resolved_path}") from exc
