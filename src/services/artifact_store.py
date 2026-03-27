from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.contracts.schemas import BlockRenderArtifact, CoderArtifact, PagePlan, StructuredPaper, TemplateProfile

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"

def load_cached_structured_data(path: Path) -> StructuredPaper | None:
    if not path.exists():
        return None

    print("[PaperAlchemy] Found cached structured paper, loading...")
    try:
        with open(path, "r", encoding="utf-8") as file:
            data_dict = json.load(file)
        structured_data = StructuredPaper(**data_dict)
        print(f"[PaperAlchemy] Structured paper loaded: {structured_data.paper_title}")
        return structured_data
    except Exception as exc:
        print(f"[PaperAlchemy] Structured cache is invalid, rerunning Reader: {exc}")
        return None

def save_structured_data(path: Path, structured_data: StructuredPaper) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(structured_data.model_dump(), file, indent=2, ensure_ascii=False)

def save_page_plan(path: Path, page_plan: PagePlan) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(page_plan.model_dump(), file, indent=2, ensure_ascii=False)

def save_coder_artifact(path: Path, artifact: CoderArtifact) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(artifact.model_dump(), file, indent=2, ensure_ascii=False)

def save_template_profile(path: Path, profile: TemplateProfile) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(profile.model_dump(), file, indent=2, ensure_ascii=False)

def get_output_paths(paper_folder_name: str) -> tuple[Path, Path, Path, Path]:
    output_dir = OUTPUT_DIR / paper_folder_name
    structured_json_path = output_dir / "structured_paper.json"
    planner_json_path = output_dir / "page_plan.json"
    coder_json_path = output_dir / "coder_artifact.json"
    return output_dir, structured_json_path, planner_json_path, coder_json_path

def get_template_profile_output_path(paper_folder_name: str) -> Path:
    output_dir = OUTPUT_DIR / paper_folder_name
    return output_dir / "template_profile.json"

def load_block_render_artifacts_from_disk(paper_folder_name: str) -> list[BlockRenderArtifact]:
    output_dir = OUTPUT_DIR / paper_folder_name
    block_renders_dir = output_dir / "block_renders"
    if not block_renders_dir.exists():
        return []

    artifacts: list[BlockRenderArtifact] = []
    for json_path in sorted(block_renders_dir.glob("*.json"), key=lambda item: item.name.lower()):
        try:
            artifact = BlockRenderArtifact.model_validate_json(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        artifacts.append(artifact)
    return artifacts
