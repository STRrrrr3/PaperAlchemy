from collections.abc import Callable
from pathlib import Path
import re
from typing import Any

from src.schemas import CoderArtifact, CoderCriticReport
from src.state import CoderState

MAX_CODER_RETRY_DEFAULT = 1


def _normalize_coder_artifact(artifact: Any) -> CoderArtifact | None:
    if artifact is None:
        return None
    if isinstance(artifact, CoderArtifact):
        return artifact
    try:
        return CoderArtifact.model_validate(artifact)
    except Exception:
        return None


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

    if "PaperAlchemy Generated Body Start" not in html_text or "PaperAlchemy Generated Body End" not in html_text:
        critiques.append("Generated body markers are missing in entry html.")

    body_start_pattern = re.compile(
        r"<body[^>]*>\s*<!--\s*PaperAlchemy Generated Body Start\s*-->",
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not body_start_pattern.search(html_text):
        critiques.append("Generated body marker is not at body start; template content leakage is likely.")

    title_count = len(re.findall(r"<title\b", html_text, flags=re.IGNORECASE))
    if title_count != 1:
        critiques.append(f"Expected exactly one <title> tag, found {title_count}.")

    for rel_asset in artifact.copied_assets:
        asset_path = site_dir / rel_asset
        if not asset_path.exists():
            critiques.append(f"Copied asset missing: {asset_path}")
        if rel_asset not in html_text:
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


def build_coder_critic_router(max_retry: int = MAX_CODER_RETRY_DEFAULT) -> Callable[[CoderState], str]:
    def _router(state: CoderState) -> str:
        if state.get("coder_critic_passed"):
            return "end"
        if int(state.get("coder_retry_count", 0)) >= max_retry:
            print(f"[PaperAlchemy-CoderCritic] reached max retry limit ({max_retry}), stop.")
            return "end"
        return "retry"

    return _router
