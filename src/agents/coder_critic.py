from collections.abc import Callable
from pathlib import Path
import re
from typing import Any

from bs4 import BeautifulSoup

from src.contracts.schemas import CoderArtifact, PagePlan
from src.contracts.state import CoderState
from src.services.artifact_store import load_template_profile
from src.validators.page_manifest import build_page_manifest_path, extract_page_manifest, load_page_manifest
from src.validators.page_validation import collect_allowed_asset_web_paths, validate_local_image_references

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


def _normalize_page_plan(plan: Any) -> PagePlan | None:
    if plan is None:
        return None
    if isinstance(plan, PagePlan):
        return plan
    try:
        return PagePlan.model_validate(plan)
    except Exception:
        return None


def _load_template_profile_from_artifact(artifact: CoderArtifact | None):
    if artifact is None:
        return None
    template_profile_path = str(artifact.template_profile_path or "").strip()
    if not template_profile_path:
        return None
    return load_template_profile(Path(template_profile_path))


def _available_asset_manifest_from_artifact(artifact: CoderArtifact) -> list[dict[str, str]]:
    site_dir = Path(artifact.site_dir)
    entry_html_parent = Path(artifact.entry_html).parent
    manifest: list[dict[str, str]] = []
    for rel_path in artifact.copied_assets:
        asset_path = site_dir / rel_path
        if not asset_path.exists():
            continue
        try:
            web_path = asset_path.relative_to(entry_html_parent).as_posix()
        except ValueError:
            web_path = Path(
                Path(*asset_path.parts[len(entry_html_parent.parts) :]).as_posix()
            ).as_posix()
        if not web_path.startswith((".", "/")):
            web_path = f"./{web_path}"
        manifest.append({"web_path": web_path})
    return manifest


def run_coder_code_critic(artifact: CoderArtifact | None, page_plan: PagePlan | None) -> list[str]:
    critiques: list[str] = []
    if not artifact:
        critiques.append("Coder output is empty or failed schema validation.")
        return critiques
    if not page_plan:
        critiques.append("Page plan is missing, so anchored revision structure cannot be verified.")
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

    try:
        soup = BeautifulSoup(html_text, "html.parser")
    except Exception as exc:
        critiques.append(f"Cannot parse entry html: {exc}")
        return critiques

    body_tag = soup.body
    if body_tag is None:
        critiques.append("Entry html does not contain a <body> element.")

    if "PaperAlchemy Generated Body Start" not in html_text or "PaperAlchemy Generated Body End" not in html_text:
        critiques.append("Generated body markers are missing in entry html.")

    body_start_pattern = re.compile(
        r"<body[^>]*>\s*<!--\s*PaperAlchemy Generated Body Start\s*-->",
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not body_start_pattern.search(html_text):
        critiques.append("Generated body marker is not at body start; template content leakage is likely.")

    manifest_path = build_page_manifest_path(entry_html)
    manifest = load_page_manifest(manifest_path)
    if manifest is None:
        critiques.append(f"Anchored revision manifest is missing or invalid: {manifest_path}")
    else:
        template_profile = _load_template_profile_from_artifact(artifact)
        try:
            rebuilt_manifest = extract_page_manifest(
                html_text=html_text,
                entry_html=entry_html,
                selected_template_id=artifact.selected_template_id,
                page_plan=page_plan,
                require_expected_globals=str(manifest.schema_version or "").strip() != "1.0",
                template_profile=template_profile,
            )
            if manifest.model_dump() != rebuilt_manifest.model_dump():
                critiques.append("page_manifest.json is out of sync with current entry html anchors.")
        except Exception as exc:
            critiques.append(f"Anchored revision validation failed: {exc}")

        expected_block_order = [item.block_id for item in sorted(page_plan.page_outline, key=lambda item: item.order)]
        actual_block_order = [item.block_id for item in manifest.blocks]
        if actual_block_order != expected_block_order:
            critiques.append(
                "Generated data-pa-block order does not match approved page_outline order. "
                f"expected={expected_block_order}, actual={actual_block_order}"
            )

    critiques.extend(
        validate_local_image_references(
            html_text=html_text,
            entry_html_path=entry_html,
            site_dir=site_dir,
            allowed_asset_web_paths=collect_allowed_asset_web_paths(
                _available_asset_manifest_from_artifact(artifact)
            ),
            enforce_paper_asset_whitelist=True,
        )
    )

    title_count = len(re.findall(r"<title\b", html_text, flags=re.IGNORECASE))
    if title_count != 1:
        critiques.append(f"Expected exactly one <title> tag, found {title_count}.")

    referenced_values: list[str] = []
    for element in soup.find_all(True):
        for attr_name in ("src", "href", "data-src", "poster"):
            attr_value = element.get(attr_name)
            if isinstance(attr_value, str) and attr_value.strip():
                referenced_values.append(attr_value.strip().replace("\\", "/"))

    for rel_asset in artifact.copied_assets:
        asset_path = site_dir / rel_asset
        if not asset_path.exists():
            critiques.append(f"Copied asset missing: {asset_path}")
        rel_asset_norm = rel_asset.replace("\\", "/")
        if not any(rel_asset_norm in value for value in referenced_values):
            critiques.append(f"Copied asset is not referenced in entry html: {rel_asset}")

    return critiques


def coder_critic_node(state: CoderState) -> dict[str, Any]:
    print("[PaperAlchemy-CoderCritic] running build checks...")
    artifact = _normalize_coder_artifact(state.get("coder_artifact"))
    page_plan = _normalize_page_plan(state.get("page_plan"))
    critiques = run_coder_code_critic(artifact, page_plan)

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
