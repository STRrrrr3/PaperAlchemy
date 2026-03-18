from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from src.planner_template_catalog import find_entry_html_candidates
from src.template_resources import load_template_tags


FEATURE_WEIGHTS: dict[str, float] = {
    "background_color": 1.0,
    "has_hero_section": 0.75,
    "Page density": 0.85,
    "image_layout": 0.65,
    "title_color": 0.6,
    "has_navigation": 0.7,
}

FEATURE_KEY_ALIASES: dict[str, str] = {
    "background_color": "background_color",
    "background color": "background_color",
    "background": "background_color",
    "has_hero_section": "has_hero_section",
    "hero_section": "has_hero_section",
    "hero section": "has_hero_section",
    "hero": "has_hero_section",
    "page density": "Page density",
    "page_density": "Page density",
    "density": "Page density",
    "image_layout": "image_layout",
    "image layout": "image_layout",
    "layout": "image_layout",
    "title_color": "title_color",
    "title color": "title_color",
    "has_navigation": "has_navigation",
    "navigation": "has_navigation",
    "has navigation": "has_navigation",
    "nav": "has_navigation",
}

VALUE_ALIASES: dict[str, str] = {
    "true": "yes",
    "false": "no",
    "1": "yes",
    "0": "no",
}


def _normalize_constraint_value(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    return VALUE_ALIASES.get(normalized, normalized)


def normalize_user_constraints(user_constraints: Mapping[str, Any] | None) -> dict[str, str]:
    if not user_constraints:
        return {}

    normalized: dict[str, str] = {}
    for raw_key, raw_value in user_constraints.items():
        key = FEATURE_KEY_ALIASES.get(str(raw_key or "").strip().lower())
        value = _normalize_constraint_value(raw_value)
        if not key or not value or value in {"any", "auto", "default", "none"}:
            continue
        normalized[key] = value
    return normalized


def _build_reason_lines(
    matched_features: list[str],
    mismatched_features: list[str],
    score: float,
    max_possible_score: float,
) -> list[str]:
    reasons = [
        f"matched:{feature}"
        for feature in matched_features
    ]
    reasons.extend(f"mismatch:{item}" for item in mismatched_features[:3])
    reasons.append(f"score:{round(score, 4)}/{round(max_possible_score, 4)}")
    return reasons


def _rank_templates_from_tags(
    normalized_constraints: dict[str, str],
    tags_json_path: Path,
) -> list[dict[str, Any]]:
    template_tags = load_template_tags(tags_json_path)
    templates_root = tags_json_path.resolve().parent / "templates"

    ranked: list[dict[str, Any]] = []
    max_possible_score = round(
        sum(FEATURE_WEIGHTS.get(feature, 0.0) for feature in normalized_constraints),
        4,
    )

    for template_id, tag_map in template_tags.items():
        template_root = templates_root / template_id
        if not template_root.exists():
            continue

        entry_candidates = find_entry_html_candidates(template_root, max_candidates=3)
        if not entry_candidates:
            continue

        matched_features: list[str] = []
        mismatched_features: list[str] = []
        score = 0.0

        for feature, desired_value in normalized_constraints.items():
            actual_value = _normalize_constraint_value(tag_map.get(feature, ""))
            if actual_value == desired_value:
                matched_features.append(feature)
                score += FEATURE_WEIGHTS.get(feature, 0.0)
            else:
                mismatched_features.append(f"{feature}:{desired_value}!={actual_value or 'missing'}")

        score = round(score, 4)
        reasons = _build_reason_lines(
            matched_features=matched_features,
            mismatched_features=mismatched_features,
            score=score,
            max_possible_score=max_possible_score,
        )

        ranked.append(
            {
                "template_id": template_id,
                "template_path": str(template_root.resolve()),
                "entry_html": entry_candidates[0],
                "entry_html_candidates": entry_candidates,
                "score": score,
                "max_possible_score": max_possible_score,
                "matched_features": matched_features,
                "mismatched_features": mismatched_features,
                "template_tags": tag_map,
                "selection_source": "tags_json",
                "reasons": reasons,
            }
        )

    ranked.sort(
        key=lambda item: (
            -float(item.get("score", 0.0)),
            -len(item.get("matched_features") or []),
            len(item.get("mismatched_features") or []),
            str(item.get("template_id") or "").lower(),
        )
    )
    return ranked


def _fallback_rank_from_filesystem(tags_json_path: Path) -> list[dict[str, Any]]:
    templates_root = tags_json_path.resolve().parent / "templates"
    if not templates_root.exists():
        raise FileNotFoundError(f"Template directory not found: {templates_root}")

    ranked: list[dict[str, Any]] = []
    for template_root in sorted((p for p in templates_root.iterdir() if p.is_dir()), key=lambda p: p.name.lower()):
        entry_candidates = find_entry_html_candidates(template_root, max_candidates=3)
        if not entry_candidates:
            continue

        ranked.append(
            {
                "template_id": template_root.name,
                "template_path": str(template_root.resolve()),
                "entry_html": entry_candidates[0],
                "entry_html_candidates": entry_candidates,
                "score": 0.0,
                "max_possible_score": 0.0,
                "matched_features": [],
                "mismatched_features": [],
                "template_tags": {},
                "selection_source": "filesystem_fallback",
                "reasons": [
                    "fallback:tags_unavailable_or_invalid",
                    "score:0.0/0.0",
                ],
            }
        )

    if not ranked:
        raise FileNotFoundError(f"No usable templates found under: {templates_root}")

    return ranked


def score_and_select_template(
    user_constraints: Mapping[str, Any] | None,
    tags_json_path: str | Path,
) -> dict[str, Any]:
    path = Path(tags_json_path)
    normalized_constraints = normalize_user_constraints(user_constraints)

    try:
        ranked = _rank_templates_from_tags(normalized_constraints, path)
        if not ranked:
            ranked = _fallback_rank_from_filesystem(path)
            ranked[0]["reasons"].append("warning:empty_ranked_results")
    except Exception as exc:
        ranked = _fallback_rank_from_filesystem(path)
        ranked[0]["reasons"].append(f"warning:{type(exc).__name__}")

    if not ranked:
        raise ValueError("No ranked templates available for selection.")

    winner = dict(ranked[0])
    matched_count = len(winner.get("matched_features") or [])
    total_count = len(normalized_constraints)

    winner["selection_rationale"] = (
        f"Matched {matched_count}/{total_count} provided constraints; "
        f"selected '{winner['template_id']}' with score {winner['score']}."
    )
    winner["normalized_user_constraints"] = normalized_constraints
    winner["selected_template_id"] = winner["template_id"]
    winner["selected_template_path"] = winner["template_path"]
    winner["selected_entry_html"] = winner["entry_html"]
    winner["ranking"] = ranked
    return winner
