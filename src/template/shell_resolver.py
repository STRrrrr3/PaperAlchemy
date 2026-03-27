from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from bs4 import BeautifulSoup, Tag

from src.services.preview_service import (
    build_layout_compose_section_preview_path,
    build_layout_compose_template_preview_path,
    build_paper_figure_preview_path,
    take_labeled_template_screenshot,
    take_local_screenshot,
    take_selector_screenshot,
)
from src.validators.page_manifest import (
    _SHELL_CONTAINER_TAGS,
    _capture_wrapper_chain,
    _tag_classes,
    _tag_ids,
    _tag_tokens,
)
from src.contracts.schemas import (
    BlockPlan,
    BlockShellContract,
    LayoutComposeBlock,
    LayoutComposeSession,
    LayoutComposeUpdate,
    LayoutFigureOption,
    LayoutSectionOption,
    PagePlan,
    ShellBindingReview,
    ShellResolutionCandidate,
    StructuredPaper,
    TemplateProfile,
)

AUTO_ACCEPT_MIN_SCORE = 8.0
AUTO_ACCEPT_MARGIN = 1.5
MAX_REVIEW_CANDIDATES = 5
MAX_COMPOSE_SECTION_GALLERY = 6


@dataclass
class _CandidateRoot:
    tag: Tag
    selector_hint: str
    dom_index: int
    has_media: bool
    has_heading: bool
    stable_score: float


def _selector_tokens(selector_hint: str) -> set[str]:
    return {
        token
        for token in re.split(r"[^a-z0-9]+", str(selector_hint or "").lower())
        if token
    }


def _is_meaningful_candidate(tag: Tag) -> bool:
    tag_name = str(tag.name or "")
    if tag_name not in _SHELL_CONTAINER_TAGS:
        return False
    classes = _tag_classes(tag)
    ids = _tag_ids(tag)
    if ids or classes:
        return True
    return any(isinstance(child, Tag) for child in tag.children)


def _selector_segment(tag: Tag) -> str:
    tag_name = str(tag.name or "div")
    tag_id = str(tag.get("id") or "").strip()
    if tag_id:
        return f"{tag_name}#{tag_id}"

    classes = [name for name in _tag_classes(tag)[:2] if name]
    if classes:
        return tag_name + "".join(f".{name}" for name in classes)

    siblings = [
        sibling
        for sibling in tag.find_previous_siblings(tag_name)
        if isinstance(sibling, Tag)
    ]
    if siblings:
        return f"{tag_name}:nth-of-type({len(siblings) + 1})"
    return tag_name


def _build_unique_selector(tag: Tag, soup: BeautifulSoup) -> str:
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


def _build_shell_contract_from_root(
    block: BlockPlan,
    root: Tag,
    selector_hint: str,
) -> BlockShellContract:
    return BlockShellContract(
        root_tag=str(root.name or "div"),
        required_classes=_tag_classes(root),
        preserve_ids=_tag_ids(root),
        wrapper_chain=_capture_wrapper_chain(root),
        actionable_root_selector=selector_hint,
    )


def _dom_index_for_tag(template_soup: BeautifulSoup, target: Tag) -> int:
    root = template_soup.body or template_soup
    for dom_index, tag in enumerate(root.find_all(True)):
        if tag is target:
            return dom_index
    return 10_000


def _iter_candidate_roots(template_soup: BeautifulSoup) -> list[_CandidateRoot]:
    root = template_soup.body or template_soup
    candidates: list[_CandidateRoot] = []
    seen: set[str] = set()

    for dom_index, tag in enumerate(root.find_all(True)):
        if not _is_meaningful_candidate(tag):
            continue
        selector_hint = _build_unique_selector(tag, template_soup)
        if not selector_hint or selector_hint in seen:
            continue
        seen.add(selector_hint)
        selector_tokens = _selector_tokens(selector_hint)
        candidates.append(
            _CandidateRoot(
                tag=tag,
                selector_hint=selector_hint,
                dom_index=dom_index,
                has_media=tag.find(["img", "video", "figure", "canvas", "svg", "table"]) is not None,
                has_heading=tag.find(re.compile(r"^h[1-6]$")) is not None,
                stable_score=(
                    2.0
                    if "#" in selector_hint
                    else 1.0 if "." in selector_hint else 0.0
                )
                + (0.5 if ":nth-of-type" not in selector_hint and selector_tokens else 0.0),
            )
        )
    return candidates


def _profile_candidate_roots(
    template_soup: BeautifulSoup,
    template_profile: TemplateProfile,
) -> list[_CandidateRoot]:
    roots: list[_CandidateRoot] = []
    seen: set[str] = set()
    for candidate in template_profile.shell_candidates:
        selector_hint = str(candidate.selector or "").strip()
        if not selector_hint or selector_hint in seen:
            continue
        try:
            matches = [match for match in template_soup.select(selector_hint) if isinstance(match, Tag)]
        except Exception:
            matches = []
        if len(matches) != 1:
            continue
        tag = matches[0]
        seen.add(selector_hint)
        roots.append(
            _CandidateRoot(
                tag=tag,
                selector_hint=selector_hint,
                dom_index=int(candidate.dom_index),
                has_media=tag.find(["img", "video", "figure", "canvas", "svg", "table"]) is not None,
                has_heading=tag.find(re.compile(r"^h[1-6]$")) is not None,
                stable_score=max(0.0, float(candidate.confidence or 0.0) * 3.0),
            )
        )
    roots.sort(key=lambda item: (item.dom_index, item.selector_hint))
    return roots


def _candidate_role_for_block(block_role: str, candidate: _CandidateRoot) -> tuple[str, float] | None:
    tag = candidate.tag
    tag_name = str(tag.name or "")
    tokens = _tag_tokens(tag)
    inferred_role = "section"
    if tag_name == "footer" or "footer" in tokens:
        inferred_role = "footer"
    elif tag_name == "nav" or tokens & {"nav", "navbar", "menu"}:
        inferred_role = "nav"
    elif tag_name == "header" or tokens & {"hero", "lead", "intro", "banner", "masthead"}:
        inferred_role = "hero"
    elif tag.find("table") is not None or tokens & {"table", "metric", "metrics", "results", "benchmark"}:
        inferred_role = "table"
    elif tag.find(["img", "video", "figure", "canvas", "svg"]) is not None or tokens & {
        "gallery",
        "carousel",
        "media",
        "video",
        "image",
        "figure",
    }:
        inferred_role = "gallery"

    if block_role == "hero":
        if inferred_role == "hero":
            return "hero", 4.0
        if tag_name == "header":
            return "hero", 3.0
        if inferred_role == "section":
            return "section", 2.0
        return None

    if block_role == "gallery":
        if inferred_role == "gallery":
            return "gallery", 4.0
        if inferred_role == "section":
            return "section", 2.0
        return None

    if block_role == "table":
        if inferred_role == "table":
            return "table", 4.0
        if inferred_role == "section":
            return "section", 2.0
        return None

    if block_role == "section":
        if tag_name in {"section", "article", "div", "main"} or inferred_role == "section":
            return "section", 4.0
        return None

    if block_role == "nav":
        if inferred_role == "nav":
            return "nav", 4.0
        if tag_name == "header":
            return "nav", 2.5
        return None

    if block_role == "footer":
        if inferred_role == "footer" or tag_name == "footer":
            return "footer", 4.0
        return None

    return None


def _content_fit_score(block: BlockPlan, candidate: _CandidateRoot) -> float:
    score = 0.0
    has_figures = bool(block.asset_binding.figure_paths)
    interaction_pattern = str(block.interaction.pattern or "")
    if has_figures and candidate.has_media:
        score += 2.0
    elif has_figures and not candidate.has_media:
        score -= 1.0
    elif not has_figures and not candidate.has_media:
        score += 0.5

    if interaction_pattern in {"carousel", "comparison-slider", "hover-detail"} and candidate.has_media:
        score += 1.0
    if str(block.target_template_region.region_role or "") in {"hero", "section"} and candidate.has_heading:
        score += 0.5
    return score


def _order_score(candidate: _CandidateRoot, last_dom_index: int) -> float:
    if candidate.dom_index < last_dom_index:
        return -6.0
    gap = candidate.dom_index - max(last_dom_index, 0)
    return max(0.0, 5.0 - min(float(gap), 10.0) * 0.5)


def _selector_similarity_score(block: BlockPlan, candidate: _CandidateRoot) -> float:
    original_tokens = _selector_tokens(block.target_template_region.selector_hint)
    candidate_tokens = _selector_tokens(candidate.selector_hint)
    if not original_tokens or not candidate_tokens:
        return 0.0
    return min(3.0, float(len(original_tokens & candidate_tokens)))


def _describe_reason(parts: list[str]) -> str:
    meaningful = [part.strip() for part in parts if part.strip()]
    return "; ".join(meaningful[:4]) or "closest matching template shell"


def _score_candidates_for_block(
    block: BlockPlan,
    candidates: list[_CandidateRoot],
    used_selectors: set[str],
    last_dom_index: int,
) -> list[tuple[_CandidateRoot, str, float, str]]:
    scored: list[tuple[_CandidateRoot, str, float, str]] = []
    for candidate in candidates:
        if candidate.selector_hint in used_selectors:
            continue
        role_result = _candidate_role_for_block(str(block.target_template_region.region_role or ""), candidate)
        if role_result is None:
            continue
        compatible_role, role_score = role_result
        order_score = _order_score(candidate, last_dom_index)
        similarity_score = _selector_similarity_score(block, candidate)
        content_score = _content_fit_score(block, candidate)
        stability_score = candidate.stable_score
        total = role_score + order_score + similarity_score + content_score + stability_score
        reasons: list[str] = [f"order={order_score:.1f}", f"role={compatible_role}"]
        if similarity_score > 0:
            reasons.append(f"selector_overlap={similarity_score:.1f}")
        if content_score > 0:
            reasons.append(f"content_fit={content_score:.1f}")
        if candidate.stable_score > 0:
            reasons.append("stable_selector")
        scored.append((candidate, compatible_role, total, _describe_reason(reasons)))
    scored.sort(key=lambda item: (item[2], -item[0].dom_index), reverse=True)
    return scored


def _resolve_root_for_selector(block: BlockPlan, template_soup: BeautifulSoup) -> tuple[Tag, str] | None:
    try:
        matches = [match for match in template_soup.select(str(block.target_template_region.selector_hint or "")) if isinstance(match, Tag)]
    except Exception:
        matches = []
    if not matches:
        return None

    candidate_roots: list[tuple[int, Tag]] = []
    for match in matches:
        candidate_roots.append((0, match))
        for depth, ancestor in enumerate(match.parents, start=1):
            if not isinstance(ancestor, Tag):
                continue
            if str(ancestor.name or "") in {"html", "body"}:
                break
            if not _is_meaningful_candidate(ancestor):
                continue
            candidate_roots.append((depth, ancestor))
            if depth >= 4:
                break

    if not candidate_roots:
        return None

    block_role = str(block.target_template_region.region_role or "")
    scored: list[tuple[float, int, Tag]] = []
    for depth, root in candidate_roots:
        candidate = _CandidateRoot(
            tag=root,
            selector_hint=_build_unique_selector(root, template_soup),
            dom_index=0,
            has_media=root.find(["img", "video", "figure", "canvas", "svg", "table"]) is not None,
            has_heading=root.find(re.compile(r"^h[1-6]$")) is not None,
            stable_score=0.0,
        )
        role_result = _candidate_role_for_block(block_role, candidate)
        if role_result is None:
            continue
        role_score = role_result[1]
        scored.append((role_score - float(depth), -depth, root))

    if not scored:
        return None

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    root = scored[0][2]
    return root, _build_unique_selector(root, template_soup)


def _apply_selector_hint(
    block: BlockPlan,
    selector_hint: str,
    template_soup: BeautifulSoup,
    *,
    rewrite_selector_hint: bool,
) -> tuple[BlockPlan, int] | None:
    updated_region = block.target_template_region.model_copy(
        update={"selector_hint": selector_hint},
        deep=True,
    )
    updated_block = block.model_copy(update={"target_template_region": updated_region}, deep=True)
    resolved = _resolve_root_for_selector(updated_block, template_soup)
    if resolved is None:
        return None
    root, canonical_selector = resolved
    stored_selector = canonical_selector if rewrite_selector_hint else str(selector_hint or "").strip()
    canonical_region = updated_region.model_copy(update={"selector_hint": stored_selector}, deep=True)
    updated_block = updated_block.model_copy(
        update={
            "target_template_region": canonical_region,
            "shell_contract": _build_shell_contract_from_root(updated_block, root, canonical_selector),
        },
        deep=True,
    )
    return updated_block, _dom_index_for_tag(template_soup, root)


def resolve_page_plan_shells(
    page_plan: PagePlan,
    template_reference_html: str,
    template_entry_html_path: str | Path,
    template_profile: TemplateProfile | None = None,
) -> tuple[PagePlan, ShellBindingReview | None]:
    template_soup = BeautifulSoup(str(template_reference_html or ""), "html.parser")
    candidate_roots = (
        _profile_candidate_roots(template_soup, template_profile)
        if template_profile is not None
        else _iter_candidate_roots(template_soup)
    )
    outline_lookup = {item.block_id: item for item in page_plan.page_outline}
    block_lookup = {block.block_id: block for block in page_plan.blocks}

    used_selectors: set[str] = set()
    last_dom_index = -1
    resolved_blocks: dict[str, BlockPlan] = {}

    for outline_item in sorted(page_plan.page_outline, key=lambda item: item.order):
        block = block_lookup.get(outline_item.block_id)
        if block is None:
            continue

        current_result = _apply_selector_hint(
            block,
            str(block.target_template_region.selector_hint or "").strip(),
            template_soup,
            rewrite_selector_hint=False,
        )
        if current_result is not None:
            updated_block, current_dom_index = current_result
            current_selector = str(
                updated_block.shell_contract.actionable_root_selector
                if updated_block.shell_contract is not None
                else updated_block.target_template_region.selector_hint
                or ""
            ).strip()
            if current_selector not in used_selectors:
                used_selectors.add(current_selector)
                last_dom_index = max(last_dom_index, current_dom_index)
                resolved_blocks[updated_block.block_id] = updated_block
                continue

        scored_candidates = _score_candidates_for_block(
            block=block,
            candidates=candidate_roots,
            used_selectors=used_selectors,
            last_dom_index=last_dom_index,
        )

        if scored_candidates:
            top_candidate, _, top_score, _ = scored_candidates[0]
            second_score = scored_candidates[1][2] if len(scored_candidates) > 1 else float("-inf")
            if top_score >= AUTO_ACCEPT_MIN_SCORE and (top_score - second_score) >= AUTO_ACCEPT_MARGIN:
                rebound = _apply_selector_hint(
                    block,
                    top_candidate.selector_hint,
                    template_soup,
                    rewrite_selector_hint=True,
                )
                if rebound is not None:
                    updated_block, rebound_dom_index = rebound
                    selector_hint = str(
                        updated_block.shell_contract.actionable_root_selector
                        if updated_block.shell_contract is not None
                        else updated_block.target_template_region.selector_hint
                        or ""
                    ).strip()
                    used_selectors.add(selector_hint)
                    last_dom_index = max(last_dom_index, rebound_dom_index)
                    resolved_blocks[updated_block.block_id] = updated_block
                    continue

        review_candidates = [
            ShellResolutionCandidate(
                selector_hint=candidate.selector_hint,
                region_role=compatible_role,  # type: ignore[arg-type]
                score=round(score, 2),
                reason=reason,
            )
            for candidate, compatible_role, score, reason in scored_candidates[:MAX_REVIEW_CANDIDATES]
        ]
        failure_reason = (
            "No compatible shell candidates were found in the selected template."
            if not review_candidates
            else "Automatic shell rebinding was too ambiguous and requires human review."
        )
        updated_blocks: list[BlockPlan] = []
        for candidate_block in page_plan.blocks:
            updated_blocks.append(resolved_blocks.get(candidate_block.block_id, candidate_block))
        updated_plan = page_plan.model_copy(update={"blocks": updated_blocks}, deep=True)
        return (
            updated_plan,
            ShellBindingReview(
                block_id=block.block_id,
                block_title=str(outline_lookup.get(block.block_id).title if outline_lookup.get(block.block_id) else block.block_id),
                original_selector_hint=str(block.target_template_region.selector_hint or "").strip(),
                failure_reason=failure_reason,
                template_entry_html=str(template_entry_html_path),
                candidates=review_candidates,
            ),
        )

    updated_blocks = [resolved_blocks.get(block.block_id, block) for block in page_plan.blocks]
    return page_plan.model_copy(update={"blocks": updated_blocks}, deep=True), None


def _sorted_compose_blocks(session: LayoutComposeSession) -> list[LayoutComposeBlock]:
    return sorted(session.blocks, key=lambda item: (item.current_order, item.block_id))


def _selector_label_map(options: list[LayoutSectionOption]) -> dict[str, LayoutSectionOption]:
    return {option.selector_hint: option for option in options}


def _suggest_block_selector(
    block: BlockPlan,
    candidate_roots: list[_CandidateRoot],
    template_soup: BeautifulSoup,
    used_selectors: set[str],
    last_dom_index: int,
) -> tuple[str, int]:
    current_result = _apply_selector_hint(
        block,
        str(block.target_template_region.selector_hint or "").strip(),
        template_soup,
        rewrite_selector_hint=True,
    )
    if current_result is not None:
        updated_block, current_dom_index = current_result
        current_selector = str(updated_block.target_template_region.selector_hint or "").strip()
        if current_selector and current_selector not in used_selectors and current_dom_index > last_dom_index:
            return current_selector, current_dom_index

    scored_candidates = _score_candidates_for_block(
        block=block,
        candidates=candidate_roots,
        used_selectors=used_selectors,
        last_dom_index=last_dom_index,
    )
    if not scored_candidates:
        return "", last_dom_index
    top_candidate = scored_candidates[0][0]
    return top_candidate.selector_hint, top_candidate.dom_index


def _build_figure_options_for_block(
    source_sections: list[str],
    structured_paper: StructuredPaper,
    paper_folder_name: str,
) -> list[LayoutFigureOption]:
    wanted_sections = {section_title.strip() for section_title in source_sections if section_title.strip()}
    seen_paths: set[str] = set()
    figure_options: list[LayoutFigureOption] = []

    for section in structured_paper.sections:
        section_title = str(section.section_title or "").strip()
        if section_title not in wanted_sections:
            continue
        for figure in section.related_figures:
            image_path = str(figure.image_path or "").strip().replace("\\", "/")
            if not image_path or image_path in seen_paths:
                continue
            seen_paths.add(image_path)
            figure_options.append(
                LayoutFigureOption(
                    image_path=image_path,
                    caption=str(figure.caption or "").strip() or None,
                    type=str(figure.type or "").strip(),
                    source_section=section_title,
                    preview_image_path=build_paper_figure_preview_path(paper_folder_name, image_path),
                )
            )
    return figure_options


def validate_layout_compose_session(session: LayoutComposeSession) -> list[str]:
    errors: list[str] = []
    ordered_blocks = _sorted_compose_blocks(session)
    selected_by_selector: dict[str, list[str]] = {}
    last_dom_index: int | None = None

    for block in ordered_blocks:
        selected_selector = str(block.selected_selector_hint or "").strip()
        options_by_selector = _selector_label_map(block.section_options)
        if not selected_selector:
            errors.append(f"Block '{block.title}' must select exactly one template section.")
            continue

        selected_option = options_by_selector.get(selected_selector)
        if selected_option is None:
            errors.append(
                f"Block '{block.title}' selected an unavailable section `{selected_selector}`."
            )
            continue

        selected_by_selector.setdefault(selected_selector, []).append(block.title)
        if last_dom_index is not None and selected_option.dom_index <= last_dom_index:
            errors.append(
                "Block order must follow template DOM order: "
                f"'{block.title}' is mapped to `{selected_selector}` at dom_index={selected_option.dom_index}."
            )
        last_dom_index = selected_option.dom_index

    for selector_hint, block_titles in selected_by_selector.items():
        if len(block_titles) <= 1:
            continue
        errors.append(
            f"Template section `{selector_hint}` is assigned more than once: {', '.join(block_titles)}."
        )

    return errors


def apply_layout_compose_update(
    session: LayoutComposeSession,
    update: LayoutComposeUpdate,
) -> LayoutComposeSession:
    blocks = [block.model_copy(deep=True) for block in session.blocks]
    target_block_id = str(update.active_block_id or session.active_block_id or "").strip()

    if target_block_id:
        for index, block in enumerate(blocks):
            if block.block_id != target_block_id:
                continue

            if update.selected_selector_hint is not None:
                block.selected_selector_hint = str(update.selected_selector_hint or "").strip()

            if update.selected_figure_paths is not None:
                valid_figure_paths = {
                    option.image_path for option in block.figure_options if option.image_path
                }
                block.selected_figure_paths = [
                    image_path
                    for image_path in update.selected_figure_paths
                    if image_path in valid_figure_paths
                ]

            if update.order_action == "move_up" and index > 0:
                blocks[index - 1].current_order, block.current_order = (
                    block.current_order,
                    blocks[index - 1].current_order,
                )
            elif update.order_action == "move_down" and index < len(blocks) - 1:
                blocks[index + 1].current_order, block.current_order = (
                    block.current_order,
                    blocks[index + 1].current_order,
                )
            break

    ordered_blocks = sorted(blocks, key=lambda item: (item.current_order, item.block_id))
    for order, block in enumerate(ordered_blocks, start=1):
        block.current_order = order

    updated_session = session.model_copy(
        update={
            "blocks": ordered_blocks,
            "active_block_id": target_block_id or session.active_block_id,
        },
        deep=True,
    )
    validation_errors = validate_layout_compose_session(updated_session)
    return updated_session.model_copy(update={"validation_errors": validation_errors}, deep=True)


def build_layout_compose_session(
    page_plan: PagePlan,
    structured_paper: StructuredPaper,
    template_reference_html: str,
    template_entry_html_path: str | Path,
    template_profile: TemplateProfile | None = None,
    *,
    paper_folder_name: str = "",
) -> LayoutComposeSession:
    template_entry_path = Path(template_entry_html_path).resolve()
    template_soup = BeautifulSoup(str(template_reference_html or ""), "html.parser")
    candidate_roots = (
        _profile_candidate_roots(template_soup, template_profile)
        if template_profile is not None
        else _iter_candidate_roots(template_soup)
    )
    outline_lookup = {item.block_id: item for item in page_plan.page_outline}
    block_lookup = {block.block_id: block for block in page_plan.blocks}

    compatible_candidates_by_block: dict[str, list[tuple[_CandidateRoot, str, float, str]]] = {}
    selector_overlay_lookup: dict[str, str] = {}
    compatible_roots_by_selector: dict[str, _CandidateRoot] = {}

    for outline_item in sorted(page_plan.page_outline, key=lambda item: item.order):
        block = block_lookup.get(outline_item.block_id)
        if block is None:
            continue
        compatible_candidates = _score_candidates_for_block(
            block=block,
            candidates=candidate_roots,
            used_selectors=set(),
            last_dom_index=-1,
        )
        compatible_candidates_by_block[block.block_id] = compatible_candidates
        for candidate, _, _, _ in compatible_candidates:
            compatible_roots_by_selector.setdefault(candidate.selector_hint, candidate)

    for overlay_index, candidate in enumerate(
        sorted(compatible_roots_by_selector.values(), key=lambda item: (item.dom_index, item.selector_hint)),
        start=1,
    ):
        selector_overlay_lookup[candidate.selector_hint] = str(overlay_index)

    template_preview_path = ""
    if template_entry_path.exists():
        selector_labels = [
            {"selector": selector_hint, "label": selector_overlay_lookup[selector_hint]}
            for selector_hint in sorted(
                selector_overlay_lookup,
                key=lambda item: (
                    compatible_roots_by_selector[item].dom_index,
                    item,
                ),
            )
        ]
        template_preview_path = take_labeled_template_screenshot(
            str(template_entry_path),
            selector_labels,
            str(build_layout_compose_template_preview_path(template_entry_path)),
        )
        if not template_preview_path:
            template_preview_path = take_local_screenshot(
                str(template_entry_path),
                str(build_layout_compose_template_preview_path(template_entry_path)),
            )

    used_selectors: set[str] = set()
    last_dom_index = -1
    compose_blocks: list[LayoutComposeBlock] = []

    for outline_item in sorted(page_plan.page_outline, key=lambda item: item.order):
        block = block_lookup.get(outline_item.block_id)
        if block is None:
            continue

        compatible_candidates = compatible_candidates_by_block.get(block.block_id, [])
        section_options: list[LayoutSectionOption] = []
        for rank, (candidate, compatible_role, score, reason) in enumerate(compatible_candidates, start=1):
            preview_image_path = ""
            if rank <= MAX_COMPOSE_SECTION_GALLERY and template_entry_path.exists():
                preview_image_path = take_selector_screenshot(
                    str(template_entry_path),
                    candidate.selector_hint,
                    str(build_layout_compose_section_preview_path(template_entry_path, candidate.selector_hint)),
                )
            section_options.append(
                LayoutSectionOption(
                    selector_hint=candidate.selector_hint,
                    region_role=compatible_role,  # type: ignore[arg-type]
                    dom_index=candidate.dom_index,
                    score=round(score, 2),
                    reason=reason,
                    preview_image_path=preview_image_path,
                    overlay_label=selector_overlay_lookup.get(candidate.selector_hint, ""),
                )
            )

        selected_selector_hint, selected_dom_index = _suggest_block_selector(
            block,
            candidate_roots,
            template_soup,
            used_selectors,
            last_dom_index,
        )
        if selected_selector_hint:
            used_selectors.add(selected_selector_hint)
            last_dom_index = max(last_dom_index, selected_dom_index)

        figure_options = _build_figure_options_for_block(
            list(outline_item.source_sections),
            structured_paper,
            paper_folder_name,
        )
        valid_figure_paths = {option.image_path for option in figure_options}
        selected_figure_paths = [
            image_path
            for image_path in block.asset_binding.figure_paths
            if image_path in valid_figure_paths
        ]

        compose_blocks.append(
            LayoutComposeBlock(
                block_id=block.block_id,
                title=str(outline_item.title or block.block_id),
                source_sections=list(outline_item.source_sections),
                current_order=int(outline_item.order),
                selected_selector_hint=selected_selector_hint,
                selected_figure_paths=selected_figure_paths,
                section_options=section_options,
                figure_options=figure_options,
            )
        )

    active_block_id = compose_blocks[0].block_id if compose_blocks else None
    session = LayoutComposeSession(
        template_entry_html=str(template_entry_path),
        template_preview_path=template_preview_path or "",
        blocks=compose_blocks,
        active_block_id=active_block_id,
        validation_errors=[],
    )
    validation_errors = validate_layout_compose_session(session)
    return session.model_copy(update={"validation_errors": validation_errors}, deep=True)


def apply_layout_compose_session_to_page_plan(
    page_plan: PagePlan,
    session: LayoutComposeSession,
    template_reference_html: str,
) -> PagePlan:
    validation_errors = validate_layout_compose_session(session)
    if validation_errors:
        raise ValueError("; ".join(validation_errors))

    template_soup = BeautifulSoup(str(template_reference_html or ""), "html.parser")
    outline_lookup = {item.block_id: item for item in page_plan.page_outline}
    block_lookup = {block.block_id: block for block in page_plan.blocks}

    updated_outline = []
    updated_blocks = []
    for order, compose_block in enumerate(_sorted_compose_blocks(session), start=1):
        outline_item = outline_lookup.get(compose_block.block_id)
        block = block_lookup.get(compose_block.block_id)
        if outline_item is None or block is None:
            raise ValueError(f"Layout compose referenced unknown block '{compose_block.block_id}'.")

        resolved = _apply_selector_hint(
            block,
            compose_block.selected_selector_hint,
            template_soup,
            rewrite_selector_hint=True,
        )
        if resolved is None:
            raise ValueError(
                f"Selected template section `{compose_block.selected_selector_hint}` could not be resolved for block "
                f"'{compose_block.block_id}'."
            )

        resolved_block, _ = resolved
        updated_outline.append(
            outline_item.model_copy(update={"order": order}, deep=True)
        )
        updated_blocks.append(
            resolved_block.model_copy(
                update={
                    "asset_binding": resolved_block.asset_binding.model_copy(
                        update={"figure_paths": list(compose_block.selected_figure_paths)},
                        deep=True,
                    ),
                    "responsive_rules": resolved_block.responsive_rules.model_copy(
                        update={"mobile_order": order},
                        deep=True,
                    ),
                },
                deep=True,
            )
        )

    return page_plan.model_copy(
        update={
            "page_outline": updated_outline,
            "blocks": updated_blocks,
        },
        deep=True,
    )

