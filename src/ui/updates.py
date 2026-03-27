from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr

from src.contracts.schemas import LayoutComposeSession, ShellBindingReview
from src.services.preview_service import (
    build_binding_candidate_preview_path,
    build_binding_template_preview_path,
    take_local_screenshot,
    take_selector_screenshot,
)

def _review_accordion_updates(stage: str) -> tuple[dict[str, Any], dict[str, Any]]:
    normalized_stage = str(stage or "").strip().lower()
    if normalized_stage == "overview":
        return gr.update(open=True), gr.update(open=False)
    if normalized_stage == "outline":
        return gr.update(open=False), gr.update(open=True)
    return gr.update(open=False), gr.update(open=False)

def _format_shell_binding_review(review: ShellBindingReview) -> str:
    lines = [
        f"### Shell Binding Review: {review.block_title}",
        f"- `block_id`: `{review.block_id}`",
        f"- original selector: `{review.original_selector_hint or '(none)'}`",
        f"- reason: {review.failure_reason}",
        "",
        "Select one candidate shell below to continue generating the first draft.",
    ]
    return "\n".join(lines)

def _build_shell_binding_preview_assets(
    review: ShellBindingReview,
) -> tuple[str | None, list[tuple[str, str]]]:
    template_entry_path = Path(str(review.template_entry_html or "")).resolve()
    if not template_entry_path.exists():
        return None, []

    template_preview_path = take_local_screenshot(
        str(template_entry_path),
        str(build_binding_template_preview_path(template_entry_path, review.block_id)),
    )
    gallery_items: list[tuple[str, str]] = []
    for rank, candidate in enumerate(review.candidates, start=1):
        preview_path = take_selector_screenshot(
            str(template_entry_path),
            candidate.selector_hint,
            str(build_binding_candidate_preview_path(template_entry_path, review.block_id, rank)),
        )
        if not preview_path:
            continue
        caption = (
            f"{rank}. {candidate.selector_hint}\n"
            f"role={candidate.region_role} | score={candidate.score:.2f}\n"
            f"{candidate.reason}"
        )
        gallery_items.append((preview_path, caption))
    return template_preview_path or None, gallery_items

def _binding_ui_hidden() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    return (
        gr.update(value="", visible=False),
        gr.update(value=[], visible=False),
        gr.update(choices=[], value=None, interactive=False, visible=False),
        gr.update(interactive=False, visible=False),
        gr.update(interactive=False, visible=False),
    )

def _binding_ui_active(
    review: ShellBindingReview,
) -> tuple[str | None, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    template_preview_path, gallery_items = _build_shell_binding_preview_assets(review)
    return (
        template_preview_path,
        gr.update(value=_format_shell_binding_review(review), visible=True),
        gr.update(value=gallery_items, visible=bool(gallery_items)),
        gr.update(
            choices=[candidate.selector_hint for candidate in review.candidates],
            value=review.candidates[0].selector_hint if review.candidates else None,
            interactive=bool(review.candidates),
            visible=True,
        ),
        gr.update(interactive=bool(review.candidates), visible=True),
        gr.update(interactive=True, visible=True),
    )

def _visible_preview_update(value: str | None) -> dict[str, Any]:
    return gr.update(value=value, visible=True)

def _hidden_preview_update() -> dict[str, Any]:
    return gr.update(value=None, visible=False)

def _normalize_manual_layout_compose_enabled(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)

def _stage_action_updates(
    stage: str,
    *,
    feedback_text_value: str = "",
    feedback_images_value: Any = None,
    manual_layout_compose_enabled: bool = False,
) -> tuple[dict[str, Any], ...]:
    normalized_stage = str(stage or "").strip().lower()
    is_overview = normalized_stage == "overview"
    is_outline = normalized_stage == "outline"
    is_webpage = normalized_stage == "webpage"
    group_visible = is_overview or is_outline or is_webpage

    if is_overview:
        placeholder = "Describe missing or incorrect source material from the Reader extraction."
    elif is_outline:
        placeholder = "Describe section-level changes: add, remove, merge, split, rename, or reorder."
    elif is_webpage:
        placeholder = "Describe the visual issue or requested frontend change. Attach screenshots below if helpful."
    else:
        placeholder = ""

    return (
        gr.update(visible=group_visible),
        gr.update(
            value=str(feedback_text_value or ""),
            visible=group_visible,
            interactive=group_visible,
            placeholder=placeholder,
        ),
        gr.update(
            value=feedback_images_value if is_webpage else None,
            visible=is_webpage,
            interactive=is_webpage,
        ),
        gr.update(
            value=bool(manual_layout_compose_enabled),
            visible=is_outline,
            interactive=is_outline,
        ),
        gr.update(interactive=is_overview, visible=is_overview),
        gr.update(interactive=is_overview, visible=is_overview),
        gr.update(interactive=is_outline, visible=is_outline),
        gr.update(interactive=is_outline, visible=is_outline),
        gr.update(interactive=is_webpage, visible=is_webpage),
        gr.update(interactive=is_webpage, visible=is_webpage),
    )

def _ordered_layout_compose_blocks(session: LayoutComposeSession) -> list[Any]:
    return sorted(session.blocks, key=lambda item: (item.current_order, item.block_id))

def _active_layout_compose_block(session: LayoutComposeSession) -> Any | None:
    active_block_id = str(session.active_block_id or "").strip()
    for block in _ordered_layout_compose_blocks(session):
        if block.block_id == active_block_id:
            return block
    blocks = _ordered_layout_compose_blocks(session)
    return blocks[0] if blocks else None

def _format_layout_compose_block_summary(session: LayoutComposeSession) -> str:
    lines = [
        "### Layout Compose",
        "Review the suggested section bindings, reorder blocks, and choose block-local figures before generating the first draft.",
        "",
    ]
    for block in _ordered_layout_compose_blocks(session):
        selected_section = str(block.selected_selector_hint or "").strip() or "(unselected)"
        lines.extend(
            [
                f"#### {block.current_order}. {block.title}",
                f"- `block_id`: `{block.block_id}`",
                "- Source sections: "
                + (", ".join(block.source_sections) if block.source_sections else "(none)"),
                f"- Selected section: `{selected_section}`",
                f"- Selected images: {len(block.selected_figure_paths)}",
                "",
            ]
        )
    return "\n".join(lines).rstrip()

def _format_layout_compose_editor(block: Any | None) -> str:
    if block is None:
        return "### Layout Compose Editor\nNo block is available for editing."

    lines = [
        f"### Editing: {block.title}",
        f"- `block_id`: `{block.block_id}`",
        "- Source sections: " + (", ".join(block.source_sections) if block.source_sections else "(none)"),
        f"- Current order: {block.current_order}",
    ]
    if block.selected_selector_hint:
        lines.append(f"- Saved section: `{block.selected_selector_hint}`")
    else:
        lines.append("- Saved section: `(unselected)`")
    lines.append(f"- Saved image count: {len(block.selected_figure_paths)}")
    return "\n".join(lines)

def _format_layout_compose_validation(session: LayoutComposeSession) -> str:
    if not session.validation_errors:
        return (
            "### Compose Validation\n"
            "- All blocks have a valid section selection.\n"
            "- Section usage is unique and follows template DOM order.\n"
            "- `Continue To Draft` is enabled."
        )

    lines = ["### Compose Validation"]
    lines.extend(f"- {error}" for error in session.validation_errors)
    return "\n".join(lines)

def _layout_compose_section_caption(option: Any) -> str:
    overlay = f"[{option.overlay_label}] " if str(option.overlay_label or "").strip() else ""
    return (
        f"{overlay}{option.selector_hint}\n"
        f"role={option.region_role} | score={option.score:.2f}\n"
        f"{option.reason}"
    )

def _layout_compose_figure_caption(option: Any) -> str:
    section_prefix = f"{option.source_section} | " if str(option.source_section or "").strip() else ""
    figure_type = str(option.type or "").strip() or "figure"
    caption = str(option.caption or "").strip()
    if caption:
        return f"{section_prefix}{figure_type}\n{caption}"
    return f"{section_prefix}{figure_type}\n{option.image_path}"

def _layout_compose_ui_hidden() -> tuple[dict[str, Any], ...]:
    return (
        gr.update(value="", visible=False),
        gr.update(choices=[], value=None, interactive=False, visible=False),
        gr.update(value=None, visible=False),
        gr.update(value="", visible=False),
        gr.update(value=[], visible=False),
        gr.update(choices=[], value=None, interactive=False, visible=False),
        gr.update(value=[], visible=False),
        gr.update(choices=[], value=[], interactive=False, visible=False),
        gr.update(value="", visible=False),
        gr.update(interactive=False, visible=False),
        gr.update(interactive=False, visible=False),
        gr.update(interactive=False, visible=False),
        gr.update(interactive=False, visible=False),
        gr.update(interactive=False, visible=False),
    )

def _layout_compose_ui_active(session: LayoutComposeSession) -> tuple[dict[str, Any], ...]:
    active_block = _active_layout_compose_block(session)
    ordered_blocks = _ordered_layout_compose_blocks(session)

    block_choices = [
        (
            f"{block.current_order}. {block.title}",
            block.block_id,
        )
        for block in ordered_blocks
    ]

    section_gallery_items = []
    section_choices: list[tuple[str, str]] = []
    section_value: str | None = None
    figure_gallery_items = []
    figure_choices: list[tuple[str, str]] = []
    figure_values: list[str] = []
    move_up_interactive = False
    move_down_interactive = False

    if active_block is not None:
        section_choices = [
            (_layout_compose_section_caption(option), option.selector_hint)
            for option in active_block.section_options
        ]
        section_value = str(active_block.selected_selector_hint or "").strip() or None
        section_gallery_items = [
            (option.preview_image_path, _layout_compose_section_caption(option))
            for option in active_block.section_options[:6]
            if str(option.preview_image_path or "").strip()
        ]
        figure_choices = [
            (_layout_compose_figure_caption(option), option.image_path)
            for option in active_block.figure_options
        ]
        figure_values = list(active_block.selected_figure_paths)
        figure_gallery_items = [
            (option.preview_image_path, _layout_compose_figure_caption(option))
            for option in active_block.figure_options
            if str(option.preview_image_path or "").strip()
        ]
        move_up_interactive = active_block.current_order > 1
        move_down_interactive = active_block.current_order < len(ordered_blocks)

    return (
        gr.update(value=_format_layout_compose_block_summary(session), visible=True),
        gr.update(
            choices=block_choices,
            value=active_block.block_id if active_block is not None else None,
            interactive=bool(block_choices),
            visible=True,
        ),
        gr.update(
            value=session.template_preview_path or None,
            visible=bool(session.template_preview_path),
        ),
        gr.update(value=_format_layout_compose_editor(active_block), visible=True),
        gr.update(value=section_gallery_items, visible=bool(section_gallery_items)),
        gr.update(
            choices=section_choices,
            value=section_value,
            interactive=bool(section_choices),
            visible=True,
        ),
        gr.update(value=figure_gallery_items, visible=bool(figure_gallery_items)),
        gr.update(
            choices=figure_choices,
            value=figure_values,
            interactive=True,
            visible=True,
        ),
        gr.update(value=_format_layout_compose_validation(session), visible=True),
        gr.update(interactive=move_up_interactive, visible=True),
        gr.update(interactive=move_down_interactive, visible=True),
        gr.update(interactive=active_block is not None, visible=True),
        gr.update(interactive=not session.validation_errors and active_block is not None, visible=True),
        gr.update(interactive=True, visible=True),
    )
