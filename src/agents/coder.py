import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup, Tag
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from src.agents.coder_critic import (
    build_coder_critic_router,
    build_vision_qa_router,
    coder_critic_node,
    take_screenshot_action,
    vision_critic_node,
)
from src.utils.html_utils import (
    extract_html_document,
    extract_html_fragment,
    message_content_to_text,
    normalize_html_document_whitespace,
    read_current_page_html,
    read_text_with_fallback,
)
from src.services.human_feedback import extract_human_feedback_text, normalize_human_feedback
from src.utils.json_utils import to_pretty_json
from src.services.llm import get_llm
from src.validators.page_manifest import (
    annotate_global_anchors,
    build_expected_global_anchors,
    build_page_manifest_path,
    enrich_page_plan_shell_contracts,
    extract_page_manifest,
    missing_shell_contract_block_ids,
    save_page_manifest,
)
from src.validators.page_validation import (
    collect_allowed_asset_web_paths,
    validate_fragment_local_image_sources,
    validate_local_image_references,
)
from src.prompts import (
    BLOCK_RENDER_SYSTEM_PROMPT,
    BLOCK_RENDER_USER_PROMPT_TEMPLATE,
    CODER_SYSTEM_PROMPT,
    CODER_USER_PROMPT_TEMPLATE,
)
from src.contracts.schemas import (
    BlockRenderArtifact,
    BlockRenderSpec,
    BlockShellContract,
    CoderArtifact,
    PagePlan,
    ResolvedBlockBinding,
    StructuredPaper,
    TemplateProfile,
    VisualSmokeReport,
)
from src.contracts.state import CoderState


def _normalize_page_plan(plan: Any) -> PagePlan | None:
    if isinstance(plan, PagePlan):
        return plan
    if plan is None:
        return None
    try:
        return PagePlan.model_validate(plan)
    except Exception:
        return None


def _normalize_structured_paper(paper: Any) -> StructuredPaper | None:
    if isinstance(paper, StructuredPaper):
        return paper
    if paper is None:
        return None
    try:
        return StructuredPaper.model_validate(paper)
    except Exception:
        return None


def _normalize_coder_artifact(artifact: Any) -> CoderArtifact | None:
    if isinstance(artifact, CoderArtifact):
        return artifact
    if artifact is None:
        return None
    try:
        return CoderArtifact.model_validate(artifact)
    except Exception:
        return None


def _normalize_visual_smoke_report(report: Any) -> VisualSmokeReport | None:
    if isinstance(report, VisualSmokeReport):
        return report
    if report is None:
        return None
    try:
        return VisualSmokeReport.model_validate(report)
    except Exception:
        return None


def _normalize_template_profile(profile: Any) -> TemplateProfile | None:
    if isinstance(profile, TemplateProfile):
        return profile
    if profile is None:
        return None
    try:
        return TemplateProfile.model_validate(profile)
    except Exception:
        return None


def _safe_slug(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", text.strip())
    slug = slug.strip("-").lower()
    return slug or "asset"


def _to_html_relative_path(target_path: Path, base_dir: Path) -> str:
    rel_path = os.path.relpath(target_path, start=base_dir)
    web_path = str(rel_path).replace("\\", "/")
    if not web_path.startswith((".", "/")):
        web_path = f"./{web_path}"
    return web_path


def _ensure_body_markers(html_text: str) -> str:
    result = str(html_text or "")
    if "PaperAlchemy Generated Body Start" not in result:
        result = re.sub(
            r"(<body\b[^>]*>)",
            r"\1\n<!-- PaperAlchemy Generated Body Start -->",
            result,
            count=1,
            flags=re.IGNORECASE,
        )
    if "PaperAlchemy Generated Body End" not in result:
        result = re.sub(
            r"(</body>)",
            r"<!-- PaperAlchemy Generated Body End -->\n\1",
            result,
            count=1,
            flags=re.IGNORECASE,
        )
    return result


def _ensure_doctype(html_text: str) -> str:
    normalized = str(html_text or "").strip()
    if normalized and "<!doctype" not in normalized.lower():
        normalized = "<!DOCTYPE html>\n" + normalized
    return normalized


def _normalize_asset_key(value: str) -> str:
    return str(value or "").strip().replace("\\", "/")


def _collect_figure_paths(page_plan: PagePlan, structured_paper: StructuredPaper) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for block in page_plan.blocks:
        for path in block.asset_binding.figure_paths:
            clean = _normalize_asset_key(path)
            if clean and clean not in seen:
                seen.add(clean)
                ordered.append(clean)
    if ordered:
        return ordered

    for section in structured_paper.sections:
        for fig in section.related_figures:
            clean = _normalize_asset_key(fig.image_path)
            if clean and clean not in seen:
                seen.add(clean)
                ordered.append(clean)
    return ordered


def _build_asset_lookup(structured_paper: StructuredPaper) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    for section in structured_paper.sections:
        for fig in section.related_figures:
            key = _normalize_asset_key(fig.image_path)
            if not key or key in lookup:
                continue
            lookup[key] = {
                "caption": str(fig.caption or "").strip(),
                "type": str(fig.type or "").strip(),
                "section_title": str(section.section_title or "").strip(),
            }
    return lookup


def _copy_paper_assets(
    project_root: Path,
    paper_folder_name: str,
    site_dir: Path,
    entry_html_path: Path,
    structured_paper: StructuredPaper,
    figure_paths: list[str],
) -> tuple[list[dict[str, str]], list[str]]:
    asset_manifest: list[dict[str, str]] = []
    copied_assets: list[str] = []
    if not figure_paths:
        return asset_manifest, copied_assets

    source_root = project_root / "data" / "output" / paper_folder_name
    target_dir = site_dir / "assets" / "paper"
    target_dir.mkdir(parents=True, exist_ok=True)
    asset_lookup = _build_asset_lookup(structured_paper)
    used_names: set[str] = set()

    for rel_path in figure_paths:
        clean_rel_path = _normalize_asset_key(rel_path)
        source_path = source_root / clean_rel_path
        if not source_path.exists() or not source_path.is_file():
            continue

        base_name = _safe_slug(source_path.stem)[:60]
        suffix = source_path.suffix or ".png"
        target_name = f"{base_name}{suffix}"
        disambiguation = 2
        while target_name in used_names:
            target_name = f"{base_name}-{disambiguation}{suffix}"
            disambiguation += 1
        used_names.add(target_name)

        target_path = target_dir / target_name
        shutil.copy2(source_path, target_path)
        copied_rel_path = str(target_path.relative_to(site_dir)).replace("\\", "/")
        web_path = _to_html_relative_path(target_path, entry_html_path.parent)
        metadata = asset_lookup.get(clean_rel_path, {})
        asset_manifest.append(
            {
                "source_path": clean_rel_path,
                "web_path": web_path,
                "filename": target_name,
                "caption": str(metadata.get("caption") or ""),
                "type": str(metadata.get("type") or ""),
                "section_title": str(metadata.get("section_title") or ""),
            }
        )
        copied_assets.append(copied_rel_path)

    return asset_manifest, copied_assets


def _format_feedback_block(value: Any) -> str:
    if not isinstance(value, list):
        return "(none)"
    lines: list[str] = []
    for index, item in enumerate(value, start=1):
        clean = str(item or "").strip()
        if clean:
            lines.append(f"{index}. {clean}")
    return "\n".join(lines) if lines else "(none)"


def _read_previous_generated_html(state: CoderState) -> str:
    artifact = _normalize_coder_artifact(state.get("coder_artifact"))
    if not artifact:
        return "(none)"
    previous_html = read_current_page_html(artifact, missing_value="").strip()
    return previous_html or "(none)"


def _sanitized_page_plan_for_prompt(page_plan: PagePlan) -> dict[str, Any]:
    payload = page_plan.model_dump()
    payload["dom_mapping"] = {
        selector: "[compat_global_anchor]"
        for selector in page_plan.dom_mapping
    }
    return payload


def _with_shell_enriched_page_plan(page_plan: PagePlan, template_reference_html: str) -> PagePlan:
    enriched_plan = enrich_page_plan_shell_contracts(page_plan, template_reference_html)
    missing_blocks = missing_shell_contract_block_ids(enriched_plan)
    if missing_blocks:
        raise ValueError("Template shell extraction failed for block(s): " + ", ".join(missing_blocks))
    return enriched_plan


def _output_dir(project_root: Path, paper_folder_name: str) -> Path:
    return project_root / "data" / "output" / paper_folder_name


def _template_profile_output_path(output_dir: Path) -> Path:
    return output_dir / "template_profile.json"


def _block_specs_dir(output_dir: Path) -> Path:
    return output_dir / "block_specs"


def _block_renders_dir(output_dir: Path) -> Path:
    return output_dir / "block_renders"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _selector_tag(soup: BeautifulSoup, selector: str) -> Tag:
    try:
        matches = [match for match in soup.select(str(selector or "").strip()) if isinstance(match, Tag)]
    except Exception as exc:
        raise ValueError(f"Invalid selector '{selector}': {exc}") from exc
    if len(matches) != 1:
        raise ValueError(f"Expected one template shell for selector '{selector}', found {len(matches)}.")
    return matches[0]


def _extract_single_root_tag(html_fragment: str) -> Tag:
    fragment = BeautifulSoup(str(html_fragment or ""), "html.parser")
    nodes = [
        node
        for node in (list(fragment.body.contents) if fragment.body is not None else list(fragment.contents))
        if not (isinstance(node, str) and not node.strip())
    ]
    if len(nodes) != 1 or not isinstance(nodes[0], Tag):
        raise ValueError("Expected exactly one root element in block fragment HTML.")
    return nodes[0]


def _allowed_slot_ids() -> set[str]:
    return {"title", "summary", "body", "media", "meta", "actions"}


def _build_render_specs(
    *,
    page_plan: PagePlan,
    template_profile: TemplateProfile,
    template_reference_html: str,
) -> tuple[list[BlockRenderSpec], PagePlan]:
    template_soup = BeautifulSoup(str(template_reference_html or ""), "html.parser")
    candidate_by_selector = {
        str(candidate.selector or "").strip(): candidate
        for candidate in template_profile.shell_candidates
        if str(candidate.selector or "").strip()
    }
    outline_lookup = {item.block_id: item for item in page_plan.page_outline}
    updated_blocks = []
    specs: list[BlockRenderSpec] = []

    for block in page_plan.blocks:
        selector = str(block.target_template_region.selector_hint or "").strip()
        if not selector:
            raise ValueError(f"Block '{block.block_id}' is missing selector_hint for block assembly.")

        shell_tag = _selector_tag(template_soup, selector)
        candidate = candidate_by_selector.get(selector)
        shell_contract = block.shell_contract
        if shell_contract is None and candidate is not None:
            shell_contract = BlockShellContract(
                root_tag=candidate.root_tag,
                required_classes=list(candidate.required_classes),
                preserve_ids=list(candidate.preserve_ids),
                wrapper_chain=list(candidate.wrapper_chain),
                actionable_root_selector=candidate.selector,
            )
        if shell_contract is None:
            raise ValueError(f"Block '{block.block_id}' is missing shell_contract for block assembly.")

        updated_block = block.model_copy(update={"shell_contract": shell_contract}, deep=True)
        updated_blocks.append(updated_block)

        outline_item = outline_lookup.get(block.block_id)
        order = int(outline_item.order if outline_item is not None else updated_block.responsive_rules.mobile_order)
        title = str(outline_item.title if outline_item is not None else updated_block.block_id)
        source_sections = (
            list(outline_item.source_sections)
            if outline_item is not None
            else []
        )
        specs.append(
            BlockRenderSpec(
                block_id=updated_block.block_id,
                order=order,
                title=title,
                source_sections=source_sections,
                binding=ResolvedBlockBinding(
                    block_id=updated_block.block_id,
                    selector=selector,
                    region_role=updated_block.target_template_region.region_role,
                    root_tag=shell_contract.root_tag,
                    required_classes=list(shell_contract.required_classes),
                    preserve_ids=list(shell_contract.preserve_ids),
                    wrapper_chain=list(shell_contract.wrapper_chain),
                    actionable_root_selector=shell_contract.actionable_root_selector,
                    dom_index=int(candidate.dom_index if candidate is not None else order),
                ),
                content_contract=updated_block.content_contract,
                asset_binding=updated_block.asset_binding,
                interaction=updated_block.interaction,
                responsive_rules=updated_block.responsive_rules,
                shell_contract=shell_contract,
                shell_html=str(shell_tag),
            )
        )

    specs.sort(key=lambda item: (item.order, item.block_id))
    updated_plan = page_plan.model_copy(update={"blocks": updated_blocks}, deep=True)
    return specs, updated_plan


def _render_block_fragment(
    *,
    spec: BlockRenderSpec,
    structured_paper: StructuredPaper,
    template_reference_html: str,
    available_paper_assets: list[dict[str, str]],
    coder_instructions: str,
    human_directives: str,
    retry_feedback: str = "",
) -> str:
    llm = get_llm(temperature=0.15, use_smart_model=True)
    effective_instructions = coder_instructions
    if retry_feedback:
        effective_instructions = (effective_instructions + "\n\nRetry feedback:\n" + retry_feedback).strip()
    response = llm.invoke(
        [
            SystemMessage(content=BLOCK_RENDER_SYSTEM_PROMPT),
            HumanMessage(
                content=BLOCK_RENDER_USER_PROMPT_TEMPLATE.format(
                    block_render_spec_json=to_pretty_json(spec),
                    structured_paper_json=to_pretty_json(structured_paper),
                    template_shell_html=spec.shell_html,
                    template_reference_html=template_reference_html,
                    available_paper_assets_json=to_pretty_json(available_paper_assets),
                    coder_instructions=effective_instructions or "(none)",
                    human_directives=human_directives or "(none)",
                )
            ),
        ]
    )
    return extract_html_fragment(message_content_to_text(response))


def _validate_block_fragment(
    *,
    fragment_html: str,
    spec: BlockRenderSpec,
    allowed_asset_web_paths: set[str],
) -> list[str]:
    critiques: list[str] = []
    try:
        root = _extract_single_root_tag(fragment_html)
    except ValueError as exc:
        return [str(exc)]

    block_id = str(root.get("data-pa-block") or "").strip()
    if block_id != spec.block_id:
        critiques.append(
            f"Fragment root data-pa-block must equal '{spec.block_id}', got '{block_id or '(missing)'}'."
        )
    if str(root.name or "") != spec.binding.root_tag:
        critiques.append(
            f"Fragment root tag '{root.name}' does not match required shell root '{spec.binding.root_tag}'."
        )
    actual_classes = {item for item in root.get("class", []) if str(item).strip()}
    missing_classes = [item for item in spec.binding.required_classes if item not in actual_classes]
    if missing_classes:
        critiques.append(f"Fragment root is missing required shell classes {missing_classes}.")
    if spec.binding.preserve_ids:
        actual_id = str(root.get("id") or "").strip()
        if actual_id not in set(spec.binding.preserve_ids):
            critiques.append(
                f"Fragment root id '{actual_id or '(missing)'}' must preserve one of {spec.binding.preserve_ids}."
            )

    slot_ids: list[str] = []
    for slot_tag in root.select("[data-pa-slot]"):
        slot_id = str(slot_tag.get("data-pa-slot") or "").strip()
        if slot_id:
            slot_ids.append(slot_id)
    if not slot_ids:
        critiques.append("Fragment must expose at least one data-pa-slot.")
    unsupported_slots = sorted(set(slot_ids) - _allowed_slot_ids())
    if unsupported_slots:
        critiques.append(f"Fragment uses unsupported data-pa-slot values: {unsupported_slots}.")

    critiques.extend(
        validate_fragment_local_image_sources(
            html_text=str(root),
            allowed_asset_web_paths=allowed_asset_web_paths,
            allowed_existing_local_sources=set(),
        )
    )
    return critiques


def _write_block_render_artifact(
    *,
    output_dir: Path,
    spec: BlockRenderSpec,
    fragment_html: str,
    validation_errors: list[str],
    notes: list[str],
) -> BlockRenderArtifact:
    specs_dir = _block_specs_dir(output_dir)
    renders_dir = _block_renders_dir(output_dir)
    specs_dir.mkdir(parents=True, exist_ok=True)
    renders_dir.mkdir(parents=True, exist_ok=True)

    spec_path = specs_dir / f"{spec.block_id}.json"
    render_html_path = renders_dir / f"{spec.block_id}.html"
    render_meta_path = renders_dir / f"{spec.block_id}.json"
    _write_json(spec_path, spec.model_dump())
    render_html_path.write_text(fragment_html, encoding="utf-8")
    _write_json(
        render_meta_path,
        {
            "block_id": spec.block_id,
            "order": spec.order,
            "selector": spec.binding.selector,
            "render_mode": "compiled_block_assembly",
            "html_path": str(render_html_path),
            "metadata_path": str(render_meta_path),
            "validation_errors": validation_errors,
            "notes": notes,
        },
    )
    return BlockRenderArtifact(
        block_id=spec.block_id,
        order=spec.order,
        selector=spec.binding.selector,
        render_mode="compiled_block_assembly",
        html=fragment_html,
        html_path=str(render_html_path),
        metadata_path=str(render_meta_path),
        validation_errors=validation_errors,
        notes=notes,
    )


def _render_blocks(
    *,
    output_dir: Path,
    specs: list[BlockRenderSpec],
    structured_paper: StructuredPaper,
    template_reference_html: str,
    available_paper_assets: list[dict[str, str]],
    coder_instructions: str,
    human_directives: str,
) -> list[BlockRenderArtifact]:
    allowed_asset_web_paths = collect_allowed_asset_web_paths(available_paper_assets)
    artifacts: list[BlockRenderArtifact] = []

    for spec in specs:
        fragment_html = ""
        last_errors: list[str] = []
        notes: list[str] = []
        for attempt in range(1, 3):
            fragment_html = _render_block_fragment(
                spec=spec,
                structured_paper=structured_paper,
                template_reference_html=template_reference_html,
                available_paper_assets=available_paper_assets,
                coder_instructions=coder_instructions,
                human_directives=human_directives,
                retry_feedback="\n".join(last_errors) if last_errors else "",
            )
            last_errors = _validate_block_fragment(
                fragment_html=fragment_html,
                spec=spec,
                allowed_asset_web_paths=allowed_asset_web_paths,
            )
            if not last_errors:
                notes.append(f"rendered_on_attempt_{attempt}")
                break
        if last_errors:
            raise ValueError(f"Block renderer failed for '{spec.block_id}': {'; '.join(last_errors)}")
        artifacts.append(
            _write_block_render_artifact(
                output_dir=output_dir,
                spec=spec,
                fragment_html=fragment_html,
                validation_errors=last_errors,
                notes=notes,
            )
        )

    artifacts.sort(key=lambda item: (item.order, item.block_id))
    return artifacts


def _assemble_page(
    *,
    page_plan: PagePlan,
    template_profile: TemplateProfile,
    template_reference_html: str,
    block_artifacts: list[BlockRenderArtifact],
) -> str:
    soup = BeautifulSoup(str(template_reference_html or ""), "html.parser")
    bound_selectors = {artifact.selector for artifact in block_artifacts}

    for artifact in sorted(block_artifacts, key=lambda item: (item.order, item.block_id)):
        target = _selector_tag(soup, artifact.selector)
        replacement_root = _extract_single_root_tag(artifact.html)
        target.replace_with(replacement_root)

    for selector in page_plan.selectors_to_remove:
        try:
            matches = [match for match in soup.select(str(selector or "").strip()) if isinstance(match, Tag)]
        except Exception:
            matches = []
        for match in matches:
            match.decompose()

    for selector in template_profile.unsafe_selectors:
        if selector in bound_selectors or selector in set(template_profile.global_preserve_selectors):
            continue
        try:
            matches = [match for match in soup.select(str(selector or "").strip()) if isinstance(match, Tag)]
        except Exception:
            matches = []
        for match in matches:
            match.decompose()

    html_text = _ensure_doctype(str(soup))
    html_text = _ensure_body_markers(html_text)
    html_text = annotate_global_anchors(html_text, page_plan)
    return normalize_html_document_whitespace(html_text)


def _persist_generated_page(
    *,
    output_dir: Path,
    site_dir: Path,
    generated_entry_html_path: Path,
    template_entry_rel: str,
    template_profile: TemplateProfile | None,
    generated_html: str,
    page_manifest: Any,
) -> tuple[str | None, str]:
    generated_entry_html_path.parent.mkdir(parents=True, exist_ok=True)
    generated_entry_html_path.write_text(generated_html, encoding="utf-8")
    manifest_path = build_page_manifest_path(generated_entry_html_path)
    save_page_manifest(manifest_path, page_manifest)

    mirrored_entry_path = site_dir / template_entry_rel if template_entry_rel else generated_entry_html_path
    if mirrored_entry_path.resolve() != generated_entry_html_path.resolve():
        mirrored_entry_path.parent.mkdir(parents=True, exist_ok=True)
        mirrored_entry_path.write_text(generated_html, encoding="utf-8")

    template_profile_path = None
    if template_profile is not None:
        template_profile_path = str(_template_profile_output_path(output_dir))
        _write_json(Path(template_profile_path), template_profile.model_dump())
    return template_profile_path, str(manifest_path)


def _build_compiled_artifact(
    *,
    output_dir: Path,
    site_dir: Path,
    generated_entry_html_path: Path,
    template_entry_rel: str,
    page_plan: PagePlan,
    template_profile_path: str | None,
    page_manifest_path: str,
    copied_assets: list[str],
) -> CoderArtifact:
    edited_files = ["index.html"]
    mirrored_entry_path = site_dir / template_entry_rel if template_entry_rel else generated_entry_html_path
    if mirrored_entry_path.resolve() != generated_entry_html_path.resolve():
        edited_files.append(str(mirrored_entry_path.relative_to(site_dir)).replace("\\", "/"))
    return CoderArtifact(
        site_dir=str(site_dir),
        entry_html=str(generated_entry_html_path),
        selected_template_id=page_plan.template_selection.selected_template_id,
        copied_assets=copied_assets,
        edited_files=edited_files,
        notes=(
            "v8-compiled-block-assembly: rendered block specs, assembled template shells programmatically, "
            "preserved global anchors, and validated page manifest compatibility."
        ),
        render_mode="compiled_block_assembly",
        template_profile_path=template_profile_path,
        page_manifest_path=page_manifest_path,
        block_artifact_dir=str(_block_renders_dir(output_dir)),
    )


def _run_compiled_block_assembly(
    *,
    paper_folder_name: str,
    structured_paper: StructuredPaper,
    page_plan: PagePlan,
    template_profile: TemplateProfile,
    human_directives: str,
    coder_instructions: str,
) -> tuple[CoderArtifact, PagePlan, list[BlockRenderSpec], list[BlockRenderArtifact]]:
    project_root = Path(__file__).resolve().parents[2]
    template_root = project_root / page_plan.template_selection.selected_root_dir
    template_entry_rel = str(page_plan.template_selection.selected_entry_html or "").strip()
    template_entry_path = template_root / template_entry_rel
    output_dir = _output_dir(project_root, paper_folder_name)
    site_dir = output_dir / "site"
    generated_entry_html_path = site_dir / "index.html"

    if site_dir.exists():
        shutil.rmtree(site_dir)
    if not template_root.exists():
        raise FileNotFoundError(f"Template root not found: {template_root}")
    if not template_entry_path.exists():
        raise FileNotFoundError(f"Template entry html not found: {template_entry_path}")

    template_reference_html = read_text_with_fallback(template_entry_path)
    shutil.copytree(template_root, site_dir)
    figure_paths = _collect_figure_paths(page_plan, structured_paper)
    available_paper_assets, copied_assets = _copy_paper_assets(
        project_root=project_root,
        paper_folder_name=paper_folder_name,
        site_dir=site_dir,
        entry_html_path=generated_entry_html_path,
        structured_paper=structured_paper,
        figure_paths=figure_paths,
    )

    block_render_specs, updated_page_plan = _build_render_specs(
        page_plan=page_plan,
        template_profile=template_profile,
        template_reference_html=template_reference_html,
    )
    block_render_artifacts = _render_blocks(
        output_dir=output_dir,
        specs=block_render_specs,
        structured_paper=structured_paper,
        template_reference_html=template_reference_html,
        available_paper_assets=available_paper_assets,
        coder_instructions=coder_instructions,
        human_directives=human_directives,
    )
    generated_html = _assemble_page(
        page_plan=updated_page_plan,
        template_profile=template_profile,
        template_reference_html=template_reference_html,
        block_artifacts=block_render_artifacts,
    )
    asset_critiques = validate_local_image_references(
        html_text=generated_html,
        entry_html_path=generated_entry_html_path,
        site_dir=site_dir,
        allowed_asset_web_paths=collect_allowed_asset_web_paths(available_paper_assets),
        enforce_paper_asset_whitelist=True,
    )
    if asset_critiques:
        raise ValueError("Compiled assembly failed local image validation: " + " | ".join(asset_critiques))

    page_manifest = extract_page_manifest(
        html_text=generated_html,
        entry_html=generated_entry_html_path,
        selected_template_id=page_plan.template_selection.selected_template_id,
        page_plan=updated_page_plan,
    )
    template_profile_path, page_manifest_path = _persist_generated_page(
        output_dir=output_dir,
        site_dir=site_dir,
        generated_entry_html_path=generated_entry_html_path,
        template_entry_rel=template_entry_rel,
        template_profile=template_profile,
        generated_html=generated_html,
        page_manifest=page_manifest,
    )
    artifact = _build_compiled_artifact(
        output_dir=output_dir,
        site_dir=site_dir,
        generated_entry_html_path=generated_entry_html_path,
        template_entry_rel=template_entry_rel,
        page_plan=updated_page_plan,
        template_profile_path=template_profile_path,
        page_manifest_path=page_manifest_path,
        copied_assets=copied_assets,
    )
    return artifact, updated_page_plan, block_render_specs, block_render_artifacts


def _run_legacy_fullpage_render(
    *,
    paper_folder_name: str,
    structured_paper: StructuredPaper,
    page_plan: PagePlan,
    human_directives: str,
    coder_instructions: str,
    state: CoderState,
    template_profile: TemplateProfile | None,
) -> tuple[CoderArtifact, PagePlan]:
    previous_generated_html = _read_previous_generated_html(state)
    project_root = Path(__file__).resolve().parents[2]
    template_root = project_root / page_plan.template_selection.selected_root_dir
    template_entry_rel = str(page_plan.template_selection.selected_entry_html or "").strip()
    template_entry_path = template_root / template_entry_rel
    output_dir = _output_dir(project_root, paper_folder_name)
    site_dir = output_dir / "site"
    generated_entry_html_path = site_dir / "index.html"

    if site_dir.exists():
        shutil.rmtree(site_dir)
    if not template_root.exists():
        raise FileNotFoundError(f"Template root not found: {template_root}")
    if not template_entry_path.exists():
        raise FileNotFoundError(f"Template entry html not found: {template_entry_path}")

    template_reference_html = read_text_with_fallback(template_entry_path)
    page_plan = _with_shell_enriched_page_plan(page_plan, template_reference_html)
    shutil.copytree(template_root, site_dir)

    figure_paths = _collect_figure_paths(page_plan, structured_paper)
    asset_manifest, copied_assets = _copy_paper_assets(
        project_root=project_root,
        paper_folder_name=paper_folder_name,
        site_dir=site_dir,
        entry_html_path=generated_entry_html_path,
        structured_paper=structured_paper,
        figure_paths=figure_paths,
    )

    llm = get_llm(temperature=0.2, use_smart_model=True)
    response = llm.invoke(
        [
            SystemMessage(content=CODER_SYSTEM_PROMPT),
            HumanMessage(
                content=CODER_USER_PROMPT_TEMPLATE.format(
                    structured_paper_json=to_pretty_json(structured_paper),
                    page_plan_json=json.dumps(
                        _sanitized_page_plan_for_prompt(page_plan),
                        indent=2,
                        ensure_ascii=False,
                    ),
                    template_reference_html=template_reference_html,
                    coder_instructions=coder_instructions or "(none)",
                    human_directives=human_directives or "(none)",
                    available_paper_assets_json=to_pretty_json(asset_manifest),
                    global_anchor_requirements_json=json.dumps(
                        build_expected_global_anchors(
                            page_plan,
                            reference_html_text=template_reference_html,
                        ),
                        indent=2,
                        ensure_ascii=False,
                    ),
                    prior_coder_feedback=_format_feedback_block(state.get("coder_feedback_history")),
                    prior_visual_feedback=_format_feedback_block(state.get("visual_feedback")),
                    previous_generated_html=previous_generated_html,
                )
            ),
        ]
    )

    generated_html = extract_html_document(message_content_to_text(response))
    if not generated_html:
        raise ValueError("Legacy fullpage coder did not return a valid HTML document.")
    generated_html = _ensure_body_markers(generated_html)
    generated_html = normalize_html_document_whitespace(generated_html)
    generated_html = annotate_global_anchors(generated_html, page_plan)

    asset_critiques = validate_local_image_references(
        html_text=generated_html,
        entry_html_path=generated_entry_html_path,
        site_dir=site_dir,
        allowed_asset_web_paths=collect_allowed_asset_web_paths(asset_manifest),
        enforce_paper_asset_whitelist=True,
    )
    if asset_critiques:
        raise ValueError("Legacy fullpage coder failed local image validation: " + " | ".join(asset_critiques))

    page_manifest = extract_page_manifest(
        html_text=generated_html,
        entry_html=generated_entry_html_path,
        selected_template_id=page_plan.template_selection.selected_template_id,
        page_plan=page_plan,
    )
    template_profile_path, page_manifest_path = _persist_generated_page(
        output_dir=output_dir,
        site_dir=site_dir,
        generated_entry_html_path=generated_entry_html_path,
        template_entry_rel=template_entry_rel,
        template_profile=template_profile,
        generated_html=generated_html,
        page_manifest=page_manifest,
    )
    edited_files = ["index.html"]
    mirrored_entry_path = site_dir / template_entry_rel if template_entry_rel else generated_entry_html_path
    if mirrored_entry_path.resolve() != generated_entry_html_path.resolve():
        edited_files.append(str(mirrored_entry_path.relative_to(site_dir)).replace("\\", "/"))
    artifact = CoderArtifact(
        site_dir=str(site_dir),
        entry_html=str(generated_entry_html_path),
        selected_template_id=page_plan.template_selection.selected_template_id,
        copied_assets=copied_assets,
        edited_files=edited_files,
        notes=(
            "v8-legacy-fullpage-fallback: generated shell-constrained HTML via fullpage coder and "
            "validated stable data-pa-block, data-pa-slot, and data-pa-global anchors."
        ),
        render_mode="legacy_fullpage",
        template_profile_path=template_profile_path,
        page_manifest_path=page_manifest_path,
        block_artifact_dir=None,
    )
    return artifact, page_plan


def coder_node(state: CoderState) -> dict[str, Any]:
    print(
        f"[PaperAlchemy-Coder] building site "
        f"(attempt {state.get('coder_retry_count', 0) + 1})..."
    )
    page_plan = _normalize_page_plan(state.get("page_plan"))
    structured_paper = _normalize_structured_paper(state.get("structured_paper"))
    template_profile = _normalize_template_profile(state.get("template_profile"))
    paper_folder_name = str(state.get("paper_folder_name") or "").strip()
    human_directives = extract_human_feedback_text(state.get("human_directives"))
    coder_instructions = str(state.get("coder_instructions") or "").strip()
    if not page_plan or not structured_paper or not paper_folder_name:
        print("[PaperAlchemy-Coder] missing page_plan/structured_paper/paper_folder_name.")
        return {}

    requested_strategy = str(page_plan.plan_meta.render_strategy or "compiled_block_assembly").strip()
    try:
        if requested_strategy == "compiled_block_assembly" and template_profile is not None:
            artifact, resolved_page_plan, block_specs, block_artifacts = _run_compiled_block_assembly(
                paper_folder_name=paper_folder_name,
                structured_paper=structured_paper,
                page_plan=page_plan,
                template_profile=template_profile,
                human_directives=human_directives,
                coder_instructions=coder_instructions,
            )
            return {
                "coder_artifact": artifact,
                "page_plan": resolved_page_plan,
                "block_render_specs": block_specs,
                "block_render_artifacts": block_artifacts,
            }

        artifact, resolved_page_plan = _run_legacy_fullpage_render(
            paper_folder_name=paper_folder_name,
            structured_paper=structured_paper,
            page_plan=page_plan,
            human_directives=human_directives,
            coder_instructions=coder_instructions,
            state=state,
            template_profile=template_profile,
        )
        return {
            "coder_artifact": artifact,
            "page_plan": resolved_page_plan,
            "block_render_specs": [],
            "block_render_artifacts": [],
        }
    except Exception as exc:
        if requested_strategy == "compiled_block_assembly":
            print(f"[PaperAlchemy-Coder] compiled block assembly failed, falling back to legacy fullpage: {exc}")
            try:
                artifact, resolved_page_plan = _run_legacy_fullpage_render(
                    paper_folder_name=paper_folder_name,
                    structured_paper=structured_paper,
                    page_plan=page_plan,
                    human_directives=human_directives,
                    coder_instructions=coder_instructions,
                    state=state,
                    template_profile=template_profile,
                )
                return {
                    "coder_artifact": artifact,
                    "page_plan": resolved_page_plan,
                    "block_render_specs": [],
                    "block_render_artifacts": [],
                }
            except Exception as fallback_exc:
                print(f"[PaperAlchemy-Coder] legacy fallback failed: {fallback_exc}")
                return {}
        print(f"[PaperAlchemy-Coder] build failed: {exc}")
        return {}


def build_coder_graph(max_retry: int = 1):
    workflow = StateGraph(CoderState)
    workflow.add_node("coder", coder_node)
    workflow.add_node("coder_critic", coder_critic_node)
    workflow.add_node("take_screenshot", take_screenshot_action)
    workflow.add_node("vision_critic", vision_critic_node)

    workflow.set_entry_point("coder")
    workflow.add_edge("coder", "coder_critic")
    workflow.add_conditional_edges(
        "coder_critic",
        build_coder_critic_router(max_retry=max_retry),
        {"retry": "coder", "visual_qa": "take_screenshot", "end": END},
    )
    workflow.add_edge("take_screenshot", "vision_critic")
    workflow.add_conditional_edges(
        "vision_critic",
        build_vision_qa_router(),
        {"retry": "coder", "end": END},
    )

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


def run_coder_agent(
    paper_folder_name: str,
    structured_data: StructuredPaper,
    page_plan: PagePlan,
    human_directives: str | dict = "",
    coder_instructions: str = "",
    previous_coder_artifact: CoderArtifact | None = None,
    max_retry: int = 2,
    template_profile: TemplateProfile | None = None,
) -> CoderArtifact | None:
    artifact, _, _ = run_coder_agent_with_diagnostics(
        paper_folder_name=paper_folder_name,
        structured_data=structured_data,
        page_plan=page_plan,
        human_directives=human_directives,
        coder_instructions=coder_instructions,
        previous_coder_artifact=previous_coder_artifact,
        max_retry=max_retry,
        template_profile=template_profile,
    )
    return artifact


def run_coder_agent_with_diagnostics(
    paper_folder_name: str,
    structured_data: StructuredPaper,
    page_plan: PagePlan,
    human_directives: str | dict = "",
    coder_instructions: str = "",
    previous_coder_artifact: CoderArtifact | None = None,
    max_retry: int = 2,
    template_profile: TemplateProfile | None = None,
) -> tuple[CoderArtifact | None, VisualSmokeReport | None, PagePlan | None]:
    app = build_coder_graph(max_retry=max_retry)
    thread = {"configurable": {"thread_id": f"coder_{paper_folder_name}"}}
    initial_state: CoderState = {
        "paper_folder_name": paper_folder_name,
        "human_directives": normalize_human_feedback(human_directives),
        "coder_instructions": str(coder_instructions or "").strip(),
        "structured_paper": structured_data,
        "page_plan": page_plan,
        "template_profile": template_profile,
        "block_render_specs": [],
        "block_render_artifacts": [],
        "coder_feedback_history": [],
        "visual_feedback": [],
        "visual_screenshot_path": "",
        "visual_iterations": 0,
        "is_visually_approved": False,
        "visual_smoke_report": None,
        "coder_artifact": previous_coder_artifact,
        "coder_critic_passed": False,
        "coder_retry_count": 0,
    }

    print("[PaperAlchemy-Coder] running Coder + CoderCritic graph...")
    for _ in app.stream(initial_state, thread):
        pass

    final_state = app.get_state(thread)
    artifact_result = final_state.values.get("coder_artifact")
    resolved_page_plan = _normalize_page_plan(final_state.values.get("page_plan"))
    visual_smoke_report = _normalize_visual_smoke_report(final_state.values.get("visual_smoke_report"))
    normalized_artifact = _normalize_coder_artifact(artifact_result)

    if not normalized_artifact or not final_state.values.get("coder_critic_passed"):
        print("[PaperAlchemy-Coder] coder completed but critic did not fully pass.")
        return None, visual_smoke_report, resolved_page_plan

    if not final_state.values.get("is_visually_approved") and int(final_state.values.get("visual_iterations", 0)) > 0:
        print("[PaperAlchemy-Coder] visual smoke test flagged issues; returning the latest artifact for human review.")

    print(f"[PaperAlchemy-Coder] build completed: {normalized_artifact.entry_html}")
    return normalized_artifact, visual_smoke_report, resolved_page_plan

