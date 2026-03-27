import json
import shutil
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch
from uuid import uuid4

from src.contracts.schemas import CoderArtifact, LayoutComposeUpdate, PagePlan, StructuredPaper
from src.patching.patch_pipeline import LEGACY_PAGE_ERROR, patch_agent_node, patch_executor_node
from src.services.human_feedback import empty_human_feedback
from src.template.shell_resolver import (
    apply_layout_compose_session_to_page_plan,
    apply_layout_compose_update,
    build_layout_compose_session,
    resolve_page_plan_shells,
)
from src.validators.page_manifest import (
    annotate_global_anchors,
    build_page_manifest_path,
    extract_page_manifest,
    save_page_manifest,
)
from src.validators.page_validation import REVISION_OVERRIDE_STYLE_TAG_ID

try:
    import app
except Exception as exc:  # pragma: no cover - skip when optional UI deps are unavailable
    app = None
    APP_IMPORT_ERROR = exc
else:
    APP_IMPORT_ERROR = None


def _sample_structured_paper() -> StructuredPaper:
    return StructuredPaper.model_validate(
        {
            "paper_title": "Demo Paper",
            "overall_summary": "Authors: Demo. Affiliations: Demo Lab. Summary.",
            "sections": [
                {
                    "section_title": "Abstract",
                    "rich_web_content": "Abstract content",
                    "related_figures": [],
                },
                {
                    "section_title": "Results",
                    "rich_web_content": "Results content",
                    "related_figures": [],
                },
            ],
        }
    )


def _sample_structured_paper_with_figures() -> StructuredPaper:
    return StructuredPaper.model_validate(
        {
            "paper_title": "Demo Paper",
            "overall_summary": "Demo summary",
            "sections": [
                {
                    "section_title": "Abstract",
                    "rich_web_content": "Abstract content",
                    "related_figures": [
                        {
                            "image_path": "assets/abstract-figure.png",
                            "caption": "Abstract figure",
                            "type": "chart",
                        }
                    ],
                },
                {
                    "section_title": "Results",
                    "rich_web_content": "Results content",
                    "related_figures": [
                        {
                            "image_path": "assets/results-figure.png",
                            "caption": "Results figure",
                            "type": "table",
                        }
                    ],
                },
            ],
        }
    )


def _sample_page_plan(selected_root_dir: str, include_globals: bool = False) -> PagePlan:
    dom_mapping = {"#hero": "Hero", "#results": "Results"}
    if include_globals:
        dom_mapping.update(
            {
                "header h1": "Brand",
                "header nav": "Nav",
                "header .button-group": "Primary action",
                "footer": "Footer",
            }
        )

    return PagePlan.model_validate(
        {
            "plan_meta": {
                "plan_version": "1.0",
                "planning_mode": "hybrid_template_bind",
                "target_framework": "static-html",
                "confidence": 0.95,
            },
            "template_selection": {
                "selected_template_id": "demo-template",
                "selected_root_dir": selected_root_dir,
                "selected_entry_html": "index.html",
                "fallback_template_id": None,
                "selection_rationale": "demo",
            },
            "decision_summary": {
                "design_goal": "demo",
                "novelty_points": ["demo"],
                "tradeoffs": ["demo"],
            },
            "adaptation_strategy": {
                "preserve_from_template": ["layout"],
                "replace_content_areas": ["hero", "results"],
                "style_override_level": "light",
                "asset_policy": "mixed",
            },
            "global_design": {
                "style_keywords": ["clean"],
                "color_strategy": {
                    "background": "#fff",
                    "surface": "#fff",
                    "text": "#111",
                    "accent": "#00f",
                },
                "typography_strategy": "sans",
                "motion_level": "none",
                "density": "balanced",
            },
            "page_outline": [
                {
                    "block_id": "hero",
                    "order": 1,
                    "title": "Hero",
                    "objective": "Intro",
                    "source_sections": ["Abstract"],
                    "estimated_height": "M",
                },
                {
                    "block_id": "results",
                    "order": 2,
                    "title": "Results",
                    "objective": "Evidence",
                    "source_sections": ["Results"],
                    "estimated_height": "M",
                },
            ],
            "blocks": [
                {
                    "block_id": "hero",
                    "target_template_region": {
                        "selector_hint": "#hero",
                        "region_role": "hero",
                        "operation": "replace_text",
                    },
                    "component_recipe": [
                        {
                            "slot": "content",
                            "module_id": None,
                            "component_id": None,
                            "style_id": None,
                            "token_set_id": None,
                            "reason": "demo",
                        }
                    ],
                    "content_contract": {
                        "headline": "Hero",
                        "body_points": ["intro"],
                        "cta": None,
                    },
                    "asset_binding": {
                        "figure_paths": [],
                        "template_asset_fallback": None,
                    },
                    "interaction": {
                        "pattern": "none",
                        "behavior_note": "demo",
                    },
                    "responsive_rules": {
                        "mobile_order": 1,
                        "desktop_layout": "stack",
                    },
                    "shell_contract": {
                        "root_tag": "section",
                        "required_classes": [],
                        "preserve_ids": ["hero"],
                        "wrapper_chain": [],
                        "actionable_root_selector": "#hero",
                    },
                    "a11y_notes": [],
                    "acceptance_checks": [],
                },
                {
                    "block_id": "results",
                    "target_template_region": {
                        "selector_hint": "#results",
                        "region_role": "section",
                        "operation": "replace_text",
                    },
                    "component_recipe": [
                        {
                            "slot": "content",
                            "module_id": None,
                            "component_id": None,
                            "style_id": None,
                            "token_set_id": None,
                            "reason": "demo",
                        }
                    ],
                    "content_contract": {
                        "headline": "Results",
                        "body_points": ["evidence"],
                        "cta": None,
                    },
                    "asset_binding": {
                        "figure_paths": [],
                        "template_asset_fallback": None,
                    },
                    "interaction": {
                        "pattern": "none",
                        "behavior_note": "demo",
                    },
                    "responsive_rules": {
                        "mobile_order": 2,
                        "desktop_layout": "stack",
                    },
                    "shell_contract": {
                        "root_tag": "section",
                        "required_classes": [],
                        "preserve_ids": ["results"],
                        "wrapper_chain": [],
                        "actionable_root_selector": "#results",
                    },
                    "a11y_notes": [],
                    "acceptance_checks": [],
                },
            ],
            "dom_mapping": dom_mapping,
            "selectors_to_remove": [],
            "coder_handoff": {
                "implementation_order": ["edit index.html"],
                "file_touch_plan": [
                    {"path": "index.html", "action": "edit", "reason": "demo"}
                ],
                "hard_constraints": [],
                "known_risks": [],
            },
            "quality_checks": [
                {"name": "grounding_check", "passed": True, "note": "demo"},
                {"name": "consistency_check", "passed": True, "note": "demo"},
                {"name": "feasibility_check", "passed": True, "note": "demo"},
                {"name": "template_path_check", "passed": True, "note": "demo"},
            ],
            "open_questions": [],
        }
    )


def _sample_html(include_globals: bool = False) -> str:
    header_html = ""
    footer_html = ""
    if include_globals:
        header_html = """
    <header>
      <h1><a href="/" data-pa-global="header_brand">Demo Brand</a></h1>
      <nav data-pa-global="header_nav"><a href="#hero">Hero</a><a href="#results">Results</a></nav>
      <div class="button-group"><a href="#paper" data-pa-global="header_primary_action">Paper</a></div>
    </header>
"""
        footer_html = """
    <footer data-pa-global="footer_meta"><p>Footer copy</p></footer>
"""

    return """<!DOCTYPE html>
<html>
  <head><title>Demo</title></head>
  <body>
    <!-- PaperAlchemy Generated Body Start -->
{header_html}
    <section id="hero" data-pa-block="hero">
      <h1 data-pa-slot="title">Old Title</h1>
      <div data-pa-slot="body"><p>Old hero body</p></div>
    </section>
    <section id="results" data-pa-block="results">
      <h2 data-pa-slot="title">Results</h2>
      <div data-pa-slot="body"><p>Keep me</p></div>
    </section>
{footer_html}
    <!-- PaperAlchemy Generated Body End -->
  </body>
</html>
""".format(header_html=header_html, footer_html=footer_html)


def _update_block_target(
    page_plan: PagePlan,
    block_id: str,
    *,
    selector_hint: str | None = None,
    region_role: str | None = None,
) -> PagePlan:
    updated_blocks = []
    for block in page_plan.blocks:
        if block.block_id != block_id:
            updated_blocks.append(block)
            continue
        update_payload: dict[str, Any] = {}
        if selector_hint is not None:
            update_payload["selector_hint"] = selector_hint
        if region_role is not None:
            update_payload["region_role"] = region_role
        updated_region = block.target_template_region.model_copy(
            update=update_payload,
            deep=True,
        )
        updated_blocks.append(
            block.model_copy(
                update={
                    "target_template_region": updated_region,
                    "shell_contract": None,
                },
                deep=True,
            )
        )
    return page_plan.model_copy(update={"blocks": updated_blocks}, deep=True)


def _get_block(page_plan: PagePlan, block_id: str):
    for block in page_plan.blocks:
        if block.block_id == block_id:
            return block
    raise AssertionError(f"Block not found in page plan: {block_id}")


def _update_block_figure_paths(page_plan: PagePlan, block_id: str, figure_paths: list[str]) -> PagePlan:
    updated_blocks = []
    for block in page_plan.blocks:
        if block.block_id != block_id:
            updated_blocks.append(block)
            continue
        updated_blocks.append(
            block.model_copy(
                update={
                    "asset_binding": block.asset_binding.model_copy(
                        update={"figure_paths": figure_paths},
                        deep=True,
                    )
                },
                deep=True,
            )
        )
    return page_plan.model_copy(update={"blocks": updated_blocks}, deep=True)


class ManifestTests(unittest.TestCase):
    def test_extract_page_manifest_from_anchored_html(self) -> None:
        page_plan = _sample_page_plan("data/templates/demo-template")
        manifest = extract_page_manifest(
            html_text=_sample_html(),
            entry_html=Path("demo/site/index.html"),
            selected_template_id="demo-template",
            page_plan=page_plan,
        )

        self.assertEqual(2, len(manifest.blocks))
        self.assertEqual("hero", manifest.blocks[0].block_id)
        self.assertEqual("title", manifest.blocks[0].slots[0].slot_id)

    def test_extract_page_manifest_rejects_missing_required_block(self) -> None:
        page_plan = _sample_page_plan("data/templates/demo-template")
        bad_html = _sample_html().replace('data-pa-block="results"', 'data-pa-block="other"')

        with self.assertRaisesRegex(ValueError, "Missing required data-pa-block ids"):
            extract_page_manifest(
                html_text=bad_html,
                entry_html=Path("demo/site/index.html"),
                selected_template_id="demo-template",
                page_plan=page_plan,
            )

    def test_extract_page_manifest_records_globals(self) -> None:
        page_plan = _sample_page_plan("data/templates/demo-template", include_globals=True)
        manifest = extract_page_manifest(
            html_text=_sample_html(include_globals=True),
            entry_html=Path("demo/site/index.html"),
            selected_template_id="demo-template",
            page_plan=page_plan,
        )

        self.assertEqual(
            ["header_brand", "header_nav", "header_primary_action", "footer_meta"],
            [item.global_id for item in manifest.globals],
        )

    def test_annotate_global_anchors_promotes_clickable_target(self) -> None:
        page_plan = _sample_page_plan("data/templates/demo-template", include_globals=True)
        raw_html = """<!DOCTYPE html>
<html>
  <body>
    <header>
      <div class="button-group"><a href="#paper"><span>Paper</span></a></div>
    </header>
  </body>
</html>
"""

        annotated_html = annotate_global_anchors(raw_html, page_plan)

        self.assertIn('data-pa-global="header_primary_action"', annotated_html)
        self.assertIn('<a data-pa-global="header_primary_action" href="#paper">', annotated_html)
        self.assertNotIn('<span data-pa-global="header_primary_action">', annotated_html)

    def test_extract_page_manifest_rejects_missing_required_global(self) -> None:
        page_plan = _sample_page_plan("data/templates/demo-template", include_globals=True)

        with self.assertRaisesRegex(ValueError, "Missing required data-pa-global ids"):
            extract_page_manifest(
                html_text=_sample_html(include_globals=False),
                entry_html=Path("demo/site/index.html"),
                selected_template_id="demo-template",
                page_plan=page_plan,
            )


class TemplateShellResolverTests(unittest.TestCase):
    def test_preserves_original_selector_when_it_already_resolves(self) -> None:
        page_plan = _sample_page_plan("data/templates/demo-template")

        resolved_plan, review = resolve_page_plan_shells(
            page_plan=page_plan,
            template_reference_html=_sample_html(),
            template_entry_html_path=Path("demo/template/index.html"),
        )

        self.assertIsNone(review)
        hero_block = _get_block(resolved_plan, "hero")
        results_block = _get_block(resolved_plan, "results")
        self.assertEqual("#hero", hero_block.target_template_region.selector_hint)
        self.assertEqual("#results", results_block.target_template_region.selector_hint)
        self.assertIsNotNone(hero_block.shell_contract)
        self.assertEqual("section#hero", hero_block.shell_contract.actionable_root_selector)

    def test_rebinds_to_template_shell_when_original_selector_is_missing(self) -> None:
        page_plan = _update_block_target(
            _sample_page_plan("data/templates/demo-template"),
            "results",
            selector_hint="#missing-results",
        )
        template_html = """<!DOCTYPE html>
<html>
  <body>
    <header id="hero"><h1>Hero</h1></header>
    <section class="results-shell"><h2>Results</h2><p>Metrics</p></section>
    <div class="content-shell"><p>Appendix</p></div>
  </body>
</html>
"""

        resolved_plan, review = resolve_page_plan_shells(
            page_plan=page_plan,
            template_reference_html=template_html,
            template_entry_html_path=Path("demo/template/index.html"),
        )

        self.assertIsNone(review)
        results_block = _get_block(resolved_plan, "results")
        self.assertNotEqual("#missing-results", results_block.target_template_region.selector_hint)
        self.assertIn("results-shell", results_block.target_template_region.selector_hint)
        self.assertIsNotNone(results_block.shell_contract)

    def test_assigns_distinct_shells_in_outline_order(self) -> None:
        page_plan = _sample_page_plan("data/templates/demo-template")
        page_plan = _update_block_target(
            _update_block_target(page_plan, "hero", selector_hint="#intro", region_role="section"),
            "results",
            selector_hint="#results",
            region_role="section",
        )
        template_html = """<!DOCTYPE html>
<html>
  <body>
    <section class="intro-shell"><h1>Intro</h1></section>
    <section class="results-shell"><h2>Results</h2></section>
    <section class="appendix-shell"><h2>Appendix</h2></section>
  </body>
</html>
"""

        resolved_plan, review = resolve_page_plan_shells(
            page_plan=page_plan,
            template_reference_html=template_html,
            template_entry_html_path=Path("demo/template/index.html"),
        )

        self.assertIsNone(review)
        hero_block = _get_block(resolved_plan, "hero")
        results_block = _get_block(resolved_plan, "results")
        self.assertIn("intro-shell", hero_block.target_template_region.selector_hint)
        self.assertIn("results-shell", results_block.target_template_region.selector_hint)
        self.assertNotEqual(
            hero_block.target_template_region.selector_hint,
            results_block.target_template_region.selector_hint,
        )

    def test_rejects_illegal_role_candidates(self) -> None:
        page_plan = _update_block_target(
            _sample_page_plan("data/templates/demo-template"),
            "results",
            selector_hint="#missing-footer",
            region_role="footer",
        )
        template_html = """<!DOCTYPE html>
<html>
  <body>
    <header id="hero"><h1>Hero</h1></header>
    <section class="results-shell"><h2>Results</h2></section>
    <section class="content-shell"><p>Body</p></section>
  </body>
</html>
"""

        resolved_plan, review = resolve_page_plan_shells(
            page_plan=page_plan,
            template_reference_html=template_html,
            template_entry_html_path=Path("demo/template/index.html"),
        )

        self.assertIsNotNone(review)
        self.assertEqual("results", review.block_id)
        self.assertEqual([], review.candidates)
        unresolved_results = _get_block(resolved_plan, "results")
        self.assertEqual("#missing-footer", unresolved_results.target_template_region.selector_hint)

    def test_prefers_best_scored_candidate_when_rebind_options_are_close(self) -> None:
        page_plan = _update_block_target(
            _sample_page_plan("data/templates/demo-template"),
            "results",
            selector_hint="#content",
        )
        template_html = """<!DOCTYPE html>
<html>
  <body>
    <header id="hero"><h1>Hero</h1></header>
    <section class="content-shell"><h2>Section A</h2><p>Alpha</p></section>
    <section class="content-panel"><h2>Section B</h2><p>Beta</p></section>
  </body>
</html>
"""

        resolved_plan, review = resolve_page_plan_shells(
            page_plan=page_plan,
            template_reference_html=template_html,
            template_entry_html_path=Path("demo/template/index.html"),
        )

        self.assertIsNone(review)
        resolved_results = _get_block(resolved_plan, "results")
        self.assertIn("content-shell", resolved_results.target_template_region.selector_hint)

    def test_layout_compose_session_builds_candidates_and_source_scoped_figures(self) -> None:
        page_plan = _sample_page_plan("data/templates/demo-template")
        page_plan = _update_block_target(page_plan, "hero", selector_hint="#missing-hero")
        page_plan = _update_block_target(page_plan, "results", selector_hint="#missing-results")
        page_plan = _update_block_figure_paths(
            page_plan,
            "hero",
            ["assets/abstract-figure.png", "assets/results-figure.png"],
        )
        page_plan = _update_block_figure_paths(
            page_plan,
            "results",
            ["assets/results-figure.png"],
        )
        structured_paper = _sample_structured_paper_with_figures()
        template_html = """<!DOCTYPE html>
<html>
  <body>
    <header id="hero-shell"><h1>Hero</h1></header>
    <section class="intro-shell"><h2>Abstract</h2></section>
    <section class="results-shell"><h2>Results</h2><img src="chart.png" /></section>
  </body>
</html>
"""

        session = build_layout_compose_session(
            page_plan=page_plan,
            structured_paper=structured_paper,
            template_reference_html=template_html,
            template_entry_html_path=Path("demo/template/index.html"),
        )

        self.assertEqual(2, len(session.blocks))
        hero_block = next(block for block in session.blocks if block.block_id == "hero")
        results_block = next(block for block in session.blocks if block.block_id == "results")
        self.assertGreaterEqual(len(hero_block.section_options), 1)
        self.assertGreaterEqual(len(results_block.section_options), 1)
        self.assertTrue(hero_block.selected_selector_hint)
        self.assertTrue(results_block.selected_selector_hint)
        self.assertEqual(["assets/abstract-figure.png"], hero_block.selected_figure_paths)
        self.assertEqual(
            ["assets/abstract-figure.png"],
            [option.image_path for option in hero_block.figure_options],
        )
        self.assertEqual(
            ["assets/results-figure.png"],
            [option.image_path for option in results_block.figure_options],
        )
        self.assertEqual(
            sorted((option.score for option in results_block.section_options), reverse=True),
            [option.score for option in results_block.section_options],
        )
        self.assertTrue(all(option.dom_index >= 0 for option in results_block.section_options))

    def test_layout_compose_validation_catches_missing_duplicate_and_reverse_order(self) -> None:
        page_plan = _sample_page_plan("data/templates/demo-template")
        page_plan = _update_block_target(page_plan, "hero", selector_hint="#missing-hero")
        page_plan = _update_block_target(page_plan, "results", selector_hint="#missing-results")
        structured_paper = _sample_structured_paper_with_figures()
        template_html = """<!DOCTYPE html>
<html>
  <body>
    <section class="hero-shell"><h1>Hero</h1></section>
    <section class="middle-shell"><h2>Middle</h2></section>
    <section class="results-shell"><h2>Results</h2></section>
  </body>
</html>
"""

        session = build_layout_compose_session(
            page_plan=page_plan,
            structured_paper=structured_paper,
            template_reference_html=template_html,
            template_entry_html_path=Path("demo/template/index.html"),
        )
        hero_block = next(block for block in session.blocks if block.block_id == "hero")
        results_block = next(block for block in session.blocks if block.block_id == "results")
        shared_selector = next(
            selector
            for selector in (option.selector_hint for option in hero_block.section_options)
            if selector in {option.selector_hint for option in results_block.section_options}
        )

        missing_session = apply_layout_compose_update(
            session,
            LayoutComposeUpdate(
                active_block_id="hero",
                selected_selector_hint="",
                action="save_block",
            ),
        )
        self.assertTrue(any("must select exactly one template section" in error for error in missing_session.validation_errors))

        duplicate_session = apply_layout_compose_update(
            session,
            LayoutComposeUpdate(
                active_block_id="results",
                selected_selector_hint=shared_selector,
                action="save_block",
            ),
        )
        self.assertTrue(any("assigned more than once" in error for error in duplicate_session.validation_errors))

        results_earliest = min(results_block.section_options, key=lambda option: option.dom_index).selector_hint
        hero_latest = max(hero_block.section_options, key=lambda option: option.dom_index).selector_hint
        reverse_session = apply_layout_compose_update(
            session,
            LayoutComposeUpdate(
                active_block_id="hero",
                selected_selector_hint=hero_latest,
                action="save_block",
            ),
        )
        reverse_session = apply_layout_compose_update(
            reverse_session,
            LayoutComposeUpdate(
                active_block_id="results",
                selected_selector_hint=results_earliest,
                action="save_block",
            ),
        )
        self.assertTrue(any("must follow template DOM order" in error for error in reverse_session.validation_errors))

    def test_apply_layout_compose_session_rewrites_order_selectors_and_figures(self) -> None:
        page_plan = _sample_page_plan("data/templates/demo-template")
        page_plan = _update_block_target(page_plan, "hero", selector_hint="#missing-hero")
        page_plan = _update_block_target(page_plan, "results", selector_hint="#missing-results")
        page_plan = _update_block_figure_paths(
            page_plan,
            "results",
            ["assets/results-figure.png"],
        )
        structured_paper = _sample_structured_paper_with_figures()
        template_html = """<!DOCTYPE html>
<html>
  <body>
    <section class="hero-shell"><h1>Hero</h1></section>
    <section class="middle-shell"><h2>Middle</h2></section>
    <section class="results-shell"><h2>Results</h2></section>
  </body>
</html>
"""

        session = build_layout_compose_session(
            page_plan=page_plan,
            structured_paper=structured_paper,
            template_reference_html=template_html,
            template_entry_html_path=Path("demo/template/index.html"),
        )
        hero_block = next(block for block in session.blocks if block.block_id == "hero")
        results_block = next(block for block in session.blocks if block.block_id == "results")
        results_earliest = min(results_block.section_options, key=lambda option: option.dom_index).selector_hint
        hero_latest = max(hero_block.section_options, key=lambda option: option.dom_index).selector_hint
        moved_session = apply_layout_compose_update(
            session,
            LayoutComposeUpdate(
                active_block_id="results",
                selected_selector_hint=results_earliest,
                selected_figure_paths=["assets/results-figure.png"],
                order_action="move_up",
                action="move_up",
            ),
        )
        moved_session = apply_layout_compose_update(
            moved_session,
            LayoutComposeUpdate(
                active_block_id="hero",
                selected_selector_hint=hero_latest,
                action="save_block",
            ),
        )

        updated_plan = apply_layout_compose_session_to_page_plan(
            page_plan,
            moved_session,
            template_html,
        )

        self.assertEqual(["results", "hero"], [item.block_id for item in updated_plan.page_outline])
        self.assertEqual([1, 2], [item.order for item in updated_plan.page_outline])
        self.assertEqual(["results", "hero"], [block.block_id for block in updated_plan.blocks])
        self.assertEqual([1, 2], [block.responsive_rules.mobile_order for block in updated_plan.blocks])
        self.assertEqual(results_earliest, updated_plan.blocks[0].target_template_region.selector_hint)
        self.assertEqual(hero_latest, updated_plan.blocks[1].target_template_region.selector_hint)
        self.assertEqual(["assets/results-figure.png"], updated_plan.blocks[0].asset_binding.figure_paths)


class PatchExecutorTests(unittest.TestCase):
    def _setup_workspace(self, include_globals: bool = False) -> tuple[Path, PagePlan, StructuredPaper, CoderArtifact]:
        temp_root = Path.cwd() / "data" / "output" / "_tmp_test_workspace"
        temp_root.mkdir(parents=True, exist_ok=True)
        root = temp_root / uuid4().hex
        root.mkdir(parents=True, exist_ok=False)
        site_dir = root / "output" / "site"
        site_dir.mkdir(parents=True, exist_ok=True)
        entry_html = site_dir / "index.html"
        entry_html.write_text(_sample_html(include_globals=include_globals), encoding="utf-8")

        template_dir = root / "templates" / "demo-template"
        template_dir.mkdir(parents=True, exist_ok=True)
        (template_dir / "index.html").write_text("<html><body>template</body></html>", encoding="utf-8")

        project_root = Path.cwd().resolve()
        selected_root_dir = str(template_dir.resolve().relative_to(project_root)).replace("\\", "/")
        page_plan = _sample_page_plan(selected_root_dir, include_globals=include_globals)
        structured_paper = _sample_structured_paper()
        manifest = extract_page_manifest(
            html_text=entry_html.read_text(encoding="utf-8"),
            entry_html=entry_html,
            selected_template_id="demo-template",
            page_plan=page_plan,
        )
        save_page_manifest(build_page_manifest_path(entry_html), manifest)

        artifact = CoderArtifact(
            site_dir=str(site_dir),
            entry_html=str(entry_html),
            selected_template_id="demo-template",
            copied_assets=[],
            edited_files=["index.html"],
            notes="v6-layered-anchored-render",
        )
        return root, page_plan, structured_paper, artifact

    def test_patch_executor_applies_slot_replacement_without_touching_other_blocks(self) -> None:
        root, page_plan, structured_paper, artifact = self._setup_workspace()
        try:
            state = {
                "paper_folder_name": "demo-paper",
                "page_plan": page_plan,
                "structured_paper": structured_paper,
                "coder_artifact": artifact,
                "patch_error": "",
                "revision_plan": {
                    "edits": [
                        {
                            "block_id": "hero",
                            "slot_id": "body",
                            "scope": "slot",
                            "change_request": "update hero body",
                            "preserve_requirements": ["keep title"],
                            "acceptance_hint": "hero body changes",
                        }
                    ]
                },
                "targeted_replacement_plan": {
                    "replacements": [
                        {
                            "block_id": "hero",
                            "slot_id": "body",
                            "scope": "slot",
                            "html": "<p>New hero body</p>",
                        }
                    ],
                    "fallback_blocks": [],
                },
            }

            result = patch_executor_node(state)

            self.assertEqual("", result.get("patch_error"))
            updated_html = Path(artifact.entry_html).read_text(encoding="utf-8")
            self.assertIn("New hero body", updated_html)
            self.assertIn("Keep me", updated_html)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_patch_agent_salvages_valid_replacement_when_style_change_is_empty(self) -> None:
        root, page_plan, structured_paper, artifact = self._setup_workspace()
        try:
            state = {
                "paper_folder_name": "demo-paper",
                "page_plan": page_plan,
                "structured_paper": structured_paper,
                "coder_artifact": artifact,
                "patch_error": "",
                "revision_plan": {
                    "edits": [
                        {
                            "block_id": "hero",
                            "slot_id": "body",
                            "scope": "slot",
                            "change_request": "update hero body",
                            "preserve_requirements": ["keep title"],
                            "acceptance_hint": "hero body changes",
                        }
                    ]
                },
            }

            response_payload = {
                "replacements": [
                    {
                        "block_id": "hero",
                        "slot_id": "body",
                        "scope": "slot",
                        "html": "<p>Updated abstract body</p>",
                    }
                ],
                "style_changes": [
                    {
                        "block_id": "hero",
                        "slot_id": "body",
                        "scope": "slot",
                        "declarations": {},
                    }
                ],
                "attribute_changes": [],
                "override_css_rules": [],
                "fallback_blocks": [],
            }

            class _FakeLLM:
                def invoke(self, _: Any) -> Any:
                    return type("Response", (), {"content": json.dumps(response_payload)})()

            with patch("src.patching.patch_pipeline.get_llm", return_value=_FakeLLM()):
                result = patch_agent_node(state)

            self.assertEqual("", result.get("patch_error"))
            targeted_plan = result.get("targeted_replacement_plan")
            self.assertIsNotNone(targeted_plan)
            self.assertEqual(1, len(targeted_plan.replacements))
            self.assertEqual(0, len(targeted_plan.style_changes))
            self.assertIn("replacements=1", str(result.get("patch_agent_output") or ""))
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_patch_executor_falls_back_to_block_regeneration(self) -> None:
        root, page_plan, structured_paper, artifact = self._setup_workspace()
        try:
            state = {
                "paper_folder_name": "demo-paper",
                "page_plan": page_plan,
                "structured_paper": structured_paper,
                "coder_artifact": artifact,
                "patch_error": "",
                "revision_plan": {
                    "edits": [
                        {
                            "block_id": "hero",
                            "slot_id": "summary",
                            "scope": "slot",
                            "change_request": "add summary",
                            "preserve_requirements": ["keep title"],
                            "acceptance_hint": "summary appears",
                        }
                    ]
                },
                "targeted_replacement_plan": {
                    "replacements": [
                        {
                            "block_id": "hero",
                            "slot_id": "summary",
                            "scope": "slot",
                            "html": "<p>Summary</p>",
                        }
                    ],
                    "fallback_blocks": [{"block_id": "hero", "reason": "slot missing"}],
                },
            }

            regenerated_html = (
                '<section id="hero" data-pa-block="hero">'
                '<h1 data-pa-slot="title">Old Title</h1>'
                '<div data-pa-slot="summary"><p>Summary</p></div>'
                '<div data-pa-slot="body"><p>Old hero body</p></div>'
                "</section>"
            )
            with patch("src.patching.patch_pipeline._regenerate_block_html", return_value=regenerated_html):
                result = patch_executor_node(state)

            self.assertEqual("", result.get("patch_error"))
            updated_html = Path(artifact.entry_html).read_text(encoding="utf-8")
            self.assertIn('data-pa-slot="summary"', updated_html)
            self.assertIn("Summary", updated_html)
            self.assertIn("Keep me", updated_html)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_patch_executor_applies_global_replacement(self) -> None:
        root, page_plan, structured_paper, artifact = self._setup_workspace(include_globals=True)
        try:
            state = {
                "paper_folder_name": "demo-paper",
                "page_plan": page_plan,
                "structured_paper": structured_paper,
                "coder_artifact": artifact,
                "patch_error": "",
                "revision_plan": {
                    "edits": [
                        {
                            "global_id": "header_primary_action",
                            "scope": "global",
                            "change_request": "rename the top button",
                            "preserve_requirements": ["keep header layout"],
                            "acceptance_hint": "button text updates",
                        }
                    ]
                },
                "targeted_replacement_plan": {
                    "replacements": [
                        {
                            "global_id": "header_primary_action",
                            "scope": "global",
                            "html": 'Read Paper',
                        }
                    ],
                    "style_changes": [],
                    "attribute_changes": [
                        {
                            "global_id": "header_primary_action",
                            "scope": "global",
                            "attributes": {
                                "href": "https://example.com/paper",
                                "target": "_blank",
                            },
                        }
                    ],
                    "override_css_rules": [],
                    "fallback_blocks": [],
                },
            }

            result = patch_executor_node(state)

            self.assertEqual("", result.get("patch_error"))
            updated_html = Path(artifact.entry_html).read_text(encoding="utf-8")
            self.assertIn("Read Paper", updated_html)
            self.assertIn('data-pa-global="header_primary_action"', updated_html)
            self.assertIn('href="https://example.com/paper"', updated_html)
            self.assertIn('target="_blank"', updated_html)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_patch_executor_applies_block_id_attribute_change(self) -> None:
        root, page_plan, structured_paper, artifact = self._setup_workspace()
        try:
            page_plan = page_plan.model_copy(
                deep=True,
                update={
                    "blocks": [
                        block.model_copy(
                            deep=True,
                            update={
                                "shell_contract": block.shell_contract.model_copy(
                                    deep=True,
                                    update={"preserve_ids": []},
                                )
                                if block.block_id == "results" and block.shell_contract is not None
                                else block.shell_contract
                            },
                        )
                        for block in page_plan.blocks
                    ]
                },
            )
            state = {
                "paper_folder_name": "demo-paper",
                "page_plan": page_plan,
                "structured_paper": structured_paper,
                "coder_artifact": artifact,
                "patch_error": "",
                "revision_plan": {
                    "edits": [
                        {
                            "block_id": "results",
                            "scope": "block",
                            "change_request": "make this section linkable from the header",
                            "preserve_requirements": ["keep the current shell"],
                            "acceptance_hint": "the results block has a stable in-page anchor",
                        }
                    ]
                },
                "targeted_replacement_plan": {
                    "replacements": [],
                    "style_changes": [],
                    "attribute_changes": [
                        {
                            "block_id": "results",
                            "scope": "block",
                            "attributes": {
                                "id": "experimental_evaluation",
                            },
                        }
                    ],
                    "override_css_rules": [],
                    "fallback_blocks": [],
                },
            }

            result = patch_executor_node(state)

            self.assertEqual("", result.get("patch_error"))
            updated_html = Path(artifact.entry_html).read_text(encoding="utf-8")
            self.assertIn('id="experimental_evaluation"', updated_html)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_patch_executor_applies_style_change_and_override_rule(self) -> None:
        root, page_plan, structured_paper, artifact = self._setup_workspace(include_globals=True)
        try:
            state = {
                "paper_folder_name": "demo-paper",
                "page_plan": page_plan,
                "structured_paper": structured_paper,
                "coder_artifact": artifact,
                "patch_error": "",
                "revision_plan": {
                    "edits": [
                        {
                            "block_id": "hero",
                            "slot_id": "body",
                            "scope": "slot",
                            "change_request": "increase body font size and spacing",
                            "preserve_requirements": ["keep copy"],
                            "acceptance_hint": "hero body is more readable",
                        }
                    ]
                },
                "targeted_replacement_plan": {
                    "replacements": [],
                    "style_changes": [
                        {
                            "block_id": "hero",
                            "slot_id": "body",
                            "scope": "slot",
                            "declarations": {
                                "font-size": "1.2rem",
                                "margin-bottom": "2rem",
                            },
                        }
                    ],
                    "override_css_rules": [
                        {
                            "selector": '[data-pa-block="hero"] p',
                            "declarations": {
                                "line-height": "1.8",
                            },
                        }
                    ],
                    "fallback_blocks": [],
                },
            }

            result = patch_executor_node(state)

            self.assertEqual("", result.get("patch_error"))
            updated_html = Path(artifact.entry_html).read_text(encoding="utf-8")
            self.assertIn("font-size: 1.2rem;", updated_html)
            self.assertIn("margin-bottom: 2rem;", updated_html)
            self.assertIn(REVISION_OVERRIDE_STYLE_TAG_ID, updated_html)
            self.assertIn('[data-pa-block="hero"] p', updated_html)
            self.assertIn("line-height: 1.8;", updated_html)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_patch_executor_rejects_invalid_local_paper_image_path(self) -> None:
        root, page_plan, structured_paper, artifact = self._setup_workspace(include_globals=True)
        try:
            state = {
                "paper_folder_name": "demo-paper",
                "page_plan": page_plan,
                "structured_paper": structured_paper,
                "coder_artifact": artifact,
                "patch_error": "",
                "revision_plan": {
                    "edits": [
                        {
                            "block_id": "hero",
                            "slot_id": "body",
                            "scope": "slot",
                            "change_request": "add image",
                            "preserve_requirements": [],
                            "acceptance_hint": "image appears",
                        }
                    ]
                },
                "targeted_replacement_plan": {
                    "replacements": [
                        {
                            "block_id": "hero",
                            "slot_id": "body",
                            "scope": "slot",
                            "html": '<p><img src="./assets/paper/missing.png" alt="missing"></p>',
                        }
                    ],
                    "style_changes": [],
                    "override_css_rules": [],
                    "fallback_blocks": [],
                },
            }

            result = patch_executor_node(state)

            self.assertIn("Post-revision asset validation failed", str(result.get("patch_error") or ""))
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_patch_executor_rejects_regenerated_block_that_breaks_shell_contract(self) -> None:
        root, page_plan, structured_paper, artifact = self._setup_workspace()
        try:
            state = {
                "paper_folder_name": "demo-paper",
                "page_plan": page_plan,
                "structured_paper": structured_paper,
                "coder_artifact": artifact,
                "patch_error": "",
                "revision_plan": {
                    "edits": [
                        {
                            "block_id": "hero",
                            "scope": "block",
                            "change_request": "rebuild hero",
                            "preserve_requirements": ["keep hero shell"],
                            "acceptance_hint": "hero is updated",
                        }
                    ]
                },
                "targeted_replacement_plan": {
                    "replacements": [],
                    "style_changes": [],
                    "attribute_changes": [],
                    "override_css_rules": [],
                    "fallback_blocks": [{"block_id": "hero", "reason": "force regen"}],
                },
            }

            bad_regenerated_html = (
                '<div data-pa-block="hero">'
                '<h1 data-pa-slot="title">Old Title</h1>'
                '<div data-pa-slot="body"><p>Broken shell</p></div>'
                '</div>'
            )
            with patch("src.patching.patch_pipeline._regenerate_block_html", return_value=bad_regenerated_html):
                result = patch_executor_node(state)

            self.assertIn("shell_contract", str(result.get("patch_error") or ""))
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_patch_executor_rejects_legacy_page_without_manifest(self) -> None:
        root, page_plan, structured_paper, artifact = self._setup_workspace()
        try:
            build_page_manifest_path(artifact.entry_html).unlink(missing_ok=True)
            state = {
                "paper_folder_name": "demo-paper",
                "page_plan": page_plan,
                "structured_paper": structured_paper,
                "coder_artifact": artifact,
                "patch_error": "",
                "revision_plan": {"edits": []},
                "targeted_replacement_plan": {"replacements": [], "fallback_blocks": []},
            }

            result = patch_executor_node(state)

            self.assertEqual(LEGACY_PAGE_ERROR, result.get("patch_error"))
        finally:
            shutil.rmtree(root, ignore_errors=True)


@unittest.skipIf(app is None, f"app import failed: {APP_IMPORT_ERROR}")
class ReviewFormatterTests(unittest.TestCase):
    def test_review_accordion_updates_follow_stage_defaults(self) -> None:
        overview_reader, overview_outline = app._review_accordion_updates("overview")
        outline_reader, outline_outline = app._review_accordion_updates("outline")
        webpage_reader, webpage_outline = app._review_accordion_updates("webpage")

        self.assertEqual(True, overview_reader.get("open"))
        self.assertEqual(False, overview_outline.get("open"))
        self.assertEqual(False, outline_reader.get("open"))
        self.assertEqual(True, outline_outline.get("open"))
        self.assertEqual(False, webpage_reader.get("open"))
        self.assertEqual(False, webpage_outline.get("open"))

    def test_page_plan_markdown_uses_outline_and_diagnostics(self) -> None:
        page_plan = _sample_page_plan("data/templates/demo")
        structured_paper = _sample_structured_paper()

        rendered = app.format_page_plan_to_markdown(
            page_plan.model_dump(),
            structured_paper.model_dump(),
        )

        self.assertIn("# Planned Webpage Outline", rendered)
        self.assertIn("### 1. Hero", rendered)
        self.assertIn("`block_id`: `hero`", rendered)
        self.assertIn("Unused source sections: None", rendered)

    def test_stage_action_updates_follow_stage_visibility_rules(self) -> None:
        outline_updates = app._stage_action_updates(
            "outline",
            feedback_text_value="Refine the sections",
            manual_layout_compose_enabled=True,
        )
        webpage_updates = app._stage_action_updates(
            "webpage",
            feedback_text_value="Tighten the spacing",
            feedback_images_value=["shot.png"],
        )
        hidden_updates = app._stage_action_updates("none")

        self.assertEqual(True, outline_updates[0].get("visible"))
        self.assertEqual("Refine the sections", outline_updates[1].get("value"))
        self.assertEqual(True, outline_updates[3].get("visible"))
        self.assertEqual(True, outline_updates[3].get("value"))
        self.assertEqual(False, outline_updates[2].get("visible"))
        self.assertEqual(False, outline_updates[4].get("visible"))
        self.assertEqual(True, outline_updates[6].get("visible"))
        self.assertEqual(True, outline_updates[7].get("visible"))

        self.assertEqual(True, webpage_updates[0].get("visible"))
        self.assertEqual(True, webpage_updates[2].get("visible"))
        self.assertEqual(False, webpage_updates[3].get("visible"))
        self.assertEqual(True, webpage_updates[8].get("visible"))
        self.assertEqual(True, webpage_updates[9].get("visible"))
        self.assertEqual(False, webpage_updates[6].get("visible"))

        self.assertEqual(False, hidden_updates[0].get("visible"))
        self.assertEqual(False, hidden_updates[4].get("visible"))
        self.assertEqual(False, hidden_updates[8].get("visible"))


@unittest.skipIf(app is None, f"app import failed: {APP_IMPORT_ERROR}")
class CoderPhaseStateTests(unittest.TestCase):
    def test_coder_phase_node_persists_shell_enriched_page_plan(self) -> None:
        resolved_page_plan = _sample_page_plan("data/templates/demo-template")
        unresolved_page_plan = resolved_page_plan.model_copy(
            update={
                "blocks": [
                    block.model_copy(update={"shell_contract": None}, deep=True)
                    for block in resolved_page_plan.blocks
                ]
            },
            deep=True,
        )
        artifact = CoderArtifact(
            site_dir="demo/site",
            entry_html="demo/site/index.html",
            selected_template_id="demo-template",
            copied_assets=[],
            edited_files=["index.html"],
            notes="demo",
        )
        state = {
            "paper_folder_name": "demo",
            "structured_paper": _sample_structured_paper(),
            "approved_page_plan": unresolved_page_plan,
            "page_plan": unresolved_page_plan,
            "coder_artifact": None,
            "human_directives": empty_human_feedback(),
            "coder_instructions": "",
        }

        with (
            patch.object(app, "run_coder_agent_with_diagnostics", return_value=(artifact, None, resolved_page_plan)),
            patch.object(
                app,
                "get_output_paths",
                return_value=(
                    Path("demo/structured.json"),
                    Path("demo/paper.md"),
                    Path("demo/page_plan.json"),
                    Path("demo/coder_artifact.json"),
                ),
            ),
            patch.object(app, "save_page_plan") as save_page_plan,
            patch.object(app, "save_coder_artifact") as save_coder_artifact,
        ):
            result = app.coder_phase_node(state)

        returned_page_plan = PagePlan.model_validate(result["page_plan"])
        self.assertTrue(all(block.shell_contract is not None for block in returned_page_plan.blocks))
        self.assertEqual(returned_page_plan, PagePlan.model_validate(result["approved_page_plan"]))
        self.assertEqual("demo/site/index.html", result["coder_artifact"].entry_html)
        save_page_plan.assert_called_once_with(Path("demo/page_plan.json"), resolved_page_plan)
        save_coder_artifact.assert_called_once()


@unittest.skipIf(app is None, f"app import failed: {APP_IMPORT_ERROR}")
class WorkflowPatchRoutingTests(unittest.TestCase):
    def _compose_session_payload(self) -> dict[str, Any]:
        return {
            "template_entry_html": "E:/demo/index.html",
            "template_preview_path": "",
            "blocks": [
                {
                    "block_id": "hero",
                    "title": "Hero",
                    "source_sections": ["Abstract"],
                    "current_order": 1,
                    "selected_selector_hint": "section.hero-shell",
                    "selected_figure_paths": [],
                    "section_options": [],
                    "figure_options": [],
                },
                {
                    "block_id": "results",
                    "title": "Results",
                    "source_sections": ["Results"],
                    "current_order": 2,
                    "selected_selector_hint": "section.results-shell",
                    "selected_figure_paths": [],
                    "section_options": [],
                    "figure_options": [],
                },
            ],
            "active_block_id": "hero",
            "validation_errors": [],
        }

    def _initial_state(self) -> dict[str, Any]:
        return {
            "paper_folder_name": "demo-paper",
            "user_constraints": {},
            "generation_constraints": {},
            "manual_layout_compose_enabled": False,
            "human_directives": empty_human_feedback(),
            "coder_instructions": "",
            "edit_intent": None,
            "edit_intent_reason": "",
            "patch_agent_output": "",
            "revision_plan": None,
            "targeted_replacement_plan": None,
            "patch_error": "",
            "paper_overview": "",
            "outline_overview": "",
            "is_approved": False,
            "is_outline_approved": False,
            "is_webpage_approved": False,
            "review_stage": "overview",
            "template_candidates": [],
            "selected_template": None,
            "template_profile": None,
            "template_profile_path": "",
            "template_compile_cache_hit": False,
            "block_render_artifacts": [],
            "structured_paper": None,
            "page_plan": None,
            "approved_page_plan": None,
            "coder_artifact": None,
            "shell_binding_review": None,
            "shell_manual_selection": None,
            "layout_compose_session": None,
            "layout_compose_update": None,
            "visual_smoke_report": None,
        }

    def _build_workflow(
        self,
        patch_agent_result: dict[str, Any],
        patch_executor_result: dict[str, Any] | None = None,
        layout_compose_prepare_results: list[dict[str, Any]] | None = None,
        coder_result: dict[str, Any] | None = None,
        translator_result: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, int]]:
        counts = {
            "reader": 0,
            "overview": 0,
            "template_compile": 0,
            "planner": 0,
            "outline": 0,
            "layout_compose_prepare": 0,
            "layout_compose_review": 0,
            "coder": 0,
            "translator": 0,
            "patch_agent": 0,
            "patch_executor": 0,
        }

        def reader_node(_: Any) -> dict[str, Any]:
            counts["reader"] += 1
            return {"structured_paper": {"title": "paper"}}

        def overview_node(_: Any) -> dict[str, Any]:
            counts["overview"] += 1
            return {"paper_overview": "overview", "review_stage": "overview"}

        def template_compile_node(_: Any) -> dict[str, Any]:
            counts["template_compile"] += 1
            return {
                "template_candidates": [
                    {
                        "template_id": "demo-template",
                        "root_dir": "data/templates/demo-template",
                        "chosen_entry_html": "index.html",
                        "score": 1.0,
                        "reasons": ["test"],
                    }
                ],
                "selected_template": {
                    "template_id": "demo-template",
                    "root_dir": "data/templates/demo-template",
                    "chosen_entry_html": "index.html",
                    "score": 1.0,
                    "reasons": ["test"],
                },
                "template_profile": {
                    "template_id": "demo-template",
                    "template_root_dir": "data/templates/demo-template",
                    "entry_html": "index.html",
                    "archetype": "generic_multi_section",
                    "global_preserve_selectors": [],
                    "shell_candidates": [],
                    "optional_widgets": [],
                    "removable_demo_selectors": [],
                    "unsafe_selectors": [],
                    "compile_confidence": 0.95,
                    "risk_flags": [],
                    "notes": [],
                    "source_fingerprint": "test",
                },
                "template_profile_path": "demo/template_profile.json",
                "template_compile_cache_hit": False,
            }

        def planner_node(_: Any) -> dict[str, Any]:
            counts["planner"] += 1
            return {"page_plan": {"template": "x"}}

        def outline_node(_: Any) -> dict[str, Any]:
            counts["outline"] += 1
            return {"outline_overview": "outline", "review_stage": "outline"}

        layout_compose_prepare_queue = iter(
            layout_compose_prepare_results
            or [
                {
                    "approved_page_plan": {"template": "x"},
                    "layout_compose_session": self._compose_session_payload(),
                    "layout_compose_update": None,
                }
            ]
        )

        def layout_compose_prepare_node(_: Any) -> dict[str, Any]:
            counts["layout_compose_prepare"] += 1
            try:
                return next(layout_compose_prepare_queue)
            except StopIteration:
                return {
                    "approved_page_plan": {"template": "x"},
                    "layout_compose_session": self._compose_session_payload(),
                    "layout_compose_update": None,
                }

        def layout_compose_review_node(_: Any) -> dict[str, Any]:
            counts["layout_compose_review"] += 1
            return {"review_stage": "layout_compose"}

        def coder_node(_: Any) -> dict[str, Any]:
            counts["coder"] += 1
            return coder_result or {
                "coder_artifact": f"artifact-{counts['coder']}",
                "patch_error": "",
                "visual_smoke_report": None,
            }

        def translator_node(_: Any) -> dict[str, Any]:
            counts["translator"] += 1
            return translator_result or {
                "revision_plan": {
                    "edits": [
                        {
                            "block_id": "hero",
                            "slot_id": "body",
                            "scope": "slot",
                            "change_request": "demo",
                            "preserve_requirements": [],
                            "acceptance_hint": "demo",
                        }
                    ]
                }
            }

        def fake_patch_agent_node(_: Any) -> dict[str, Any]:
            counts["patch_agent"] += 1
            return patch_agent_result

        def fake_patch_executor_node(_: Any) -> dict[str, Any]:
            counts["patch_executor"] += 1
            return patch_executor_result or {"coder_artifact": "patched-artifact", "patch_error": ""}

        with (
            patch.object(app, "reader_phase_node", reader_node),
            patch.object(app, "overview_node", overview_node),
            patch.object(app, "template_compile_phase_node", template_compile_node),
            patch.object(app, "planner_phase_node", planner_node),
            patch.object(app, "outline_review_node", outline_node),
            patch.object(app, "layout_compose_prepare_node", layout_compose_prepare_node),
            patch.object(app, "layout_compose_review_node", layout_compose_review_node),
            patch.object(app, "coder_phase_node", coder_node),
            patch.object(app, "translator_node", translator_node),
            patch.object(app, "patch_agent_node", fake_patch_agent_node),
            patch.object(app, "patch_executor_node", fake_patch_executor_node),
        ):
            workflow = app.build_hitl_workflow()

        return workflow, counts

    def _pause_at_outline_review(self, workflow: Any) -> dict[str, Any]:
        config = {"configurable": {"thread_id": f"test::{uuid4().hex}"}}
        workflow.invoke(self._initial_state(), config=config)
        overview_state = workflow.get_state(config)
        self.assertEqual("overview", overview_state.values.get("review_stage"))

        workflow.update_state(
            config,
            {
                "human_directives": empty_human_feedback(),
                "coder_instructions": "",
                "patch_agent_output": "",
                "revision_plan": None,
                "targeted_replacement_plan": None,
                "patch_error": "",
                "is_approved": True,
                "is_outline_approved": False,
                "is_webpage_approved": False,
                "shell_binding_review": None,
                "shell_manual_selection": None,
                "layout_compose_session": None,
                "layout_compose_update": None,
                "visual_smoke_report": None,
            },
            as_node="overview",
        )
        workflow.invoke(None, config=config)

        outline_state = workflow.get_state(config)
        self.assertEqual("outline", outline_state.values.get("review_stage"))
        return config

    def _pause_at_webpage_review(self, workflow: Any, *, manual_layout_compose_enabled: bool = True) -> dict[str, Any]:
        config = self._pause_at_outline_review(workflow)

        workflow.update_state(
            config,
            {
                "human_directives": empty_human_feedback(),
                "coder_instructions": "",
                "patch_agent_output": "",
                "revision_plan": None,
                "targeted_replacement_plan": None,
                "patch_error": "",
                "approved_page_plan": {"template": "x"},
                "manual_layout_compose_enabled": manual_layout_compose_enabled,
                "is_outline_approved": True,
                "is_webpage_approved": False,
                "shell_binding_review": None,
                "shell_manual_selection": None,
                "layout_compose_session": None,
                "layout_compose_update": None,
                "visual_smoke_report": None,
            },
            as_node="outline_review",
        )
        workflow.invoke(None, config=config)

        paused_state = workflow.get_state(config)
        if manual_layout_compose_enabled:
            self.assertEqual("layout_compose", paused_state.values.get("review_stage"))
            workflow.update_state(
                config,
                {
                    "layout_compose_session": paused_state.values.get("layout_compose_session") or self._compose_session_payload(),
                    "layout_compose_update": None,
                },
                as_node="layout_compose_review",
            )
            workflow.invoke(None, config=config)
            paused_state = workflow.get_state(config)

        self.assertEqual("webpage", paused_state.values.get("review_stage"))
        return config

    def test_workflow_pauses_at_outline_review_before_coder(self) -> None:
        workflow, counts = self._build_workflow(
            {
                "patch_agent_output": "",
                "targeted_replacement_plan": None,
                "patch_error": "",
            }
        )

        config = self._pause_at_outline_review(workflow)
        outline_state = workflow.get_state(config)

        self.assertEqual("outline", outline_state.values.get("review_stage"))
        self.assertEqual(1, counts["reader"])
        self.assertEqual(1, counts["overview"])
        self.assertEqual(1, counts["template_compile"])
        self.assertEqual(1, counts["planner"])
        self.assertEqual(1, counts["outline"])
        self.assertEqual(0, counts["layout_compose_prepare"])
        self.assertEqual(0, counts["coder"])

    def test_outline_approval_pauses_at_layout_compose_review(self) -> None:
        workflow, counts = self._build_workflow(
            {
                "patch_agent_output": "",
                "targeted_replacement_plan": None,
                "patch_error": "",
            },
            layout_compose_prepare_results=[
                {
                    "approved_page_plan": {"template": "x"},
                    "layout_compose_session": self._compose_session_payload(),
                    "layout_compose_update": None,
                }
            ],
        )
        config = self._pause_at_outline_review(workflow)

        workflow.update_state(
            config,
            {
                "human_directives": empty_human_feedback(),
                "coder_instructions": "",
                "patch_agent_output": "",
                "revision_plan": None,
                "targeted_replacement_plan": None,
                "patch_error": "",
                "approved_page_plan": {"template": "x"},
                "manual_layout_compose_enabled": True,
                "is_outline_approved": True,
                "is_webpage_approved": False,
                "shell_binding_review": None,
                "shell_manual_selection": None,
                "layout_compose_session": None,
                "layout_compose_update": None,
                "visual_smoke_report": None,
            },
            as_node="outline_review",
        )
        workflow.invoke(None, config=config)

        compose_state = workflow.get_state(config)
        self.assertEqual("layout_compose", compose_state.values.get("review_stage"))
        self.assertEqual(1, counts["layout_compose_prepare"])
        self.assertEqual(1, counts["layout_compose_review"])
        self.assertEqual(0, counts["coder"])

    def test_outline_approval_without_manual_compose_reaches_webpage_review(self) -> None:
        workflow, counts = self._build_workflow(
            {
                "patch_agent_output": "",
                "targeted_replacement_plan": None,
                "patch_error": "",
            }
        )
        config = self._pause_at_outline_review(workflow)

        workflow.update_state(
            config,
            {
                "human_directives": empty_human_feedback(),
                "coder_instructions": "",
                "patch_agent_output": "",
                "revision_plan": None,
                "targeted_replacement_plan": None,
                "patch_error": "",
                "approved_page_plan": {"template": "x"},
                "manual_layout_compose_enabled": False,
                "is_outline_approved": True,
                "is_webpage_approved": False,
                "shell_binding_review": None,
                "shell_manual_selection": None,
                "layout_compose_session": None,
                "layout_compose_update": None,
                "visual_smoke_report": None,
            },
            as_node="outline_review",
        )
        workflow.invoke(None, config=config)

        webpage_state = workflow.get_state(config)
        self.assertEqual("webpage", webpage_state.values.get("review_stage"))
        self.assertEqual(0, counts["layout_compose_prepare"])
        self.assertEqual(0, counts["layout_compose_review"])
        self.assertEqual(1, counts["coder"])

    def test_layout_compose_resume_reaches_webpage_after_confirmation(self) -> None:
        workflow, counts = self._build_workflow(
            {
                "patch_agent_output": "",
                "targeted_replacement_plan": None,
                "patch_error": "",
            },
            layout_compose_prepare_results=[
                {
                    "approved_page_plan": {"template": "x"},
                    "layout_compose_session": self._compose_session_payload(),
                    "layout_compose_update": None,
                },
            ],
        )
        config = self._pause_at_outline_review(workflow)

        workflow.update_state(
            config,
            {
                "human_directives": empty_human_feedback(),
                "coder_instructions": "",
                "patch_agent_output": "",
                "revision_plan": None,
                "targeted_replacement_plan": None,
                "patch_error": "",
                "approved_page_plan": {"template": "x"},
                "manual_layout_compose_enabled": True,
                "is_outline_approved": True,
                "is_webpage_approved": False,
                "shell_binding_review": None,
                "shell_manual_selection": None,
                "layout_compose_session": None,
                "layout_compose_update": None,
                "visual_smoke_report": None,
            },
            as_node="outline_review",
        )
        workflow.invoke(None, config=config)
        compose_state = workflow.get_state(config)
        self.assertEqual("layout_compose", compose_state.values.get("review_stage"))

        workflow.update_state(
            config,
            {
                "layout_compose_session": compose_state.values.get("layout_compose_session") or self._compose_session_payload(),
                "layout_compose_update": None,
            },
            as_node="layout_compose_review",
        )
        workflow.invoke(None, config=config)

        webpage_state = workflow.get_state(config)
        self.assertEqual("webpage", webpage_state.values.get("review_stage"))
        self.assertEqual(1, counts["layout_compose_prepare"])
        self.assertEqual(1, counts["layout_compose_review"])
        self.assertEqual(1, counts["coder"])

    def test_structural_visual_smoke_routes_back_to_planner_instead_of_webpage_review(self) -> None:
        workflow, counts = self._build_workflow(
            {
                "patch_agent_output": "",
                "targeted_replacement_plan": None,
                "patch_error": "",
            },
            layout_compose_prepare_results=[
                {
                    "approved_page_plan": {"template": "x"},
                    "layout_compose_session": self._compose_session_payload(),
                    "layout_compose_update": None,
                },
            ],
            coder_result={
                "coder_artifact": "artifact-1",
                "patch_error": "",
                "visual_smoke_report": {
                    "passed": False,
                    "issue_class": "structure",
                    "suggested_recovery": "rerun_planner",
                    "issues": ["hero structure does not match the selected template"],
                    "selectors_to_remove": [],
                    "css_rules_to_inject": [],
                },
            },
        )
        config = self._pause_at_outline_review(workflow)

        workflow.update_state(
            config,
            {
                "human_directives": empty_human_feedback(),
                "coder_instructions": "",
                "patch_agent_output": "",
                "revision_plan": None,
                "targeted_replacement_plan": None,
                "patch_error": "",
                "approved_page_plan": {"template": "x"},
                "manual_layout_compose_enabled": True,
                "is_outline_approved": True,
                "is_webpage_approved": False,
                "shell_binding_review": None,
                "shell_manual_selection": None,
                "layout_compose_session": None,
                "layout_compose_update": None,
                "visual_smoke_report": None,
            },
            as_node="outline_review",
        )
        workflow.invoke(None, config=config)
        compose_state = workflow.get_state(config)
        self.assertEqual("layout_compose", compose_state.values.get("review_stage"))

        workflow.update_state(
            config,
            {
                "layout_compose_session": compose_state.values.get("layout_compose_session") or self._compose_session_payload(),
                "layout_compose_update": None,
            },
            as_node="layout_compose_review",
        )
        workflow.invoke(None, config=config)

        rerouted_state = workflow.get_state(config)
        self.assertEqual("outline", rerouted_state.values.get("review_stage"))
        self.assertEqual(2, counts["planner"])
        self.assertEqual(1, counts["coder"])
        self.assertEqual(0, counts["translator"])

    def test_revision_flow_calls_patch_executor_after_patch_agent(self) -> None:
        workflow, counts = self._build_workflow(
            {
                "patch_agent_output": "replacements=1; fallback_blocks=0",
                "targeted_replacement_plan": {
                    "replacements": [
                        {
                            "block_id": "hero",
                            "slot_id": "body",
                            "scope": "slot",
                            "html": "<p>Updated</p>",
                        }
                    ],
                    "fallback_blocks": [],
                },
                "patch_error": "",
            }
        )
        config = self._pause_at_webpage_review(workflow)

        workflow.update_state(
            config,
            {
                "human_directives": {"text": "fix hero body", "images": []},
                "coder_instructions": "",
                "patch_agent_output": "",
                "revision_plan": None,
                "targeted_replacement_plan": None,
                "patch_error": "",
                "is_webpage_approved": False,
            },
            as_node="webpage_review",
        )
        workflow.invoke(None, config=config)

        self.assertEqual(1, counts["translator"])
        self.assertEqual(1, counts["patch_agent"])
        self.assertEqual(1, counts["patch_executor"])
        self.assertEqual(1, counts["coder"])

    def test_patch_executor_runs_even_when_patch_agent_sets_safe_fail(self) -> None:
        workflow, counts = self._build_workflow(
            {
                "patch_agent_output": "",
                "targeted_replacement_plan": None,
                "patch_error": LEGACY_PAGE_ERROR,
            },
            patch_executor_result={"patch_error": LEGACY_PAGE_ERROR},
        )
        config = self._pause_at_webpage_review(workflow)

        workflow.update_state(
            config,
            {
                "human_directives": {"text": "fix hero body", "images": []},
                "coder_instructions": "",
                "patch_agent_output": "",
                "revision_plan": None,
                "targeted_replacement_plan": None,
                "patch_error": "",
                "is_webpage_approved": False,
            },
            as_node="webpage_review",
        )
        workflow.invoke(None, config=config)

        state = workflow.get_state(config)
        self.assertEqual(LEGACY_PAGE_ERROR, state.values.get("patch_error"))
        self.assertEqual(1, counts["patch_executor"])

    def test_non_patch_feedback_is_intercepted_before_patch_agent(self) -> None:
        workflow, counts = self._build_workflow(
            {
                "patch_agent_output": "should-not-run",
                "targeted_replacement_plan": {
                    "replacements": [
                        {
                            "block_id": "hero",
                            "slot_id": "body",
                            "scope": "slot",
                            "html": "<p>Updated</p>",
                        }
                    ],
                    "fallback_blocks": [],
                },
                "patch_error": "",
            },
            translator_result={"revision_plan": {"edits": []}},
        )
        config = self._pause_at_webpage_review(workflow)

        workflow.update_state(
            config,
            {
                "human_directives": {"text": "overall redesign the page rhythm and switch template", "images": []},
                "coder_instructions": "",
                "edit_intent": None,
                "edit_intent_reason": "",
                "patch_agent_output": "",
                "revision_plan": None,
                "targeted_replacement_plan": None,
                "patch_error": "",
                "is_webpage_approved": False,
            },
            as_node="webpage_review",
        )
        workflow.invoke(None, config=config)

        state = workflow.get_state(config)
        self.assertEqual("non_patch", state.values.get("edit_intent"))
        self.assertIn("non_patch", str(state.values.get("patch_error") or ""))
        self.assertEqual(1, counts["translator"])
        self.assertEqual(0, counts["patch_agent"])
        self.assertEqual(0, counts["patch_executor"])









