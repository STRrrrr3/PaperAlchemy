import shutil
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch
from uuid import uuid4

from src.agent_patch import LEGACY_PAGE_ERROR, patch_executor_node
from src.human_feedback import empty_human_feedback
from src.page_manifest import (
    annotate_global_anchors,
    build_page_manifest_path,
    extract_page_manifest,
    save_page_manifest,
)
from src.page_validation import REVISION_OVERRIDE_STYLE_TAG_ID
from src.schemas import CoderArtifact, PagePlan, StructuredPaper

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
            with patch("src.agent_patch._regenerate_block_html", return_value=regenerated_html):
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
            with patch("src.agent_patch._regenerate_block_html", return_value=bad_regenerated_html):
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


@unittest.skipIf(app is None, f"app import failed: {APP_IMPORT_ERROR}")
class WorkflowPatchRoutingTests(unittest.TestCase):
    def _initial_state(self) -> dict[str, Any]:
        return {
            "paper_folder_name": "demo-paper",
            "user_constraints": {},
            "generation_constraints": {},
            "human_directives": empty_human_feedback(),
            "coder_instructions": "",
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
            "structured_paper": None,
            "page_plan": None,
            "approved_page_plan": None,
            "coder_artifact": None,
        }

    def _build_workflow(
        self,
        patch_agent_result: dict[str, Any],
        patch_executor_result: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, int]]:
        counts = {
            "reader": 0,
            "overview": 0,
            "planner": 0,
            "outline": 0,
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

        def planner_node(_: Any) -> dict[str, Any]:
            counts["planner"] += 1
            return {"page_plan": {"template": "x"}}

        def outline_node(_: Any) -> dict[str, Any]:
            counts["outline"] += 1
            return {"outline_overview": "outline", "review_stage": "outline"}

        def coder_node(_: Any) -> dict[str, Any]:
            counts["coder"] += 1
            return {"coder_artifact": f"artifact-{counts['coder']}", "patch_error": ""}

        def translator_node(_: Any) -> dict[str, Any]:
            counts["translator"] += 1
            return {
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
            patch.object(app, "planner_phase_node", planner_node),
            patch.object(app, "outline_review_node", outline_node),
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
            },
            as_node="overview",
        )
        workflow.invoke(None, config=config)

        outline_state = workflow.get_state(config)
        self.assertEqual("outline", outline_state.values.get("review_stage"))
        return config

    def _pause_at_webpage_review(self, workflow: Any) -> dict[str, Any]:
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
                "is_outline_approved": True,
                "is_webpage_approved": False,
            },
            as_node="outline_review",
        )
        workflow.invoke(None, config=config)

        webpage_state = workflow.get_state(config)
        self.assertEqual("webpage", webpage_state.values.get("review_stage"))
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
        self.assertEqual(1, counts["planner"])
        self.assertEqual(1, counts["outline"])
        self.assertEqual(0, counts["coder"])

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









