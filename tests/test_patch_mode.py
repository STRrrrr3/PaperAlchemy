import shutil
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch
from uuid import uuid4

from src.agent_patch import LEGACY_PAGE_ERROR, patch_executor_node
from src.human_feedback import empty_human_feedback
from src.page_manifest import build_page_manifest_path, extract_page_manifest, save_page_manifest
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


def _sample_page_plan(selected_root_dir: str) -> PagePlan:
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
                    "a11y_notes": [],
                    "acceptance_checks": [],
                },
            ],
            "dom_mapping": {"#hero": "Hero", "#results": "Results"},
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


def _sample_html() -> str:
    return """<!DOCTYPE html>
<html>
  <head><title>Demo</title></head>
  <body>
    <!-- PaperAlchemy Generated Body Start -->
    <section data-pa-block="hero">
      <h1 data-pa-slot="title">Old Title</h1>
      <div data-pa-slot="body"><p>Old hero body</p></div>
    </section>
    <section data-pa-block="results">
      <h2 data-pa-slot="title">Results</h2>
      <div data-pa-slot="body"><p>Keep me</p></div>
    </section>
    <!-- PaperAlchemy Generated Body End -->
  </body>
</html>
"""


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


class PatchExecutorTests(unittest.TestCase):
    def _setup_workspace(self) -> tuple[Path, PagePlan, StructuredPaper, CoderArtifact]:
        temp_root = Path.cwd() / "data" / "output" / "_tmp_test_workspace"
        temp_root.mkdir(parents=True, exist_ok=True)
        root = temp_root / uuid4().hex
        root.mkdir(parents=True, exist_ok=False)
        site_dir = root / "output" / "site"
        site_dir.mkdir(parents=True, exist_ok=True)
        entry_html = site_dir / "index.html"
        entry_html.write_text(_sample_html(), encoding="utf-8")

        template_dir = root / "templates" / "demo-template"
        template_dir.mkdir(parents=True, exist_ok=True)
        (template_dir / "index.html").write_text("<html><body>template</body></html>", encoding="utf-8")

        project_root = Path.cwd().resolve()
        selected_root_dir = str(template_dir.resolve().relative_to(project_root)).replace("\\", "/")
        page_plan = _sample_page_plan(selected_root_dir)
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
            notes="v5-anchored-llm-render",
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
                '<section data-pa-block="hero">'
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
            "is_approved": False,
            "is_webpage_approved": False,
            "review_stage": "overview",
            "structured_paper": None,
            "page_plan": None,
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
            patch.object(app, "coder_phase_node", coder_node),
            patch.object(app, "translator_node", translator_node),
            patch.object(app, "patch_agent_node", fake_patch_agent_node),
            patch.object(app, "patch_executor_node", fake_patch_executor_node),
        ):
            workflow = app.build_hitl_workflow()

        return workflow, counts

    def _pause_at_webpage_review(self, workflow: Any) -> dict[str, Any]:
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
                "is_webpage_approved": False,
            },
            as_node="overview",
        )
        workflow.invoke(None, config=config)

        webpage_state = workflow.get_state(config)
        self.assertEqual("webpage", webpage_state.values.get("review_stage"))
        return config

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





