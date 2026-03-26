import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

from bs4 import BeautifulSoup

from src import agent_coder, agent_planner
from src.human_feedback import empty_human_feedback
from src.schemas import PagePlan, StructuredPaper, TemplateCandidate
from src.template_compile import compile_template_profile, hydrate_template_candidate_from_root


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
                "plan_version": "1.1",
                "planning_mode": "hybrid_template_bind",
                "target_framework": "static-html",
                "confidence": 0.9,
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
                        "selector_hint": "section.missing-hero",
                        "region_role": "hero",
                        "operation": "replace_text",
                    },
                    "component_recipe": [
                        {"slot": "content", "reason": "demo"}
                    ],
                    "content_contract": {
                        "headline": "Hero",
                        "body_points": ["intro"],
                        "cta": None,
                    },
                    "asset_binding": {"figure_paths": [], "template_asset_fallback": None},
                    "interaction": {"pattern": "none", "behavior_note": "demo"},
                    "responsive_rules": {"mobile_order": 1, "desktop_layout": "stack"},
                    "shell_contract": None,
                    "a11y_notes": [],
                    "acceptance_checks": [],
                },
                {
                    "block_id": "results",
                    "target_template_region": {
                        "selector_hint": "section.missing-results",
                        "region_role": "section",
                        "operation": "replace_text",
                    },
                    "component_recipe": [
                        {"slot": "content", "reason": "demo"}
                    ],
                    "content_contract": {
                        "headline": "Results",
                        "body_points": ["evidence"],
                        "cta": None,
                    },
                    "asset_binding": {"figure_paths": [], "template_asset_fallback": None},
                    "interaction": {"pattern": "none", "behavior_note": "demo"},
                    "responsive_rules": {"mobile_order": 2, "desktop_layout": "stack"},
                    "shell_contract": None,
                    "a11y_notes": [],
                    "acceptance_checks": [],
                },
            ],
            "dom_mapping": {},
            "selectors_to_remove": [".demo-copy"],
            "coder_handoff": {
                "implementation_order": ["index.html"],
                "file_touch_plan": [{"path": "index.html", "action": "edit", "reason": "demo"}],
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


class TemplateCompileTests(unittest.TestCase):
    def _workspace_tmp_dir(self, name: str) -> Path:
        temp_root = Path(__file__).resolve().parent / "_tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        target = temp_root / name
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        target.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(target, ignore_errors=True))
        return target

    def test_compile_profile_cache_hit_and_invalidation(self) -> None:
        project_root = self._workspace_tmp_dir("compile_cache_case")
        template_root = project_root / "templates" / "demo-template"
        template_root.mkdir(parents=True, exist_ok=True)
        (template_root / "index.html").write_text(
            "<!DOCTYPE html><html><body><header><nav></nav></header><section class='hero'></section><footer></footer></body></html>",
            encoding="utf-8",
        )
        candidate = hydrate_template_candidate_from_root(
            project_root=project_root,
            template_root=template_root,
            template_id="demo-template",
        )

        profile_a, cache_path_a, cache_hit_a = compile_template_profile(
            project_root=project_root,
            candidate=candidate,
            allow_llm=False,
        )
        profile_b, cache_path_b, cache_hit_b = compile_template_profile(
            project_root=project_root,
            candidate=candidate,
            allow_llm=False,
        )
        self.assertFalse(cache_hit_a)
        self.assertTrue(cache_hit_b)
        self.assertEqual(cache_path_a, cache_path_b)

        (template_root / "index.html").write_text(
            "<!DOCTYPE html><html><body><header><nav></nav></header><section class='hero'></section><section class='results'></section><footer></footer></body></html>",
            encoding="utf-8",
        )
        profile_c, cache_path_c, cache_hit_c = compile_template_profile(
            project_root=project_root,
            candidate=candidate,
            allow_llm=False,
        )
        self.assertFalse(cache_hit_c)
        self.assertNotEqual(profile_a.source_fingerprint, profile_c.source_fingerprint)
        self.assertNotEqual(cache_path_a, cache_path_c)

    def test_compile_profile_detects_expected_archetypes_and_safe_selectors(self) -> None:
        cases = [
            (
                "hero_bulma",
                "<!DOCTYPE html><html><body><section class='hero is-fullheight'><div class='hero-body'>Hero</div></section></body></html>",
                "body { } /* bulma */",
            ),
            (
                "bootstrap_navbar",
                "<!DOCTYPE html><html><body><nav class='navbar navbar-expand-lg'></nav><main><section class='content'></section></main></body></html>",
                "/* bootstrap */ .navbar{}",
            ),
            (
                "single_column_article",
                "<!DOCTYPE html><html><body><article class='prose'><section class='content'></section></article></body></html>",
                ".prose{max-width:70ch;}",
            ),
            (
                "chart_fetch_dashboard",
                "<!DOCTYPE html><html><body><div class='dashboard'><canvas class='chart'></canvas></div></body></html>",
                ".dashboard{display:grid;}",
            ),
        ]

        project_root = self._workspace_tmp_dir("compile_archetype_case")
        for expected_archetype, html_text, css_text in cases:
            template_root = project_root / expected_archetype
            template_root.mkdir(parents=True, exist_ok=True)
            (template_root / "index.html").write_text(html_text, encoding="utf-8")
            (template_root / "style.css").write_text(css_text, encoding="utf-8")
            if expected_archetype == "chart_fetch_dashboard":
                (template_root / "app.js").write_text("fetch('/api');", encoding="utf-8")

            candidate = TemplateCandidate(
                template_id=expected_archetype,
                root_dir=str(template_root.relative_to(project_root)).replace("\\", "/"),
                chosen_entry_html="index.html",
                score=1.0,
                reasons=["test"],
            )
            profile, _, _ = compile_template_profile(
                project_root=project_root,
                candidate=candidate,
                allow_llm=False,
            )

            self.assertEqual(expected_archetype, profile.archetype)
            soup = BeautifulSoup(html_text, "html.parser")
            for shell_candidate in profile.shell_candidates:
                self.assertEqual(1, len(soup.select(shell_candidate.selector)))
            self.assertFalse(set(profile.global_preserve_selectors) & set(profile.removable_demo_selectors))
            self.assertFalse(set(profile.global_preserve_selectors) & set(profile.unsafe_selectors))


class PlannerRefactorTests(unittest.TestCase):
    def test_low_confidence_template_forces_legacy_render_strategy(self) -> None:
        structured_paper = _sample_structured_paper()
        candidate = TemplateCandidate(
            template_id="demo-template",
            root_dir="templates/demo-template",
            chosen_entry_html="index.html",
            score=1.0,
            reasons=["test"],
        )
        template_profile = agent_planner.TemplateProfile.model_validate(
            {
                "template_id": "demo-template",
                "template_root_dir": "templates/demo-template",
                "entry_html": "index.html",
                "archetype": "generic_multi_section",
                "global_preserve_selectors": ["header", "footer"],
                "shell_candidates": [
                    {
                        "selector": "section.hero-shell",
                        "role": "hero",
                        "root_tag": "section",
                        "required_classes": ["hero-shell"],
                        "preserve_ids": [],
                        "wrapper_chain": [],
                        "dom_index": 1,
                        "confidence": 0.9,
                        "signals": ["hero_token"],
                    },
                    {
                        "selector": "section.results-shell",
                        "role": "section",
                        "root_tag": "section",
                        "required_classes": ["results-shell"],
                        "preserve_ids": [],
                        "wrapper_chain": [],
                        "dom_index": 2,
                        "confidence": 0.88,
                        "signals": ["section"],
                    },
                ],
                "optional_widgets": [],
                "removable_demo_selectors": [],
                "unsafe_selectors": [],
                "compile_confidence": 0.42,
                "risk_flags": [],
                "notes": [],
                "source_fingerprint": "demo",
            }
        )
        planner_result = _sample_page_plan("templates/demo-template")

        class _FakeStructuredLLM:
            def invoke(self, _: list[object]) -> PagePlan:
                return planner_result

        class _FakeLLM:
            def with_structured_output(self, _: object) -> _FakeStructuredLLM:
                return _FakeStructuredLLM()

        with patch.object(agent_planner, "get_llm", return_value=_FakeLLM()):
            result = agent_planner.unified_planner_node(
                {
                    "structured_paper": structured_paper,
                    "previous_page_plan": None,
                    "template_catalog": [
                        {
                            "template_id": "demo-template",
                            "root_dir": "templates/demo-template",
                            "entry_html_candidates": ["index.html"],
                        }
                    ],
                    "template_link_map": {},
                    "module_index": {},
                    "generation_constraints": {},
                    "user_constraints": {},
                    "human_directives": empty_human_feedback(),
                    "template_candidates": [candidate],
                    "selected_template": candidate,
                    "template_profile": template_profile,
                    "planner_feedback_history": [],
                    "page_plan": None,
                    "planner_critic_passed": False,
                    "planner_retry_count": 0,
                }
            )

        page_plan = PagePlan.model_validate(result["page_plan"])
        self.assertEqual("legacy_fullpage", page_plan.plan_meta.render_strategy)
        self.assertEqual("section.hero-shell", page_plan.blocks[0].target_template_region.selector_hint)
        self.assertEqual("section.results-shell", page_plan.blocks[1].target_template_region.selector_hint)
        self.assertEqual({"header", "footer"}, set(page_plan.dom_mapping))


class CoderAssemblyTests(unittest.TestCase):
    def test_page_assembler_preserves_global_shell_and_outline_order(self) -> None:
        page_plan = _sample_page_plan("templates/demo-template").model_copy(
            update={
                "blocks": [
                    _sample_page_plan("templates/demo-template").blocks[0].model_copy(
                        update={
                            "target_template_region": _sample_page_plan("templates/demo-template").blocks[0].target_template_region.model_copy(
                                update={"selector_hint": "section.hero-shell"},
                                deep=True,
                            ),
                            "shell_contract": {
                                "root_tag": "section",
                                "required_classes": ["hero-shell"],
                                "preserve_ids": [],
                                "wrapper_chain": [],
                                "actionable_root_selector": "section.hero-shell",
                            },
                        },
                        deep=True,
                    ),
                    _sample_page_plan("templates/demo-template").blocks[1].model_copy(
                        update={
                            "target_template_region": _sample_page_plan("templates/demo-template").blocks[1].target_template_region.model_copy(
                                update={"selector_hint": "section.results-shell"},
                                deep=True,
                            ),
                            "shell_contract": {
                                "root_tag": "section",
                                "required_classes": ["results-shell"],
                                "preserve_ids": [],
                                "wrapper_chain": [],
                                "actionable_root_selector": "section.results-shell",
                            },
                        },
                        deep=True,
                    ),
                ],
                "dom_mapping": {"header": "preserve_global_anchor", "footer": "preserve_global_anchor"},
            },
            deep=True,
        )
        template_profile = agent_coder.TemplateProfile.model_validate(
            {
                "template_id": "demo-template",
                "template_root_dir": "templates/demo-template",
                "entry_html": "index.html",
                "archetype": "generic_multi_section",
                "global_preserve_selectors": ["header", "footer"],
                "shell_candidates": [
                    {
                        "selector": "section.hero-shell",
                        "role": "hero",
                        "root_tag": "section",
                        "required_classes": ["hero-shell"],
                        "preserve_ids": [],
                        "wrapper_chain": [],
                        "dom_index": 1,
                        "confidence": 0.9,
                        "signals": ["hero_token"],
                    },
                    {
                        "selector": "section.results-shell",
                        "role": "section",
                        "root_tag": "section",
                        "required_classes": ["results-shell"],
                        "preserve_ids": [],
                        "wrapper_chain": [],
                        "dom_index": 2,
                        "confidence": 0.88,
                        "signals": ["section"],
                    },
                ],
                "optional_widgets": [],
                "removable_demo_selectors": [".demo-copy"],
                "unsafe_selectors": [],
                "compile_confidence": 0.95,
                "risk_flags": [],
                "notes": [],
                "source_fingerprint": "demo",
            }
        )
        template_html = """<!DOCTYPE html>
<html>
  <body>
    <header><nav>Keep nav</nav></header>
    <section class="hero-shell"><p class="demo-copy">demo</p></section>
    <section class="results-shell"></section>
    <footer>Keep footer</footer>
  </body>
</html>"""
        block_artifacts = [
            agent_coder.BlockRenderArtifact(
                block_id="hero",
                order=1,
                selector="section.hero-shell",
                html='<section class="hero-shell" data-pa-block="hero"><h1 data-pa-slot="title">Hero</h1></section>',
            ),
            agent_coder.BlockRenderArtifact(
                block_id="results",
                order=2,
                selector="section.results-shell",
                html='<section class="results-shell" data-pa-block="results"><div data-pa-slot="body">Results</div></section>',
            ),
        ]

        assembled = agent_coder._assemble_page(
            page_plan=page_plan,
            template_profile=template_profile,
            template_reference_html=template_html,
            block_artifacts=block_artifacts,
        )

        self.assertIn("<header>", assembled)
        self.assertIn("<footer", assembled)
        self.assertNotIn("demo-copy", assembled)
        self.assertLess(assembled.index('data-pa-block="hero"'), assembled.index('data-pa-block="results"'))


if __name__ == "__main__":
    unittest.main()
