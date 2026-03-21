import unittest
from typing import Any
from unittest.mock import patch
from uuid import uuid4

from src.agent_patch import (
    FULL_REGENERATE_REQUIRED,
    PatchApplicationError,
    PatchParseError,
    apply_patch_output,
    build_patch_operations,
    parse_search_replace_blocks,
)
from src.human_feedback import empty_human_feedback

try:
    import app
except Exception as exc:  # pragma: no cover - skip when optional UI deps are unavailable
    app = None
    APP_IMPORT_ERROR = exc
else:
    APP_IMPORT_ERROR = None


class PatchModeTests(unittest.TestCase):
    def test_parse_single_block(self) -> None:
        patch_text = (
            "<<<<<<< SEARCH\n"
            "<div>Old</div>\n"
            "=======\n"
            "<div>New</div>\n"
            ">>>>>>> REPLACE"
        )

        blocks = parse_search_replace_blocks(patch_text)

        self.assertEqual(1, len(blocks))
        self.assertEqual("<div>Old</div>", blocks[0].search)
        self.assertEqual("<div>New</div>", blocks[0].replace)

    def test_parse_multiple_blocks(self) -> None:
        patch_text = (
            "<<<<<<< SEARCH\n"
            "<div>One</div>\n"
            "=======\n"
            "<div>Alpha</div>\n"
            ">>>>>>> REPLACE\n\n"
            "<<<<<<< SEARCH\n"
            "<div>Two</div>\n"
            "=======\n"
            "<div>Beta</div>\n"
            ">>>>>>> REPLACE"
        )

        blocks = parse_search_replace_blocks(patch_text)

        self.assertEqual(2, len(blocks))
        self.assertEqual("<div>One</div>", blocks[0].search)
        self.assertEqual("<div>Beta</div>", blocks[1].replace)

    def test_rejects_prose_outside_patch_blocks(self) -> None:
        patch_text = "Please apply this.\n<<<<<<< SEARCH\nA\n=======\nB\n>>>>>>> REPLACE"

        with self.assertRaisesRegex(PatchParseError, "outside Search/Replace blocks"):
            parse_search_replace_blocks(patch_text)

    def test_rejects_code_fences(self) -> None:
        patch_text = "```text\n<<<<<<< SEARCH\nA\n=======\nB\n>>>>>>> REPLACE\n```"

        with self.assertRaisesRegex(PatchParseError, "code fences"):
            parse_search_replace_blocks(patch_text)

    def test_rejects_malformed_markers(self) -> None:
        patch_text = "<<<<<<< SEARCH\nA\n=======\nB"

        with self.assertRaisesRegex(PatchParseError, "missing a REPLACE terminator"):
            parse_search_replace_blocks(patch_text)

    def test_rejects_empty_search_block(self) -> None:
        patch_text = "<<<<<<< SEARCH\n\n=======\nB\n>>>>>>> REPLACE"

        with self.assertRaisesRegex(PatchParseError, "SEARCH snippet is empty"):
            parse_search_replace_blocks(patch_text)

    def test_apply_patch_output_supports_multiple_blocks(self) -> None:
        current_html = (
            "<header>\n"
            "  <div>Placeholder</div>\n"
            "</header>\n"
            "<main>\n"
            "  <p>Old body</p>\n"
            "</main>\n"
        )
        patch_text = (
            "<<<<<<< SEARCH\n"
            "<header>\n"
            "  <div>Placeholder</div>\n"
            "</header>\n"
            "=======\n"
            "<header>\n"
            "  <div>Final title</div>\n"
            "</header>\n"
            ">>>>>>> REPLACE\n\n"
            "<<<<<<< SEARCH\n"
            "<main>\n"
            "  <p>Old body</p>\n"
            "</main>\n"
            "=======\n"
            "<main>\n"
            "  <p>New body</p>\n"
            "</main>\n"
            ">>>>>>> REPLACE"
        )

        patched_html, blocks = apply_patch_output(current_html, patch_text)

        self.assertEqual(2, len(blocks))
        self.assertIn("<div>Final title</div>", patched_html)
        self.assertIn("<p>New body</p>", patched_html)
        self.assertNotIn("Placeholder", patched_html)

    def test_zero_match_fails_safely(self) -> None:
        current_html = "<div>Only current content</div>\n"
        patch_text = "<<<<<<< SEARCH\n<div>Missing</div>\n=======\n<div>New</div>\n>>>>>>> REPLACE"

        with self.assertRaisesRegex(PatchApplicationError, "could not find an exact SEARCH match"):
            apply_patch_output(current_html, patch_text)

        self.assertEqual("<div>Only current content</div>\n", current_html)

    def test_multiple_match_fails_safely(self) -> None:
        current_html = "<li>Item</li>\n<li>Item</li>\n"
        patch_text = "<<<<<<< SEARCH\n<li>Item</li>\n=======\n<li>Updated</li>\n>>>>>>> REPLACE"

        with self.assertRaisesRegex(PatchApplicationError, "found multiple exact SEARCH matches"):
            apply_patch_output(current_html, patch_text)

    def test_overlapping_matches_fail_safely(self) -> None:
        current_html = "abcdef"
        blocks = parse_search_replace_blocks(
            "<<<<<<< SEARCH\nabcde\n=======\nABCDE\n>>>>>>> REPLACE\n\n"
            "<<<<<<< SEARCH\ncde\n=======\nCDE\n>>>>>>> REPLACE"
        )

        with self.assertRaisesRegex(PatchApplicationError, "overlapping SEARCH ranges"):
            build_patch_operations(current_html, blocks)

    def test_atomic_validation_prevents_partial_application(self) -> None:
        current_html = "<div>A</div>\n<div>B</div>\n"
        patch_text = (
            "<<<<<<< SEARCH\n<div>A</div>\n=======\n<div>X</div>\n>>>>>>> REPLACE\n\n"
            "<<<<<<< SEARCH\n<div>C</div>\n=======\n<div>Y</div>\n>>>>>>> REPLACE"
        )

        with self.assertRaises(PatchApplicationError):
            apply_patch_output(current_html, patch_text)

        self.assertEqual("<div>A</div>\n<div>B</div>\n", current_html)


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
        patch_agent_output: str,
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
            return {"coder_instructions": "1. Remove the empty nav placeholder."}

        def fake_patch_agent_node(_: Any) -> dict[str, Any]:
            counts["patch_agent"] += 1
            return {"patch_agent_output": patch_agent_output}

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

    def test_full_regenerate_route_calls_coder_instead_of_patch_executor(self) -> None:
        workflow, counts = self._build_workflow(FULL_REGENERATE_REQUIRED)
        config = self._pause_at_webpage_review(workflow)

        workflow.update_state(
            config,
            {
                "human_directives": {"text": "remove the empty nav item", "images": []},
                "coder_instructions": "",
                "patch_agent_output": "",
                "patch_error": "",
                "is_webpage_approved": False,
            },
            as_node="webpage_review",
        )
        workflow.invoke(None, config=config)

        state = workflow.get_state(config)
        self.assertEqual("webpage", state.values.get("review_stage"))
        self.assertEqual(1, counts["translator"])
        self.assertEqual(1, counts["patch_agent"])
        self.assertEqual(0, counts["patch_executor"])
        self.assertEqual(2, counts["coder"])

    def test_patch_route_calls_patch_executor_without_full_regenerate(self) -> None:
        workflow, counts = self._build_workflow(
            "<<<<<<< SEARCH\n<nav>Old</nav>\n=======\n<nav>New</nav>\n>>>>>>> REPLACE"
        )
        config = self._pause_at_webpage_review(workflow)

        workflow.update_state(
            config,
            {
                "human_directives": {"text": "fix the nav copy", "images": []},
                "coder_instructions": "",
                "patch_agent_output": "",
                "patch_error": "",
                "is_webpage_approved": False,
            },
            as_node="webpage_review",
        )
        workflow.invoke(None, config=config)

        state = workflow.get_state(config)
        self.assertEqual("webpage", state.values.get("review_stage"))
        self.assertEqual(1, counts["translator"])
        self.assertEqual(1, counts["patch_agent"])
        self.assertEqual(1, counts["patch_executor"])
        self.assertEqual(1, counts["coder"])
        self.assertEqual("", state.values.get("patch_error"))

    def test_safe_fail_returns_to_webpage_review_with_patch_error(self) -> None:
        workflow, counts = self._build_workflow(
            "<<<<<<< SEARCH\n<nav>Old</nav>\n=======\n<nav>New</nav>\n>>>>>>> REPLACE",
            patch_executor_result={"patch_error": "Patch Executor could not find an exact SEARCH match for block 1."},
        )
        config = self._pause_at_webpage_review(workflow)

        workflow.update_state(
            config,
            {
                "human_directives": {"text": "remove the stale nav entry", "images": []},
                "coder_instructions": "",
                "patch_agent_output": "",
                "patch_error": "",
                "is_webpage_approved": False,
            },
            as_node="webpage_review",
        )
        workflow.invoke(None, config=config)

        state = workflow.get_state(config)
        self.assertEqual("webpage", state.values.get("review_stage"))
        self.assertEqual(
            "Patch Executor could not find an exact SEARCH match for block 1.",
            state.values.get("patch_error"),
        )
        self.assertEqual(1, counts["patch_executor"])
        self.assertEqual(1, counts["coder"])

    def test_webpage_review_pause_can_approve_to_end(self) -> None:
        workflow, counts = self._build_workflow(FULL_REGENERATE_REQUIRED)
        config = self._pause_at_webpage_review(workflow)

        workflow.update_state(
            config,
            {
                "human_directives": empty_human_feedback(),
                "coder_instructions": "",
                "patch_agent_output": "",
                "patch_error": "",
                "is_webpage_approved": True,
            },
            as_node="webpage_review",
        )
        workflow.invoke(None, config=config)

        state = workflow.get_state(config)
        self.assertEqual("webpage", state.values.get("review_stage"))
        self.assertEqual(0, counts["translator"])
        self.assertEqual(0, counts["patch_agent"])
        self.assertEqual(0, counts["patch_executor"])
        self.assertEqual(1, counts["coder"])


if __name__ == "__main__":
    unittest.main()
