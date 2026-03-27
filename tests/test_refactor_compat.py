import ast
import importlib
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"


class RefactorCompatibilityTests(unittest.TestCase):
    def test_structured_import_paths_work(self) -> None:
        app_module = importlib.import_module("app")
        from src.agents import coder as agent_coder
        from src.agents import planner as agent_planner
        from src.contracts.schemas import PagePlan
        from src.template.compile import compile_template_profile

        self.assertTrue(hasattr(app_module, "build_app"))
        self.assertTrue(hasattr(agent_planner, "run_planner_agent"))
        self.assertTrue(hasattr(agent_coder, "run_coder_agent_with_diagnostics"))
        self.assertTrue(hasattr(PagePlan, "model_validate"))
        self.assertTrue(callable(compile_template_profile))

    def test_app_entrypoints_still_build(self) -> None:
        app_module = importlib.import_module("app")
        workflow = app_module.build_hitl_workflow()
        demo = app_module.build_app()

        self.assertIsNotNone(workflow)
        self.assertIsNotNone(demo)

    def test_src_import_graph_has_no_cycles(self) -> None:
        graph: dict[str, set[str]] = {}

        for path in sorted(SRC_ROOT.rglob("*.py")):
            module_name = ".".join(path.relative_to(REPO_ROOT).with_suffix("").parts)
            source = path.read_text(encoding="utf-8-sig")
            tree = ast.parse(source)
            deps: set[str] = set()

            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("src."):
                    deps.add(node.module)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.startswith("src."):
                            deps.add(alias.name)

            graph[module_name] = deps

        visited: set[str] = set()
        stack: list[str] = []
        active: set[str] = set()
        found_cycles: list[list[str]] = []

        def dfs(module_name: str) -> None:
            visited.add(module_name)
            active.add(module_name)
            stack.append(module_name)

            for dep in sorted(graph.get(module_name, set())):
                if dep not in graph:
                    continue
                if dep not in visited:
                    dfs(dep)
                    continue
                if dep in active:
                    cycle_start = stack.index(dep)
                    found_cycles.append(stack[cycle_start:] + [dep])

            stack.pop()
            active.remove(module_name)

        for module_name in sorted(graph):
            if module_name not in visited:
                dfs(module_name)

        self.assertEqual([], found_cycles)
