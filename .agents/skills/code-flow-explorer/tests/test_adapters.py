from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
SKILL_DIR = HERE.parent
sys.path.insert(0, str(SKILL_DIR / "scripts"))

from code_flow.adapters import make_adapter
from code_flow.hierarchy import build_views, validate_semantic
from code_flow.model import FlowGraph, FlowNode
from code_flow.renderer import _to_dot, render_html


class AdapterTests(unittest.TestCase):
    def test_python_adapter_builds_branches_and_loops(self) -> None:
        path = HERE / "fixtures" / "python_sample.py"
        source = path.read_text(encoding="utf-8")
        graph = make_adapter("python").analyze(path, source, "classify_and_sum")
        kinds = {node.kind for node in graph.nodes}
        self.assertFalse(graph.validate())
        self.assertIn("decision", kinds)
        self.assertIn("loop", kinds)
        self.assertIn("return", kinds)
        self.assertTrue(any(edge.kind == "loop_back" for edge in graph.edges))

    def test_jinja_adapter_distinguishes_query_and_return(self) -> None:
        path = HERE / "fixtures" / "jinja_sample.sql"
        source = path.read_text(encoding="utf-8")
        graph = make_adapter("jinja-dbt").analyze(path, source, "choose_query")
        kinds = {node.kind for node in graph.nodes}
        self.assertFalse(graph.validate())
        self.assertIn("decision", kinds)
        self.assertIn("loop", kinds)
        self.assertIn("warehouse_query", kinds)
        self.assertIn("return", kinds)
        self.assertTrue(
            any("elif queries" in node.label for node in graph.nodes),
            "Jinja elif branches should remain explicit decisions.",
        )

    def test_jinja_adapter_expands_local_macros_from_the_same_file(self) -> None:
        path = HERE / "fixtures" / "jinja_expanded.sql"
        source = path.read_text(encoding="utf-8")
        adapter = make_adapter("jinja-dbt")
        compact_graph = adapter.analyze(path, source, "main")
        expanded_graph = adapter.analyze(
            path,
            source,
            "main",
            expand_local_macros=True,
        )

        compact_macros = {node.metadata.get("macro") for node in compact_graph.nodes}
        expanded_macros = {node.metadata.get("macro") for node in expanded_graph.nodes}
        self.assertEqual(compact_macros, {"main"})
        self.assertEqual(expanded_macros, {"main", "helper", "unused_helper"})
        self.assertFalse(expanded_graph.validate())
        self.assertTrue(any(edge.kind == "call" for edge in expanded_graph.edges))
        self.assertTrue(
            any(edge.kind == "call_return" for edge in expanded_graph.edges)
        )

        call_site = next(
            node
            for node in expanded_graph.nodes
            if node.metadata.get("expanded_local_macro") == "helper"
        )
        helper_entry = next(
            node
            for node in expanded_graph.nodes
            if node.kind == "entry" and node.metadata.get("macro") == "helper"
        )
        helper_exit = next(
            node
            for node in expanded_graph.nodes
            if node.kind == "exit" and node.metadata.get("macro") == "helper"
        )
        self.assertTrue(
            any(
                edge.source == call_site.id
                and edge.target == helper_entry.id
                and edge.kind == "call"
                for edge in expanded_graph.edges
            )
        )
        self.assertTrue(
            any(
                edge.source == helper_exit.id and edge.kind == "call_return"
                for edge in expanded_graph.edges
            )
        )
        self.assertFalse(
            any(
                edge.source == call_site.id and edge.target != helper_entry.id
                for edge in expanded_graph.edges
            )
        )

    def test_semantic_group_count_is_adaptive(self) -> None:
        graph = FlowGraph(
            schema_version=1,
            language="python",
            source_path="adaptive.py",
            symbol="adaptive",
            title="Adaptive grouping",
            target_start_line=1,
            target_end_line=15,
            nodes=[
                FlowNode("entry", "entry", "Enter", 1, 1),
                *[
                    FlowNode(f"operation-{line}", "operation", f"Line {line}", line, line)
                    for line in range(2, 15)
                ],
                FlowNode("exit", "exit", "Exit", 15, 15),
            ],
        )
        phases = [
            {
                "id": f"phase-{line}",
                "label": f"Responsibility {line}",
                "summary": f"Explain the responsibility at line {line}.",
                "category": "work",
                "line_start": line,
                "line_end": line,
                "steps": [
                    {
                        "id": f"step-{line}",
                        "label": f"Handle line {line}",
                        "summary": f"Describe the grounded operation at line {line}.",
                        "line_start": line,
                        "line_end": line,
                    }
                ],
            }
            for line in range(2, 15)
        ]
        semantic = {
            "purpose": "Prove that natural grouping has no twelve-node ceiling.",
            "legend": [{"id": "work", "label": "Work", "color": "blue"}],
            "phases": phases,
        }
        self.assertEqual(len(phases), 13)
        self.assertFalse(validate_semantic(graph, semantic))

    def test_shared_hierarchy_and_renderer(self) -> None:
        path = HERE / "fixtures" / "jinja_sample.sql"
        source = path.read_text(encoding="utf-8")
        graph = make_adapter("jinja-dbt").analyze(path, source, "choose_query")
        semantic = {
            "purpose": "Build queries and either execute or return them.",
            "inputs": ["items"],
            "outputs": ["query list or warehouse side effect"],
            "side_effects": ["may execute run_query"],
            "legend": [
                {"id": "preparation", "label": "Preparation", "color": "blue"},
                {"id": "outcome", "label": "Outcome", "color": "amber"},
            ],
            "phases": [
                {
                    "id": "build",
                    "label": "Build candidate queries",
                    "summary": "Collect enabled item queries.",
                    "category": "preparation",
                    "line_start": graph.target_start_line,
                    "line_end": 7,
                    "steps": [
                        {
                            "id": "prepare-query-list",
                            "label": "Prepare an empty query list",
                            "summary": "Create the collection that will hold candidate queries.",
                            "line_start": graph.target_start_line,
                            "line_end": 2,
                        },
                        {
                            "id": "collect-enabled-items",
                            "label": "Collect queries for enabled items",
                            "summary": "Inspect each item and keep a query only when it is enabled.",
                            "line_start": 3,
                            "line_end": 7,
                        },
                    ],
                },
                {
                    "id": "finish",
                    "label": "Execute or return",
                    "summary": "Choose the execution behavior.",
                    "category": "outcome",
                    "line_start": 8,
                    "line_end": graph.target_end_line,
                    "steps": [
                        {
                            "id": "execute-first-query",
                            "label": "Execute the first available query",
                            "summary": "Run one query when execution is enabled and work exists.",
                            "line_start": 8,
                            "line_end": 9,
                        },
                        {
                            "id": "return-query-result",
                            "label": "Return prepared queries or an empty list",
                            "summary": "Return generated work during parsing, or an empty fallback.",
                            "line_start": 10,
                            "line_end": graph.target_end_line,
                        },
                    ],
                },
            ],
        }
        self.assertFalse(validate_semantic(graph, semantic))
        invalid_semantic = json.loads(json.dumps(semantic))
        invalid_semantic["legend"][0]["color"] = "#3157d5"
        self.assertTrue(
            any(
                "unsupported color" in issue
                for issue in validate_semantic(graph, invalid_semantic)
            )
        )
        views = build_views(graph, semantic)
        self.assertEqual([view.name for view in views], ["overview", "logic", "exact"])
        self.assertEqual(len(views[0].nodes), 4)
        self.assertEqual(len(views[1].nodes), 6)
        overview_phase = next(node for node in views[0].nodes if node.kind == "phase")
        self.assertEqual(overview_phase.metadata["drilldown_view"], "logic")
        self.assertEqual(overview_phase.metadata["color_token"], "blue")
        self.assertIn("Collect enabled item queries.", overview_phase.label)
        overview_dot = _to_dot(views[0])
        self.assertIn('fillcolor="#dbeafe", color="#2563eb"', overview_dot)
        logic_step = next(node for node in views[1].nodes if node.kind == "logic_step")
        self.assertEqual(logic_step.metadata["drilldown_view"], "exact")
        self.assertEqual(logic_step.metadata["color_token"], "blue")
        self.assertTrue(logic_step.metadata["covered_node_ids"])
        logic_dot = _to_dot(views[1])
        self.assertIn('fillcolor="#eff6ff", color="#60a5fa"', logic_dot)
        self.assertIn("Return prepared queries or an\\nempty list", logic_dot)
        self.assertNotIn("Return prepared queries or an\\\\nempty list", logic_dot)
        exact_dot = _to_dot(views[2])
        self.assertIn('fillcolor="#f8fafc", color="#64748b"', exact_dot)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "flow.html"
            render_html(
                graph,
                semantic,
                source,
                SKILL_DIR / "assets" / "viewer-template.html",
                output,
            )
            html = output.read_text(encoding="utf-8")
            self.assertIn('data-graph="overview"', html)
            self.assertIn('data-graph="exact"', html)
            self.assertIn("Build candidate queries", html)
            self.assertIn("Prepare an empty query list", html)
            self.assertIn("const focusedSvgs =", html)
            self.assertIn("const semanticLegend =", html)
            self.assertIn("Preparation", html)
            self.assertIn("data-trail-index", html)
            self.assertIn("drilldown_view", html)
            self.assertIn("data.metadata?.short_label || data.label", html)
            self.assertIn("const cameraStates = new Map()", html)
            self.assertIn('return `${state.view}:${state.focus || "all"}`', html)
            self.assertNotIn("zoom[activeView] = null", html)
            self.assertIn('pane.addEventListener("wheel"', html)
            self.assertIn("if (!event.ctrlKey", html)
            self.assertIn("{passive: false}", html)
            self.assertIn('classList.toggle("inspector-open", open)', html)
            self.assertIn(".canvas-shell.inspector-open .graph-pane", html)
            self.assertIn("Math.hypot(deltaX, deltaY) < 4", html)
            self.assertIn('pane.addEventListener("click", event => {', html)
            self.assertNotIn('event.target.closest("g.node")', html)
            self.assertIn('id="drilldown-action"', html)
            self.assertIn('group.addEventListener("dblclick"', html)
            self.assertIn('event.key === "Enter"', html)
            self.assertIn('group.addEventListener("click", () => inspectNode(id))', html)
            self.assertIn('class="graph-pane active"', html)
            self.assertIn('class="canvas-shell"', html)
            self.assertIn('id="context-drawer"', html)
            self.assertIn('id="inspector"', html)
            self.assertIn('data-language="jinja-dbt"', html)
            self.assertIn('<span class="k">macro</span>', html)
            self.assertIn(".source code .k", html)
            self.assertNotIn("__SOURCE_STYLE__", html)
            self.assertNotIn('document.querySelectorAll(".graph")', html)
            self.assertNotIn("https://", html)


if __name__ == "__main__":
    unittest.main()
