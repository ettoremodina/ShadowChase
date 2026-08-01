from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from code_flow.model import Edge, FlowGraph, FlowNode


SEMANTIC_COLOR_TOKENS = frozenset(
    {"blue", "teal", "violet", "amber", "rose", "green", "slate"}
)


@dataclass(slots=True)
class GraphView:
    name: str
    nodes: list[FlowNode]
    edges: list[Edge]


def validate_semantic(graph: FlowGraph, semantic: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    if not str(semantic.get("purpose", "")).strip():
        issues.append("semantic.purpose must be a non-empty string.")
    legend = semantic.get("legend")
    if not isinstance(legend, list) or not legend:
        issues.append("semantic.legend must be a non-empty list.")
        legend = []
    legend_ids: set[str] = set()
    for index, item in enumerate(legend):
        if not isinstance(item, dict):
            issues.append(f"Legend item {index} must be an object.")
            continue
        legend_id = str(item.get("id", "")).strip()
        label = str(item.get("label", "")).strip()
        color = str(item.get("color", "")).strip()
        if not legend_id:
            issues.append(f"Legend item {index} has no id.")
        elif legend_id in legend_ids:
            issues.append(f"Duplicate legend id: {legend_id}")
        legend_ids.add(legend_id)
        if not label:
            issues.append(f"Legend item {legend_id or index} has no label.")
        if color not in SEMANTIC_COLOR_TOKENS:
            allowed = ", ".join(sorted(SEMANTIC_COLOR_TOKENS))
            issues.append(
                f"Legend item {legend_id or index} uses unsupported color "
                f"{color!r}; choose one of: {allowed}."
            )
    phases = semantic.get("phases")
    if not isinstance(phases, list) or not phases:
        return [*issues, "semantic.phases must be a non-empty list."]

    phase_ids: set[str] = set()
    step_ids: set[str] = set()
    phase_ranges: list[tuple[int, int, str]] = []
    step_ranges: list[tuple[int, int, str, str]] = []
    for index, phase in enumerate(phases):
        if not isinstance(phase, dict):
            issues.append(f"Phase {index} must be an object.")
            continue
        phase_id = str(phase.get("id", "")).strip()
        label = str(phase.get("label", "")).strip()
        summary = str(phase.get("summary", "")).strip()
        category = str(phase.get("category", "")).strip()
        if not phase_id:
            issues.append(f"Phase {index} has no id.")
        elif phase_id in phase_ids:
            issues.append(f"Duplicate phase id: {phase_id}")
        phase_ids.add(phase_id)
        if not label:
            issues.append(f"Phase {phase_id or index} has no label.")
        if not summary:
            issues.append(f"Phase {phase_id or index} has no summary.")
        if not category:
            issues.append(f"Phase {phase_id or index} has no category.")
        elif category not in legend_ids:
            issues.append(
                f"Phase {phase_id or index} references unknown legend category "
                f"{category!r}."
            )
        try:
            start = int(phase["line_start"])
            end = int(phase["line_end"])
        except (KeyError, TypeError, ValueError):
            issues.append(f"Phase {phase_id or index} needs integer line_start and line_end.")
            continue
        if start > end:
            issues.append(f"Phase {phase_id or index} starts after it ends.")
        if start < graph.target_start_line or end > graph.target_end_line:
            issues.append(
                f"Phase {phase_id or index} range {start}-{end} is outside "
                f"target {graph.target_start_line}-{graph.target_end_line}."
            )
        phase_key = phase_id or str(index)
        phase_ranges.append((start, end, phase_key))

        steps = phase.get("steps")
        if not isinstance(steps, list) or not steps:
            issues.append(f"Phase {phase_key} needs a non-empty steps list.")
            continue
        local_ranges: list[tuple[int, int, str]] = []
        for step_index, step in enumerate(steps):
            if not isinstance(step, dict):
                issues.append(f"Step {step_index} in phase {phase_key} must be an object.")
                continue
            step_id = str(step.get("id", "")).strip()
            step_label = str(step.get("label", "")).strip()
            step_summary = str(step.get("summary", "")).strip()
            step_key = step_id or f"{phase_key}-{step_index}"
            if not step_id:
                issues.append(f"Step {step_index} in phase {phase_key} has no id.")
            elif step_id in step_ids:
                issues.append(f"Duplicate step id: {step_id}")
            step_ids.add(step_id)
            if not step_label:
                issues.append(f"Step {step_key} has no label.")
            if not step_summary:
                issues.append(f"Step {step_key} has no summary.")
            try:
                step_start = int(step["line_start"])
                step_end = int(step["line_end"])
            except (KeyError, TypeError, ValueError):
                issues.append(f"Step {step_key} needs integer line_start and line_end.")
                continue
            if step_start > step_end:
                issues.append(f"Step {step_key} starts after it ends.")
            if step_start < start or step_end > end:
                issues.append(
                    f"Step {step_key} range {step_start}-{step_end} is outside "
                    f"phase {phase_key} range {start}-{end}."
                )
            local_ranges.append((step_start, step_end, step_key))
            step_ranges.append((step_start, step_end, step_key, phase_key))

        issues.extend(_overlap_issues(local_ranges, "Logic steps"))

    issues.extend(_overlap_issues(phase_ranges, "Semantic phases"))

    executable = [node for node in graph.nodes if node.kind not in {"entry", "exit"}]
    for start, end, phase_id in phase_ranges:
        if not any(start <= node.start_line <= end for node in executable):
            issues.append(f"Phase {phase_id} does not cover any exact node.")
    for start, end, step_id, _ in step_ranges:
        if not any(start <= node.start_line <= end for node in executable):
            issues.append(f"Step {step_id} does not cover any exact node.")
    for node in executable:
        covering_phases = [
            phase_id
            for start, end, phase_id in phase_ranges
            if start <= node.start_line <= end
        ]
        if not covering_phases:
            issues.append(
                f"Node {node.id} at line {node.start_line} is not covered by any phase."
            )
            continue
        if not any(
            start <= node.start_line <= end and phase_id in covering_phases
            for start, end, _, phase_id in step_ranges
        ):
            issues.append(
                f"Node {node.id} at line {node.start_line} is not covered by any logic step."
            )
    return issues


def build_views(graph: FlowGraph, semantic: dict[str, Any]) -> list[GraphView]:
    return [
        _build_overview(graph, semantic),
        _build_logic(graph, semantic),
        GraphView("exact", list(graph.nodes), list(graph.edges)),
    ]


def _build_overview(graph: FlowGraph, semantic: dict[str, Any]) -> GraphView:
    phases = semantic["phases"]
    legend = {
        str(item["id"]): item
        for item in semantic.get("legend", [])
        if isinstance(item, dict) and item.get("id")
    }
    phase_nodes: list[FlowNode] = []
    node_to_phase: dict[str, str] = {}

    for phase in phases:
        phase_id = f"phase-{phase['id']}"
        category = str(phase["category"])
        color_token = str(legend.get(category, {}).get("color", "blue"))
        start = int(phase["line_start"])
        end = int(phase["line_end"])
        covered = [
            node.id
            for node in graph.nodes
            if node.kind not in {"entry", "exit"} and start <= node.start_line <= end
        ]
        child_node_ids = [
            f"logic-{step['id']}"
            for step in phase["steps"]
        ]
        for node_id in covered:
            node_to_phase[node_id] = phase_id
        phase_nodes.append(
            FlowNode(
                id=phase_id,
                kind="phase",
                label=_overview_label(phase),
                start_line=start,
                end_line=end,
                detail=str(phase.get("summary", phase["label"])),
                metadata={
                    "covered_node_ids": covered,
                    "child_node_ids": child_node_ids,
                    "drilldown_view": "logic",
                    "drilldown_focus": phase_id,
                    "short_label": str(phase["label"]),
                    "category": category,
                    "color_token": color_token,
                },
            )
        )

    boundary_nodes = _map_boundaries(graph, node_to_phase)
    return GraphView(
        "overview",
        [*boundary_nodes, *phase_nodes],
        _collapse_edges(graph, node_to_phase),
    )


def _build_logic(graph: FlowGraph, semantic: dict[str, Any]) -> GraphView:
    step_nodes: list[FlowNode] = []
    node_to_step: dict[str, str] = {}
    legend = {
        str(item["id"]): item
        for item in semantic.get("legend", [])
        if isinstance(item, dict) and item.get("id")
    }
    for phase in semantic["phases"]:
        parent_id = f"phase-{phase['id']}"
        category = str(phase["category"])
        color_token = str(legend.get(category, {}).get("color", "blue"))
        for step in phase["steps"]:
            step_id = f"logic-{step['id']}"
            start = int(step["line_start"])
            end = int(step["line_end"])
            covered = [
                node.id
                for node in graph.nodes
                if node.kind not in {"entry", "exit"} and start <= node.start_line <= end
            ]
            for node_id in covered:
                node_to_step[node_id] = step_id
            step_nodes.append(
                FlowNode(
                    id=step_id,
                    kind="logic_step",
                    label=str(step["label"]),
                    start_line=start,
                    end_line=end,
                    detail=str(step.get("summary", step["label"])),
                    metadata={
                        "covered_node_ids": covered,
                        "parent_node_id": parent_id,
                        "drilldown_view": "exact",
                        "drilldown_focus": step_id,
                        "short_label": str(step["label"]),
                        "category": category,
                        "color_token": color_token,
                    },
                )
            )

    boundary_nodes = _map_boundaries(graph, node_to_step)
    return GraphView(
        "logic",
        [*boundary_nodes, *step_nodes],
        _collapse_edges(graph, node_to_step),
    )


def focused_view(view: GraphView, node_ids: list[str]) -> GraphView:
    """Return a deterministically filtered view with a fresh Graphviz layout."""
    selected = set(node_ids)
    return GraphView(
        name=view.name,
        nodes=[node for node in view.nodes if node.id in selected],
        edges=[
            edge
            for edge in view.edges
            if edge.source in selected and edge.target in selected
        ],
    )


def _map_boundaries(
    graph: FlowGraph,
    node_mapping: dict[str, str],
) -> list[FlowNode]:
    boundary_nodes = [node for node in graph.nodes if node.kind in {"entry", "exit"}]
    for node in boundary_nodes:
        node_mapping[node.id] = node.id
    return boundary_nodes


def _collapse_edges(
    graph: FlowGraph,
    node_mapping: dict[str, str],
) -> list[Edge]:
    merged: dict[tuple[str, str], Edge] = {}
    labels: defaultdict[tuple[str, str], list[str]] = defaultdict(list)
    for edge in graph.edges:
        source = node_mapping.get(edge.source)
        target = node_mapping.get(edge.target)
        if not source or not target or source == target:
            continue
        key = (source, target)
        if key not in merged:
            merged[key] = Edge(source, target, "", edge.kind)
        if edge.label and edge.label not in labels[key]:
            labels[key].append(edge.label)
    for key, edge in merged.items():
        edge.label = " / ".join(labels[key][:3])
    return list(merged.values())


def _overview_label(phase: dict[str, Any]) -> str:
    label = str(phase["label"]).strip()
    summary = " ".join(str(phase.get("summary", "")).split())
    if not summary:
        return label
    return f"{label}\n{summary}"


def _overlap_issues(
    ranges: list[tuple[int, int, str]],
    label: str,
) -> list[str]:
    issues: list[str] = []
    ordered = sorted(ranges)
    for index, (start, _, current_id) in enumerate(ordered):
        if index and start <= ordered[index - 1][1]:
            previous = ordered[index - 1][2]
            issues.append(f"{label} overlap: {previous} and {current_id}.")
    return issues
