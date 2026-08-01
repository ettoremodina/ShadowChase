from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any
import json
import re
import shutil
import subprocess
import textwrap

from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name
from pygments.lexers.special import TextLexer

from code_flow.hierarchy import GraphView, build_views, focused_view
from code_flow.model import FlowGraph, FlowNode


NODE_STYLES: dict[str, tuple[str, str, str]] = {
    "entry": ("oval", "#dcfce7", "#15803d"),
    "exit": ("oval", "#fee2e2", "#b91c1c"),
    "phase": ("box", "#e0e7ff", "#4338ca"),
    "logic_step": ("box", "#e0f2fe", "#0369a1"),
    "decision": ("diamond", "#fef3c7", "#b45309"),
    "loop": ("hexagon", "#dbeafe", "#1d4ed8"),
    "warehouse_query": ("cylinder", "#ffe4e6", "#be123c"),
    "sql_capture": ("folder", "#ede9fe", "#6d28d9"),
    "sql_emit": ("note", "#f3e8ff", "#7e22ce"),
    "return": ("oval", "#fee2e2", "#dc2626"),
    "raise": ("octagon", "#fecaca", "#991b1b"),
    "adapter_call": ("component", "#cffafe", "#0e7490"),
    "dbt_reference": ("box", "#ccfbf1", "#0f766e"),
    "macro_call": ("box", "#e0f2fe", "#0369a1"),
    "call": ("box", "#e0f2fe", "#0369a1"),
    "log": ("box", "#f1f5f9", "#64748b"),
    "assignment": ("box", "#f8fafc", "#64748b"),
    "mutation": ("box", "#fae8ff", "#a21caf"),
}

SEMANTIC_NODE_STYLES: dict[str, dict[str, tuple[str, str]]] = {
    "blue": {
        "overview": ("#dbeafe", "#2563eb"),
        "logic": ("#eff6ff", "#60a5fa"),
    },
    "teal": {
        "overview": ("#ccfbf1", "#0f766e"),
        "logic": ("#f0fdfa", "#2dd4bf"),
    },
    "violet": {
        "overview": ("#ede9fe", "#7c3aed"),
        "logic": ("#f5f3ff", "#a78bfa"),
    },
    "amber": {
        "overview": ("#fef3c7", "#b45309"),
        "logic": ("#fffbeb", "#f59e0b"),
    },
    "rose": {
        "overview": ("#ffe4e6", "#be123c"),
        "logic": ("#fff1f2", "#fb7185"),
    },
    "green": {
        "overview": ("#dcfce7", "#15803d"),
        "logic": ("#f0fdf4", "#4ade80"),
    },
    "slate": {
        "overview": ("#e2e8f0", "#475569"),
        "logic": ("#f8fafc", "#94a3b8"),
    },
}


def render_html(
    graph: FlowGraph,
    semantic: dict[str, Any],
    source: str,
    template_path: str | Path,
    output_path: str | Path,
) -> Path:
    if not shutil.which("dot"):
        raise RuntimeError("Graphviz 'dot' was not found on PATH.")

    views = build_views(graph, semantic)
    svg_by_name = {view.name: _render_view_svg(view) for view in views}
    view_by_name = {view.name: view for view in views}
    focused_svgs: dict[str, dict[str, str]] = {"logic": {}, "exact": {}}
    for node in view_by_name["overview"].nodes:
        child_ids = node.metadata.get("child_node_ids", [])
        if child_ids:
            focused_svgs["logic"][node.id] = _render_view_svg(
                focused_view(view_by_name["logic"], child_ids)
            )
    for node in view_by_name["logic"].nodes:
        covered_ids = node.metadata.get("covered_node_ids", [])
        if covered_ids:
            focused_svgs["exact"][node.id] = _render_view_svg(
                focused_view(view_by_name["exact"], covered_ids)
            )
    node_data = _node_data(graph, views)
    source_html, source_style = _render_source(source, graph.language)
    template = Path(template_path).read_text(encoding="utf-8")

    replacements = {
        "__TITLE__": escape(graph.title),
        "__PURPOSE__": escape(str(semantic.get("purpose", ""))),
        "__LANGUAGE__": escape(graph.language),
        "__SYMBOL__": escape(graph.symbol),
        "__OVERVIEW_SVG__": svg_by_name["overview"],
        "__LOGIC_SVG__": svg_by_name["logic"],
        "__EXACT_SVG__": svg_by_name["exact"],
        "__SOURCE_HTML__": source_html,
        "__SOURCE_STYLE__": source_style,
        "__NODE_DATA_JSON__": json.dumps(node_data, ensure_ascii=False).replace("</", "<\\/"),
        "__FOCUSED_SVG_JSON__": json.dumps(
            focused_svgs,
            ensure_ascii=False,
        ).replace("</", "<\\/"),
        "__LEGEND_JSON__": json.dumps(
            _semantic_legend(semantic),
            ensure_ascii=False,
        ).replace("</", "<\\/"),
        "__SUMMARY_JSON__": json.dumps(
            {
                "inputs": semantic.get("inputs", []),
                "outputs": semantic.get("outputs", []),
                "side_effects": semantic.get("side_effects", []),
                "warnings": graph.warnings,
            },
            ensure_ascii=False,
        ).replace("</", "<\\/"),
    }
    rendered = template
    for placeholder, value in replacements.items():
        rendered = rendered.replace(placeholder, value)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    return output


def _render_view_svg(view: GraphView) -> str:
    dot = _to_dot(view)
    completed = subprocess.run(
        ["dot", "-Tsvg"],
        input=dot,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Graphviz failed for {view.name}: {completed.stderr.strip()}")
    svg = completed.stdout
    svg = re.sub(r"<\?xml[^>]*>\s*", "", svg, count=1)
    svg = re.sub(r"<!DOCTYPE[^>]*(?:\[[\s\S]*?\]\s*)?>\s*", "", svg, count=1)
    svg = svg.replace("<svg ", f'<svg data-view="{escape(view.name)}" ')
    return svg


def _to_dot(view: GraphView) -> str:
    lines = [
        "digraph flow {",
        '  graph [rankdir=TB, bgcolor="transparent", pad="0.25", nodesep="0.32", ranksep="0.55", splines=polyline];',
        '  node [fontname="Segoe UI", fontsize=10, margin="0.14,0.09", style="rounded,filled", penwidth=1.4];',
        '  edge [fontname="Segoe UI", fontsize=9, color="#64748b", fontcolor="#475569", arrowsize=0.7, penwidth=1.2];',
    ]
    for node in view.nodes:
        shape, fill, stroke = _node_style(node, view.name)
        label = _dot_escape(_wrap(node.label))
        tooltip = _dot_escape(f"{node.kind} · lines {node.start_line}-{node.end_line}")
        svg_id = _safe_svg_id(node.id)
        lines.append(
            f'  "{_dot_escape(node.id)}" [label="{label}", shape={shape}, '
            f'fillcolor="{fill}", color="{stroke}", tooltip="{tooltip}", '
            f'id="{svg_id}", class="{_dot_escape(node.kind)}"];'
        )
    for edge in view.edges:
        attributes: list[str] = []
        if edge.label:
            attributes.append(f'label="{_dot_escape(_wrap(edge.label, 28))}"')
        if edge.kind in {"loop_back", "continue"}:
            attributes.extend(['style="dashed"', 'color="#2563eb"'])
        elif edge.kind == "call":
            attributes.extend(['color="#0369a1"', 'penwidth="1.8"'])
        elif edge.kind == "call_return":
            attributes.extend(['style="dashed"', 'color="#0284c7"'])
        elif edge.kind in {"raise", "exception"}:
            attributes.extend(['style="dashed"', 'color="#dc2626"'])
        elif edge.kind == "branch":
            attributes.append('color="#a16207"')
        suffix = f" [{', '.join(attributes)}]" if attributes else ""
        lines.append(
            f'  "{_dot_escape(edge.source)}" -> "{_dot_escape(edge.target)}"{suffix};'
        )
    lines.append("}")
    return "\n".join(lines)


def _node_style(node: FlowNode, view_name: str) -> tuple[str, str, str]:
    shape, default_fill, default_stroke = NODE_STYLES.get(
        node.kind, ("box", "#f8fafc", "#64748b")
    )
    if node.kind not in {"phase", "logic_step"}:
        return shape, default_fill, default_stroke
    color_token = str(node.metadata.get("color_token", "blue"))
    fill, stroke = SEMANTIC_NODE_STYLES.get(
        color_token,
        SEMANTIC_NODE_STYLES["blue"],
    ).get(view_name, (default_fill, default_stroke))
    return shape, fill, stroke


def _semantic_legend(semantic: dict[str, Any]) -> list[dict[str, str]]:
    used_categories = {
        str(phase.get("category", ""))
        for phase in semantic.get("phases", [])
        if isinstance(phase, dict)
    }
    resolved: list[dict[str, str]] = []
    for item in semantic.get("legend", []):
        if not isinstance(item, dict) or str(item.get("id", "")) not in used_categories:
            continue
        color = str(item.get("color", "blue"))
        fill, stroke = SEMANTIC_NODE_STYLES.get(
            color,
            SEMANTIC_NODE_STYLES["blue"],
        )["overview"]
        resolved.append(
            {
                "id": str(item["id"]),
                "label": str(item["label"]),
                "color": color,
                "fill": fill,
                "stroke": stroke,
            }
        )
    return resolved


def _node_data(graph: FlowGraph, views: list[GraphView]) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for node in graph.nodes:
        data[node.id] = _serialize_node(node)
    for view in views:
        for node in view.nodes:
            data.setdefault(node.id, _serialize_node(node))
    return data


def _serialize_node(node: FlowNode) -> dict[str, Any]:
    return {
        "id": node.id,
        "kind": node.kind,
        "label": node.label,
        "detail": node.detail,
        "start_line": node.start_line,
        "end_line": node.end_line,
        "metadata": node.metadata,
    }


def _render_source(source: str, language: str) -> tuple[str, str]:
    lexer_name = {
        "python": "python",
        "jinja-dbt": "sql+jinja",
    }.get(language)
    try:
        lexer = get_lexer_by_name(lexer_name) if lexer_name else TextLexer()
    except ValueError:
        lexer = TextLexer()
    formatter = HtmlFormatter(nowrap=True, style="monokai")
    highlighted_lines = highlight(source, lexer, formatter).splitlines()
    source_lines = source.splitlines()
    if len(highlighted_lines) < len(source_lines):
        highlighted_lines.extend([""] * (len(source_lines) - len(highlighted_lines)))

    rows: list[str] = []
    for number, highlighted_line in enumerate(
        highlighted_lines[: len(source_lines)],
        start=1,
    ):
        rows.append(
            f'<span class="source-line" id="source-line-{number}" data-line="{number}">'
            f'<span class="line-number">{number}</span>'
            f'<code>{highlighted_line or " "}</code></span>'
        )
    return "\n".join(rows), formatter.get_style_defs(".source code")


def _wrap(text: str, width: int = 34) -> str:
    wrapped: list[str] = []
    for paragraph in text.splitlines() or [""]:
        wrapped.extend(textwrap.wrap(" ".join(paragraph.split()), width=width) or [""])
    return "\n".join(wrapped)


def _dot_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _safe_svg_id(node_id: str) -> str:
    return "flow-" + re.sub(r"[^A-Za-z0-9_-]+", "-", node_id)
