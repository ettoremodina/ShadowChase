from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from code_flow.adapters import detect_language, make_adapter
from code_flow.hierarchy import validate_semantic
from code_flow.model import FlowGraph
from code_flow.renderer import render_html


SKILL_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE = SKILL_DIR / "assets" / "viewer-template.html"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create hierarchical flow artifacts for small code targets."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="Extract exact control flow to JSON.")
    analyze.add_argument("source")
    analyze.add_argument("--language", default="auto", choices=["auto", "python", "jinja-dbt"])
    analyze.add_argument("--symbol")
    analyze.add_argument(
        "--expand-local-macros",
        action="store_true",
        help="Include every macro in the same Jinja/dbt file and connect local calls.",
    )
    analyze.add_argument("--output", required=True)

    validate = subparsers.add_parser("validate", help="Validate flow IR and semantic phases.")
    validate.add_argument("flow_ir")
    validate.add_argument("--semantic", required=True)

    render = subparsers.add_parser("render", help="Render validated flow data to HTML.")
    render.add_argument("flow_ir")
    render.add_argument("--source", required=True)
    render.add_argument("--semantic", required=True)
    render.add_argument("--template", default=str(DEFAULT_TEMPLATE))
    render.add_argument("--output", required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "analyze":
            return _analyze(args)
        if args.command == "validate":
            return _validate(args)
        if args.command == "render":
            return _render(args)
    except (OSError, ValueError, RuntimeError, SyntaxError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 2


def _analyze(args: argparse.Namespace) -> int:
    source_path = Path(args.source).resolve()
    source = source_path.read_text(encoding="utf-8")
    language = (
        detect_language(source_path, source)
        if args.language == "auto"
        else args.language
    )
    graph = make_adapter(language).analyze(
        source_path,
        source,
        args.symbol,
        expand_local_macros=args.expand_local_macros,
    )
    issues = graph.validate()
    if issues:
        raise ValueError("Invalid extracted graph:\n- " + "\n- ".join(issues))
    graph.write(args.output)
    print(
        f"Wrote {args.output}: {len(graph.nodes)} nodes, "
        f"{len(graph.edges)} edges, language={graph.language}, symbol={graph.symbol}"
    )
    for warning in graph.warnings:
        print(f"warning: {warning}")
    return 0


def _validate(args: argparse.Namespace) -> int:
    graph = FlowGraph.read(args.flow_ir)
    semantic = _read_json(args.semantic)
    issues = [*graph.validate(), *validate_semantic(graph, semantic)]
    if issues:
        print("Validation failed:", file=sys.stderr)
        for issue in issues:
            print(f"- {issue}", file=sys.stderr)
        return 1
    step_count = sum(len(phase.get("steps", [])) for phase in semantic["phases"])
    print(
        f"Validation passed: {len(graph.nodes)} nodes, "
        f"{len(graph.edges)} edges, {len(semantic['phases'])} overview phases, "
        f"{step_count} logic steps."
    )
    return 0


def _render(args: argparse.Namespace) -> int:
    graph = FlowGraph.read(args.flow_ir)
    semantic = _read_json(args.semantic)
    issues = [*graph.validate(), *validate_semantic(graph, semantic)]
    if issues:
        raise ValueError("Cannot render invalid inputs:\n- " + "\n- ".join(issues))
    source = Path(args.source).read_text(encoding="utf-8")
    output = render_html(graph, semantic, source, args.template, args.output)
    print(f"Wrote interactive HTML: {output}")
    return 0


def _read_json(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return data


if __name__ == "__main__":
    raise SystemExit(main())
