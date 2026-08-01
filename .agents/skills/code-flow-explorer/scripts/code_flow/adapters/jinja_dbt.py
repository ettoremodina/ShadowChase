from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from jinja2 import Environment, nodes

from code_flow.adapters.base import LanguageAdapter
from code_flow.model import Edge, FlowGraph, FlowNode
from code_flow.source import SourceLocator, compact


@dataclass(slots=True)
class Completion:
    node_id: str
    kind: str = "normal"
    label: str = ""


@dataclass(slots=True)
class Fragment:
    entry: str | None = None
    completions: list[Completion] = field(default_factory=list)


class JinjaDbtAdapter(LanguageAdapter):
    language = "jinja-dbt"

    def analyze(
        self,
        source_path: str | Path,
        source: str,
        symbol: str | None,
        *,
        expand_local_macros: bool = False,
    ) -> FlowGraph:
        environment = Environment(
            extensions=["jinja2.ext.do", "jinja2.ext.loopcontrols"],
            keep_trailing_newline=True,
        )
        tree = environment.parse(source, name=Path(source_path).name, filename=str(source_path))
        macros = list(tree.find_all(nodes.Macro))

        target: nodes.Node
        target_body: list[nodes.Node]
        target_name: str
        if macros:
            if symbol:
                match = next((macro for macro in macros if macro.name == symbol), None)
                if match is None:
                    names = ", ".join(macro.name for macro in macros)
                    raise ValueError(f"Macro {symbol!r} was not found. Available macros: {names}")
                target = match
            else:
                target = macros[0]
            target_body = list(target.body)
            target_name = target.name
        else:
            if symbol:
                raise ValueError(f"No macros were found, so symbol {symbol!r} cannot be selected.")
            target = tree
            target_body = list(tree.body)
            target_name = Path(source_path).stem

        locator = SourceLocator(source)
        builder = _JinjaFlowBuilder(locator)
        selected_targets: list[tuple[nodes.Node, list[nodes.Node], str]] = [
            (target, target_body, target_name)
        ]
        if expand_local_macros and macros:
            selected_targets.extend(
                (macro, list(macro.body), macro.name)
                for macro in macros
                if macro is not target
            )

        boundaries: dict[str, tuple[str, str]] = {}
        ranges: list[tuple[int, int]] = []
        for selected, body, name in selected_targets:
            boundaries[name], target_range = _build_target_flow(
                builder,
                locator,
                selected,
                body,
                name,
            )
            ranges.append(target_range)

        if expand_local_macros and macros:
            _expand_local_macro_calls(builder, boundaries)

        start_line = min(start for start, _ in ranges)
        end_line = max(end for _, end in ranges)
        title_prefix = "Expanded flow" if expand_local_macros and macros else "Flow"

        graph = FlowGraph(
            schema_version=1,
            language=self.language,
            source_path=str(Path(source_path)),
            symbol=target_name,
            title=f"{title_prefix} of dbt/Jinja macro {target_name}",
            target_start_line=start_line,
            target_end_line=end_line,
            nodes=builder.nodes,
            edges=builder.edges,
            warnings=[
                "This is static template flow. Runtime dbt context, adapter dispatch, and warehouse results are not resolved.",
                *(
                    [
                        f"Expanded {len(selected_targets)} local macros from the same source file. "
                        "Call/return edges are context-insensitive when a macro has multiple callers."
                    ]
                    if expand_local_macros and macros
                    else []
                ),
                *builder.warnings,
            ],
        )
        return graph


def _build_target_flow(
    builder: "_JinjaFlowBuilder",
    locator: SourceLocator,
    target: nodes.Node,
    target_body: list[nodes.Node],
    target_name: str,
) -> tuple[tuple[str, str], tuple[int, int]]:
    builder.current_macro = target_name
    start_line = max(getattr(target, "lineno", 1) or 1, 1)
    end_line = max(_max_lineno(target), start_line)
    entry = builder.add_node(
        "entry",
        f"Enter {target_name}",
        start_line,
        start_line,
        locator.line(start_line).strip(),
    )
    exit_node = builder.add_node(
        "exit",
        f"Exit {target_name}",
        end_line,
        end_line,
        "Template rendering or macro evaluation completes.",
    )
    body = builder.build_sequence(target_body)
    if body.entry:
        builder.add_edge(entry, body.entry)
        for completion in body.completions:
            label = completion.label
            if completion.kind == "return" and not label:
                label = "return"
            elif completion.kind == "raise" and not label:
                label = "error"
            elif completion.kind in {"break", "continue"}:
                builder.warnings.append(
                    f"Unresolved {completion.kind} at {completion.node_id}; "
                    f"linked to {target_name} exit."
                )
            builder.add_edge(completion.node_id, exit_node, label, completion.kind)
    else:
        builder.add_edge(entry, exit_node)
    return (entry, exit_node), (start_line, end_line)


def _expand_local_macro_calls(
    builder: "_JinjaFlowBuilder",
    boundaries: dict[str, tuple[str, str]],
) -> None:
    caller_counts: dict[str, int] = {}
    for node in list(builder.nodes):
        local_calls = [
            name
            for name in node.metadata.get("calls", [])
            if name in boundaries
        ]
        if not local_calls:
            continue
        if len(local_calls) > 1:
            builder.warnings.append(
                f"Node {node.id} calls multiple local macros ({', '.join(local_calls)}); "
                "their evaluation order remains unresolved."
            )
            continue

        callee = local_calls[0]
        caller_counts[callee] = caller_counts.get(callee, 0) + 1
        callee_entry, callee_exit = boundaries[callee]
        outgoing = [edge for edge in builder.edges if edge.source == node.id]
        builder.edges = [edge for edge in builder.edges if edge.source != node.id]
        builder.add_edge(node.id, callee_entry, f"call {callee}", "call")
        caller = str(node.metadata.get("macro", "caller"))
        for edge in outgoing:
            return_label = f"return to {caller}"
            if edge.label:
                return_label = f"{return_label} · {edge.label}"
            builder.add_edge(callee_exit, edge.target, return_label, "call_return")
        node.metadata["expanded_local_macro"] = callee

    repeated = sorted(name for name, count in caller_counts.items() if count > 1)
    if repeated:
        builder.warnings.append(
            "Context-insensitive returns may over-approximate control flow for macros "
            f"with multiple local callers: {', '.join(repeated)}."
        )


class _JinjaFlowBuilder:
    def __init__(self, locator: SourceLocator) -> None:
        self.locator = locator
        self.nodes: list[FlowNode] = []
        self.edges: list[Edge] = []
        self.warnings: list[str] = []
        self._counter = 0
        self.current_macro = ""

    def add_node(
        self,
        kind: str,
        label: str,
        start_line: int,
        end_line: int | None = None,
        detail: str = "",
        metadata: dict | None = None,
    ) -> str:
        self._counter += 1
        node_id = f"{kind}-{self._counter}"
        resolved_metadata = {"macro": self.current_macro}
        resolved_metadata.update(metadata or {})
        self.nodes.append(
            FlowNode(
                id=node_id,
                kind=kind,
                label=compact(label),
                start_line=max(start_line, 1),
                end_line=max(end_line or start_line, start_line),
                detail=detail or label,
                metadata=resolved_metadata,
            )
        )
        return node_id

    def add_edge(
        self,
        source: str,
        target: str,
        label: str = "",
        kind: str = "flow",
    ) -> None:
        edge = Edge(source=source, target=target, label=label, kind=kind)
        if edge not in self.edges:
            self.edges.append(edge)

    def build_sequence(self, statements: Iterable[nodes.Node]) -> Fragment:
        result = Fragment()
        active_normals: list[Completion] = []
        abnormal: list[Completion] = []

        for statement in statements:
            fragment = self.build_statement(statement)
            if not fragment.entry:
                continue
            if result.entry is None:
                result.entry = fragment.entry
            for completion in active_normals:
                self.add_edge(completion.node_id, fragment.entry, completion.label)
            active_normals = [
                completion
                for completion in fragment.completions
                if completion.kind == "normal"
            ]
            abnormal.extend(
                completion
                for completion in fragment.completions
                if completion.kind != "normal"
            )

        result.completions = [*active_normals, *abnormal]
        return result

    def build_statement(self, statement: nodes.Node) -> Fragment:
        if isinstance(statement, nodes.If):
            return self._build_if(statement)
        if isinstance(statement, nodes.For):
            return self._build_for(statement)
        if isinstance(statement, nodes.AssignBlock):
            return self._build_assign_block(statement)
        if isinstance(statement, nodes.Assign):
            return self._build_assignment(statement)
        if isinstance(statement, nodes.ExprStmt):
            return self._build_expression(statement.node, statement)
        if isinstance(statement, nodes.Output):
            return self._build_output(statement)
        if isinstance(statement, nodes.CallBlock):
            return self._build_call_block(statement)
        if isinstance(statement, nodes.Break):
            return self._build_terminal(statement, "break")
        if isinstance(statement, nodes.Continue):
            return self._build_terminal(statement, "continue")
        if isinstance(statement, (nodes.Import, nodes.FromImport, nodes.Include, nodes.Extends)):
            return self._build_simple(statement, "template_dependency")
        if isinstance(statement, nodes.FilterBlock):
            return self._build_wrapped(statement, "filter")
        if isinstance(statement, nodes.Block):
            return self._build_wrapped(statement, "template_block")
        if isinstance(statement, nodes.With):
            return self._build_wrapped(statement, "scope")
        if isinstance(statement, nodes.Scope):
            return self.build_sequence(statement.body)

        return self._build_simple(statement, "operation")

    def _tag_detail(self, statement: nodes.Node) -> tuple[str, int]:
        start = max(getattr(statement, "lineno", 1) or 1, 1)
        tag, tag_end = self.locator.next_tag(start)
        return tag or self.locator.line(start).strip(), tag_end

    def _build_simple(self, statement: nodes.Node, kind: str) -> Fragment:
        detail, end = self._tag_detail(statement)
        node_id = self.add_node(kind, detail or type(statement).__name__, statement.lineno, end, detail)
        return Fragment(node_id, [Completion(node_id)])

    def _build_terminal(self, statement: nodes.Node, kind: str) -> Fragment:
        detail, end = self._tag_detail(statement)
        node_id = self.add_node(kind, detail or kind.title(), statement.lineno, end, detail)
        return Fragment(node_id, [Completion(node_id, kind)])

    def _build_if(self, statement: nodes.If) -> Fragment:
        detail, end = self._tag_detail(statement)
        decision = self.add_node(
            "decision",
            detail or "Jinja condition",
            statement.lineno,
            end,
            detail,
            {"construct": "if"},
        )
        body = self.build_sequence(statement.body)
        else_statements = [*getattr(statement, "elif_", []), *statement.else_]
        else_body = self.build_sequence(else_statements)
        completions: list[Completion] = []

        if body.entry:
            self.add_edge(decision, body.entry, "true", "branch")
            completions.extend(body.completions)
        else:
            completions.append(Completion(decision, "normal", "true"))

        if else_body.entry:
            self.add_edge(decision, else_body.entry, "false", "branch")
            completions.extend(else_body.completions)
        else:
            completions.append(Completion(decision, "normal", "false"))

        return Fragment(decision, completions)

    def _build_for(self, statement: nodes.For) -> Fragment:
        detail, end = self._tag_detail(statement)
        loop_node = self.add_node(
            "loop",
            detail or "Jinja loop",
            statement.lineno,
            end,
            detail,
            {"construct": "for"},
        )
        body = self.build_sequence(statement.body)
        else_body = self.build_sequence(statement.else_)
        completions: list[Completion] = []

        if body.entry:
            self.add_edge(loop_node, body.entry, "next item", "loop")
            for completion in body.completions:
                if completion.kind in {"normal", "continue"}:
                    self.add_edge(completion.node_id, loop_node, "repeat", "loop_back")
                elif completion.kind == "break":
                    completions.append(Completion(completion.node_id, "normal", "break"))
                else:
                    completions.append(completion)

        if else_body.entry:
            self.add_edge(loop_node, else_body.entry, "empty / complete", "loop_exit")
            completions.extend(else_body.completions)
        else:
            completions.append(Completion(loop_node, "normal", "complete"))

        return Fragment(loop_node, completions)

    def _build_assign_block(self, statement: nodes.AssignBlock) -> Fragment:
        detail, tag_end = self._tag_detail(statement)
        end = max(_max_lineno(statement), tag_end)
        target = _expression_name(statement.target)
        block_text = self.locator.range_text(statement.lineno, end)
        kind = "sql_capture" if _looks_like_sql(block_text) else "capture"
        label = f"Capture SQL in {target}" if kind == "sql_capture" else f"Capture block in {target}"
        node_id = self.add_node(
            kind,
            label,
            statement.lineno,
            end,
            block_text,
            {"target": target},
        )
        body = self.build_sequence(statement.body)
        if not body.entry:
            return Fragment(node_id, [Completion(node_id)])

        self.add_edge(node_id, body.entry, "capture body")
        finish = self.add_node(
            "capture_end",
            f"Finish capture {target}",
            end,
            end,
            f"Complete the captured value for {target}.",
            {"target": target},
        )
        completions: list[Completion] = []
        for completion in body.completions:
            if completion.kind == "normal":
                self.add_edge(completion.node_id, finish, completion.label)
            else:
                completions.append(completion)
        completions.append(Completion(finish))
        return Fragment(node_id, completions)

    def _build_assignment(self, statement: nodes.Assign) -> Fragment:
        detail, end = self._tag_detail(statement)
        calls = _call_names(statement.node)
        if calls:
            kind, _, completion_kind = _classify_calls(calls, detail)
            if kind == "macro_call":
                kind = "assignment"
        else:
            kind, completion_kind = "assignment", "normal"
        node_id = self.add_node(
            kind,
            detail or "Assign value",
            statement.lineno,
            end,
            detail,
            {
                "calls": calls,
                "target": _expression_name(statement.target),
            },
        )
        return Fragment(node_id, [Completion(node_id, completion_kind)])

    def _build_expression(self, expression: nodes.Node, owner: nodes.Node) -> Fragment:
        detail, end = self._tag_detail(owner)
        calls = _call_names(expression)
        kind, label, completion_kind = _classify_calls(calls, detail)
        node_id = self.add_node(
            kind,
            label,
            owner.lineno,
            end,
            detail,
            {"calls": calls},
        )
        return Fragment(node_id, [Completion(node_id, completion_kind)])

    def _build_output(self, statement: nodes.Output) -> Fragment:
        calls = _call_names(statement)
        start = max(statement.lineno or 1, 1)
        end = max(_max_lineno(statement), start)
        text = self.locator.range_text(start, end)
        template_text = "".join(
            child.data
            for child in statement.nodes
            if isinstance(child, nodes.TemplateData)
        )
        has_template_data = bool(template_text.strip())

        if not has_template_data and not calls:
            return Fragment()

        if has_template_data and _looks_like_sql(text):
            kind = "sql_emit"
            label = f"Emit SQL: {compact(template_text, 72)}"
            completion_kind = "normal"
        elif calls:
            kind, label, completion_kind = _classify_calls(calls, compact(text))
        else:
            stripped = compact(template_text)
            if not stripped:
                return Fragment()
            kind = "template_output"
            label = f"Emit template text: {stripped}"
            completion_kind = "normal"

        node_id = self.add_node(
            kind,
            label,
            start,
            end,
            text,
            {"calls": calls},
        )
        return Fragment(node_id, [Completion(node_id, completion_kind)])

    def _build_call_block(self, statement: nodes.CallBlock) -> Fragment:
        detail, end = self._tag_detail(statement)
        calls = _call_names(statement.call)
        kind, label, completion_kind = _classify_calls(calls, detail)
        call_node = self.add_node(
            kind,
            label,
            statement.lineno,
            end,
            detail,
            {"calls": calls, "construct": "call_block"},
        )
        body = self.build_sequence(statement.body)
        if body.entry:
            self.add_edge(call_node, body.entry, "callback body", "call")
            completions = body.completions
        else:
            completions = [Completion(call_node, completion_kind)]
        return Fragment(call_node, completions)

    def _build_wrapped(self, statement: nodes.Node, kind: str) -> Fragment:
        detail, end = self._tag_detail(statement)
        wrapper = self.add_node(kind, detail or kind.title(), statement.lineno, end, detail)
        body = self.build_sequence(getattr(statement, "body", []))
        if body.entry:
            self.add_edge(wrapper, body.entry)
            return Fragment(wrapper, body.completions)
        return Fragment(wrapper, [Completion(wrapper)])


def _max_lineno(node: nodes.Node) -> int:
    maximum = max(getattr(node, "lineno", 1) or 1, 1)
    for child in node.iter_child_nodes():
        maximum = max(maximum, _max_lineno(child))
    return maximum


def _expression_name(expression: nodes.Node | None) -> str:
    if expression is None:
        return "value"
    if isinstance(expression, nodes.Name):
        return expression.name
    if isinstance(expression, nodes.Getattr):
        base = _expression_name(expression.node)
        return f"{base}.{expression.attr}"
    if isinstance(expression, nodes.Getitem):
        return f"{_expression_name(expression.node)}[...]"
    if isinstance(expression, (nodes.Tuple, nodes.List)):
        return ", ".join(_expression_name(item) for item in expression.items)
    return type(expression).__name__


def _call_names(node: nodes.Node) -> list[str]:
    calls: list[str] = []
    candidates = [node, *node.find_all(nodes.Call)]
    for candidate in candidates:
        if isinstance(candidate, nodes.Call):
            name = _expression_name(candidate.node)
            if name not in calls:
                calls.append(name)
    return calls


def _classify_calls(calls: list[str], fallback: str) -> tuple[str, str, str]:
    normalized = [name.lower() for name in calls]
    primary = calls[0] if calls else ""
    if any(name == "return" or name.endswith(".return") for name in normalized):
        return "return", fallback or "Return from macro", "return"
    if any(name == "run_query" or name.endswith(".run_query") for name in normalized):
        return "warehouse_query", fallback or "Execute warehouse query", "normal"
    if any("raise_compiler_error" in name for name in normalized):
        return "raise", fallback or "Raise dbt compiler error", "raise"
    if any(name == "log" or name.endswith(".log") for name in normalized):
        return "log", fallback or "Write dbt log message", "normal"
    if any(name in {"ref", "source"} for name in normalized):
        return "dbt_reference", fallback or f"Resolve dbt {primary}", "normal"
    if any(name.startswith("adapter.") for name in normalized):
        return "adapter_call", fallback or f"Call {primary}", "normal"
    if any(name.endswith(".append") or name.endswith(".update") for name in normalized):
        return "mutation", fallback or f"Mutate collection with {primary}", "normal"
    if calls:
        return "macro_call", fallback or f"Call {primary}", "normal"
    return "operation", fallback or "Evaluate expression", "normal"


def _looks_like_sql(text: str) -> bool:
    uppercase = f" {text.upper()} "
    sql_markers = (
        " SELECT ",
        " FROM ",
        " UPDATE ",
        " DELETE ",
        " INSERT ",
        " MERGE ",
        " WHERE ",
        " CREATE ",
        " DROP ",
    )
    return any(marker in uppercase for marker in sql_markers)
