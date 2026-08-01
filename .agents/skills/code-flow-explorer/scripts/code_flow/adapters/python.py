from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

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


class PythonAdapter(LanguageAdapter):
    language = "python"

    def analyze(
        self,
        source_path: str | Path,
        source: str,
        symbol: str | None,
        *,
        expand_local_macros: bool = False,
    ) -> FlowGraph:
        if expand_local_macros:
            raise ValueError("Local macro expansion is only supported for jinja-dbt.")
        tree = ast.parse(source, filename=str(source_path), type_comments=True)
        callables = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]

        target: ast.AST
        body: list[ast.stmt]
        target_name: str
        if symbol:
            match = next((node for node in callables if node.name == symbol), None)
            if match is None:
                names = ", ".join(node.name for node in callables) or "(none)"
                raise ValueError(f"Function {symbol!r} was not found. Available: {names}")
            target = match
            body = list(match.body)
            target_name = match.name
        elif callables:
            target = callables[0]
            body = list(target.body)
            target_name = target.name
        else:
            target = tree
            body = list(tree.body)
            target_name = Path(source_path).stem

        locator = SourceLocator(source)
        builder = _PythonFlowBuilder(source, locator)
        start_line = max(getattr(target, "lineno", 1), 1)
        end_line = max(getattr(target, "end_lineno", len(locator.lines)), start_line)
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
            "Function or module execution completes.",
        )
        fragment = builder.build_sequence(body)
        if fragment.entry:
            builder.add_edge(entry, fragment.entry)
            for completion in fragment.completions:
                label = completion.label or (
                    completion.kind if completion.kind in {"return", "raise"} else ""
                )
                builder.add_edge(completion.node_id, exit_node, label, completion.kind)
        else:
            builder.add_edge(entry, exit_node)

        return FlowGraph(
            schema_version=1,
            language=self.language,
            source_path=str(Path(source_path)),
            symbol=target_name,
            title=f"Flow of Python target {target_name}",
            target_start_line=start_line,
            target_end_line=end_line,
            nodes=builder.nodes,
            edges=builder.edges,
            warnings=builder.warnings,
        )


class _PythonFlowBuilder:
    def __init__(self, source: str, locator: SourceLocator) -> None:
        self.source = source
        self.locator = locator
        self.nodes: list[FlowNode] = []
        self.edges: list[Edge] = []
        self.warnings: list[str] = []
        self._counter = 0

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
        self.nodes.append(
            FlowNode(
                id=node_id,
                kind=kind,
                label=compact(label),
                start_line=max(start_line, 1),
                end_line=max(end_line or start_line, start_line),
                detail=detail or label,
                metadata=metadata or {},
            )
        )
        return node_id

    def add_edge(self, source: str, target: str, label: str = "", kind: str = "flow") -> None:
        edge = Edge(source=source, target=target, label=label, kind=kind)
        if edge not in self.edges:
            self.edges.append(edge)

    def build_sequence(self, statements: Iterable[ast.stmt]) -> Fragment:
        result = Fragment()
        active: list[Completion] = []
        abnormal: list[Completion] = []
        for statement in statements:
            fragment = self.build_statement(statement)
            if not fragment.entry:
                continue
            if result.entry is None:
                result.entry = fragment.entry
            for completion in active:
                self.add_edge(completion.node_id, fragment.entry, completion.label)
            active = [item for item in fragment.completions if item.kind == "normal"]
            abnormal.extend(item for item in fragment.completions if item.kind != "normal")
        result.completions = [*active, *abnormal]
        return result

    def build_statement(self, statement: ast.stmt) -> Fragment:
        if isinstance(statement, ast.If):
            return self._build_if(statement)
        if isinstance(statement, (ast.For, ast.AsyncFor, ast.While)):
            return self._build_loop(statement)
        if isinstance(statement, ast.Return):
            return self._build_terminal(statement, "return")
        if isinstance(statement, ast.Raise):
            return self._build_terminal(statement, "raise")
        if isinstance(statement, ast.Break):
            return self._build_terminal(statement, "break")
        if isinstance(statement, ast.Continue):
            return self._build_terminal(statement, "continue")
        if isinstance(statement, (ast.Try, ast.TryStar)):
            return self._build_try(statement)
        if isinstance(statement, ast.Match):
            return self._build_match(statement)
        if isinstance(statement, (ast.With, ast.AsyncWith)):
            return self._build_wrapped(statement, "context")
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return self._build_simple(statement, "definition")
        if isinstance(statement, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
            return self._build_simple(statement, "assignment")
        if isinstance(statement, ast.Expr):
            return self._build_expression(statement)
        if isinstance(statement, (ast.Import, ast.ImportFrom)):
            return self._build_simple(statement, "import")
        if isinstance(statement, (ast.Assert, ast.Delete, ast.Pass, ast.Global, ast.Nonlocal)):
            return self._build_simple(statement, "operation")
        return self._build_simple(statement, "operation")

    def _source(self, node: ast.AST) -> str:
        return ast.get_source_segment(self.source, node) or self.locator.line(
            getattr(node, "lineno", 1)
        ).strip()

    def _range(self, node: ast.AST) -> tuple[int, int]:
        start = max(getattr(node, "lineno", 1), 1)
        end = max(getattr(node, "end_lineno", start), start)
        return start, end

    def _build_simple(self, statement: ast.AST, kind: str) -> Fragment:
        start, end = self._range(statement)
        detail = self._source(statement)
        node_id = self.add_node(kind, detail, start, end, detail)
        return Fragment(node_id, [Completion(node_id)])

    def _build_terminal(self, statement: ast.AST, kind: str) -> Fragment:
        start, end = self._range(statement)
        detail = self._source(statement)
        node_id = self.add_node(kind, detail or kind.title(), start, end, detail)
        return Fragment(node_id, [Completion(node_id, kind)])

    def _build_expression(self, statement: ast.Expr) -> Fragment:
        start, end = self._range(statement)
        detail = self._source(statement)
        calls = [_call_name(call.func) for call in ast.walk(statement) if isinstance(call, ast.Call)]
        kind = "call" if calls else "operation"
        node_id = self.add_node(kind, detail, start, end, detail, {"calls": calls})
        return Fragment(node_id, [Completion(node_id)])

    def _build_if(self, statement: ast.If) -> Fragment:
        start = statement.lineno
        condition = ast.get_source_segment(self.source, statement.test) or "condition"
        decision = self.add_node(
            "decision",
            f"if {condition}",
            start,
            getattr(statement.test, "end_lineno", start),
            condition,
        )
        body = self.build_sequence(statement.body)
        else_body = self.build_sequence(statement.orelse)
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

    def _build_loop(self, statement: ast.For | ast.AsyncFor | ast.While) -> Fragment:
        if isinstance(statement, (ast.For, ast.AsyncFor)):
            target = ast.get_source_segment(self.source, statement.target) or "item"
            iterator = ast.get_source_segment(self.source, statement.iter) or "iterable"
            label = f"for {target} in {iterator}"
            condition_end = getattr(statement.iter, "end_lineno", statement.lineno)
        else:
            condition = ast.get_source_segment(self.source, statement.test) or "condition"
            label = f"while {condition}"
            condition_end = getattr(statement.test, "end_lineno", statement.lineno)
        loop_node = self.add_node(
            "loop",
            label,
            statement.lineno,
            condition_end,
            label,
        )
        body = self.build_sequence(statement.body)
        else_body = self.build_sequence(statement.orelse)
        completions: list[Completion] = []
        if body.entry:
            self.add_edge(loop_node, body.entry, "next iteration", "loop")
            for completion in body.completions:
                if completion.kind in {"normal", "continue"}:
                    self.add_edge(completion.node_id, loop_node, "repeat", "loop_back")
                elif completion.kind == "break":
                    completions.append(Completion(completion.node_id, "normal", "break"))
                else:
                    completions.append(completion)
        if else_body.entry:
            self.add_edge(loop_node, else_body.entry, "complete", "loop_exit")
            completions.extend(else_body.completions)
        else:
            completions.append(Completion(loop_node, "normal", "complete"))
        return Fragment(loop_node, completions)

    def _build_try(self, statement: ast.Try | ast.TryStar) -> Fragment:
        try_node = self.add_node(
            "decision",
            "try block",
            statement.lineno,
            statement.lineno,
            self.locator.line(statement.lineno).strip(),
            {"construct": "try"},
        )
        body = self.build_sequence(statement.body)
        completions: list[Completion] = []
        if body.entry:
            self.add_edge(try_node, body.entry, "normal", "branch")
            completions.extend(body.completions)
        for handler in statement.handlers:
            handler_type = (
                ast.get_source_segment(self.source, handler.type)
                if handler.type is not None
                else "any exception"
            )
            handler_body = self.build_sequence(handler.body)
            if handler_body.entry:
                self.add_edge(try_node, handler_body.entry, f"except {handler_type}", "exception")
                completions.extend(handler_body.completions)
        else_body = self.build_sequence(statement.orelse)
        if else_body.entry:
            normal = [item for item in completions if item.kind == "normal"]
            completions = [item for item in completions if item.kind != "normal"]
            for completion in normal:
                self.add_edge(completion.node_id, else_body.entry)
            completions.extend(else_body.completions)
        final_body = self.build_sequence(statement.finalbody)
        if final_body.entry:
            previous = completions
            completions = []
            for completion in previous:
                self.add_edge(completion.node_id, final_body.entry, completion.label, "finally")
            completions.extend(final_body.completions)
            self.warnings.append(
                "Python finally control flow is simplified; abrupt completions are shown entering finally."
            )
        return Fragment(try_node, completions or [Completion(try_node)])

    def _build_match(self, statement: ast.Match) -> Fragment:
        subject = ast.get_source_segment(self.source, statement.subject) or "value"
        match_node = self.add_node(
            "decision",
            f"match {subject}",
            statement.lineno,
            getattr(statement.subject, "end_lineno", statement.lineno),
            subject,
            {"construct": "match"},
        )
        completions: list[Completion] = []
        for case in statement.cases:
            pattern = ast.get_source_segment(self.source, case.pattern) or ast.dump(case.pattern)
            body = self.build_sequence(case.body)
            if body.entry:
                self.add_edge(match_node, body.entry, f"case {compact(pattern, 40)}", "branch")
                completions.extend(body.completions)
        return Fragment(match_node, completions or [Completion(match_node)])

    def _build_wrapped(self, statement: ast.With | ast.AsyncWith, kind: str) -> Fragment:
        start, end = self._range(statement)
        header = self.locator.line(start).strip()
        wrapper = self.add_node(kind, header, start, start, header)
        body = self.build_sequence(statement.body)
        if body.entry:
            self.add_edge(wrapper, body.entry)
            return Fragment(wrapper, body.completions)
        return Fragment(wrapper, [Completion(wrapper)])


def _call_name(expression: ast.AST) -> str:
    if isinstance(expression, ast.Name):
        return expression.id
    if isinstance(expression, ast.Attribute):
        return f"{_call_name(expression.value)}.{expression.attr}"
    if isinstance(expression, ast.Subscript):
        return f"{_call_name(expression.value)}[...]"
    return type(expression).__name__
