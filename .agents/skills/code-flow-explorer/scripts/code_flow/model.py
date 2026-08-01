from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json


@dataclass(slots=True)
class FlowNode:
    id: str
    kind: str
    label: str
    start_line: int
    end_line: int
    detail: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Edge:
    source: str
    target: str
    label: str = ""
    kind: str = "flow"


@dataclass(slots=True)
class FlowGraph:
    schema_version: int
    language: str
    source_path: str
    symbol: str
    title: str
    target_start_line: int
    target_end_line: int
    nodes: list[FlowNode] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def read(cls, path: str | Path) -> "FlowGraph":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        raw["nodes"] = [FlowNode(**node) for node in raw.get("nodes", [])]
        raw["edges"] = [Edge(**edge) for edge in raw.get("edges", [])]
        return cls(**raw)

    def validate(self) -> list[str]:
        issues: list[str] = []
        node_ids = [node.id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            issues.append("Node IDs are not unique.")
        known = set(node_ids)
        for edge in self.edges:
            if edge.source not in known:
                issues.append(f"Edge source does not exist: {edge.source}")
            if edge.target not in known:
                issues.append(f"Edge target does not exist: {edge.target}")
        for node in self.nodes:
            if node.start_line < 1 or node.end_line < node.start_line:
                issues.append(f"Invalid source range for {node.id}.")
        if not any(node.kind == "entry" for node in self.nodes):
            issues.append("Graph has no entry node.")
        if not any(node.kind == "exit" for node in self.nodes):
            issues.append("Graph has no exit node.")
        return issues

