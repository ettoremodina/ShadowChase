from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from code_flow.model import FlowGraph


class LanguageAdapter(ABC):
    language: str

    @abstractmethod
    def analyze(
        self,
        source_path: str | Path,
        source: str,
        symbol: str | None,
        *,
        expand_local_macros: bool = False,
    ) -> FlowGraph:
        """Convert one small source target into the common flow graph."""
