from __future__ import annotations

from dataclasses import dataclass, field
import re


TAG_PATTERN = re.compile(r"({[{%].*?(?:}}|%}))", re.DOTALL)


@dataclass(slots=True)
class SourceLocator:
    source: str
    lines: list[str] = field(init=False)
    _tag_offsets: dict[int, int] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.lines = self.source.splitlines()

    def line(self, number: int) -> str:
        if 1 <= number <= len(self.lines):
            return self.lines[number - 1]
        return ""

    def range_text(self, start: int, end: int) -> str:
        start = max(start, 1)
        end = min(max(end, start), len(self.lines))
        return "\n".join(self.lines[start - 1 : end])

    def next_tag(self, line_number: int) -> tuple[str, int]:
        """Return the next Jinja tag starting on or after a source line."""
        if not self.lines:
            return "", max(line_number, 1)
        start_index = sum(len(line) + 1 for line in self.lines[: max(line_number - 1, 0)])
        offset = self._tag_offsets.get(line_number, start_index)
        match = TAG_PATTERN.search(self.source, offset)
        if not match:
            return self.line(line_number).strip(), line_number
        match_line = self.source.count("\n", 0, match.start()) + 1
        if match_line > line_number + 1:
            return self.line(line_number).strip(), line_number
        self._tag_offsets[line_number] = match.end()
        end_line = self.source.count("\n", 0, match.end()) + 1
        return " ".join(match.group(1).split()), end_line

    def first_nonempty(self, start: int, end: int) -> str:
        for number in range(max(start, 1), min(end, len(self.lines)) + 1):
            text = self.line(number).strip()
            if text and not text.startswith("{#"):
                return text
        return ""


def compact(text: str, limit: int = 90) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: max(1, limit - 1)].rstrip() + "…"

