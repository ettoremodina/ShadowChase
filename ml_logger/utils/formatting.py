"""Regex-based Rich highlighting for project log messages."""

import re
from rich.text import Text

class Highlighter:
    """Compile configured regex rules and apply their Rich styles."""

    def __init__(self, highlight_rules: list):
        self.highlight_rules = []
        for rule in highlight_rules:
            # Pre-compile the regex patterns for performance
            self.highlight_rules.append((re.compile(rule['pattern'], flags=re.IGNORECASE), rule['style']))

    def apply(self, text: str) -> Text:
        """Applies highlighting rules and returns a Rich Text object."""
        text_obj = Text.from_markup(text)
        return self(text_obj)

    def __call__(self, text: Text) -> Text:
        """Apply configured styles to a Rich log message."""
        text_obj = text.copy()
        for pattern, style in self.highlight_rules:
            for match in pattern.finditer(text_obj.plain):
                text_obj.stylize(style, match.start(), match.end())
        return text_obj
