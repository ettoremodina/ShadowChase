#!/usr/bin/env python3
"""Render Markdown-like LLM output as a self-contained HTML reader."""

from __future__ import annotations

import argparse
import html
import re
import sys
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Match


SAFE_LINK = re.compile(r"^(?:https?://|mailto:|#|/|\.\.?/)", re.IGNORECASE)
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
LIST_ITEM = re.compile(r"^\s*([-+*]|\d+[.)])\s+(.+)$")
TABLE_DIVIDER = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$")


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "section"


def strip_markdown(value: str) -> str:
    value = re.sub(r"`([^`]+)`", r"\1", value)
    value = re.sub(r"\[([^]]+)]\([^)]+\)", r"\1", value)
    return re.sub(r"[*_~]", "", value).strip()


def render_inline(value: str) -> str:
    code_tokens: list[str] = []

    def stash_code(match: Match[str]) -> str:
        code_tokens.append(f"<code>{html.escape(match.group(1))}</code>")
        return f"\x00CODE{len(code_tokens) - 1}\x00"

    value = re.sub(r"`([^`\n]+)`", stash_code, value)
    value = html.escape(value, quote=False)

    def render_link(match: Match[str]) -> str:
        label, target = match.group(1), html.unescape(match.group(2)).strip()
        if not SAFE_LINK.match(target):
            return label
        href = html.escape(target, quote=True)
        external = ' target="_blank" rel="noopener noreferrer"' if target.startswith(("http://", "https://")) else ""
        return f'<a href="{href}"{external}>{label}</a>'

    value = re.sub(r"\[([^]]+)]\(([^)\s]+)\)", render_link, value)
    value = re.sub(r"\*\*(.+?)\*\*|__(.+?)__", lambda match: f"<strong>{match.group(1) or match.group(2)}</strong>", value)
    value = re.sub(r"~~(.+?)~~", r"<del>\1</del>", value)
    value = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)|(?<!_)_([^_\n]+)_(?!_)", lambda match: f"<em>{match.group(1) or match.group(2)}</em>", value)

    for index, token in enumerate(code_tokens):
        value = value.replace(f"\x00CODE{index}\x00", token)
    return value


class MarkdownRenderer:
    def __init__(self) -> None:
        self.heading_ids: dict[str, int] = {}
        self.toc: list[tuple[int, str, str]] = []

    def unique_heading_id(self, title: str) -> str:
        base = slugify(strip_markdown(title))
        count = self.heading_ids.get(base, 0)
        self.heading_ids[base] = count + 1
        return base if count == 0 else f"{base}-{count + 1}"

    def starts_block(self, lines: list[str], index: int) -> bool:
        line = lines[index]
        if not line.strip():
            return True
        if HEADING.match(line) or LIST_ITEM.match(line) or line.lstrip().startswith((">", "```", "~~~")):
            return True
        if re.match(r"^\s*(?:---+|___+|\*\*\*+)\s*$", line):
            return True
        return index + 1 < len(lines) and "|" in line and TABLE_DIVIDER.match(lines[index + 1]) is not None

    def render(self, source: str) -> str:
        lines = source.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        blocks: list[str] = []
        index = 0
        while index < len(lines):
            line = lines[index]
            if not line.strip():
                index += 1
                continue
            if line.lstrip().startswith(("```", "~~~")):
                block, index = self.render_code_block(lines, index)
            elif HEADING.match(line):
                block, index = self.render_heading(line), index + 1
            elif re.match(r"^\s*(?:---+|___+|\*\*\*+)\s*$", line):
                block, index = "<hr>", index + 1
            elif line.lstrip().startswith(">"):
                block, index = self.render_quote(lines, index)
            elif index + 1 < len(lines) and "|" in line and TABLE_DIVIDER.match(lines[index + 1]):
                block, index = self.render_table(lines, index)
            elif LIST_ITEM.match(line):
                block, index = self.render_list(lines, index)
            else:
                block, index = self.render_paragraph(lines, index)
            blocks.append(block)
        return "\n".join(blocks)

    def render_code_block(self, lines: list[str], index: int) -> tuple[str, int]:
        opening = lines[index].lstrip()
        marker = opening[:3]
        language = opening[3:].strip()
        index += 1
        body: list[str] = []
        while index < len(lines) and not lines[index].lstrip().startswith(marker):
            body.append(lines[index])
            index += 1
        if index < len(lines):
            index += 1
        language_label = html.escape(language or "text")
        language_class = re.sub(r"[^a-zA-Z0-9_-]", "", language)
        return (
            '<div class="code-shell">'
            f'<div class="code-bar"><span>{language_label}</span><button type="button" class="copy-code">Copy</button></div>'
            f'<pre><code class="language-{language_class}">{html.escape(chr(10).join(body))}</code></pre></div>',
            index,
        )

    def render_heading(self, line: str) -> str:
        match = HEADING.match(line)
        assert match is not None
        level = len(match.group(1))
        title = match.group(2)
        heading_id = self.unique_heading_id(title)
        if level <= 3:
            self.toc.append((level, strip_markdown(title), heading_id))
        return f'<h{level} id="{heading_id}"><a class="heading-anchor" href="#{heading_id}">{render_inline(title)}</a></h{level}>'

    def render_quote(self, lines: list[str], index: int) -> tuple[str, int]:
        quote_lines: list[str] = []
        while index < len(lines) and lines[index].lstrip().startswith(">"):
            quote_lines.append(re.sub(r"^\s*>\s?", "", lines[index]))
            index += 1
        nested = MarkdownRenderer().render("\n".join(quote_lines))
        return f"<blockquote>{nested}</blockquote>", index

    def render_table(self, lines: list[str], index: int) -> tuple[str, int]:
        headers = self.split_table_row(lines[index])
        index += 2
        rows: list[list[str]] = []
        while index < len(lines) and lines[index].strip() and "|" in lines[index]:
            rows.append(self.split_table_row(lines[index]))
            index += 1
        header_html = "".join(f"<th>{render_inline(cell)}</th>" for cell in headers)
        body_html = "".join(
            "<tr>" + "".join(f"<td>{render_inline(cell)}</td>" for cell in row) + "</tr>" for row in rows
        )
        return f'<div class="table-wrap"><table><thead><tr>{header_html}</tr></thead><tbody>{body_html}</tbody></table></div>', index

    @staticmethod
    def split_table_row(line: str) -> list[str]:
        return [cell.strip() for cell in line.strip().strip("|").split("|")]

    def render_list(self, lines: list[str], index: int) -> tuple[str, int]:
        first = LIST_ITEM.match(lines[index])
        assert first is not None
        ordered = first.group(1)[0].isdigit()
        tag = "ol" if ordered else "ul"
        items: list[str] = []
        while index < len(lines):
            match = LIST_ITEM.match(lines[index])
            if match is None or match.group(1)[0].isdigit() != ordered:
                break
            item = match.group(2)
            task = re.match(r"^\[([ xX])]\s+(.+)$", item)
            if task:
                checked = " checked" if task.group(1).lower() == "x" else ""
                item_html = f'<input type="checkbox" disabled{checked}> <span>{render_inline(task.group(2))}</span>'
                items.append(f'<li class="task-item">{item_html}</li>')
            else:
                items.append(f"<li>{render_inline(item)}</li>")
            index += 1
        return f"<{tag}>" + "".join(items) + f"</{tag}>", index

    def render_paragraph(self, lines: list[str], index: int) -> tuple[str, int]:
        paragraph: list[str] = []
        while index < len(lines) and (not paragraph or not self.starts_block(lines, index)):
            if not lines[index].strip():
                break
            paragraph.append(lines[index].strip())
            index += 1
        return f"<p>{render_inline(' '.join(paragraph))}</p>", index


def render_toc(items: list[tuple[int, str, str]]) -> str:
    if not items:
        return '<p class="toc-empty">No headings found.</p>'
    return "\n".join(
        f'<a class="toc-link toc-level-{level}" href="#{heading_id}">{html.escape(title)}</a>'
        for level, title, heading_id in items
    )


def infer_title(source: str, source_label: str) -> str:
    for line in source.splitlines():
        match = re.match(r"^#\s+(.+)$", line.strip())
        if match:
            return strip_markdown(match.group(1))
    if source_label == "standard input":
        return "LLM Output"
    return Path(source_label).stem.replace("-", " ").replace("_", " ").title()


def build_document(source: str, source_label: str, title: str | None, subtitle: str | None, template_path: Path) -> str:
    renderer = MarkdownRenderer()
    content = renderer.render(source)
    document_title = title or infer_title(source, source_label)
    word_count = len(re.findall(r"\b\w+\b", source))
    reading_time = max(1, round(word_count / 220))
    generated_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    replacements = {
        "{{TITLE}}": html.escape(document_title),
        "{{SUBTITLE}}": html.escape(subtitle or "Structured long-form output"),
        "{{SOURCE_LABEL}}": html.escape(source_label),
        "{{GENERATED_AT}}": html.escape(generated_at),
        "{{WORD_COUNT}}": f"{word_count:,}",
        "{{READING_TIME}}": str(reading_time),
        "{{TOC}}": render_toc(renderer.toc),
        "{{CONTENT}}": content,
    }
    document = template_path.read_text(encoding="utf-8")
    for placeholder, replacement in replacements.items():
        document = document.replace(placeholder, replacement)
    return document


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="UTF-8 Markdown/text file, or - for standard input")
    parser.add_argument("--output", "-o", help="Output HTML path; defaults beside the input")
    parser.add_argument("--title", help="Document title; inferred when omitted")
    parser.add_argument("--subtitle", help="Short label displayed below the title")
    parser.add_argument("--open", action="store_true", dest="open_browser", help="Open the result in the default browser")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.input == "-":
        source = sys.stdin.read()
        source_label = "standard input"
        output_path = Path(args.output or "llm-output.html")
    else:
        input_path = Path(args.input).expanduser().resolve()
        source = input_path.read_text(encoding="utf-8")
        source_label = input_path.name
        output_path = Path(args.output).expanduser() if args.output else input_path.with_suffix(".html")
    output_path = output_path.resolve()
    template_path = Path(__file__).resolve().parents[1] / "assets" / "reader-template.html"
    document = build_document(source, source_label, args.title, args.subtitle, template_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")
    print(output_path)
    if args.open_browser:
        webbrowser.open(output_path.as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
