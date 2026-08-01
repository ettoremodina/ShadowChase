---
name: html-output-viewer
description: Render a long LLM response, implementation plan, analysis, or Markdown/text file as a polished self-contained HTML reader. Use only when the user explicitly invokes `$html-output-viewer` or explicitly asks to use the html-output-viewer skill. Do not trigger automatically for ordinary plans, reports, long answers, Markdown, or HTML requests.
compatibility: Requires Python 3.9 or newer; the renderer uses only the Python standard library.
metadata:
  category: documents
  scope: global
  retention-class: capability
  maintenance-policy: to_test
  status: active
  origin: local
---

# HTML Output Viewer

Turn long-form LLM output into an easy-to-navigate HTML document while spending tokens on content rather than repeatedly designing a page.

## Invocation boundary

Continue only when the user explicitly named or invoked this skill. The narrow trigger is intentional: most plans and answers are clearer directly in the conversation and should not create an artifact by default.

## Workflow

1. Identify the source content.
   - If the user supplied a Markdown or text file, use it directly.
   - If the content is in the conversation, preserve its wording and structure in a UTF-8 Markdown file. Do not summarize or rewrite it unless requested.
   - If this skill is invoked before the content exists, produce the requested plan or answer as Markdown first.
2. Choose an output path near the source or in the user's requested directory. Use a descriptive kebab-case filename ending in `.html`.
3. Run the bundled renderer:

```powershell
python scripts/render_html.py <input.md> --output <output.html> --title "Document title"
```

Paths are relative to this skill directory. Omit `--title` to infer it from the first level-one heading or the source filename. Use `--open` only when the environment has an interactive browser and opening it helps the user.

4. Verify that the command succeeds and that the output file exists. For important artifacts, also run the bundled tests before delivery:

```powershell
python -m unittest discover -s tests -v
```

5. Return a link to the HTML file. Mention the source Markdown only when it is useful to the user.

## Content guidance

- Prefer meaningful headings because they become the navigation index.
- use colors and bold to highlight words and phrases that are important to the user.
- Use checklists for actionable plans, tables for comparisons, and fenced code blocks for commands or code.
- Keep the document self-contained. The template intentionally uses no external fonts, scripts, stylesheets, or network requests.
- Treat the source as untrusted text. The renderer escapes raw HTML and accepts only safe link schemes.
- Reuse `assets/reader-template.html`; do not redesign the page unless the user explicitly requests a custom visual treatment.

## Script interface

```text
python scripts/render_html.py INPUT [--output FILE] [--title TEXT]
                                    [--subtitle TEXT] [--open]
```

Use `INPUT` as `-` to read UTF-8 content from standard input. The renderer supports headings, paragraphs, links, emphasis, inline code, fenced code blocks, blockquotes, ordered and unordered lists, task lists, tables, and horizontal rules.
