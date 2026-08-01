---
name: paper-summarizer-visual
description: >-
  Preprocesses academic PDFs into markdown and images, classifies paper type
  (textbook, research article, review, mathematical, case study, technical report),
  summarizes key concepts into a structured JSON using pre-built visual exposition
  components (Mermaid diagrams, formulas, comparison tables, timelines, stats),
  and renders a self-contained visual infographic HTML page. Always use this skill
  whenever the user asks to summarize a PDF paper, explain a research manuscript,
  visualize an arXiv paper, create an infographic or visual report for a document,
  or study an academic paper visually.
metadata:
  category: research-analysis
  scope: global
  retention-class: hybrid
  maintenance-policy: to_improve
  status: active
  origin: local
---

# Paper Summarizer Visual

Converts a PDF academic paper into a highly informative, self-contained HTML visual summary page. Designed specifically for **efficient learning and optimal technical exposition**.

> **Zero UI Token Waste Principle**: All styling, layouts, search filters, and interactive renderers (MathJax latex, Mermaid diagrams, code copy buttons, accordions, tables) are pre-built inside `template.html`. The agent generates ONLY the structured `summary.json`.

## Dependencies

- **`uv` skill** — Required to run the CLI script with inline dependencies (`pymupdf`, `pymupdf4llm`).

## Quick Start

When the user provides a PDF path or asks to summarize a paper:

```
1. Preprocess the PDF   →  uv run paper_tool.py preprocess --pdf <input.pdf> --output-dir <work-dir>
2. Classify & Summarize →  Read content.md, classify paper, write summary.json
3. Render HTML          →  uv run paper_tool.py render --summary <work-dir>/summary.json --images-dir <work-dir>/images --template <template.html> --output <output.html>
```

## Utility Script

**Location:** `scripts/paper_tool.py` (relative to this skill's directory)

### Subcommand: `preprocess`

Converts a PDF to markdown, extracts images with hash deduplication, and extracts figure captions.

```bash
uv run <skill-dir>/scripts/paper_tool.py preprocess \
  --pdf <input.pdf> \
  --output-dir <work-dir>
```

**Outputs:**
- `<work-dir>/content.md` — Full markdown text
- `<work-dir>/images/` — Extracted unique figures (`fig_001.png`, etc.)
- `<work-dir>/metadata.json` — Metadata, page count, word count, image details, extracted captions

### Subcommand: `render`

Fills the HTML template with the structured `summary.json`.

```bash
uv run <skill-dir>/scripts/paper_tool.py render \
  --summary <work-dir>/summary.json \
  --images-dir <work-dir>/images \
  --template <skill-dir>/resources/template.html \
  --output <output-dir>/<paper_name>.html
```

## Detailed Workflow

### Step 1: Preprocess the PDF

Run the `preprocess` subcommand. Choose a working directory under `summarized_papers/_work_<name>/`.

### Step 2: Classify the Paper Type

Read `references/paper_types.md`. Read the first ~300 lines of `<work-dir>/content.md` and classify into one of 6 types:
`textbook`, `research_article`, `review_survey`, `mathematical`, `case_study`, `technical_report`.

### Step 3: Summarize the Paper (`summary.json`)

Read `<work-dir>/content.md`. Select appropriate section types from `references/paper_types.md`:
- `mermaid` — For architecture, flow, taxonomy
- `formula` — For core equations and mathematical formulations
- `stats` — For key empirical findings/metrics
- `table` — For baseline vs proposed comparisons
- `timeline` — For algorithm steps or historical evolution
- `two_column` — For problem vs solution, pros vs cons
- `accordion` — For proofs, deep dives, pseudo-code
- `card` — For general explanatory text

Write `<work-dir>/summary.json` matching the schema:

```json
{
  "title": "Full Paper Title",
  "authors": "Author1, Author2",
  "year": "2025",
  "paper_type": "research_article",
  "color_theme": "blue",
  "tags": ["Tag1", "Tag2"],
  "tldr": "One-sentence summary of the paper's core contribution.",
  "sections": [
    {
      "heading": "Architecture Flow",
      "icon": "🔄",
      "type": "mermaid",
      "code": "graph LR\n  A --> B"
    },
    {
      "heading": "Core Equation",
      "icon": "📐",
      "type": "formula",
      "name": "Objective Function",
      "latex": "L(\\theta) = \\mathbb{E}[\\log P(y|x)]",
      "explanation": "Maximizes expected log-likelihood.",
      "breakdown": [
        {"symbol": "\\theta", "description": "Model parameters"}
      ]
    }
  ],
  "key_figures": [
    {"image": "fig_001.png", "caption": "System Architecture Overview"}
  ],
  "key_takeaways": [
    "First core insight",
    "Second core insight"
  ]
}
```

### Step 4: Render the HTML

Run `paper_tool.py render`. Save to `summarized_papers/<paper_name>.html`.

### Step 5: Report to User

Inform the user:
1. Detected paper type and color theme
2. Output HTML file location
3. Summary of key sections generated
