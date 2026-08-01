# Paper Type Definitions & Summary Structures

This document defines how to classify academic papers and what summary structure
to use for each type. The agent reads the preprocessed markdown and matches
against the classification criteria below.

---

## 🆕 Pre-built Section Types (v3 - High-Exposition Learning Suite)

When generating the `summary.json`, use pre-built section types to provide rich, visual exposition for fast learning. All UI rendering logic is handled automatically by `template.html`.

Supported section types and their exact schema:

1. **`card` (Default Text Card)**
   ```json
   { "heading": "Title", "icon": "📌", "content": "Markdown text with **bold**, *italic*, lists, and $latex$ math." }
   ```

2. **`mermaid` (Diagram / Architecture Flowchart)**
   ```json
   {
     "heading": "System Architecture",
     "icon": "🔄",
     "type": "mermaid",
     "description": "High-level overview of model components",
     "code": "graph TD\n  A[Raw Data] --> B[Encoder]\n  B --> C[Decoder]\n  C --> D[Output]"
   }
   ```

3. **`formula` (Mathematical Equation Card)**
   ```json
   {
     "heading": "Core Objective Function",
     "icon": "📐",
     "type": "formula",
     "name": "Bi-level Objective",
     "latex": "\\max_{x, r} \\sum_{i} Y(x_i, r_i)",
     "explanation": "Maximizes total profit across all stations subject to equilibrium constraints.",
     "breakdown": [
       { "symbol": "x_i", "description": "Binary site location indicator" },
       { "symbol": "r_i", "description": "Station vehicle capacity" }
     ]
   }
   ```

4. **`stats` (Metrics Grid)**
   ```json
   {
     "heading": "Key Metrics",
     "icon": "📊",
     "type": "stats",
     "stats": [
       {"icon": "📈", "value": "95.4%", "label": "Accuracy"},
       {"icon": "⚡", "value": "1.2s", "label": "Latency"}
     ]
   }
   ```

5. **`timeline` (Step-by-Step Procedure / History)**
   ```json
   {
     "heading": "Algorithm Pipeline",
     "icon": "⏱️",
     "type": "timeline",
     "steps": [
       {"title": "Initialization", "description": "Set initial parameters..."},
       {"title": "Equilibrium Solve", "description": "Compute SUE traffic flows..."}
     ]
   }
   ```

6. **`table` (Comparison Table)**
   ```json
   {
     "heading": "Method Comparison",
     "icon": "⚖️",
     "type": "table",
     "columns": ["Method", "Accuracy", "Speed"],
     "rows": [
       ["Baseline", "80%", "Fast"],
       ["Proposed", "95%", "Slow"]
     ]
   }
   ```

7. **`quote` (Key Definition / Theorem / Quote)**
   ```json
   {
     "heading": "Core Motivation",
     "icon": "💬",
     "type": "quote",
     "quote": "The fundamental bottleneck is multi-modal traffic congestion under asymmetric demand.",
     "attribution": "Section 2.1"
   }
   ```

8. **`flowchart` (Process Flow Nodes)**
   ```json
   {
     "heading": "Workflow Steps",
     "icon": "🔄",
     "type": "flowchart",
     "steps": [
       {"icon": "📥", "label": "Input", "description": "Raw data"},
       {"icon": "⚙️", "label": "Process", "description": "Filtering"},
       {"icon": "📤", "label": "Output", "description": "Results"}
     ]
   }
   ```

9. **`two_column` (Side-by-Side Comparison / Pros & Cons)**
   ```json
   {
     "heading": "Tradeoff Analysis",
     "icon": "⚖️",
     "type": "two_column",
     "left": {"title": "Operator Benefit", "content": "- Higher revenue\n- Optimal fleet sizing"},
     "right": {"title": "Social Benefit", "content": "- Reduced emissions\n- Lower travel delay"}
   }
   ```

10. **`accordion` (Deep Dive Details)**
    ```json
    {
      "heading": "Technical Deep Dive",
      "icon": "🔍",
      "type": "accordion",
      "items": [
        {"icon": "📐", "title": "Proof Sketch", "content": "Equations..."},
        {"icon": "💻", "title": "Algorithm Pseudo-code", "content": "```python\n...```"}
      ]
    }
    ```

---

## Classification Criteria & Recommended Layouts

Read the preprocessed markdown and classify the paper into one of 6 types.

### 1. `textbook` — Textbook / Practical Guide
- **Color theme:** `purple`
- **Recommended components:** `accordion` for deep dives, `formula` for key equations, `timeline` for learning sequence.

### 2. `research_article` — Research Article
- **Color theme:** `blue`
- **Recommended components:** `mermaid` for architecture, `formula` for loss/objectives, `two_column` for problem vs approach, `stats` for key results.

### 3. `review_survey` — Review / Survey Paper
- **Color theme:** `green`
- **Recommended components:** `table` for taxonomy comparison, `timeline` for historical evolution, `mermaid` for domain classification tree.

### 4. `mathematical` — Mathematical / Methods Paper
- **Color theme:** `red`
- **Recommended components:** `formula` for key theorems/equations, `accordion` for proof sketches, `quote` for primary definitions.

### 5. `case_study` — Case Study / Simulation
- **Color theme:** `orange`
- **Recommended components:** `two_column` for context vs goals, `mermaid` / `flowchart` for model architecture, `stats` for empirical findings.

### 6. `technical_report` — Technical Report / Policy Document
- **Color theme:** `gold`
- **Recommended components:** `table` for policy impact, `two_column` for guidelines vs implementation, `stats` for metrics.
