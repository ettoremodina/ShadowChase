---
name: code-flow-explorer
description: Create a hierarchical, source-linked flow visualization for one function, script, dbt/Jinja macro, or small module. Use this skill whenever the user wants to understand or diagram local code execution, branches, loops, generated SQL, macro behavior, or control flow—even if they only ask “how does this function work?” Do not use it for whole-codebase architecture or dependency maps.
compatibility: Requires Python 3.11+, uv, and Graphviz `dot`. The bundled runtime installs Jinja2 locally through uv; it never requires dbt or executes a dbt project.
metadata:
  category: codebase-understanding
  scope: global
  retention-class: capability
  maintenance-policy: to_test
  status: active
  origin: local
---

# Code Flow Explorer

Build a source-faithful interactive HTML explanation of a small code target. Keep deterministic control flow separate from agent-authored semantic interpretation: scripts own nodes and edges; the agent only names and groups source-backed regions.

## Scope

Use this skill for:

- one Python function, method, script, or small module;
- one Jinja template or dbt macro, optionally expanded with every local macro in the same small source file;
- explicit conditions, loops, returns, calls, SQL capture, and known side effects.

Route whole-codebase structure, architecture, or dependency questions to the `understand` family instead. Static diagrams show possible source paths, not observed runtime traces.

## Workflow

Resolve this skill directory as `SKILL_DIR`, then choose an output directory near the user's requested destination.

1. Detect the language.
   - Use `python` for `.py`.
   - Use `jinja-dbt` for `.jinja`, `.jinja2`, `.j2`, or dbt `.sql` containing Jinja delimiters.
   - For ambiguous files, inspect the source before choosing.
2. Read the relevant language guidance:
   - Python: `references/python.md`
   - Jinja/dbt: `references/jinja-dbt.md`
3. Generate the exact intermediate representation:

```powershell
uv run --project "$SKILL_DIR" python "$SKILL_DIR/scripts/code_flow_cli.py" analyze SOURCE `
  --language LANGUAGE `
  --symbol OPTIONAL_SYMBOL `
  --output OUTPUT_DIR/flow-ir.json
```

Omit `--symbol` to analyze the first function/macro, or the module body when no callable exists.
For a Jinja/dbt file that must show every local macro in one graph, add
`--expand-local-macros`. Keep `--symbol` set to the root macro so the artifact
retains a clear orchestration entry point.

4. Read the source and `flow-ir.json`. Create `semantic.json` using `references/semantic-schema.md`.
   - Describe purpose, inputs, outputs, and side effects.
   - Define adaptive semantic phases for Overview and nested semantic steps for Logic.
   - Group by natural responsibilities and alternatives in the target; never force a fixed node count.
   - Make phases and their steps collectively cover all exact nodes without overlapping.
   - Write action-oriented labels and plain-language summaries. They may explain intent, but must not invent behavior absent from the source.
5. Validate and render:

```powershell
uv run --project "$SKILL_DIR" python "$SKILL_DIR/scripts/code_flow_cli.py" render OUTPUT_DIR/flow-ir.json `
  --source SOURCE `
  --semantic OUTPUT_DIR/semantic.json `
  --output OUTPUT_DIR/code-flow.html
```

6. Verify the artifact:

```powershell
uv run --project "$SKILL_DIR" python "$SKILL_DIR/scripts/code_flow_cli.py" validate OUTPUT_DIR/flow-ir.json `
  --semantic OUTPUT_DIR/semantic.json
```

Open the HTML only when the user asked to inspect it interactively or opening it clearly helps. Return links to the HTML and semantic JSON; mention the raw IR only when useful for debugging or extension work.

## Viewer contract

The HTML has three graph levels:

- **Overview** contracts exact nodes into agent-labelled semantic phases and includes a compact explanation of each phase's outcome.
- **Logic** contracts exact nodes into agent-labelled semantic steps nested within those phases.
- **Exact** shows every extracted source operation and possible edge.

Clicking a node inspects it without navigating. Dragging from any graph surface, including a node, pans after a short movement threshold so ordinary clicks remain intact. Use the inspector action, double-click, or Enter to open an Overview phase in focused Logic or a Logic step in focused Exact. Breadcrumbs move back through the focused path, while the view tabs open any complete level. Each full or focused scope remembers its own zoom and pan position; Ctrl+wheel zooms around the pointer. On desktop, the open source inspector reserves space beside the graph so both remain visible; on narrow screens it stays an overlay. The source inspector embeds language-aware syntax highlighting and remains fully offline. The agent-authored Overview legend gives phases semantic color categories, Logic inherits lighter parent colors, and Exact keeps deterministic technical node-kind colors. Every selection remains linked to exact source lines and semantic detail. For dbt/Jinja, Exact visually distinguishes SQL generation from warehouse-executing calls.

## Safety and correctness

- Parse dbt/Jinja statically. Do not run `dbt compile`, `dbt run`, `run_query`, or any warehouse command.
- Existing compiled SQL may be read as supporting evidence, but it does not replace the template-flow graph.
- Treat `run_query` and statement blocks as potential warehouse side effects.
- Treat dynamic dispatch, adapter behavior, reflection, callbacks, and runtime-provided objects as unresolved when static evidence is insufficient.
- Local macro expansion is context-insensitive when a macro has multiple callers; report the resulting call/return paths as a static over-approximation.
- Preserve exact graph edges. If Overview or Logic grouping is poor, fix `semantic.json`, not `flow-ir.json`.

## Adding a language

Follow `references/adapter-contract.md`. Add one adapter module and one language reference file. Reuse the shared model, hierarchy, validation, and HTML renderer.

## Tests

Run deterministic tests before delivering important artifacts:

```powershell
uv run --project "$SKILL_DIR" python -m unittest discover -s "$SKILL_DIR/tests" -v
```
