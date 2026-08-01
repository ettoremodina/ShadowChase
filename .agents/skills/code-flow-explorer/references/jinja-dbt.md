# Jinja and dbt flow interpretation

Use this adapter for dbt macros and Jinja templates. It parses source with Jinja2 and never invokes dbt.

## Semantics

Jinja evaluation generates text. In dbt, that text is usually SQL. Keep three ideas distinct:

- `sql_capture`: build a SQL string inside `{% set name %}...{% endset %}`;
- `sql_emit`: emit SQL or template text to the caller;
- `warehouse_query`: call `run_query`, which may execute SQL against a warehouse in real dbt execution.

Also distinguish:

- `dbt_reference`: `ref()` or `source()`;
- `adapter_call`: calls through the dbt adapter;
- `macro_call`: another macro or unresolved callable;
- `log`: diagnostic output;
- `mutation`: namespace or collection mutation.

## dbt phases

`execute` is false while dbt parses a project and true during compilation/execution. A diagram must preserve `if execute` branches because database-dependent values and side effects can differ between those phases.

## Static limitations

- Runtime values supplied by dbt context are unknown.
- `adapter.dispatch` and overridden package macros are dynamic.
- Graph metadata, warehouse query results, relation existence, and adapter quoting cannot be resolved from one macro.
- The adapter recognizes standard Jinja plus `do`, `break`, and `continue`. Custom dbt block syntax outside ordinary macros may require a later dbt-specific extension.
- Source ranges are derived from Jinja AST line data and source tags. Multiple inline tags on one line are supported best-effort.

Do not run `dbt compile` automatically. It can connect to the warehouse and execute `run_query` calls. Existing compiled artifacts may be read as optional evidence.

## Expanding local macros

Use `--expand-local-macros` when the user wants every macro in one Jinja/dbt
source file represented in a single graph. The selected `--symbol` remains the
root explanation target, while the exact graph includes every local macro and
connects statically resolvable local call sites with call and return edges.

The expansion is context-insensitive: when the same macro has multiple callers,
its exit can connect to more than one continuation. Treat those paths as a
static over-approximation. Calls through packages, `adapter.dispatch`, imported
templates, variables, callbacks, or other runtime resolution remain unresolved.
