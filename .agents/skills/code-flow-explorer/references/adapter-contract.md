# Language adapter contract

Add languages without changing hierarchy or rendering code.

## Required module behavior

Create `scripts/code_flow/adapters/<language>.py` with a class implementing:

```python
class LanguageAdapter:
    language: str

    def analyze(
        self,
        source_path: str | Path,
        source: str,
        symbol: str | None,
        *,
        expand_local_macros: bool = False,
    ) -> FlowGraph:
        ...
```

Register the adapter and language detection rule in `adapters/__init__.py`.
Adapters that do not support local macro expansion should reject a true
`expand_local_macros` value with a clear error.

## Common IR

Return a `FlowGraph` with:

- stable, unique node IDs;
- one `entry` and one `exit`;
- source-backed `start_line` and `end_line` for every node;
- explicit branch and loop edge labels;
- warnings for semantics that static analysis cannot resolve.

Useful common node kinds:

- boundaries: `entry`, `exit`;
- control: `decision`, `loop`, `return`, `raise`, `break`, `continue`;
- operations: `assignment`, `call`, `mutation`, `operation`;
- templating: `sql_capture`, `sql_emit`, `template_output`;
- dbt: `warehouse_query`, `dbt_reference`, `adapter_call`, `macro_call`.

Add a new kind only when it changes comprehension or visual treatment. Unknown kinds render safely as ordinary process boxes.

## Completion semantics

Adapters should distinguish normal completion from `return`, `raise`, `break`, and `continue`. Sequential construction must connect only normal completions to the next statement. Loops resolve `break` and `continue`; callable boundaries resolve `return`.

## Tests and documentation

For each adapter:

1. Add a compact fixture covering a branch, loop, call, and terminal path.
2. Assert entry/exit presence, edge validity, and expected node kinds.
3. Add `references/<language>.md` explaining language-specific interpretation and limitations.
4. Add at least one skill eval prompt that naturally targets the language.
