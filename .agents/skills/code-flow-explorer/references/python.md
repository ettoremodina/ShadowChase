# Python flow interpretation

The Python adapter uses the standard-library `ast` module and does not execute or import the target.

It models:

- sequential statements and calls;
- `if` branches;
- `for` and `while` loops;
- `break`, `continue`, `return`, and `raise`;
- `try` handlers and simplified `finally` flow;
- `match` cases;
- context managers and nested definitions.

## Static limitations

- Dynamic dispatch, decorators, reflection, monkey patching, callbacks, generators, and async scheduling are not resolved.
- Calls are represented at the call site; callees are not inlined.
- Exception edges are structural approximations unless explicitly represented by `try`/`raise`.
- `finally` preserves the visible cleanup path but does not model every Python completion-state nuance.

Describe the result as source-faithful possible flow, not an observed runtime trace.

