---
summary: "Use skill-benchmarker only when the user explicitly invokes it; ordinary skill work must not launch comparative benchmarks."
created-at: 2026-07-31T14:21:50.7675051Z
updated-at: 2026-07-31T14:21:50.7675051Z
---

# Skill benchmarking is explicit opt-in

# Skill benchmarking is explicit opt-in

`skill-benchmarker` is reserved for explicit invocation by name. Ordinary skill creation, editing, validation, review, and improvement must not automatically launch baseline runs, grading agents, benchmark viewers, blind comparisons, repeated iterations, or trigger-description optimization.

The built-in Codex `skill-creator` handles normal skill creation and maintenance. The repository-local benchmarker retains the evaluation utilities adapted from Anthropic's official `skill-creator`, with provenance recorded in its frontmatter.

A benchmark defaults to one focused comparison iteration and does not modify the target skill. Blind comparison, repeated improvement, and trigger-description optimization each require an additional explicit request.

This decision reduces context and execution overhead and removes the duplicate local `skill-creator` name.

Recorded: 2026-07-31.
