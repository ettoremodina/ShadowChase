---
summary: "Why medium/simple task batches use an explicit, temporary worktree-based parallel orchestration skill."
created-at: 2026-07-31T15:59:56.9728551Z
updated-at: 2026-07-31T15:59:56.9728551Z
---

# Parallel worktree task orchestration

## Decision

Use `parallel-worktree-tasks` only after explicit user invocation for batches of medium or simple repository tasks. The user prefers control over individual tasks and does not want this capability to become a verbose planning or autonomous shipping pipeline.

## Rationale

The workflow borrows the useful primitives from Matt Pocock's engineering skills—fresh task contexts, explicit blocking edges, frontier scheduling, and parallel subagents—without adopting the full spec, tracker, TDD, review, and commit pipeline. Upstream reviewed 2026-07-31: https://github.com/mattpocock/skills/tree/main/skills/engineering (public, MIT-licensed, actively maintained, and widely used).

## Safety boundary

Write workers use separate Git worktrees and branches. A dedicated integration worktree receives task commits before the original branch changes. Expected overlapping writes are serialized; accidental Git or semantic conflicts are resolved autonomously only when both intents are clear and tests can verify the combination. Ambiguous or contradictory intent returns to the user. Never stash, reset, force-update, push, or discard dirty worktrees as part of this workflow.

## Lifecycle

This is deliberately transitional. Review periodically and retire the skill when Codex natively and reliably detects suitable task batches, isolates parallel writes in worktrees, controls inherited context, and integrates verified results without explicit workflow instructions. Do not grow it into a general planner.
