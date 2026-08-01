---
name: parallel-worktree-tasks
description: Run a user-supplied batch of medium or simple repository tasks concurrently in isolated Git worktrees, then integrate and verify them safely. Use only when the user explicitly invokes this skill or explicitly asks for parallel agents with worktrees; do not use for one small task, tightly coupled changes, or broad planning.
metadata:
  category: engineering-workflow
  scope: global
  retention-class: capability
  maintenance-policy: retire-when-native
  status: active
  origin: local
---

# Parallel Worktree Tasks

Execute a small task graph with fresh subagent contexts and isolated Git worktrees. Keep orchestration shorter than the work: no spec, tracker, extended interview, or architecture exercise unless the user separately requests one.

## Lifecycle note

Treat this skill as a candidate for future retirement. Review it periodically and retire it when Codex reliably recognizes parallelizable medium/simple tasks, creates isolated worktrees, limits context inheritance, and integrates them safely without explicit workflow instructions. Do not expand it into a general planning or shipping pipeline.

## Preflight

1. Confirm that the current directory is a Git repository, multi-agent tools are available, no merge/rebase/cherry-pick is in progress, `HEAD` resolves, and the primary worktree is clean. Never stash, reset, or commit pre-existing user changes to make the check pass.
2. Record the original worktree path, branch, and base commit. If `HEAD` is detached, stop before write work and explain the required branch decision.
3. Make one brief scheduling pass. For each task record only: id, objective, dependencies, likely write scope, and verification command. Do not create a plan artifact.
4. Use one agent when the batch has no useful parallel frontier or worktree overhead would exceed the work.

## Schedule the frontier

- Treat dependencies as a DAG. The frontier is every unfinished task whose dependencies are complete.
- Parallelize read-only tasks freely. Parallelize write tasks only when their expected file ownership is disjoint or conflicts are acceptably low-risk.
- Serialize tasks expected to edit the same files, shared schemas, central registries, lockfiles, generated artifacts, or migrations.
- Give parallel workers distinct ports, temporary directories, and test databases when they use mutable external state; serialize them when isolation is unavailable.
- Never exceed the available agent slots. Keep the primary agent as orchestrator; do not create nested workers unless the host requires it.
- Recompute the frontier after each completion. Start dependent tasks in fresh worktrees from the integrated dependency state.

## Create isolated worktrees

Resolve `scripts/worktree-batch.ps1` relative to this `SKILL.md` and use it; do not hand-roll worktree paths or recursive cleanup.

1. Create one integration worktree from the recorded base commit.
2. Create one task worktree per runnable task. Independent first-wave tasks start from the base commit. Dependent tasks start from the current integration branch after their blockers have been integrated.
3. Keep worktrees outside the repository under the helper's sibling `<repo>.worktrees/<batch>/` root. Use branches named `codex/<batch>/<task>`.

Example:

```powershell
./scripts/worktree-batch.ps1 -Action Create -RepositoryPath C:\repo -BatchId batch-20260731 -TaskId integration -Kind integration -BaseRef <base-sha>
./scripts/worktree-batch.ps1 -Action Create -RepositoryPath C:\repo -BatchId batch-20260731 -TaskId task-a -BaseRef <base-sha>
```

## Dispatch workers

Spawn each worker with no inherited conversation turns when the host supports that control. Its prompt must contain only:

- task objective and acceptance criteria;
- exact worktree path and branch;
- relevant repository instructions and known constraints;
- allowed or expected write scope;
- verification command;
- compact dependency handoff, when applicable.

Require the worker to operate only inside its worktree, preserve unrelated changes, run focused verification, create one final commit, and return: status, commit SHA, files changed, tests run, and any risks. A worker must not merge, push, remove worktrees, or edit the original worktree.

## Integrate away from the original branch

1. Validate every returned commit and inspect its changed-file list. Reassess overlap using actual diffs rather than the initial scope estimate.
2. Merge successful task branches into the integration worktree with `git merge --no-ff` in dependency order. Keeping task commits as ancestors makes later non-forced branch cleanup verifiable. Run focused verification after each risky integration.
3. Worktree isolation prevents simultaneous filesystem writes; it does not prevent Git or semantic conflicts. Resolve a conflict autonomously only when both task intents can clearly be preserved. Inspect both task briefs and both diffs, keep the combined behavior, and rerun relevant tests.
4. If requirements contradict, the intended behavior is ambiguous, or safe resolution would discard one task's work, abort the integration operation in the integration worktree and ask the user. Leave the original branch untouched.
5. Run the appropriate final test suite in the integration worktree.

## Update and clean up

1. Recheck that the original worktree is clean. If its branch advanced, rebase the integration branch onto the new tip inside the integration worktree, resolve only unambiguous conflicts, and rerun final verification. If the original has uncommitted changes, stop.
2. Fast-forward the original branch to the verified integration branch. Never force-update it.
3. After a successful fast-forward, remove only clean batch worktrees with the helper and delete only fully integrated task branches using `-DeleteBranch`. Never use force deletion. Preserve all worktrees and branches when integration or verification fails.
4. Do not push or open a pull request unless the user explicitly requested it.
5. Report the scheduling used, commits integrated, conflicts resolved or deferred, verification results, and cleanup performed.
