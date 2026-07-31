---
name: memory-manager
description: Manage durable AgentWorkflow memory whenever the user asks to remember, save, learn, record, recall, organize, index, or classify information, or when durable project decisions, reusable preferences, recurring mistakes, or workflow knowledge should persist. Use it to route information between global and project-local memory, create new categories safely, maintain every hierarchical INDEX.md through the deterministic memory control script, and keep AGENTS.md compact by moving conditional detail to guides or skills.
compatibility: Windows PowerShell or PowerShell 7; the AgentWorkflow repository and scripts/memory-control-center.ps1 must be available.
metadata:
  category: knowledge-management
  scope: global
  retention-class: durable
  maintenance-policy: review-periodically
  status: active
  origin: local
---

# Memory Manager

Keep durable knowledge easy to discover without loading the entire memory tree or turning `AGENTS.md` into a handbook.

## Route before writing

1. Decide whether the information is durable. Do not store transient task state, secrets, or facts that are easy to derive from the repository.
2. Choose scope:
   - global for cross-project preferences, standards, reusable workflows, and broadly useful knowledge;
   - local for project decisions, domain facts, recurring project-specific mistakes, and project-only sources.
3. Decide the canonical form:
   - always-on steering belongs in `AGENTS.md`;
   - a reusable procedure belongs in a skill;
   - a project-specific conditional procedure belongs in `docs/agent-guides/`;
   - deterministic maintenance belongs in a script;
   - durable facts, decisions, preferences, and lessons belong in memory.
4. Read the selected root `INDEX.md`, then only the relevant child indexes. Search existing categories and records before creating anything new.

## Write memory safely

Use the source repository recorded in `.agents/control-center.json` to locate `scripts/memory-control-center.ps1`. For work inside AgentWorkflow itself, use the repository root directly.

When adding a new record, choose a focused kebab-case relative path and a one-sentence summary. Let the script create missing directories and refresh all affected indexes:

```powershell
& "$sourceRepository\scripts\memory-control-center.ps1" `
  -Action Add `
  -MemoryRoot $memoryRoot `
  -RelativePath "decisions/api-versioning.md" `
  -Title "API versioning" `
  -Summary "Why the project keeps explicit API versions." `
  -Body "The durable decision and its reasoning."
```

If a record already exists, update it deliberately with the normal file-editing workflow, preserve useful provenance, and then run:

```powershell
& "$sourceRepository\scripts\memory-control-center.ps1" `
  -Action Reindex `
  -MemoryRoot $memoryRoot
```

Finish with `-Action Check`. A successful check is the guarantee that every directory has an `INDEX.md` and every generated inventory matches the files on disk.

## Create categories conservatively

- Reuse an existing category when its index description fits.
- Create a new category automatically when the information has a stable, distinct subject and no existing category fits.
- Prefer specific nouns in kebab-case; do not create vague buckets such as `misc`, `other`, or `notes`.
- Do not rename, merge, or delete existing categories automatically. Those operations can break references and require deliberate review.
- The script owns the bounded `memory-index` blocks. Preserve human-authored introductions outside those markers.

## Read memory progressively

Start at the global and local root indexes. Follow only the links whose summaries match the task. Prefer local memory when a project-specific record conflicts with a global default. Stop loading records when you have enough context to act.

## Keep AGENTS.md thin

Treat `AGENTS.md` as an always-loaded router. Keep project context, critical constraints, memory paths, and conditional triggers there; move detailed procedures elsewhere.

Use the script budget check after material edits:

```powershell
& "$sourceRepository\scripts\memory-control-center.ps1" `
  -Action CheckAgents `
  -AgentsPath "$projectRoot\AGENTS.md"
```

When the budget is exceeded, extract one coherent conditional section:

- cross-project executable behavior → shared skill;
- project-only procedure → `docs/agent-guides/<topic>.md`;
- durable context or decision → local/global memory.

Leave a concise trigger in `AGENTS.md` explaining exactly when the referenced file should be read. A link without a trigger is not sufficient.

## Report

Briefly state what was stored, its scope and path, which indexes changed, whether validation passed, and any classification uncertainty that remains.
