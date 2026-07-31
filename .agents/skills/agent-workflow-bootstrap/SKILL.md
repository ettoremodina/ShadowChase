---
name: agent-workflow-bootstrap
description: Initialize or refresh a new or existing repository with this AgentWorkflow control center. Use whenever the user asks to set up, bootstrap, connect, initialize, or update a project's agentic workflow, project memory, AGENTS.md, or relevant project skills. Keep discovery shallow and produce a useful first pass quickly rather than auditing the whole repository.
compatibility: Windows PowerShell or PowerShell 7; the AgentWorkflow repository and its project-control-center.ps1 script must be available.
metadata:
  category: project-template
  scope: global
  retention-class: durable
  maintenance-policy: review-periodically
  status: active
  origin: local
---

# Agent Workflow Bootstrap

Connect a repository to the shared AgentWorkflow environment with a small, reversible first pass.

## Workflow

1. Resolve the target repository to an absolute path. This can be a new directory or an existing project.
2. Inspect only enough context to describe the project and select useful skills:
   - read the root `README.md` and existing root `AGENTS.md` when present;
   - inspect root manifests such as `pyproject.toml`, `package.json`, `Cargo.toml`, solution files, or equivalent;
   - inspect top-level directory and file names;
   - do not perform a full codebase audit, dependency analysis, or knowledge-graph build.
3. Derive a short project context:
   - project name;
   - one-sentence purpose;
   - primary technology;
   - important commands already evidenced by project files;
   - important constraints, or `Not established yet` when unknown.
4. Read the active entries in `<AgentWorkflow>/registry/skills.csv`. Select a small set of directly relevant skills, normally 1–6:
   - always include `agent-workflow-bootstrap` so the project can be refreshed later;
   - always include `memory-manager` so durable memory is classified and indexed consistently;
   - add a skill only when the project files or user description provide a concrete reason;
   - do not add document, ML, frontend, deployment, or testing skills merely because they might someday be useful;
   - preserve project-local skills already present;
   - do not create a new project-specific skill during bootstrap unless the user explicitly asks for one.
5. Run the bundled script with `-Apply`. It delegates skill and memory linking to the existing Project Control Center, creates or refreshes hierarchical memory indexes through the Memory Control Center, records the applied workflow version, registers the connected project, and creates or updates the root `AGENTS.md` safely:
   - for a new file, it starts from `<AgentWorkflow>/templates/base-project/AGENTS.md`, replaces the template's placeholder project context with discovered facts and the managed AgentWorkflow block, and retains the reusable guidance sections;
   - for an existing file, it updates only the managed block and preserves all project-authored content outside the markers;
   - for a legacy bootstrap-only file containing just the standard title and managed block, it performs a one-time safe upgrade to the template-based baseline.

```powershell
./.agents/skills/agent-workflow-bootstrap/scripts/bootstrap-project.ps1 `
  -ProjectPath "C:\absolute\path\to\project" `
  -Skills agent-workflow-bootstrap,memory-manager,frontend-design,webapp-testing `
  -Providers codex `
  -ProjectName "Example" `
  -Purpose "A small web application for ..." `
  -Technology "TypeScript, React, Vite" `
  -Commands "npm install; npm test; npm run dev" `
  -Constraints "Preserve the existing API contract" `
  -Apply
```

The script always runs the Control Center preview before applying. Without `-Apply`, it performs preview only.

6. Verify the final `Status` output and briefly report:
   - selected skills and why they were selected;
   - the locations of global and local memory;
   - the applied AgentWorkflow version and whether an update remains available;
   - whether `AGENTS.md` was created or updated;
   - any unknown project detail left for later refinement.

## Memory routing

- Start from `.agents/memory/global/INDEX.md` and `.agents/memory/local/INDEX.md`; do not load every memory file by default.
- Use `memory-manager` whenever durable knowledge is added, moved, classified, or reindexed.
- Every directory in either memory tree must have a current `INDEX.md`; validate through the deterministic memory script.
- Store cross-project preferences, standards, reusable workflows, and knowledge maps in global memory.
- Store project decisions, domain facts, recurring project-specific mistakes, and local sources in local memory.
- Keep information out of memory when it is easy to derive from the repository.
- Never store secrets in either memory scope.

## Updating an existing project

Use the same workflow. The Project Control Center preserves existing local skills and the bootstrap script preserves all `AGENTS.md` content outside its managed markers. It does not remove skills that are no longer selected; report stale selections for deliberate cleanup later.

## Boundaries

- Do not configure MCP servers, credentials, hooks, services, databases, dashboards, or logging automation.
- Do not turn bootstrap into a long project assessment.
- Do not overwrite project-authored content in an existing `AGENTS.md` or any local memory file.
- Do not silently promote a project-local skill to the global source.
