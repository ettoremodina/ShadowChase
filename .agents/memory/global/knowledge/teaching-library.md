---
summary: "Routes teaching tasks to subject-scoped folders in the canonical knowledge library without exposing unrelated subjects in projects."
created-at: 2026-07-31T13:06:44.9267151Z
updated-at: 2026-07-31T13:06:44.9267151Z
---

# Shared teaching knowledge library

Teaching knowledge is canonical in the AgentWorkflow `knowledge/<subject>/` library. Projects connect the relevant subject through `.agents/knowledge/<subject>/` and reuse the single shared `knowledge/assets/` and `knowledge/templates/` collections through sibling junctions. Use the `teach` skill and its `scripts/connect-subject.ps1` helper to preview, scaffold when needed, and create all three links. Keep raw sources immutable under each subject's `raw-sources/`; keep reusable teaching behavior in the skill; and place a usage `README.md` beside every template.
