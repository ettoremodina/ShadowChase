---
summary: "Refactor incrementally around ml_logger while preserving game behavior, public commands, legacy imports, and historical pickle compatibility."
created-at: 2026-07-31T23:21:22.8356105Z
updated-at: 2026-07-31T23:21:22.8356105Z
---

# Incremental refactoring strategy

The approved refactoring will proceed one checkpoint at a time. Integrate all operational output, experiment metrics, results, and artifacts through ml_logger using a project-specific adapter. Reorganize the repository only after characterization tests exist. Preserve current gameplay and training behavior, existing top-level commands, legacy import paths, and the ability to load historical pickle saves through compatibility shims. Redesign examples and analysis around structured, versioned run data while retaining a legacy reader during migration. Keep ml_logger in this repository during stabilization; consider extracting it only after the refactor is verified.

The local training workflow must retain working NVIDIA CUDA acceleration. Do not replace the CUDA-enabled PyTorch installation with a CPU-only build during environment or dependency changes.

Approved by the user on 2026-08-01.
