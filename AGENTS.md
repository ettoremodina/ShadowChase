# Project Agent Guidelines

<!-- agent-workflow:start -->
## Agent workflow

### Project context

- **Project:** ScotlandYardRL
- **Purpose:** Not established yet
- **Technology:** Not established yet
- **Important commands:** Not established yet
- **Important constraints:** Not established yet

### Skills and memory

- Base skills (`html-output-viewer`, `code-flow-explorer`) are installed once for every project. This project also links: none. Change the set from AgentWorkflow with `scripts/project.ps1 -Action Link`.
- Project-owned skills live in `.agents/skills/` and are committed. Linked skills are shared with other projects: for a project-specific variant, create a new skill with a different name instead of editing the shared one.
- Project memory starts at `.agents/memory/local/INDEX.md`; open only relevant entries. Never store secrets or personal data in this repository.
<!-- agent-workflow:end -->

## Understand the goal (summarize)

- Ask focused questions when the real goal, constraints, or success criteria are unclear.
- Do not accept a proposed solution blindly. Check whether it makes technical sense and explain any correction clearly.
- Surface important alternatives, consequences, and edge cases that the user may not know to ask about.

## Plan large work and use checkpoints

- Begin long, complex, or ambiguous work with a reviewable plan.
- Do not start a very large implementation until the user has had a chance to review the direction.
- Divide large projects into small, useful increments.
- Add checkpoints where the user can inspect the result before the next major increment begins.

## Explain material changes
- Ask for confirmation before drastic architectural changes or changes that are difficult to reverse.
- Keep changes small and reviewable when practical.

## Design and implementation principles

- Prefer modular, reusable designs and avoid unnecessary duplication or hardcoding.
- Optimize for maintainability, robustness, and future extension without adding complexity before it is needed.
- Choose the simplest suitable technology for the task instead of defaulting to the project's main language.

## Developer environment

- When commands fail for the agent, check for missing dependencies, missing executables, and incorrect `PATH` configuration before creating workarounds. 
- Tell the user when an environment fix would prevent repeated failures or wasted effort.
- Recommend project-specific tools only when they materially improve reliability, speed, or usability.

## Machine-learning projects

Apply this section only when the project includes machine learning.

- All outputs must pass through the logger, every new module will use the logger 
- Aim for modularity and plan for future expansion
- Never hard code configs: create a config file or add to an existing one for every script



## MEMORY
- use the memory folder to keep track of what you learned about this project. It is mainly used to store distilled information about code output or insights from the user, i.e. everything that cannot be inferred from the code alone 
- Its use is to avoid commiting the same mistake twice and guide implementation of new code and experiments

## Project-specific additions

Add repository layout rules, coding conventions, test expectations, safety boundaries, release processes, and other project-specific guidance here when they become established.


## Research
- use exa for semantic search when I ask to look online

## Documentation
- always write documentation for a new function or class, if it's an important new module or feature of the project write it in the docs
