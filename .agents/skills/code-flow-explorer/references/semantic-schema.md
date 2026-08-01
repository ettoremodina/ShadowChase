# Semantic annotation schema

The agent creates semantic annotations after deterministic extraction. The code-generated `flow-ir.json` remains the authority for exact nodes, edges, kinds, and source ranges. `semantic.json` supplies readable labels and two source-backed grouping levels; it never changes exact control flow.

```json
{
  "purpose": "One sentence explaining what the target accomplishes.",
  "inputs": [
    "Input name — what it controls"
  ],
  "outputs": [
    "Returned value, generated text, or externally visible result"
  ],
  "side_effects": [
    "Files, logs, network, database, mutation, or other effects"
  ],
  "legend": [
    {
      "id": "preparation",
      "label": "Prepare and qualify work",
      "color": "blue"
    },
    {
      "id": "mutation",
      "label": "External changes",
      "color": "amber"
    }
  ],
  "phases": [
    {
      "id": "stable-kebab-case-id",
      "label": "Short action-oriented label",
      "summary": "A plain-language explanation of what this phase accomplishes.",
      "category": "preparation",
      "line_start": 1,
      "line_end": 40,
      "steps": [
        {
          "id": "globally-unique-step-id",
          "label": "Readable action or decision",
          "summary": "What happens in this smaller region and why it matters.",
          "line_start": 1,
          "line_end": 14
        },
        {
          "id": "another-step-id",
          "label": "Continue the phase",
          "summary": "The next meaningful unit of behavior.",
          "line_start": 15,
          "line_end": 40
        }
      ]
    }
  ]
}
```

## Adaptive grouping rules

- Let the code determine the number of phases and steps. There is no minimum or maximum node quota.
- Overview phases represent major responsibilities, outcomes, or large alternatives. Make Overview detailed enough to explain the target without opening Exact.
- Logic steps represent meaningful operations or decisions within a phase. Combine incidental assignments, logging, formatting, and closely related calls when they serve one outcome.
- Split a group when it contains independently meaningful branches, loops, side effects, error paths, or generated outputs.
- Merge neighboring operations when separating them would only restate syntax.
- A useful Logic view is substantially smaller than Exact, but comprehension and source fidelity take priority over a compression ratio.

## Source-grounding rules

- Use inclusive source line ranges.
- Keep ranges inside the selected function or macro.
- Keep phases non-overlapping. Within each phase, keep steps non-overlapping and inside the phase range.
- Cover every extracted non-boundary node with exactly one phase and one nested step. The ranges do not need to account for blank or comment-only lines.
- Prefer semantic outcomes such as “Discover eligible models” over syntax labels such as “First loop.”
- Preserve major alternatives. When a source region contains two large branches, give each branch its own phase.
- Use plain language in Overview and Logic. Keep identifiers only when they help connect the explanation to the source.
- Mention uncertainty explicitly rather than inventing runtime values.

## Semantic color rules

- The agent authors the Overview legend and assigns every phase to one legend category.
- Categories explain meaning in the target, such as preparation, validation, a major execution path, mutation, or an early exit. Do not use colors merely to make adjacent nodes different.
- Use only the controlled color tokens `blue`, `teal`, `violet`, `amber`, `rose`, `green`, and `slate`; do not add free-form hex values.
- Reuse a category when phases share a semantic role. The renderer shows only categories used by at least one phase.
- Logic steps inherit a lighter version of their parent phase color. Exact remains deterministic and uses technical node-kind colors.

## Navigation mapping

The renderer derives navigation from source coverage:

- each Overview phase links to its nested Logic steps;
- each Logic step links to the exact nodes whose start lines fall inside its range;
- clicking any node selects it, opens its source explanation, and does not navigate;
- the inspector action, a double-click, or Enter on a focused grouped node renders its child subgraph with a fresh layout;
- camera position and zoom are remembered independently for every full or focused scope;
- Exact always retains original extracted labels, details, and edges.
