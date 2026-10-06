# Shadow Chase refactoring plan

Durable record of the approved refactoring. It survives session loss: any future
session can read this file and continue from the first unchecked checkpoint.

Approved by the user on 2026-08-01. Working branch: `main`.

## Goals

1. Route every operational message, experiment metric, result, and artifact
   through `ml_logger`.
2. Reorganize the repository into explicit domain, agent, application,
   infrastructure, and interface layers.
3. Preserve current gameplay, training, and evaluation behavior exactly.
4. Rebuild examples and analysis on structured, versioned run data.

## Invariants

These hold at every checkpoint. A change that breaks one is rejected, not
patched afterwards.

- Game rules, reward shaping, and network topology stay numerically identical.
- Existing commands keep working: `main.py`, `train_dqn.py`, `test_agents.py`,
  `game_controls/simple_game.py`.
- Legacy import paths (`ShadowChase.*`, `agents.*`, `training.*`) keep resolving,
  through compatibility shims once files move.
- Historical pickles in `saved_games/` stay loadable; the classes they reference
  keep their original module paths.
- The CUDA-enabled PyTorch install is never replaced with a CPU-only build.
  Pinned runtime: Python 3.11, `torch 2.7.1+cu118`, RTX 4050.
- `python -m pytest` passes before a checkpoint is called done.
- `ml_logger` stays generic. Domain naming lives in
  `ShadowChase.integrations.ml_logging`, never inside `ml_logger`.

## Checkpoints

### 1. Environment and baseline — done

`.venv` rebuilt on Python 3.11.15 with CUDA-enabled PyTorch. Core imports,
a random game, PettingZoo init, and 25 recent pickles verified.

### 2. Characterization tests — done

`pytest.ini` plus `tests/characterization/` lock in observable behavior before
any structural change. Two `xfail` tests document pre-existing defects rather
than hiding them. See [baseline.md](baseline.md).

### 3. Central config and ml_logger adapter — done

`logger_config.yaml` and `ShadowChase/integrations/ml_logging.py` provide
`GameRunRecorder`, `TrainingRunRecorder`, and versioned replay serialization.
See [ml-logging-adapter.md](ml-logging-adapter.md).

### 4. Entry-point migration — done

One command owns one run. Domain code never opens a run and never touches the
global run lifecycle.

- [x] 4a. Batch evaluation — `game_controls/simple_game.py --batch N`
- [x] 4b. Training — `train_dqn.py`, `DQNTrainer`, `BaseTrainer`, `plot_utils`
- [x] 4c. Interactive and visualization CLI — `main.py`
- [x] 4d. Agent comparison script — `test_agents.py`

A comparison delegates each matchup to the evaluation command, so it owns one
`comparison` run that indexes one child `evaluation` run per matchup. The
process boundary between matchups is preserved deliberately: removing it would
change cache and agent state, which is a behavior change, not a logging one.

### 4.5 Pickle compatibility repair — done

The invariant "historical pickles stay loadable" was already broken before this
refactoring started. `tests/characterization/test_pickle_compatibility.py` reads
the real corpus and found that **all 2,521 pre-2025-08 saves failed to load**,
from renames that predate the plan:

| Failing saves | Cause |
|---|---|
| 2,100 | `ScotlandYardMovement` renamed to `ShadowChaseMovement` |
| 411 | package `ScotlandYard` renamed to `ShadowChase` |
| 6 | `Player` value `mr_x` renamed to `MrX` |
| 4 | package `cops_and_robbers` renamed to `ScotlandYard` |

A round trip through freshly created objects cannot detect this, because a new
save records whatever path is current. Only reading the corpus does.

Repaired by the same shim mechanism checkpoint 5 depends on:

- `ShadowChase/compat.py` installs a `sys.meta_path` finder that maps the
  historical package names, including the `storage` to `services` rename;
- class aliases in `ShadowChase/core/game.py` for the renamed rule classes;
- `Player._missing_` for the renamed enum value.

All 11,996 saved games now load, with class identity preserved. The finder must
stay in front of the standard path finder: behind it, a legacy name whose parent
resolves to this package gets found on disk and executed a second time, giving
duplicate classes and a second `Player` enum whose members compare unequal to
the real ones.

**Follow-up: loadable was not the same as usable.** Writing
[docs/usage.md](../usage.md) required exercising every documented command,
including video export, which surfaced a second gap in the same invariant:
unpickling restores `__dict__` directly and never calls `__init__`, so a class
whose *fields* were renamed keeps the old field names on every object saved
before the rename. About a fifth of the corpus (43 of 201 sampled) was saved
with `GameState.mr_x_tickets` / `mr_x_visible` / `mr_x_moves_log` instead of
the current `MrX_tickets` / `MrX_visible` / `MrX_moves_log`, and nothing
migrated them. The original pickle-compatibility test didn't catch it because
it checked `MrX_position` and `turn` but never the ticket or visibility
fields — the exact fields that were missing.

Fixed with `GameState.__setstate__`, which renames the legacy fields on load.
The corpus test now asserts the current field names are present and the legacy
ones are gone, and a focused unit test exercises the migration directly against
a hand-built legacy `__dict__`.

Two unrelated bugs surfaced by the same exercise, both one-line and fixed:

- `ShadowChase/services/export_video.py` computed its own path to the project
  root with a check (`current_dir.name == "ShadowChaseRL"`) that could never be
  true for a file at `ShadowChase/services/`, so every invocation failed before
  importing anything. Replaced with `Path(__file__).resolve().parents[2]`.
- `export_video_from_command_line` branched on `hasattr(game_data,
  'game_history')` before `hasattr(game_data, 'game_config')` to tell a live
  game object from a `GameRecord` — but `GameRecord` has both attributes, so
  every saved game (always a `GameRecord` on disk) took the wrong branch and
  crashed reconstructing a graph that only the live object has. Reordered the
  checks so the more specific attribute is tried first.

### 5. Physical reorganization

Move files into the target layout and leave import shims behind. Deferred until
checkpoint 4 is complete, because moving modules before the logging boundary is
settled would mix two kinds of breakage.

The application layer gets one module per verb this system actually performs.
The full set, from a survey of the working tree: play, train, evaluate, compare,
analyze, replay, export video, author boards, benchmark. The last two are
developer tooling and become scripts rather than application modules.

```
src/shadow_chase/
  domain/          rules, state, movement, win conditions
  agents/          random, heuristic, MCTS, DQN
  application/     play, training, evaluation, comparison,
                   analysis, replay, video
  infrastructure/  persistence, boards, cache, observability, compat
  interfaces/      cli/  gui/  web/  pettingzoo/
configs/           game, training, logger_config.yaml
scripts/           board authoring, profiling, cache benchmarks (was other/)
tests/             unit, integration, characterization, compatibility
docs/  examples/
```

Pickle compatibility is the hard constraint here: unpickling resolves the module
path recorded at save time, so the original module names must keep importing the
same classes.

Moved module-by-module, lowest pickle risk first, each move verified by the
full suite plus a real smoke run before the next: `agents/` first (nothing
under it is pickled directly — DQN checkpoints hold only `state_dict` tensors,
plain strings and JSON config), `ShadowChase/core/game.py` last (the highest
risk, since it's what most saved-game pickles reference).

- [x] 5a. `agents/` → `src/shadow_chase/agents/`
- [ ] 5b. `training/` → `src/shadow_chase/application/training.py` + `deep_q/`
- [ ] 5c. `game_controls/`, `main.py`, `test_agents.py` → `interfaces/cli/`, `application/`
- [ ] 5d. `webui/`, `pettingzoo_integration/` → `interfaces/web/`, `interfaces/pettingzoo/`
- [ ] 5e. `ShadowChase/services/`, `ShadowChase/ui/` → `infrastructure/`, `interfaces/gui/`
- [ ] 5f. `ShadowChase/core/game.py` → `domain/game.py`
- [ ] 5g. `other/` → `scripts/`

#### 5a. `agents/` — done

No `pyproject.toml` existed before this: every entry point resolved imports by
manually doing `sys.path.insert(0, project_root)`, the same fragile pattern
that caused the `export_video.py` bug found while writing the usage docs.
Added `pyproject.toml` (a `src`-layout package, empty `dependencies` so `uv pip
install -e . --no-deps` cannot touch the pinned CUDA torch build) and installed
it into `.venv`. New code should import `shadow_chase.agents` directly; call
sites using `agents.*` are unaffected.

`agents/__init__.py` is now a compatibility shim re-exporting every submodule
of `shadow_chase.agents`, registered in `sys.modules` so both
`from agents import AgentType` and `from agents.heuristics import
GameHeuristics` keep resolving. Two things made this non-trivial:

- **`dqn_agent` has to stay lazy.** The original package never imported it at
  `agents` init time — `agent_registry` only loads it inside a method, on
  first request for a DQN agent. `dqn_agent.py` and
  `training/deep_q/dqn_trainer.py` reference each other by absolute path
  (`from agents.dqn_agent import ...` and `from agents import AgentType`), a
  tangle that predates this move — confirmed by running the identical import
  against the untouched code extracted from git history. Importing it eagerly
  in the shim would force that tangle to resolve while the shim is still
  mid-init, which fails. Fixed with `importlib.util.LazyLoader`, matching the
  original's actual lazy timing rather than working around the tangle.
- **The lazy module needs registering under both names.** Registering it only
  as `sys.modules["agents.dqn_agent"]` left `shadow_chase.agents.dqn_agent`
  unregistered, so a later `import shadow_chase.agents.dqn_agent` re-executed
  the file and produced a second, distinct `DQNMrXAgent` class — same code,
  failing identity. A test asserting `is` (not just equal names) caught it;
  fixed by pointing both `sys.modules` keys, plus the parent package
  attribute, at the same module object.

Verified: full suite (40 passed, 2 known xfails), every entry point's
`--help`, a real headless demo, a 2-game heuristic-vs-random batch, and a
2-episode DQN training run through to a saved checkpoint and evaluation.
`tests/characterization/test_agents_package_move.py` pins the shim's identity
and lazy-loading behavior against regressions.

### 6. Examples and analysis

Replace ad-hoc scripts with small reproducible examples. Analysis reads run
metrics and versioned replays from the `ml_logger` catalog, with a legacy reader
retained for existing pickle and JSON output during migration.

### 7. Verification and documentation

Full pass over historical saves, every CLI, a real DQN run on CUDA, and the
visualization path. Then update `README.md` and the architecture docs.

## Deferred, deliberately

- Resolving `requirements.txt` (NumPy 2.3.2 conflicts with OpenCV 4.12 and
  SciPy 1.11). The working `.venv` is the reference until packaging is redone.
- The `catalog.sqlite` Windows file lock and the legacy JSON export failure,
  both currently pinned by `xfail` tests.
- Extracting `ml_logger` into a separate distributable package. It stays in this
  repository until the refactor is verified.
- The PettingZoo adapter's deprecated `observation_space` / `action_space`
  dictionary attributes.
- Cache namespace policy for agent comparison. The previous script configured
  the cache in a process that plays no games, so the settings never reached a
  game. Wiring them into the evaluation processes would change which decisions
  are cached, so it belongs to a behavior change, not to this migration.
