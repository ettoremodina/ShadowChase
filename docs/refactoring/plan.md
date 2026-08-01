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

### 4. Entry-point migration — in progress

One command owns one run. Domain code never opens a run and never touches the
global run lifecycle.

- [x] 4a. Batch evaluation — `game_controls/simple_game.py --batch N`
- [x] 4b. Training — `train_dqn.py`, `DQNTrainer`, `BaseTrainer`, `plot_utils`
- [ ] 4c. Interactive and visualization CLI — `main.py`
- [ ] 4d. Agent comparison script — `test_agents.py`

### 5. Physical reorganization

Move files into the target layout and leave import shims behind. Deferred until
checkpoint 4 is complete, because moving modules before the logging boundary is
settled would mix two kinds of breakage.

```
src/shadow_chase/
  domain/          rules, state, movement, win conditions
  agents/          random, heuristic, MCTS, DQN
  application/     gameplay, training, evaluation
  infrastructure/  persistence, boards, cache, observability
  interfaces/      cli, gui, pettingzoo
configs/           game, training, logger_config.yaml
tests/             unit, integration, characterization, compatibility
docs/  examples/  scripts/
```

Pickle compatibility is the hard constraint here: unpickling resolves the module
path recorded at save time, so the original module names must keep importing the
same classes.

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
