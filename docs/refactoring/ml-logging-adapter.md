# Shadow Chase ml_logger adapter

`ShadowChase.integrations.ml_logging` is the boundary between game concepts and
the generic `ml_logger` package. Game, agent, and training modules should not add
domain behavior to `ml_logger` itself.

## Lifecycle

One command owns one run. The recommended evaluation pattern is:

```python
from ml_logger import run
from ShadowChase.integrations import GameRunRecorder

with run("evaluation", config=effective_config) as experiment:
    recorder = GameRunRecorder(experiment)
    recorder.record_run_parameters(effective_config)

    for game_index, game in enumerate(games):
        recorder.record_game(
            game_index,
            game,
            execution_time_seconds=durations[game_index],
        )

    recorder.finalize()
```

Training uses `TrainingRunRecorder` to batch same-step metrics and register
checkpoints or plots through the artifact API. `record_metrics(step, metrics,
phase=...)` and `finalize(summary, namespace=...)` let one recorder serve a
training phase and a validation or standalone evaluation phase without
mislabeling either.

## Recording levels

- `summary`: numeric per-game metrics only;
- `actions`: metrics plus versioned state and ticket history replays;
- `full`: action data plus the graph and game configuration.

Replay files use `schema_version: 1`. Changing existing field meaning requires a
schema migration rather than silently rewriting the format.

## Metric namespaces

- `game/*`: one row per evaluated or played game;
- `evaluation/*`: aggregate batch summaries;
- `comparison/*`: agent comparison totals, with one `comparison/<matchup>/*`
  block per matchup;
- `train/*`: training-loop metrics;
- `validation/*`: validation or periodic evaluation metrics;
- `training/*`: final training summary fields.

The root `logger_config.yaml` controls persistence, dashboard, telemetry, reports,
and the default replay level.

## Migrated entry point: batch evaluation

`game_controls/simple_game.py --batch N` now owns one `evaluation` run and
records every game through `GameRunRecorder`. The legacy pickle save remains
enabled by default for compatibility. It can be disabled independently from
ml_logger output:

```powershell
python game_controls/simple_game.py `
  --batch 10 `
  --map-size test `
  --mr-x-agent random `
  --detective-agent heuristic `
  --recording-level actions `
  --no-legacy-save
```

Additional options select a run name, logger configuration, artifact root, or
disable replay files while retaining numeric metrics.

## Migrated entry point: DQN training

`train_dqn.py` owns one `training` run. `BaseTrainer._log_training_step` is the
single choke point that forwards per-episode numeric metrics, so any future
trainer inherits ml_logger recording without extra wiring.

```powershell
python train_dqn.py `
  --role MrX `
  --episodes 2000 `
  --plotting-every 500 `
  --save-dir training_results `
  --no-show-plots
```

What one training run contains:

- `train/episode_reward`, `train/epsilon`, `train/buffer_size`, `train/avg_loss`
  once per episode;
- `train/rolling_avg_reward`, `train/rolling_avg_loss`, `train/rolling_win_rate`
  every 100 episodes;
- `train/q_mean`, `train/q_std`, `train/q_min`, `train/q_max`,
  `train/q_negative_fraction` at the monitoring interval;
- `validation/win` and `validation/turns` for the post-training evaluation;
- the model checkpoint under `artifacts/model/` and each metrics plot under
  `artifacts/plot/`;
- `training/*` and `validation/*` summary fields in the manifest.

`--evaluate --model_path ...` opens a standalone `evaluation` run instead and
summarizes under `evaluation/*`.

Behavior preserved from before the migration:

- `--episodes` left unset still takes the episode count from
  `training/configs/dqn_config.json`, and `--plotting-every` still defaults to a
  third of that count, which reproduces the previous hard-coded 9001/3000 run.
- Training map defaults stay `extended` with 5 detectives and 24 turns;
  evaluation still runs on the `test` map with 2 detectives.
- Legacy checkpoint and plot files are still written to `training_results/`.
  `--save-dir` redirects them, which is what keeps tests out of the repository.

The previous `__main__` block bypassed `argparse` entirely and always trained
Mr. X, so flags such as `--role` silently did nothing. It now calls `main()`.
The default invocation reproduces the old behavior; the flags actually work.

## Migrated entry point: interactive and visualization demos

`main.py` owns one `gameplay` run. It selects a demonstration, plays or
visualizes it, and records the resulting game through `GameRunRecorder`. The
demo functions build and play games only; the command owns the run lifecycle.

```powershell
python main.py                        # extracted board, 5 detectives, GUI
python main.py --list-demos           # catalog, no run opened
python main.py --demo grid --detectives 3 --headless
```

What one gameplay run contains:

- `game/*` metrics for the single recorded game, at step 0;
- one versioned replay under `replays/`, subject to `--recording-level` and
  `--no-replays`;
- `gameplay/*` summary fields in the manifest, including `gameplay/games` and
  `gameplay/recorded_games`.

`GameRunRecorder.finalize(summary, namespace=...)` now mirrors
`TrainingRunRecorder.finalize`. Batch evaluation keeps the default
`evaluation` namespace, so existing manifests are unchanged; the gameplay
command summarizes under `gameplay/*` instead of claiming to be a batch
evaluation.

Behavior preserved from before the migration:

- `python main.py` with no arguments still runs the extracted board demo with
  5 detectives and opens the visualizer, which is what the old hard-coded
  `__main__` block did.
- Every demo keeps its original graph, positions, and detective count. Demos
  with a fixed layout reject `--detectives` with an error rather than accepting
  a flag that cannot take effect.
- `--headless` builds and records a game without constructing the Tk window,
  which is what makes the entry point testable.

One behavior change: `demo_path_game` called `solve()` on a solver that was
never implemented and could only raise `AttributeError`. It now logs that no
solver is available and visualizes the game like the other graph demos.

The recorder also accepts games from the base `Game` class, which has no ticket
history or reveal turns, so the ticketless demos record the same metric and
replay shape as full-board games.

## Migrated entry point: agent comparison

`test_agents.py` owns one `comparison` run. Each matchup is still evaluated by
`game_controls/simple_game.py --batch` in its own process, so a comparison
produces one comparison run plus one child evaluation run per matchup. The
comparison run is the index over those children.

```powershell
python test_agents.py                           # random vs random, 10 games
python test_agents.py --all-combinations        # every distinct ordered pair
python test_agents.py --mr-x-agent deep_q --detective-agent heuristic --games 50
```

`AgentComparisonRecorder` consumes what the child already recorded instead of
re-deriving it. `evaluation_summary()` strips the `evaluation/` namespace from a
child manifest's result, `record_matchup` merges it into the comparison totals,
and `finalize` writes both views:

- `comparison/<matchup>/*` metrics, one row per matchup at the matchup index;
- `comparison/<matchup>/*` summary fields, including `run_id`, which links the
  block to the evaluation run that produced it;
- `comparison/games`, `comparison/mrx_wins`, `comparison/mrx_win_rate`,
  `comparison/average_turns` and the other totals across all matchups;
- `comparison/matchups`, `comparison/failed_matchups`,
  `comparison/requested_matchups`, `comparison/analysis_artifacts`, and
  `comparison/matchup_duration_seconds`, which times the matchups only;
- every analysis plot under `artifacts/plot/` and the analysis report under
  `artifacts/report/`.

Subprocess execution is deliberate rather than incidental. Running the matchups
in-process would be simpler, but it would change what the games do: each agent
process starts with default cache namespace settings and its own MCTS state.
Preserving behavior outranks removing a process boundary.

Behavior preserved from before the migration:

- `python test_agents.py` with no arguments still plays 10 random-vs-random
  games on the `extracted` map with 5 detectives, saves them under
  `saved_games/video_exporting_test/random_vs_random/`, and then runs the
  analysis pass. Every previously hard-coded constant is now a flag with that
  value as its default.
- `--all-combinations` replaces the dead `run_all_combinations = False` local
  and still evaluates the selected matchup first, then every distinct ordered
  pair; duplicates are dropped so two matchups can never share a save
  directory.
- A failing matchup is still logged and skipped rather than aborting the
  comparison, and the command still exits 0. Failures are now visible in the
  run as `comparison/<matchup>/failed` and `comparison/failed_matchups`.

Two behavior changes:

- The script configured the cache system (`enable_cache`,
  `disable_namespace_cache(GAME_METHODS)`, and so on) in a process that plays no
  games. Those module-level flags do not cross a process boundary, so every
  child reset them to their defaults and the configuration never took effect.
  The dead configuration is removed rather than silently kept. Making cache
  policy reach the games is a behavior change, not a logging change, and is
  listed as deferred in [plan.md](plan.md).
- Child processes now run with `PYTHONIOENCODING=utf-8`. Both scripts print
  non-ASCII status characters, which raised `UnicodeEncodeError` under the
  Windows console default as soon as their output was redirected — failing a
  matchup for a reason unrelated to the games it played.
