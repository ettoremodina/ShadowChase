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
