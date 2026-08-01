# Shadow Chase — features and commands

What this project can do, and the command for each. For how the pieces are
built, see [refactoring/plan.md](refactoring/plan.md); for the web board, see
[webui.md](webui.md).

Every command runs from the repository root, because boards, saves and
calibration data are read with paths relative to it.

```bash
.venv/Scripts/python.exe main.py        # Windows
.venv/bin/python main.py                # Linux and macOS
```

`python` below means that interpreter. The environment is Python 3.11 with
`torch 2.7.1+cu118`; see [Environment](#environment).

---

## The nine things this project does

| Feature | Command |
| --- | --- |
| [Play](#play) a game — terminal, desktop or browser | `python game_controls/simple_game.py` · `python main.py` · `python -m webui` |
| [Train](#train) a DQN agent | `python train_dqn.py` |
| [Evaluate](#evaluate) one agent pairing over N games | `python game_controls/simple_game.py --batch N` |
| [Compare](#compare) many agent pairings | `python test_agents.py` |
| [Analyze](#analyze) saved games into plots and a report | `python ShadowChase/services/analyze_games.py <dir>` |
| [Replay](#replay) a finished game | `python -m webui` → Saved games |
| [Export video](#export-video) of a saved game | `python ShadowChase/services/export_video.py <game.pkl>` |
| [Author boards](#author-boards) from the scanned map | `python other/create_board_data.py` |
| [Benchmark](#benchmark) agents and the cache | `python other/profile_mcts_agent.py` |

Agents available everywhere an agent is named:
`random`, `heuristic`, `optimized_mcts`, `epsilon_greedy_mcts`, `deep_q`.

---

## Play

### Terminal

```bash
python game_controls/simple_game.py
```

Prompts for board, mode, detective count, verbosity and both agent types, then
plays one game in the terminal. Human, AI, and mixed modes are all offered.

### Desktop board

```bash
python main.py                                  # extracted board, 5 detectives
python main.py --list-demos                     # every demo, no run opened
python main.py --demo grid --detectives 3
python main.py --demo until-end --headless      # build and record, no window
```

Twelve demos, from a 3×3 grid walkthrough of the movement rules to the full
scanned London board. `--headless` skips the window and still records the game,
which is what makes the demos testable.

Demos with a fixed layout reject `--detectives` rather than ignoring it.

### Browser

```bash
python -m webui                 # start the server and open the board
python -m webui --port 8123
python -m webui --no-browser
```

The most complete way to play: pan and zoom the scanned board, ticket-aware
click targets, the reveal ruler, suspect-station shading, and replay. Needs
`fastapi` and `uvicorn`. Full description in [webui.md](webui.md).

---

## Train

```bash
python train_dqn.py                             # Mr. X, episodes from the config
python train_dqn.py --role detectives
python train_dqn.py --episodes 2000 --plotting-every 500 --no-show-plots
python train_dqn.py --device cuda
```

Defaults: the `extended` map, 5 detectives, 24 turns, checkpoints and plots
under `training_results/`. With `--episodes` unset the count comes from
`training/configs/dqn_config.json` (currently 9001), and `--plotting-every`
defaults to a third of it.

| Flag | Meaning |
| --- | --- |
| `--role {MrX,detectives}` | Which side to train |
| `--episodes N` | Override the configured episode count |
| `--plotting-every N` | Episodes between metric plots |
| `--map-size` / `--detectives` / `--max-turns` | Environment shape |
| `--save-dir DIR` | Where checkpoints and plots are written |
| `--device cuda\|cpu` | Force a device |
| `--no-show-plots` | Write plots without opening windows |

### Evaluate a trained model

```bash
python train_dqn.py --evaluate --model-path training_results/dqn_MrX_1785576154.pth
python train_dqn.py --evaluate --model-path <model> --eval-games 200
```

Opens an `evaluation` run instead of a training run. Evaluation uses the `test`
map with 2 detectives.

---

## Evaluate

One agent pairing, N games, no interaction:

```bash
python game_controls/simple_game.py --batch 100 \
  --map-size extracted --detectives 5 --max-turns 24 \
  --mr-x-agent deep_q --detective-agent heuristic \
  --save-dir my_experiment --verbosity 0
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--batch N` | — | Number of games; enables batch mode |
| `--map-size {test,full,extracted}` | `test` | Board |
| `--detectives 1..5` | `2` | Detective count |
| `--max-turns N` | `24` | Turn limit per game |
| `--mr-x-agent` / `--detective-agent` | `random` | Agent types |
| `--verbosity 0..5` | `2` | 0 is silent, 5 prints heuristics |
| `--save-dir DIR` | `fritto_misto` | Subdirectory under `saved_games/` |

Games land in `saved_games/<save-dir>/games/*.pkl` with JSON metadata beside
them, and in the run's own metrics. See [Run output](#run-output).

---

## Compare

Many pairings in one experiment. Each pairing runs as its own evaluation
process; the comparison indexes them all.

```bash
python test_agents.py                           # random vs random, 10 games
python test_agents.py --all-combinations        # every distinct ordered pair
python test_agents.py --all-combinations --agents random heuristic deep_q
python test_agents.py --mr-x-agent deep_q --detective-agent heuristic --games 50
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--games N` | `10` | Games per pairing |
| `--test-name NAME` | `video_exporting_test` | Subdirectory under `saved_games/` and analysis target |
| `--map-size` / `--detectives` / `--max-turns` | `extracted`, `5`, `24` | Board and shape |
| `--mr-x-agent` / `--detective-agent` | `random` | The pairing always evaluated |
| `--all-combinations` | off | Also every distinct ordered pair |
| `--agents A B ...` | all five | Agents used to build those pairs |
| `--no-analysis` | off | Skip the analysis pass |

The analysis pass runs automatically at the end and its plots are attached to
the run. A pairing that fails is logged and skipped; the rest still complete.

---

## Analyze

```bash
python ShadowChase/services/analyze_games.py my_experiment
```

Reads `saved_games/my_experiment/*/` and writes:

- `analysis_graphs/win_rates_by_agent.jpg`
- `analysis_graphs/game_length_analysis.jpg`
- `analysis_graphs/execution_time_analysis.jpg`
- `analysis_graphs/agent_performance_matrix.jpg`
- `analysis_graphs/temporal_analysis.jpg`
- `analysis_graphs/comprehensive_dashboard.jpg`
- `analysis_report.txt`

Win rates carry 95% Wilson confidence intervals; game lengths and execution
times carry t-based intervals. The directory argument is a name under
`saved_games/`, not a path.

---

## Replay

```bash
python -m webui         # left rail → Saved games
```

Loads every recorded state at once, so the timeline scrubs freely. Arrows step,
`space` plays, `Home` and `End` jump to the ends. Mr. X's position is always
shown, with a badge saying whether the detectives could see him at the time.

The desktop visualizer opens the same replay through its saved-games dialog.
There is no standalone replay command.

---

## Export video

```bash
python ShadowChase/services/export_video.py saved_games/my_experiment/random_vs_random/games/game_20250803_015235_41a5d12795f84770.pkl
python ShadowChase/services/export_video.py <game.pkl> --output replay.mp4 --duration 2.0
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `-o, --output PATH` | auto-generated | MP4 destination |
| `-d, --duration SEC` | `1.0` | Seconds per turn |
| `--fps N` | `1/duration` | Frame rate |
| `-v, --verbose` | off | Progress detail |

The browser UI exposes the same export from its left rail, writing to
`exports/`.

---

## Author boards

Developer tooling for the scanned board, in `other/`. These are rough scripts
rather than polished commands.

| Script | Purpose |
| --- | --- |
| `other/createBoard.py` | Interactive board construction |
| `other/create_board_data.py` | Build the board data files under `data/` |
| `other/calibrate_board_overlay.py` | Align node coordinates to the photo |

Output feeds `data/board_calibration.json` and the edge lists that
`extracted` boards load.

---

## Benchmark

| Script | Purpose |
| --- | --- |
| `other/profile_mcts_agent.py` | Profile MCTS decision cost |
| `other/test_mcts_cache_performance.py` | MCTS with and without caching |
| `other/test_random_cache_performance.py` | Cache overhead baseline |
| `other/compare_cache_performance.py` | Compare recorded cache runs |
| `other/analyze_cache_performance.py` | Plot cache statistics |

---

## Run output

Every command above opens exactly one run and writes it under `artifacts/`:

```
artifacts/
  catalog.sqlite                      index of every run
  runs/<date>/<run-type>-<timestamp>-<id>/
    manifest.json                     config, params, result, git, command
    metrics/metrics.jsonl             one row per step
    logs/                             the run's log file
    replays/                          versioned game replays
    artifacts/                        models, plots, reports
    run_report.html                   standalone summary
```

Run types: `gameplay` (a demo), `evaluation` (a batch or a model evaluation),
`training`, `comparison` (many pairings).

Flags shared by every entry point:

| Flag | Meaning |
| --- | --- |
| `--run-name NAME` | Name the run |
| `--artifact-root DIR` | Write somewhere other than `artifacts/` |
| `--logger-config FILE` | Use a different `logger_config.yaml` |
| `--recording-level {summary,actions,full}` | Replay detail |
| `--no-replays` | Metrics only, no replay files |

`summary` records numbers only, `actions` adds states and ticket history,
`full` adds the graph and configuration. Defaults live in
`logger_config.yaml`.

Legacy pickle output under `saved_games/` is written in addition, and can be
turned off with `--no-legacy-save` where it applies.

---

## Reinforcement-learning integration

The game is exposed as a PettingZoo multi-agent environment:

```bash
python pettingzoo_integration/example_usage.py
```

Use `pettingzoo_integration/shadow_chase_env.py` to drive the game from an
external RL library.

---

## Tests

```bash
python -m pytest                         # everything
python -m pytest -m "not integration"    # fast subset
python -m pytest -m cuda                 # requires an NVIDIA device
```

`tests/characterization/` pins observable behavior: game rules, persistence,
the ml_logger adapter, every entry point, and the loadability of the saved-game
corpus. Two tests are `xfail` and document known defects rather than hiding
them.

---

## Environment

Python 3.11 with a CUDA build of PyTorch (`torch 2.7.1+cu118`, tested on an
RTX 4050). The working environment is `.venv/`.

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`requirements.txt` is not currently installable as written — NumPy 2.3.2
conflicts with OpenCV 4.12 and SciPy 1.11. `.venv` is the reference
environment until packaging is redone. Do not reinstall PyTorch from
`requirements.txt`: it would replace the CUDA build with a CPU-only one.

The browser UI additionally needs:

```bash
uv pip install "fastapi>=0.115" "uvicorn[standard]>=0.30"
```

---

## Saved games and compatibility

Games are stored under `saved_games/<experiment>/<pairing>/games/` as pickles,
with JSON metadata alongside for analysis. The corpus currently holds ~12,000
games going back to earlier releases of this project, under two former package
names.

Those older names are aliased in `ShadowChase/compat.py`, so every save remains
loadable. If a class or module is renamed again, add the old name there —
otherwise the games saved before the rename become unreadable, silently and
permanently.
