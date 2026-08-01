"""Characterize the Shadow Chase adapter built on top of ml_logger."""

import json

import pytest

from ml_logger import get_logger, load_config, run
from ShadowChase.core.game import TransportType
from ShadowChase.integrations import (
    AgentComparisonRecorder,
    GameRunRecorder,
    TrainingRunRecorder,
    evaluation_summary,
    serialize_game_replay,
)


logger = get_logger(__name__)


def _write_adapter_test_config(path):
    """Write a quiet logger configuration for deterministic adapter tests."""
    path.write_text(
        """version: 2
logging:
  enabled: true
  level: INFO
  console: false
  file: true
dashboard:
  mode: off
telemetry:
  enabled: false
report:
  enabled: false
metrics:
  include: ["*"]
  exclude: []
""",
        encoding="utf-8",
    )
    return path


def test_project_logger_config_selects_shadow_chase_training_settings(project_root):
    """Verify the central configuration exposes the approved adapter policy."""
    config = load_config(
        project_root / "logger_config.yaml",
        run_type="training",
    )

    recording = config["integrations"]["shadow_chase"]["recording"]
    assert recording == {"level": "actions", "save_replays": True}
    assert config["telemetry"]["enabled"] is True
    assert "train/*" in config["metrics"]["include"]


@pytest.mark.integration
def test_game_recorder_persists_metrics_replay_and_aggregate(tmp_path, shadow_game):
    """Verify one recorded game creates analysis-ready run data."""
    assert shadow_game.make_move(
        MrX_moves=[(2, TransportType.BUS)]
    )
    config_path = _write_adapter_test_config(tmp_path / "logger.yaml")

    with run(
        "evaluation",
        root_dir=tmp_path / "artifacts",
        logger_config_path=config_path,
    ) as context:
        recorder = GameRunRecorder(context, recording_level="actions")
        recorder.record_run_parameters(
            {"map_size": "test", "mrx_agent": "random"}
        )
        recorded = recorder.record_game(
            0,
            shadow_game,
            execution_time_seconds=0.125,
            game_id="fixture/game",
            metadata={"detective_agent": "heuristic"},
        )
        aggregate = recorder.finalize()
        run_dir = context.run_dir

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    metric_record = json.loads(
        (run_dir / "metrics" / "metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    replay = json.loads(recorded.replay_path.read_text(encoding="utf-8"))

    assert metric_record["game/turns"] == 1
    assert metric_record["game/duration_seconds"] == 0.125
    assert replay["schema_version"] == 1
    assert replay["recording_level"] == "actions"
    assert len(replay["states"]) == 2
    assert "configuration" not in replay
    assert recorded.replay_path.name == "000000-fixture-game.json"
    assert aggregate["evaluation/games"] == 1
    assert manifest["result"]["evaluation/average_turns"] == 1.0


def test_full_replay_contains_versioned_graph_configuration(shadow_game):
    """Verify full recording adds portable graph and game configuration."""
    replay = serialize_game_replay(
        shadow_game,
        game_index=4,
        recording_level="full",
    )

    assert replay["schema_version"] == 1
    assert replay["configuration"]["num_detectives"] == 2
    assert replay["configuration"]["reveal_turns"] == [3, 8, 13, 18, 24]
    assert len(replay["configuration"]["graph"]["links"]) == 7


@pytest.mark.integration
def test_training_recorder_namespaces_metrics_and_copies_checkpoint(tmp_path):
    """Verify training records use stable names and canonical artifacts."""
    config_path = _write_adapter_test_config(tmp_path / "logger.yaml")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"characterization-checkpoint")

    with run(
        "training",
        root_dir=tmp_path / "artifacts",
        logger_config_path=config_path,
    ) as context:
        recorder = TrainingRunRecorder(context)
        recorder.record_run_parameters({"role": "MrX"})
        recorder.record_metrics(3, {"loss": 0.25, "epsilon": 0.9})
        artifact_path = recorder.record_checkpoint(checkpoint)
        summary = recorder.finalize({"best_reward": 12.5})
        run_dir = context.run_dir

    metric_record = json.loads(
        (run_dir / "metrics" / "metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert metric_record["train/loss"] == 0.25
    assert metric_record["train/epsilon"] == 0.9
    assert artifact_path.read_bytes() == checkpoint.read_bytes()
    assert summary == {"training/best_reward": 12.5}


def test_evaluation_summary_strips_only_top_level_namespace_fields():
    """Verify a child run's aggregate is readable without its nested blocks."""
    summary = evaluation_summary(
        {
            "evaluation/games": 4,
            "evaluation/mrx_win_rate": 0.25,
            "evaluation/random_vs_random/games": 4,
            "gameplay/games": 1,
        }
    )

    assert summary == {"games": 4, "mrx_win_rate": 0.25}


@pytest.mark.integration
def test_comparison_recorder_aggregates_matchups_and_keeps_failures(tmp_path):
    """Verify a comparison totals its child runs without hiding a failure."""
    config_path = _write_adapter_test_config(tmp_path / "logger.yaml")

    with run(
        "comparison",
        root_dir=tmp_path / "artifacts",
        logger_config_path=config_path,
    ) as context:
        recorder = AgentComparisonRecorder(context)
        recorder.record_matchup(
            0,
            "random_vs_heuristic",
            {
                "games": 4,
                "mrx_wins": 1,
                "detective_wins": 3,
                "incomplete_games": 0,
                "average_turns": 6.0,
                "average_duration_seconds": 0.5,
            },
            run_id="evaluation-fixture-a",
        )
        recorder.record_matchup(
            1,
            "heuristic_vs_random",
            {
                "games": 2,
                "mrx_wins": 2,
                "detective_wins": 0,
                "incomplete_games": 0,
                "average_turns": 3.0,
                "average_duration_seconds": 0.25,
            },
            run_id="evaluation-fixture-b",
        )
        recorder.record_failed_matchup(2, "deep_q_vs_random", "exit code 1")
        summary = recorder.finalize()
        run_dir = context.run_dir

    metrics = [
        json.loads(line)
        for line in (run_dir / "metrics" / "metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert summary["comparison/games"] == 6
    assert summary["comparison/mrx_wins"] == 3
    assert summary["comparison/mrx_win_rate"] == 0.5
    # Totals are weighted by games, not averaged across matchups.
    assert summary["comparison/average_turns"] == 5.0
    assert summary["comparison/matchups"] == 2
    assert summary["comparison/failed_matchups"] == 1
    assert summary["comparison/random_vs_heuristic/run_id"] == "evaluation-fixture-a"
    assert summary["comparison/deep_q_vs_random/error"] == "exit code 1"

    assert len(metrics) == 3
    assert metrics[1]["comparison/heuristic_vs_random/games"] == 2
    # Identifiers stay out of the metric series.
    assert "comparison/random_vs_heuristic/run_id" not in metrics[0]


@pytest.mark.integration
def test_training_recorder_supports_phase_and_summary_namespaces(tmp_path):
    """Verify a standalone evaluation reuses the recorder without mislabeling."""
    config_path = _write_adapter_test_config(tmp_path / "logger.yaml")

    with run(
        "evaluation",
        root_dir=tmp_path / "artifacts",
        logger_config_path=config_path,
    ) as context:
        recorder = TrainingRunRecorder(context)
        recorder.record_metrics(0, {"win": 1}, phase="evaluation")
        summary = recorder.finalize({"win_rate": 1.0}, namespace="evaluation")
        run_dir = context.run_dir

    metric_record = json.loads(
        (run_dir / "metrics" / "metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert metric_record["evaluation/win"] == 1
    assert summary == {"evaluation/win_rate": 1.0}
