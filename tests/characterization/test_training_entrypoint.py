"""Characterize the ml_logger-enabled DQN training entry point."""

import json
import os
import subprocess
import sys

import pytest

from ml_logger import get_logger


logger = get_logger(__name__)


def _read_metrics(run_dir):
    """Return every recorded metric row for one run directory."""
    metrics_file = run_dir / "metrics" / "metrics.jsonl"
    return [
        json.loads(line)
        for line in metrics_file.read_text(encoding="utf-8").splitlines()
    ]


@pytest.mark.integration
def test_training_cli_creates_one_completed_training_run(project_root, tmp_path):
    """Verify a tiny training command records metrics, params, and a checkpoint."""
    artifact_root = tmp_path / "artifacts"
    save_dir = tmp_path / "training_results"
    command = [
        sys.executable,
        str(project_root / "train_dqn.py"),
        "--role",
        "MrX",
        "--episodes",
        "14",
        "--plotting-every",
        "7",
        "--map-size",
        "test",
        "--detectives",
        "2",
        "--max-turns",
        "12",
        "--eval-games",
        "2",
        "--device",
        "cpu",
        "--no-show-plots",
        "--save-dir",
        str(save_dir),
        "--artifact-root",
        str(artifact_root),
        "--logger-config",
        str(project_root / "logger_config.yaml"),
        "--run-name",
        "characterization-training",
    ]
    environment = dict(os.environ, MPLBACKEND="Agg")

    completed = subprocess.run(
        command,
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    run_directories = list((artifact_root / "runs").glob("*/*"))
    assert len(run_directories) == 1
    run_dir = run_directories[0]
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["status"] == "completed"
    assert manifest["run_type"] == "training"
    assert manifest["name"] == "characterization-training"

    params = manifest["params"]
    assert params["training/player_role"] == "MrX"
    assert params["training/episodes"] == 14
    assert params["network/feature_size"] == 226
    assert params["network/parameter_count"] == 54273

    result = manifest["result"]
    assert result["training/total_episodes"] == 14
    assert result["validation/games"] == 2
    assert 0.0 <= result["validation/win_rate"] <= 1.0

    # The checkpoint is registered as a run artifact and also kept on disk.
    checkpoints = list((run_dir / "artifacts" / "model").glob("*.pth"))
    assert len(checkpoints) == 1
    assert list(save_dir.glob("*.pth"))

    metrics = _read_metrics(run_dir)
    recorded_names = {name for row in metrics for name in row}
    assert {"train/episode_reward", "train/epsilon", "train/buffer_size"} <= recorded_names
    assert {"train/q_mean", "train/q_std"} <= recorded_names
    assert {"validation/win", "validation/turns"} <= recorded_names

    # One train row per episode, plus periodic rolling and Q-value batches.
    episode_rows = [row for row in metrics if "train/episode_reward" in row]
    assert len(episode_rows) == 14


@pytest.mark.integration
def test_training_cli_writes_nothing_outside_the_requested_directories(
    project_root, tmp_path
):
    """Confirm --save-dir fully redirects checkpoints away from the repository."""
    artifact_root = tmp_path / "artifacts"
    save_dir = tmp_path / "isolated"
    command = [
        sys.executable,
        str(project_root / "train_dqn.py"),
        "--episodes",
        "2",
        "--plotting-every",
        "1",
        "--map-size",
        "test",
        "--detectives",
        "2",
        "--max-turns",
        "6",
        "--eval-games",
        "1",
        "--device",
        "cpu",
        "--no-show-plots",
        "--save-dir",
        str(save_dir),
        "--artifact-root",
        str(artifact_root),
        "--logger-config",
        str(project_root / "logger_config.yaml"),
    ]
    environment = dict(os.environ, MPLBACKEND="Agg")

    completed = subprocess.run(
        command,
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    assert list(save_dir.glob("*.pth"))
    # Two episodes are below the plotting threshold, so no plot is written.
    assert not list(save_dir.glob("*.png"))
