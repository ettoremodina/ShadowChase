"""Characterize the ml_logger-enabled terminal batch entry point."""

import json
import subprocess
import sys

import pytest

from ml_logger import get_logger


logger = get_logger(__name__)


@pytest.mark.integration
def test_batch_cli_creates_one_completed_evaluation_run(project_root, tmp_path):
    """Verify a one-game batch emits metrics without legacy pickle output."""
    artifact_root = tmp_path / "artifacts"
    command = [
        sys.executable,
        str(project_root / "game_controls" / "simple_game.py"),
        "--batch",
        "1",
        "--map-size",
        "test",
        "--detectives",
        "2",
        "--max-turns",
        "24",
        "--mr-x-agent",
        "random",
        "--detective-agent",
        "random",
        "--verbosity",
        "0",
        "--recording-level",
        "summary",
        "--no-replays",
        "--no-legacy-save",
        "--artifact-root",
        str(artifact_root),
        "--logger-config",
        str(project_root / "logger_config.yaml"),
        "--run-name",
        "characterization-batch",
    ]

    completed = subprocess.run(
        command,
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    run_directories = list((artifact_root / "runs").glob("*/*"))
    assert len(run_directories) == 1
    run_dir = run_directories[0]
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    metrics = [
        json.loads(line)
        for line in (run_dir / "metrics" / "metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert manifest["status"] == "completed"
    assert manifest["run_type"] == "evaluation"
    assert manifest["name"] == "characterization-batch"
    assert manifest["config"]["legacy_save_enabled"] is False
    assert manifest["result"]["evaluation/games"] == 1
    assert len(metrics) == 1
    assert metrics[0]["game/legacy_saved"] == 0
    assert not list((run_dir / "replays").glob("*.json"))
