"""Characterize the ml_logger-enabled interactive and visualization CLI."""

import json
import subprocess
import sys

import pytest

from ml_logger import get_logger


logger = get_logger(__name__)


def _run_main(project_root, *arguments, timeout=120):
    """Invoke main.py as its own process, the way a user would."""
    command = [sys.executable, str(project_root / "main.py"), *arguments]
    return subprocess.run(
        command,
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


@pytest.mark.integration
def test_headless_demo_creates_one_completed_gameplay_run(project_root, tmp_path):
    """Verify one demo owns one run and records its finished game."""
    artifact_root = tmp_path / "artifacts"

    completed = _run_main(
        project_root,
        "--demo",
        "until-end",
        "--headless",
        "--artifact-root",
        str(artifact_root),
        "--logger-config",
        str(project_root / "logger_config.yaml"),
        "--run-name",
        "characterization-gameplay",
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
    assert manifest["run_type"] == "gameplay"
    assert manifest["name"] == "characterization-gameplay"
    assert manifest["config"]["demo"] == "until-end"
    assert manifest["config"]["headless"] is True

    result = manifest["result"]
    assert result["gameplay/games"] == 1
    assert result["gameplay/recorded_games"] == 1
    assert result["gameplay/detective_wins"] == 1

    assert len(metrics) == 1
    assert metrics[0]["game/detective_win"] == 1
    # The scripted path game uses the ticketless base class.
    assert metrics[0]["game/ticket_events"] == 0

    replays = list((run_dir / "replays").glob("*.json"))
    assert len(replays) == 1
    replay = json.loads(replays[0].read_text(encoding="utf-8"))
    assert replay["schema_version"] == 1
    assert replay["summary"]["winner"] == "detectives"


@pytest.mark.integration
def test_headless_ticket_demo_records_a_shadow_chase_game(project_root, tmp_path):
    """Verify a full-board demo records ticket state without the GUI."""
    artifact_root = tmp_path / "artifacts"

    completed = _run_main(
        project_root,
        "--demo",
        "random-start",
        "--detectives",
        "3",
        "--headless",
        "--recording-level",
        "summary",
        "--artifact-root",
        str(artifact_root),
        "--logger-config",
        str(project_root / "logger_config.yaml"),
    )

    assert completed.returncode == 0, completed.stderr
    run_dir = next(iter((artifact_root / "runs").glob("*/*")))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["status"] == "completed"
    assert manifest["params"]["num_detectives"] == 3
    assert manifest["params"]["logging/recording_level"] == "summary"
    # An initialized but unplayed game is still one recorded, incomplete game.
    assert manifest["result"]["gameplay/games"] == 1
    assert manifest["result"]["gameplay/incomplete_games"] == 1
    assert not list((run_dir / "replays").glob("*.json"))


def test_fixed_layout_demo_rejects_a_detective_count(project_root, tmp_path):
    """Verify a flag that cannot take effect fails instead of being ignored."""
    completed = _run_main(
        project_root,
        "--demo",
        "basic",
        "--detectives",
        "3",
        "--headless",
        "--artifact-root",
        str(tmp_path / "artifacts"),
    )

    assert completed.returncode != 0
    assert "does not accept --detectives" in completed.stderr
    assert not (tmp_path / "artifacts").exists()


def test_list_demos_exits_without_opening_a_run(project_root, tmp_path):
    """Verify the catalog listing stays outside the run lifecycle."""
    completed = _run_main(
        project_root,
        "--list-demos",
        "--artifact-root",
        str(tmp_path / "artifacts"),
    )

    assert completed.returncode == 0, completed.stderr
    assert "extracted" in completed.stdout + completed.stderr
    assert not (tmp_path / "artifacts").exists()
