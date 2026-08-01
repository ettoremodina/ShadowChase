"""Characterize the ml_logger-enabled agent comparison entry point."""

import importlib.util
import json
import subprocess
import sys

import pytest

from ml_logger import get_logger


logger = get_logger(__name__)


def _load_comparison_cli(project_root):
    """Import the root script under a name that cannot collide with a test."""
    spec = importlib.util.spec_from_file_location(
        "shadow_chase_comparison_cli",
        project_root / "test_agents.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_matchup_plan_keeps_the_selected_pair_first_and_deduplicates(project_root):
    """Verify the plan reproduces the previous script's combination order."""
    cli = _load_comparison_cli(project_root)
    arguments = cli.parse_arguments(
        [
            "--mr-x-agent",
            "random",
            "--detective-agent",
            "random",
            "--all-combinations",
            "--agents",
            "random",
            "heuristic",
        ]
    )

    assert cli.plan_matchups(arguments) == [
        ("random", "random"),
        ("random", "heuristic"),
        ("heuristic", "random"),
    ]

    single = cli.parse_arguments(["--mr-x-agent", "heuristic"])
    assert cli.plan_matchups(single) == [("heuristic", "random")]


@pytest.mark.integration
def test_comparison_cli_records_every_matchup_in_one_run(project_root, tmp_path):
    """Verify one comparison run indexes and aggregates its evaluation runs."""
    artifact_root = tmp_path / "artifacts"
    command = [
        sys.executable,
        str(project_root / "test_agents.py"),
        "--games",
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
        "--all-combinations",
        "--agents",
        "random",
        "heuristic",
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
        "characterization-comparison",
    ]

    completed = subprocess.run(
        command,
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    manifests = [
        json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        for run_dir in (artifact_root / "runs").glob("*/*")
    ]
    comparisons = [
        manifest for manifest in manifests if manifest["run_type"] == "comparison"
    ]
    evaluations = [
        manifest for manifest in manifests if manifest["run_type"] == "evaluation"
    ]

    assert len(comparisons) == 1
    assert len(evaluations) == 3
    manifest = comparisons[0]
    assert manifest["status"] == "completed"
    assert manifest["name"] == "characterization-comparison"
    assert manifest["config"]["matchups"] == [
        "random_vs_random",
        "random_vs_heuristic",
        "heuristic_vs_random",
    ]
    assert manifest["config"]["analysis_enabled"] is False

    result = manifest["result"]
    assert result["comparison/matchups"] == 3
    assert result["comparison/failed_matchups"] == 0
    assert result["comparison/games"] == 3
    assert (
        result["comparison/mrx_wins"]
        + result["comparison/detective_wins"]
        + result["comparison/incomplete_games"]
        == 3
    )
    assert result["comparison/random_vs_heuristic/games"] == 1
    # Every matchup block names the evaluation run that produced it.
    child_run_ids = {evaluation["run_id"] for evaluation in evaluations}
    assert result["comparison/random_vs_heuristic/run_id"] in child_run_ids

    run_dir = artifact_root / "runs"
    comparison_dir = next(
        path.parent
        for path in run_dir.glob("*/*/manifest.json")
        if json.loads(path.read_text(encoding="utf-8"))["run_type"] == "comparison"
    )
    metrics = [
        json.loads(line)
        for line in (comparison_dir / "metrics" / "metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(metrics) == 3
    assert metrics[0]["comparison/random_vs_random/games"] == 1
