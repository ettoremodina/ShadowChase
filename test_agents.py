#!/usr/bin/env python3
"""Agent comparison entry point for Shadow Chase.

One invocation owns exactly one ``comparison`` run. Every matchup is still
evaluated by ``game_controls/simple_game.py --batch``, which owns its own
``evaluation`` run in its own process; this command records those child results
as one comparison and registers the analysis output as run artifacts.

Usage:
    python test_agents.py                          # random vs random, 10 games
    python test_agents.py --all-combinations       # every distinct matchup
    python test_agents.py --mr-x-agent deep_q --detective-agent heuristic
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

# Add the project root to sys.path for imports
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_logger import RunCatalog, get_logger, run
from ShadowChase.integrations import AgentComparisonRecorder, evaluation_summary

from agents import AgentSelector


logger = get_logger(__name__)

DEFAULT_TEST_NAME = "video_exporting_test"
DEFAULT_GAMES_PER_MATCHUP = 10
DEFAULT_MAP_SIZE = "extracted"
DEFAULT_DETECTIVES = 5
DEFAULT_MAX_TURNS = 24
LEGACY_SAVE_ROOT = Path("saved_games")
EVALUATION_SCRIPT = PROJECT_ROOT / "game_controls" / "simple_game.py"
ANALYSIS_SCRIPT = PROJECT_ROOT / "ShadowChase" / "services" / "analyze_games.py"


def child_environment() -> dict[str, str]:
    """Give child processes a UTF-8 stdout.

    Both the evaluation and analysis scripts print non-ASCII status characters.
    Under the Windows default console encoding that raises ``UnicodeEncodeError``
    as soon as their output is redirected, which would fail a matchup for a
    reason that has nothing to do with the games it played.
    """
    return {**os.environ, "PYTHONIOENCODING": "utf-8"}


def agent_types() -> list[str]:
    """Return the selectable agent identifiers, in registration order."""
    return [choice[0] for choice in AgentSelector.get_agent_choices_for_ui()]


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command line arguments"""
    choices = agent_types()
    parser = argparse.ArgumentParser(
        description='Shadow Chase agent comparison across matchups',
    )
    parser.add_argument('--test-name', type=str, default=DEFAULT_TEST_NAME,
                       help=f'Legacy saved_games subdirectory and analysis target '
                            f'(default: {DEFAULT_TEST_NAME})')
    parser.add_argument('--games', type=int, default=DEFAULT_GAMES_PER_MATCHUP,
                       help=f'Games per matchup (default: {DEFAULT_GAMES_PER_MATCHUP})')
    parser.add_argument('--map-size', choices=['test', 'full', 'extracted'],
                       default=DEFAULT_MAP_SIZE,
                       help=f'Board used by every matchup (default: {DEFAULT_MAP_SIZE})')
    parser.add_argument('--detectives', type=int, default=DEFAULT_DETECTIVES,
                       choices=[1, 2, 3, 4, 5],
                       help=f'Number of detectives (default: {DEFAULT_DETECTIVES})')
    parser.add_argument('--max-turns', type=int, default=DEFAULT_MAX_TURNS,
                       help=f'Maximum turns per game (default: {DEFAULT_MAX_TURNS})')
    parser.add_argument('--mr-x-agent', choices=choices, default='random',
                       help='Mr. X agent of the always-evaluated matchup')
    parser.add_argument('--detective-agent', choices=choices, default='random',
                       help='Detective agent of the always-evaluated matchup')
    parser.add_argument('--all-combinations', action='store_true',
                       help='Also evaluate every distinct ordered agent pair')
    parser.add_argument('--agents', nargs='+', choices=choices, default=choices,
                       help='Agents used to build --all-combinations pairs')
    parser.add_argument('--verbosity', type=int, default=0, choices=[0, 1, 2, 3, 4, 5],
                       help='Verbosity passed to each evaluation process (default: 0)')
    parser.add_argument('--no-analysis', action='store_true',
                       help='Skip the legacy analysis pass over saved_games')
    parser.add_argument('--run-name', type=str,
                       help='Optional ml_logger run name')
    parser.add_argument('--logger-config', type=str,
                       help='Path to an ml_logger YAML configuration')
    parser.add_argument('--artifact-root', type=str,
                       help='Override the ml_logger artifact root directory')
    parser.add_argument('--recording-level', choices=['summary', 'actions', 'full'],
                       help='Game replay detail stored by each evaluation run')
    parser.add_argument('--no-replays', action='store_true',
                       help='Disable ml_logger replay files while retaining metrics')
    parser.add_argument('--no-legacy-save', action='store_true',
                       help='Disable legacy saved_games output, which also skips analysis')
    return parser.parse_args(argv)


def plan_matchups(args: argparse.Namespace) -> list[tuple[str, str]]:
    """List the matchups to evaluate, keeping the selected one first."""
    matchups = [(args.mr_x_agent, args.detective_agent)]
    if args.all_combinations:
        for mrx_agent in args.agents:
            for detective_agent in args.agents:
                if mrx_agent != detective_agent:
                    matchups.append((mrx_agent, detective_agent))
    unique: list[tuple[str, str]] = []
    for matchup in matchups:
        if matchup not in unique:
            unique.append(matchup)
    return unique


def matchup_name(mrx_agent: str, detective_agent: str) -> str:
    """Return the combination label shared with the legacy save layout."""
    return f"{mrx_agent}_vs_{detective_agent}"


def evaluate_matchup(args: argparse.Namespace, mrx_agent: str,
                     detective_agent: str, *, artifact_root: Path,
                     run_name: str) -> subprocess.CompletedProcess:
    """Run one matchup as its own evaluation process, as before the migration."""
    combination = matchup_name(mrx_agent, detective_agent)
    command = [
        sys.executable, str(EVALUATION_SCRIPT),
        "--batch", str(args.games),
        "--map-size", args.map_size,
        "--detectives", str(args.detectives),
        "--max-turns", str(args.max_turns),
        "--save-dir", f"{args.test_name}/{combination}",
        "--mr-x-agent", mrx_agent,
        "--detective-agent", detective_agent,
        "--verbosity", str(args.verbosity),
        "--artifact-root", str(artifact_root),
        "--run-name", run_name,
    ]
    if args.logger_config:
        command += ["--logger-config", args.logger_config]
    if args.recording_level:
        command += ["--recording-level", args.recording_level]
    if args.no_replays:
        command.append("--no-replays")
    if args.no_legacy_save:
        command.append("--no-legacy-save")

    logger.info("Evaluating matchup %s", combination)
    return subprocess.run(command, check=False, env=child_environment())


def find_matchup_run(catalog: RunCatalog, run_name: str) -> Optional[dict]:
    """Return the manifest of the child run created under a unique name."""
    for row in catalog.list_runs(limit=500):
        if row["name"] == run_name:
            return json.loads(row["manifest_json"])
    return None


def analyze_games(test_name: str) -> Optional[Path]:
    """Run the legacy analysis pass and return the directory it wrote into."""
    analysis_dir = LEGACY_SAVE_ROOT / test_name
    if not analysis_dir.is_dir():
        logger.warning("No legacy games to analyze in %s", analysis_dir)
        return None

    logger.info("Analyzing games in %s", analysis_dir)
    completed = subprocess.run(
        [sys.executable, str(ANALYSIS_SCRIPT), test_name],
        check=False,
        env=child_environment(),
    )
    if completed.returncode != 0:
        logger.error("Analysis failed with exit code %d", completed.returncode)
        return None
    return analysis_dir


def register_analysis_artifacts(recorder: AgentComparisonRecorder,
                                analysis_dir: Path) -> int:
    """Copy the analysis plots and report into the comparison run."""
    registered = 0
    for plot in sorted((analysis_dir / "analysis_graphs").glob("*.jpg")):
        recorder.record_artifact(plot, kind="plot")
        registered += 1
    report = analysis_dir / "analysis_report.txt"
    if report.is_file():
        recorder.record_artifact(report, kind="report")
        registered += 1
    logger.info("Registered %d analysis artifacts", registered)
    return registered


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Compare agent matchups inside a single comparison run."""
    args = parse_arguments(argv)
    matchups = plan_matchups(args)
    analysis_enabled = not (args.no_analysis or args.no_legacy_save)

    effective_config = {
        "test_name": args.test_name,
        "games_per_matchup": args.games,
        "map_size": args.map_size,
        "num_detectives": args.detectives,
        "max_turns": args.max_turns,
        "verbosity": args.verbosity,
        "matchups": [matchup_name(*matchup) for matchup in matchups],
        "all_combinations": args.all_combinations,
        "analysis_enabled": analysis_enabled,
        "legacy_save_enabled": not args.no_legacy_save,
    }
    with run(
        "comparison",
        name=args.run_name or args.test_name,
        config=effective_config,
        root_dir=args.artifact_root,
        metadata={"entry_point": "test_agents"},
        logger_config_path=args.logger_config,
    ) as context:
        recorder = AgentComparisonRecorder(context)
        recorder.record_run_parameters(effective_config)
        logger.info(
            "Comparing %d matchup(s), %d games each on the %s map",
            len(matchups),
            args.games,
            args.map_size,
        )

        token = context.run_id.rsplit("-", 1)[-1]
        start_time = time.time()
        for index, (mrx_agent, detective_agent) in enumerate(matchups):
            combination = matchup_name(mrx_agent, detective_agent)
            child_name = f"{combination}-{token}"
            completed = evaluate_matchup(
                args,
                mrx_agent,
                detective_agent,
                artifact_root=context.root_dir,
                run_name=child_name,
            )
            _record_matchup_outcome(
                recorder,
                context.catalog,
                index,
                combination,
                child_name,
                completed.returncode,
            )
            recorder.record_progress(index + 1, len(matchups))
        matchup_duration_seconds = time.time() - start_time

        analysis_artifacts = 0
        if analysis_enabled:
            analysis_dir = analyze_games(args.test_name)
            if analysis_dir is not None:
                analysis_artifacts = register_analysis_artifacts(
                    recorder,
                    analysis_dir,
                )
        else:
            logger.info("Analysis skipped")

        summary = recorder.finalize(
            {
                "requested_matchups": len(matchups),
                "analysis_artifacts": analysis_artifacts,
                "matchup_duration_seconds": matchup_duration_seconds,
            }
        )
        logger.info(
            "Comparison complete: %d/%d matchups, %d games",
            summary["comparison/matchups"],
            len(matchups),
            summary["comparison/games"],
        )
        logger.info("Run artifacts: %s", context.run_dir)
    return 0


def _record_matchup_outcome(recorder: AgentComparisonRecorder,
                            catalog: RunCatalog, index: int, combination: str,
                            child_name: str, return_code: int) -> None:
    """Record a matchup from its child run, or as a failure that did not stop the run."""
    if return_code != 0:
        recorder.record_failed_matchup(
            index,
            combination,
            f"Evaluation process exited with code {return_code}",
        )
        return

    manifest = find_matchup_run(catalog, child_name)
    if manifest is None:
        recorder.record_failed_matchup(
            index,
            combination,
            f"No evaluation run named {child_name} was cataloged",
        )
        return

    summary = evaluation_summary(manifest.get("result", {}))
    if not summary:
        recorder.record_failed_matchup(
            index,
            combination,
            f"Evaluation run {manifest['run_id']} recorded no summary",
        )
        return

    recorded = recorder.record_matchup(
        index,
        combination,
        summary,
        run_id=manifest["run_id"],
    )
    logger.info(
        "Matchup %s: %d games, Mr. X win rate %.1f%%, %.1f average turns",
        combination,
        int(recorded.summary.get("games", 0)),
        100 * float(recorded.summary.get("mrx_win_rate", 0.0)),
        float(recorded.summary.get("average_turns", 0.0)),
    )


if __name__ == "__main__":
    raise SystemExit(main())
