"""Interactive and visualization entry point for Shadow Chase.

One invocation owns exactly one ``gameplay`` run. The demos in
``ShadowChase.examples.demos`` build and play games; this command opens the
run, records the resulting game through the ml_logger adapter, and closes the
run. Demo code never touches the run lifecycle.

Usage:
    python main.py                       # extracted board with 5 detectives
    python main.py --list-demos          # show every available demo
    python main.py --demo grid --headless
"""
import argparse
import os
import sys
import time
from typing import Optional, Sequence

# Add the project root to sys.path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from ml_logger import configure_logging, get_logger, run
from ShadowChase.core.game import Game
from ShadowChase.examples.demos import DEFAULT_DEMO, DEMOS
from ShadowChase.integrations import GameRunRecorder


logger = get_logger(__name__)


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Shadow Chase interactive and visualization demos',
    )
    parser.add_argument('--demo', choices=sorted(DEMOS), default=DEFAULT_DEMO,
                       help=f'Demonstration to run (default: {DEFAULT_DEMO})')
    parser.add_argument('--list-demos', action='store_true',
                       help='List the available demonstrations and exit')
    parser.add_argument('--detectives', type=int, choices=[1, 2, 3, 4, 5],
                       help='Number of detectives, for demos that accept one')
    parser.add_argument('--headless', action='store_true',
                       help='Build and record the game without opening the GUI')
    parser.add_argument('--run-name', type=str,
                       help='Optional ml_logger run name')
    parser.add_argument('--logger-config', type=str,
                       help='Path to an ml_logger YAML configuration')
    parser.add_argument('--artifact-root', type=str,
                       help='Override the ml_logger artifact root directory')
    parser.add_argument('--recording-level', choices=['summary', 'actions', 'full'],
                       help='Game replay detail stored by ml_logger')
    parser.add_argument('--no-replays', action='store_true',
                       help='Disable ml_logger replay files while retaining metrics')
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run one demonstration inside a single gameplay run."""
    args = parse_arguments(argv)

    if args.list_demos:
        configure_logging()
        _log_demo_catalog()
        return 0

    demo = DEMOS[args.demo]
    if args.detectives is not None and not demo.configurable_detectives:
        raise SystemExit(
            f"Demo '{args.demo}' uses a fixed layout of "
            f"{demo.default_detectives} detectives and does not accept "
            "--detectives"
        )
    num_detectives = args.detectives or demo.default_detectives
    headless = args.headless or not demo.uses_visualizer

    effective_config = {
        "demo": args.demo,
        "num_detectives": num_detectives,
        "headless": headless,
        "uses_visualizer": demo.uses_visualizer and not args.headless,
    }
    with run(
        "gameplay",
        name=args.run_name or args.demo,
        config=effective_config,
        root_dir=args.artifact_root,
        metadata={"entry_point": "main"},
        logger_config_path=args.logger_config,
    ) as context:
        recorder = GameRunRecorder(
            context,
            recording_level=args.recording_level,
            save_replays=False if args.no_replays else None,
        )
        recorder.record_run_parameters(effective_config)
        logger.info("Demo '%s': %s", args.demo, demo.description)

        start_time = time.time()
        game = demo.play(num_detectives=num_detectives, headless=headless)
        execution_time = time.time() - start_time

        recorded = _record_demo_game(
            recorder,
            game,
            demo_name=args.demo,
            num_detectives=num_detectives,
            execution_time=execution_time,
        )
        recorder.finalize(
            {
                "demo": args.demo,
                "recorded_games": int(recorded),
                "duration_seconds": execution_time,
            },
            namespace="gameplay",
        )
        logger.info("Run artifacts: %s", context.run_dir)
    return 0


def _record_demo_game(recorder: GameRunRecorder, game: Optional[Game], *,
                      demo_name: str, num_detectives: int,
                      execution_time: float) -> bool:
    """Record the demo's game unless it never reached an initialized state."""
    if game is None or game.game_state is None:
        logger.info("Demo '%s' produced no initialized game to record", demo_name)
        return False
    recorder.record_game(
        0,
        game,
        execution_time_seconds=execution_time,
        game_id=demo_name,
        metadata={
            "demo": demo_name,
            "num_detectives": num_detectives,
            "game_class": type(game).__name__,
        },
    )
    return True


def _log_demo_catalog() -> None:
    """Log every demo with the detective count it runs by default."""
    logger.info("Available demos:")
    for name, demo in sorted(DEMOS.items()):
        detectives = demo.default_detectives
        logger.info(
            "  %-12s %s (%d detective%s%s%s)",
            name,
            demo.description,
            detectives,
            "" if detectives == 1 else "s",
            "" if demo.configurable_detectives else ", fixed",
            ", GUI" if demo.uses_visualizer else "",
        )


if __name__ == "__main__":
    raise SystemExit(main())
