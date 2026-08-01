"""
Simple Shadow Chase Terminal Game

A clean, terminal-based implementation of Shadow Chase without GUI.
Choose between human and AI players, multiple difficulty levels, and customizable display.

Usage: 
    python simple_game.py              # Interactive single game
    python simple_game.py --batch N    # Play N games automatically
"""
import sys
import os
from argparse import Namespace

# Add the project root to sys.path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from game_controls.display_utils import VerbosityLevel
from game_controls.game_utils import (
    get_game_configuration, 
    play_single_game, 
    play_multiple_games,
    parse_arguments
)
from agents import AgentType, AgentSelector
from ml_logger import configure_logging, get_logger, run
from ShadowChase.integrations import GameRunRecorder


logger = get_logger(__name__)


def main():
    """Run batch evaluation or the existing interactive terminal workflow."""
    args = parse_arguments()
    
    # Handle batch mode
    if args.batch:
        return _run_batch(args)
    
    # Interactive single game mode
    else:
        configure_logging()
        logger.info("Shadow Chase simple terminal game")
        try:
            # Get full game configuration including agent types
            map_size, play_mode, num_detectives, verbosity, MrX_agent, detective_agent = get_game_configuration()
            
            # Play single interactive game
            game_id, turn_count, completed = play_single_game(
                map_size=map_size,
                play_mode=play_mode,
                num_detectives=num_detectives,
                verbosity=verbosity,
                auto_save=False,
                max_turns=args.max_turns,
                MrX_agent_type=MrX_agent,
                detective_agent_type=detective_agent,
                save_dir=args.save_dir
            )
            
            if completed:
                logger.info("Thanks for playing Shadow Chase")
            else:
                logger.info("Game ended early")
                
        except KeyboardInterrupt:
            logger.info("Game interrupted by user")
        except Exception as e:
            logger.exception("Terminal game failed: %s", e)


def _run_batch(args: Namespace) -> dict:
    """Own one evaluation run around the existing batch game loop."""
    effective_config = {
        "games": args.batch,
        "map_size": args.map_size,
        "num_detectives": args.detectives,
        "max_turns": args.max_turns,
        "mrx_agent": args.mr_x_agent,
        "detective_agent": args.detective_agent,
        "verbosity": args.verbosity,
        "legacy_save_enabled": not args.no_legacy_save,
        "legacy_save_directory": args.save_dir,
    }
    run_name = args.run_name or f"{args.mr_x_agent}-vs-{args.detective_agent}"
    with run(
        "evaluation",
        name=run_name,
        config=effective_config,
        root_dir=args.artifact_root,
        metadata={"entry_point": "game_controls.simple_game"},
        logger_config_path=args.logger_config,
    ) as context:
        recorder = GameRunRecorder(
            context,
            recording_level=args.recording_level,
            save_replays=False if args.no_replays else None,
        )
        recorder.record_run_parameters(effective_config)
        logger.info(
            "Batch mode: %d games, %s vs %s",
            args.batch,
            args.mr_x_agent,
            args.detective_agent,
        )
        results = play_multiple_games(
            n_games=args.batch,
            map_size=args.map_size,
            play_mode="ai_vs_ai",
            num_detectives=args.detectives,
            verbosity=args.verbosity,
            max_turns=args.max_turns,
            MrX_agent_type=args.mr_x_agent,
            detective_agent_type=args.detective_agent,
            save_dir=args.save_dir,
            save_legacy=not args.no_legacy_save,
            run_recorder=recorder,
        )
        duration_seconds = (
            results["end_time"] - results["start_time"]
        ).total_seconds()
        recorder.finalize(
            {
                "completed_games": results["completed_games"],
                "legacy_saved_games": len(results["saved_games"]),
                "batch_duration_seconds": duration_seconds,
            }
        )
        logger.info("Run artifacts: %s", context.run_dir)
        return results

if __name__ == "__main__":
    main()
