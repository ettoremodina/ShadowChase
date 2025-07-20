"""
Simple Scotland Yard Terminal Game

A clean, terminal-based implementation of Scotland Yard without GUI.
Choose between human and AI players, multiple difficulty levels, and customizable display.

Usage: 
    python simple_game.py              # Interactive single game
    python simple_game.py --batch N    # Play N games automatically
"""
import sys
import os
import argparse

# Add the project root to sys.path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from simple_play.display_utils import VerbosityLevel
from simple_play.game_utils import (
    get_game_configuration, 
    play_single_game, 
    play_multiple_games
)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Scotland Yard Terminal Game')
    parser.add_argument('--batch', type=int, metavar='N', 
                       help='Play N games automatically (AI vs AI mode)')
    parser.add_argument('--map-size', choices=['test', 'full'], default='test',
                       help='Map size: test (10 nodes) or full (199 nodes)')
    parser.add_argument('--detectives', type=int, default=2, choices=[1, 2, 3, 4],
                       help='Number of detectives (1-4)')
    parser.add_argument('--max-turns', type=int, default=24,
                       help='Maximum turns per game')
    parser.add_argument('--verbosity', type=int, default=2, choices=[1, 2, 3, 4],
                       help='Verbosity level (1=basic, 2=moves, 3=detailed, 4=debug)')
    return parser.parse_args()


def main():
    """Main entry point"""
    print("🕵️ SCOTLAND YARD - SIMPLE TERMINAL GAME 🕵️‍♂️")
    print("=" * 60)
    
    args = parse_arguments()
    
    # Handle batch mode
    if args.batch:
        print(f"🤖 Batch mode: Playing {args.batch} AI vs AI games")
        results = play_multiple_games(
            n_games=args.batch,
            map_size=args.map_size,
            play_mode="ai_vs_ai",
            num_detectives=args.detectives,
            verbosity=VerbosityLevel.BASIC,
            max_turns=args.max_turns
        )
        return results
    
    # Interactive single game mode
    try:
        map_size, play_mode, num_detectives, verbosity = get_game_configuration()
        
        # Override with command line arguments if provided
        if hasattr(args, 'map_size') and args.map_size:
            map_size = args.map_size
        if hasattr(args, 'detectives') and args.detectives:
            num_detectives = args.detectives
        if hasattr(args, 'verbosity') and args.verbosity:
            verbosity = args.verbosity
        
        # Play single interactive game
        game_id, turn_count, completed = play_single_game(
            map_size=map_size,
            play_mode=play_mode,
            num_detectives=num_detectives,
            verbosity=verbosity,
            auto_save=False,
            max_turns=args.max_turns
        )
        
        if completed:
            print("\nThanks for playing Scotland Yard! 🎮")
        else:
            print("\n👋 Game ended early. Thanks for playing!")
            
    except KeyboardInterrupt:
        print("\n\n👋 Game interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("Please check your game setup and try again.")

if __name__ == "__main__":
    main()
