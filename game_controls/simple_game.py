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


def main():
    """Main entry point"""
    print("🕵️ SCOTLAND YARD - SIMPLE TERMINAL GAME 🕵️‍♂️")
    print("=" * 60)
    
    args = parse_arguments()
    
    # Handle batch mode
    if args.batch:
        print(f"🤖 Batch mode: Playing {args.batch} AI vs AI games")
        
        print(f"Mr. X Agent: {args.mr_x_agent}")
        print(f"Detective Agent: {args.detective_agent}")
        
        results = play_multiple_games(
            n_games=args.batch,
            map_size=args.map_size,
            play_mode="ai_vs_ai",
            num_detectives=args.detectives,
            verbosity=args.verbosity,  # Use the verbosity from command line args
            max_turns=args.max_turns,
            mr_x_agent_type=args.mr_x_agent,
            detective_agent_type=args.detective_agent,
            save_dir=args.save_dir
        )
        return results
    
    # Interactive single game mode
    else:
        try:
            # Get full game configuration including agent types
            map_size, play_mode, num_detectives, verbosity, mr_x_agent, detective_agent = get_game_configuration()
            
            # Play single interactive game
            game_id, turn_count, completed = play_single_game(
                map_size=map_size,
                play_mode=play_mode,
                num_detectives=num_detectives,
                verbosity=verbosity,
                auto_save=False,
                max_turns=args.max_turns,
                mr_x_agent_type=mr_x_agent,
                detective_agent_type=detective_agent,
                save_dir=args.save_dir
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
