"""
Simple Scotland Yard Terminal Game

A clean, terminal-based implementation of Scotland Yard without GUI.
Choose between human and AI players, multiple difficulty levels, and customizable display.

Usage: python simple_game.py
"""
import sys
import os
import time

# Add the project root to sys.path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from cops_and_robbers.core.game import Player
from simple_play.display_utils import GameDisplay, VerbosityLevel
from simple_play.game_logic import GameController, GameSetup, get_game_mode, get_verbosity_level


def main():
    """Main game loop"""
    print("🕵️ SCOTLAND YARD - SIMPLE TERMINAL GAME 🕵️‍♂️")
    print("=" * 60)
    
    # Get game configuration
    # map_size, play_mode, num_detectives = "test", "human_det_vs_ai_mrx", 2
    # verbosity = 4
    map_size, play_mode, num_detectives = get_game_mode()
    verbosity = get_verbosity_level()
    
    # Create display
    display = GameDisplay(verbosity)
    
    # Create and initialize game
    if map_size == "test":
        game = GameSetup.create_test_game(num_detectives)
        GameSetup.initialize_test_positions(game)
        display.print_info("Created test game (10 nodes)")
    else:
        game = GameSetup.create_full_game(num_detectives)
        GameSetup.initialize_full_positions(game, num_detectives)
        display.print_info("Created full Scotland Yard game (199 nodes)")
    
    # Create controller
    controller = GameController(game, display)
    
    # Game mode info
    mode_descriptions = {
        "human_vs_human": "Human vs Human",
        "human_det_vs_ai_mrx": "Human Detectives vs AI Mr. X",
        "ai_det_vs_human_mrx": "AI Detectives vs Human Mr. X", 
        "ai_vs_ai": "AI vs AI"
    }
    display.print_info(f"Game mode: {mode_descriptions[play_mode]}")
    
    # Show initial help
    if play_mode != "ai_vs_ai":
        display.print_input_help()
    
    # Main game loop
    turn_count = 0
    max_turns = 24  # Safety limit
    
    while not game.is_game_over() and turn_count < max_turns:
        turn_count += 1
        
        # Clear screen for clean display
        if verbosity <= VerbosityLevel.MOVES:
            display.clear_screen()
        
        # Show game state
        display.print_title(f"Turn {turn_count}")
        display.print_game_state(game)
        
        # Debug info if requested
        display.print_debug_info(game)
        
        current_player = game.game_state.turn
        
        # Handle turn based on game mode
        if current_player == Player.COPS:
            # Detective turn
            if play_mode in ["human_vs_human", "human_det_vs_ai_mrx"]:
                # Human detectives
                print(f"\n{display.symbols['detective']} DETECTIVES' TURN")
                
                # All detectives move simultaneously
                success = controller.make_all_detective_moves()
                if not success:
                    print("\n👋 Game ended by user")
                    return
            else:
                # AI detectives
                print(f"\n{display.symbols['detective']} AI DETECTIVES' TURN")
                success = controller.make_ai_move(Player.COPS)
                if not success:
                    display.print_error("AI detectives failed to move")
                
                if play_mode == "ai_vs_ai":
                    time.sleep(2)  # Pause for AI vs AI
        
        else:
            # Mr. X turn
            if play_mode in ["human_vs_human", "ai_det_vs_human_mrx"]:
                # Human Mr. X
                print(f"\n{display.symbols['mr_x']} MR. X'S TURN")
                success = controller.make_human_move(Player.ROBBER)
                if not success:
                    print("\n👋 Game ended by user")
                    return
            else:
                # AI Mr. X
                print(f"\n{display.symbols['mr_x']} AI MR. X'S TURN")
                success = controller.make_ai_move(Player.ROBBER)
                if not success:
                    display.print_error("AI Mr. X failed to move")
                
                if play_mode == "ai_vs_ai":
                    time.sleep(2)  # Pause for AI vs AI
        
        # Reset turn state
        controller.reset_turn_state()
        
        # Pause between turns (except for AI vs AI which has its own timing)
        if play_mode != "ai_vs_ai":
            input("\n⏯️  Press Enter to continue...")
    
    # Game over
    display.clear_screen()
    display.print_title("GAME OVER")
    display.print_game_state(game)
    
    if game.is_game_over():
        winner = game.get_winner()
        if winner == Player.COPS:
            print(f"\n🏆 {display.symbols['detective']} DETECTIVES WIN!")
            print("The detectives have successfully captured Mr. X!")
        else:
            print(f"\n🏆 {display.symbols['mr_x']} MR. X WINS!")
            print("Mr. X has successfully evaded the detectives!")
    else:
        print(f"\n⏰ Game ended after {max_turns} turns")
    
    print(f"\nTotal turns played: {turn_count}")
    print("\nThanks for playing Scotland Yard! 🎮")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Game interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("Please check your game setup and try again.")
