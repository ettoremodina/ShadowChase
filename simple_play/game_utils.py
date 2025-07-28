"""
Game utilities for Scotland Yard terminal gameplay.
Contains game configuration, saving, and batch execution functions.
"""
import time
from datetime import datetime
import argparse
from typing import Tuple, Optional
from ScotlandYard.core.game import Player, ScotlandYardGame
from ScotlandYard.services.game_service import GameService
from simple_play.display_utils import GameDisplay, VerbosityLevel, display_game_start_info, display_game_over
from simple_play.game_logic import GameController, GameSetup
from tqdm import tqdm

from agents import AgentType


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Scotland Yard Terminal Game')
    from agents import AgentSelector, AgentType
    agent_types = [agent_name[0] for agent_name in AgentSelector.get_agent_choices_for_ui()]

    parser.add_argument('--batch', type=int, metavar='N', 
                       help='Play N games automatically (AI vs AI mode)')
    parser.add_argument('--map-size', choices=['test', 'full', 'extracted'], default='test',
                       help='Map size: test (10 nodes), full (199 nodes), or extracted (custom extracted board)')
    parser.add_argument('--detectives', type=int, default=2, choices=[1, 2, 3, 4, 5],
                       help='Number of detectives (1-5)')
    parser.add_argument('--max-turns', type=int, default=24,
                       help='Maximum turns per game')
    parser.add_argument('--verbosity', type=int, default=2, choices=[0, 1, 2, 3, 4, 5],
                       help='Verbosity level (0=silent, 1=basic, 2=moves, 3=detailed, 4=debug, 5=heuristics)')
    parser.add_argument('--save-dir', type=str, default='fritto_misto',
                       help='Directory for saving games (default: fritto_misto)')
    parser.add_argument('--mr-x-agent', type=str, choices=agent_types, default='random',
                       help='Mr. X agent type (default: random)')
    parser.add_argument('--detective-agent', type=str, choices=agent_types, default='random',
                       help='Detective agent type (default: random)')
    return parser.parse_args()

def get_game_configuration() -> Tuple[str, str, int, int, AgentType, AgentType]:
    """Get game configuration from user or defaults"""
    try:
        from simple_play.game_logic import get_game_mode, get_verbosity_level
        map_size, play_mode, num_detectives, mr_x_agent, detective_agent = get_game_mode()
        verbosity = get_verbosity_level()
        return map_size, play_mode, num_detectives, verbosity, mr_x_agent, detective_agent
    except Exception:
        # Default configuration if interactive selection fails
        return "test", "ai_vs_ai", 2, VerbosityLevel.MOVES, AgentType.RANDOM, AgentType.RANDOM


def create_and_initialize_game(map_size: str, num_detectives: int) -> ScotlandYardGame:
    """Create and initialize a game based on parameters"""
    if map_size == "test":
        game = GameSetup.create_test_game(num_detectives)
        GameSetup.initialize_test_positions(game)
    else:
        game = GameSetup.create_extracted_board_game(num_detectives)
        GameSetup.initialize_extracted_board_positions(game, num_detectives)
    
    return game


def save_game_session(game: ScotlandYardGame, play_mode: str, map_size: str, 
                     num_detectives: int, turn_count: int, display: GameDisplay,
                     mr_x_agent_type=None, detective_agent_type=None, save_dir: str = "fritto_misto", execution_time: float = None) -> Optional[str]:
    """Save a completed game session with metadata"""
    try:
        from ScotlandYard.storage.game_loader import GameLoader
        game_loader = GameLoader(save_dir)
        game_service = GameService(game_loader)
        
        # Convert agent types to strings if they are AgentType enums
        mr_x_agent_str = mr_x_agent_type.value if hasattr(mr_x_agent_type, 'value') else mr_x_agent_type
        detective_agent_str = detective_agent_type.value if hasattr(detective_agent_type, 'value') else detective_agent_type
        
        game_id = game_service.save_terminal_game(
            game, play_mode, map_size, num_detectives, turn_count,
            mr_x_agent_str, detective_agent_str, execution_time
        )
        if display.verbosity >= VerbosityLevel.BASIC:
            display.print_info(f"Game saved successfully! Game ID: {game_id}")
            display.print_info(f"Saved to directory: {save_dir}")
        return game_id
    except Exception as e:
        display.print_error(f"Failed to save game: {e}")
        return None


def save_game_automatically(game: ScotlandYardGame, play_mode: str, map_size: str,
                          num_detectives: int, turn_count: int, display: GameDisplay,
                          mr_x_agent_type=None, detective_agent_type=None, save_dir: str = "fritto_misto", execution_time: float = None) -> Optional[str]:
    """Save game without user confirmation"""
    if not game.game_history:  # Don't save if no moves were made
        return None
    
    game_id = save_game_session(game, play_mode, map_size, num_detectives, turn_count, display,
                               mr_x_agent_type, detective_agent_type, save_dir, execution_time)
    if game_id and display.verbosity >= VerbosityLevel.MOVES:
        print(f"✅ Game {game_id} saved automatically to {save_dir}")
    return game_id


def offer_save_option(game: ScotlandYardGame, play_mode: str, map_size: str, 
                     num_detectives: int, turn_count: int, display: GameDisplay,
                     mr_x_agent_type=None, detective_agent_type=None, save_dir: str = "fritto_misto"):
    """Offer the user option to save the game with confirmation"""
    if not game.game_history:  # Don't save if no moves were made
        return
        
    print("\n💾 SAVE GAME")
    print("=" * 30)
    
    while True:
        save_choice = input("Would you like to save this game? (y/n): ").strip().lower()
        
        if save_choice in ['y', 'yes']:
            game_id = save_game_session(game, play_mode, map_size, num_detectives, turn_count, display,
                                       mr_x_agent_type, detective_agent_type, save_dir, None)
            if game_id:
                print(f"✅ Game saved as: {game_id}")
                print(f"📁 Game files saved to: {save_dir}/")
                print(f"   - Main game file: {save_dir}/games/{game_id}.pkl")
                print(f"   - Metadata: {save_dir}/metadata/{game_id}.json")
            break
        elif save_choice in ['n', 'no']:
            print("Game not saved.")
            break
        else:
            print("Please enter 'y' for yes or 'n' for no.")


def execute_single_turn(controller: GameController, game: ScotlandYardGame, 
                       play_mode: str, display: GameDisplay) -> bool:
    """Execute a single game turn. Returns True if turn completed successfully, False if user quit"""
    current_player = game.game_state.turn
    
    # Handle turn based on game mode
    if current_player == Player.DETECTIVES:
        # Detective turn
        if play_mode in ["human_vs_human", "human_det_vs_ai_mrx"]:
            # Human detectives
            print(f"\n{display.symbols['detective']} DETECTIVES' TURN")
            success = controller.make_all_detective_moves()
            if not success:
                return False  # User quit
        else:
            # AI detectives
            if display.verbosity >= VerbosityLevel.BASIC:
                print(f"\n{display.symbols['detective']} AI DETECTIVES' TURN")
            success = controller.make_ai_move(Player.DETECTIVES)
            if not success:
                display.print_error("AI detectives failed to move")
            
            # if play_mode == "ai_vs_ai":
            #     time.sleep(1)  # Pause for AI vs AI
    
    else:
        # Mr. X turn
        if play_mode in ["human_vs_human", "ai_det_vs_human_mrx"]:
            # Human Mr. X
            print(f"\n{display.symbols['mr_x']} MR. X'S TURN")
            success = controller.make_mr_x_move()
            if not success:
                return False  # User quit
        else:
            # AI Mr. X
            if display.verbosity >= VerbosityLevel.BASIC:
                print(f"\n{display.symbols['mr_x']} AI MR. X'S TURN")
            success = controller.make_ai_move(Player.MRX)
            if not success:
                display.print_error("AI Mr. X failed to move")
            
            # if play_mode == "ai_vs_ai":
            #     time.sleep(1)  # Pause for AI vs AI
    
    # Reset turn state
    controller.reset_turn_state()
    return True




def play_single_game(map_size: str = "test", play_mode: str = "ai_vs_ai", 
                    num_detectives: int = 2, verbosity: int = VerbosityLevel.BASIC,
                    auto_save: bool = False, max_turns: int = 24,
                    mr_x_agent_type=None, detective_agent_type=None, save_dir: str = "fritto_misto") -> Tuple[Optional[str], int, bool]:
    """
    Play a single game with specified parameters.
    
    Args:
        map_size: Size of the game map
        play_mode: Game mode (human_vs_human, ai_vs_ai, etc.)
        num_detectives: Number of detectives
        verbosity: Display verbosity level
        auto_save: Whether to automatically save the game
        max_turns: Maximum number of turns
        mr_x_agent_type: Type of agent for Mr. X (defaults to RANDOM)
        detective_agent_type: Type of agent for detectives (defaults to RANDOM)
        save_dir: Directory for saving games (defaults to "fritto_misto")
    
    Returns:
        Tuple of (game_id if saved, turn_count, game_completed_normally)
    """
    # Start timing the game execution
    start_time = time.time()
    
    # Create display
    display = GameDisplay(verbosity)
    
    # Create and initialize game
    game = create_and_initialize_game(map_size, num_detectives)
    
    # Set the game in display for heuristics initialization
    display.set_game(game)
    
    # Set default agent types if not provided
    if mr_x_agent_type is None:
        mr_x_agent_type = AgentType.RANDOM
    if detective_agent_type is None:
        detective_agent_type = AgentType.RANDOM
    
    # Create controller with agent types
    controller = GameController(game, display, mr_x_agent_type, detective_agent_type)
    
    # Show game start info only if verbosity is high enough
    if verbosity >= VerbosityLevel.MOVES:
        display_game_start_info(display, play_mode, map_size)
    
    # Main game loop
    turn_count = 0
    game_completed = False
    
    while not game.is_game_over():
        turn_count += 1
        
        # Clear screen for clean display (only for interactive modes)
        if verbosity <= VerbosityLevel.MOVES and play_mode != "ai_vs_ai":
            display.clear_screen()
        
        # Show game state
        if verbosity >= VerbosityLevel.MOVES:
            display.print_title(f"Turn {turn_count}")
            display.print_game_state(game)
            display.print_debug_info(game)
        
        # Execute turn
        success = execute_single_turn(controller, game, play_mode, display)
        if not success:
            # User quit early
            break
        
        # Pause between turns (except for AI vs AI which has its own timing)
        if play_mode != "ai_vs_ai" and verbosity >= VerbosityLevel.MOVES:
            input("\n⏯️  Press Enter to continue...")
    
    # Check if game completed normally
    game_completed = game.is_game_over() 
    
    # Calculate execution time
    end_time = time.time()
    execution_time = end_time - start_time

    # Display game over info if verbosity allows
    if verbosity >= VerbosityLevel.MOVES:
        display_game_over(game, display, turn_count, max_turns)

    # Save game
    game_id = None
    if play_mode == "ai_vs_ai":
        auto_save = True  # Always auto-save in AI vs AI mode
    if auto_save:
        game_id = save_game_automatically(game, play_mode, map_size, num_detectives, turn_count, display,
                                         mr_x_agent_type, detective_agent_type, save_dir, execution_time)
    elif verbosity >= VerbosityLevel.MOVES:
        offer_save_option(game, play_mode, map_size, num_detectives, turn_count, display,
                         mr_x_agent_type, detective_agent_type, save_dir)

    return game_id, turn_count, game_completed


def play_multiple_games(n_games: int, map_size: str = "test", play_mode: str = "ai_vs_ai",
                       num_detectives: int = 2, verbosity: int = VerbosityLevel.BASIC,
                       max_turns: int = 24, mr_x_agent_type=None, detective_agent_type=None, 
                       save_dir: str = "fritto_misto") -> dict:
    """
    Play N games automatically with the specified parameters.
    All games are saved automatically without confirmation.
    
    Args:
        n_games: Number of games to play
        map_size: Size of the game map
        play_mode: Game mode (should be ai_vs_ai for batch)
        num_detectives: Number of detectives
        verbosity: Display verbosity level
        max_turns: Maximum number of turns per game
        mr_x_agent_type: Type of agent for Mr. X
        detective_agent_type: Type of agent for detectives
        save_dir: Directory for saving games
    
    Returns:
        Dictionary with game statistics
    """
    mr_x_agent = AgentType(mr_x_agent_type) 
    detective_agent = AgentType(detective_agent_type) 
    print(f"🎮 BATCH GAME EXECUTION - {n_games} games")
    if verbosity >= VerbosityLevel.BASIC:
        print(f"   Mr. X Agent: {mr_x_agent_type.title()}")
        print(f"   Detective Agent: {detective_agent_type.title()}")
        print(f"   Save Directory: {save_dir}")
        print("=" * 50)
    
    results = {
        'total_games': n_games,
        'completed_games': 0,
        'detective_wins': 0,
        'mrx_wins': 0,
        'timeout_games': 0,
        'total_turns': 0,
        'saved_games': [],
        'incomplete_games': [],  # Track incomplete game IDs
        'mr_x_agent': mr_x_agent_type,
        'detective_agent': detective_agent_type,
        'start_time': datetime.now(),
        'end_time': None
    }
    
    for game_num in tqdm(range(1, n_games + 1)):
        if verbosity >= VerbosityLevel.BASIC:
            print(f"\n🔄 Playing game {game_num}/{n_games}...")
        
        # try:
        game_id, turn_count, completed = play_single_game(
            map_size=map_size,
            play_mode=play_mode,
            num_detectives=num_detectives,
            verbosity=verbosity,  # Use the passed verbosity instead of hardcoded BASIC
            auto_save=True,
            max_turns=max_turns,
            mr_x_agent_type=mr_x_agent,
            detective_agent_type=detective_agent,
            save_dir=save_dir
        )
        
        results['total_turns'] += turn_count
        
        if completed:
            results['completed_games'] += 1
            
            # Load the saved game to check winner
            if game_id:
                results['saved_games'].append(game_id)
                # For now, we'll track this during game execution
                # Could be enhanced to check winner from saved game
        else:
            results['timeout_games'] += 1
            if game_id:
                results['incomplete_games'].append(game_id)
                # Print incomplete game ID if verbosity allows basic output or higher
                if verbosity >= VerbosityLevel.BASIC:
                    print(f"❌ Game {game_num} incomplete: {game_id}")
        
        # Progress indicator
        if game_num % 10 == 0 or game_num == n_games:
            progress = (game_num / n_games) * 100
            if verbosity >= VerbosityLevel.BASIC:
                print(f"📊 Progress: {progress:.1f}% ({game_num}/{n_games})")
        
        # except Exception as e:
        #     print(f"❌ Error in game {game_num}: {e}")
        #     continue
    
    results['end_time'] = datetime.now()
    duration = results['end_time'] - results['start_time']
    
    # Print final statistics
    print(f"\n📈 BATCH EXECUTION COMPLETE")
    if verbosity >= VerbosityLevel.BASIC:
        print("=" * 40)
        print(f"Total games: {results['total_games']}")
        print(f"Completed games: {results['completed_games']}")
        print(f"Incomplete games: {len(results['incomplete_games'])}")
        if results['incomplete_games']:
            print(f"Incomplete game IDs: {', '.join(results['incomplete_games'])}")
        print(f"Games saved: {len(results['saved_games'])}")
        print(f"Total turns played: {results['total_turns']}")
        print(f"Average turns per game: {results['total_turns']/results['completed_games']:.1f}" if results['completed_games'] > 0 else "N/A")
        print(f"Execution time: {duration}")
        print(f"Games per minute: {results['completed_games']/(duration.total_seconds()/60):.1f}" if duration.total_seconds() > 0 else "N/A")
        print(f"Mr. X Agent: {results['mr_x_agent'].title()}")
        print(f"Detective Agent: {results['detective_agent'].title()}")
    else:
        # For silent mode, still show incomplete games
        if results['incomplete_games']:
            print(f"Incomplete games ({len(results['incomplete_games'])}): {', '.join(results['incomplete_games'])}")
    
    return results
