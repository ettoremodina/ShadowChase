"""
Training environment for Scotland Yard AI agents.

This module provides the TrainingEnvironment class for standardized
game episode execution during training.
"""

import time
from typing import Dict, List, Any, Tuple

from attr import dataclass

from ScotlandYard.core.game import ScotlandYardGame, Player
from simple_play.game_utils import create_and_initialize_game, execute_single_turn
from simple_play.game_logic import GameController
from simple_play.display_utils import GameDisplay, VerbosityLevel
from agents import AgentType

from ScotlandYard.storage.game_loader import GameLoader
from ScotlandYard.services.game_service import GameService

@dataclass
class GameResult:
    """Result of a single game episode."""
    winner: str  # "mr_x", "detectives", or "timeout"
    total_turns: int
    game_length: float  # seconds
    mr_x_final_position: int
    detective_final_positions: List[int]
    moves_history: List[Dict[str, Any]]


class TrainingEnvironment:
    """
    Environment for training agents in Scotland Yard games.
    
    This class provides a standardized interface for running game episodes
    during training, with support for different agent types and configurations.
    """
    
    def __init__(self, 
                 map_size: str = "test",
                 num_detectives: int = 2,
                 max_turns: int = 24,
                 verbosity: int = VerbosityLevel.SILENT):
        """
        Initialize the training environment.
        
        Args:
            map_size: Game map size ("test", "full", "extracted")
            num_detectives: Number of detective agents
            max_turns: Maximum turns per game
            verbosity: Display verbosity level
        """
        self.map_size = map_size
        self.num_detectives = num_detectives
        self.max_turns = max_turns
        self.verbosity = verbosity
        self.save_dir = "game_saves"
        self.game_loader = GameLoader(self.save_dir)
        self.game_service = GameService(self.game_loader)
        
    def run_episode(self, 
                   mr_x_agent, 
                   detective_agent,
                   collect_experience: bool = False) -> Tuple[GameResult, List[Dict[str, Any]]]:
        """
        Run a single game episode with the given agents.
        
        Args:
            mr_x_agent: Mr. X agent instance
            detective_agent: Detective agent instance
            collect_experience: Whether to collect detailed experience data
            
        Returns:
            Tuple of (GameResult, experience_data)
        """
        start_time = time.time()
        
        # Create and initialize game
        game = create_and_initialize_game(self.map_size, self.num_detectives)
        display = GameDisplay(self.verbosity)
        display.set_game(game)
        
        # Create controller with custom agents
        controller = GameController(game, display, 
                                   AgentType.RANDOM,  # We'll replace these
                                   AgentType.RANDOM)
        
        # Replace the controller's agents with our custom ones
        controller.mr_x_agent = mr_x_agent
        controller.multi_detective_agent = detective_agent
        
        # Track experience data
        experience_data = []
        moves_history = []
        turn_count = 0
        
        # Main game loop
        while not game.is_game_over() and turn_count < self.max_turns:
            turn_count += 1
            
            # Collect pre-move state if requested
            if collect_experience:
                pre_state = {
                    'turn': turn_count,
                    'current_player': game.game_state.turn,
                    'mr_x_position': game.game_state.MrX_position,
                    'detective_positions': game.game_state.detective_positions.copy(),
                    'mr_x_visible': game.game_state.mr_x_visible,
                    'mr_x_tickets': game.get_mr_x_tickets(),
                    'game_state_detective': game.get_state_copy() if hasattr(game, 'get_state_detective') else None
                }
            
            # Execute turn
            success = execute_single_turn(controller, game, "ai_vs_ai", display)
            if not success:
                break
            
            # Collect post-move data if requested
            if collect_experience:
                post_state = {
                    'mr_x_position': game.game_state.MrX_position,
                    'detective_positions': game.game_state.detective_positions.copy(),
                    'mr_x_visible': game.game_state.mr_x_visible
                }
                
                # Record the experience
                experience_data.append({
                    'pre_state': pre_state,
                    'post_state': post_state,
                    'turn': turn_count,
                    'player': pre_state['current_player']
                })
            
            # Track moves
            if hasattr(game, 'game_history') and game.game_history:
                last_move = game.game_history[-1]
                moves_history.append(last_move)
        
        # Determine winner
        end_time = time.time()
        game_length = end_time - start_time
        
        if game.is_game_over():
            winner = game.get_winner()
            if winner == Player.MRX:
                winner_str = "mr_x"
            elif winner == Player.DETECTIVES:
                winner_str = "detectives"
            else:
                winner_str = "unknown"
        else:
            winner_str = "timeout"
        
        # Create result
        result = GameResult(
            winner=winner_str,
            total_turns=turn_count,
            game_length=game_length,
            mr_x_final_position=game.game_state.MrX_position,
            detective_final_positions=game.game_state.detective_positions.copy(),
            moves_history=moves_history
        )
            
        
        # Define missing variables for game saving
        mr_x_agent_type = type(mr_x_agent).__name__
        detective_agent_type = type(detective_agent).__name__

        # Convert agent types to strings if they are AgentType enums
        mr_x_agent_str = mr_x_agent_type.value if hasattr(mr_x_agent_type, 'value') else mr_x_agent_type
        detective_agent_str = detective_agent_type.value if hasattr(detective_agent_type, 'value') else detective_agent_type
        
        game_id = self.game_service.save_terminal_game(
            game, "ai_vs_ai", self.map_size, self.num_detectives, turn_count,
            mr_x_agent_str, detective_agent_str, game_length
        )
        return result, experience_data
    
    def run_evaluation_batch(self,
                           mr_x_agent,
                           detective_agent,
                           num_games: int) -> List[GameResult]:
        """
        Run a batch of evaluation games.
        
        Args:
            mr_x_agent: Mr. X agent instance
            detective_agent: Detective agent instance
            num_games: Number of games to run
            
        Returns:
            List of GameResult objects
        """
        results = []
        
        for i in range(num_games):
            if i % 10 == 0:
                print(f"Running evaluation game {i+1}/{num_games}")
            
            result, _ = self.run_episode(mr_x_agent, detective_agent, collect_experience=False)
            results.append(result)
        
        return results
