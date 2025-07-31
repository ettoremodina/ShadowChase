"""
Training environment for Scotland Yard AI agents.

This module provides the TrainingEnvironment class for standardized
game episode execution during training.
"""

import time
from typing import Dict, List, Any, Tuple

from dataclasses import dataclass
from ScotlandYard.core.game import Player
from game_controls.game_utils import create_and_initialize_game, execute_single_turn
from game_controls.game_logic import GameController
from game_controls.display_utils import GameDisplay, VerbosityLevel

from ScotlandYard.services.game_loader import GameLoader
from ScotlandYard.services.game_service import GameService
from agents.heuristics import GameHeuristics
from agents import AgentType

@dataclass
class GameResult:
    """Result of a single game episode."""
    winner: str  # "mr_x", "detectives", or "timeout"
    total_turns: int
    game_length: float  # seconds
    mr_x_final_position: int
    detective_final_positions: List[int]
    moves_history: List[Dict[str, Any]]
    mr_x_min_distances: List[int]  # For reward shaping, min distance to Mr. X after each turn


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
        controller.agent_mrx = mr_x_agent
        controller.agent_detectives = detective_agent
        
        # Track experience data
        experience_data = []
        moves_history = []
        turn_count = 0
        
        # For reward shaping: track min distance to Mr. X after each turn
        mr_x_min_distances = []
        heuristics = GameHeuristics(game)

        # Main game loop
        while not game.is_game_over():
            # Collect pre-move state if requested
            # if collect_experience:

            success = execute_single_turn(controller, game, "ai_vs_ai", display)
            if not success:
                break
            # Collect post-move data if requested
            # if collect_experience:

            # Track moves
            if hasattr(game, 'game_history') and game.game_history:
                last_move = game.game_history[-1]
                moves_history.append(last_move)
            # For reward shaping: record min distance to Mr. X after this turn
            if heuristics:
                min_dist = heuristics.get_minimum_distance_to_mr_x()
                mr_x_min_distances.append(min_dist)
            turn_count += 1
        
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
                winner_str = "unknown" #never reached in this setup
        else:
            winner_str = "timeout" #never reached in this setup
        
        # Create result
        result = GameResult(
            winner=winner_str,
            total_turns=turn_count,
            game_length=game_length,
            mr_x_final_position=game.game_state.MrX_position,
            detective_final_positions=game.game_state.detective_positions.copy(),
            moves_history=moves_history,
            mr_x_min_distances=mr_x_min_distances
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
