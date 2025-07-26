"""
MCTS (Monte Carlo Tree Search) agents for Scotland Yard.

This module provides MCTS-based agents that use tree search with random simulations
to make decisions in the Scotland Yard game.
"""

import os
import json
import time
import copy
import random
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

from ScotlandYard.core.game import ScotlandYardGame, Player, TransportType
from .base_agent import MrXAgent, MultiDetectiveAgent, DetectiveAgent


def load_mcts_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load MCTS configuration from JSON file."""
    if config_path is None:
        # Default to the config file in the project
        project_root = Path(__file__).parent.parent
        config_path = project_root / "training" / "configs" / "mcts_config.json"
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        # Default configuration if file doesn't exist
        raise FileNotFoundError(f"MCTS config file not found at {config_path}")


class MCTSNode:
    """
    A node in the Monte Carlo Tree Search tree.
    
    Each node represents a game state and tracks statistics
    about wins/losses from this state.
    """
    
    def __init__(self, game_state: ScotlandYardGame, move: Optional[Tuple] = None, parent: Optional['MCTSNode'] = None, max_depth: int = 20):
        """
        Initialize MCTS node.
        
        Args:
            game_state: Game state for this node
            move: Move that led to this state
            parent: Parent node in the tree
            max_depth: Maximum simulation depth
        """
        self.game_state = copy.deepcopy(game_state)
        self.move = move
        self.parent = parent
        self.max_depth = max_depth
        self.children: List['MCTSNode'] = []
        self.visits = 0
        self.wins = 0
        self.untried_moves: List[Tuple] = []
        self._initialize_untried_moves()
    
    def _initialize_untried_moves(self):
        """Initialize the list of untried moves from this state."""
        if self.game_state.is_game_over():
            self.untried_moves = []
        else:
            current_player = self.game_state.game_state.turn
            if current_player == Player.MRX:
                # For Mr. X, get moves from current position
                moves = list(self.game_state.get_valid_moves(Player.MRX))
                # Convert to Mr. X format (destination, transport, use_double)
                self.untried_moves = [(dest, transport, False) for dest, transport in moves]
            else:
                # For detectives, we need to get moves for each detective
                detective_moves = []
                for i, pos in enumerate(self.game_state.game_state.detective_positions):
                    moves = self.game_state.get_valid_moves(Player.DETECTIVES, pos)
                    detective_moves.extend(list(moves))
                self.untried_moves = detective_moves
    
    def is_fully_expanded(self) -> bool:
        """Check if all moves from this node have been tried."""
        return len(self.untried_moves) == 0
    
    def is_terminal(self) -> bool:
        """Check if this is a terminal node (game over)."""
        return self.game_state.is_game_over()
    
    def select_child(self, exploration_constant: float = 1.414) -> 'MCTSNode':
        """
        Select best child using UCB1 formula.
        
        Args:
            exploration_constant: Exploration parameter (sqrt(2) by default)
            
        Returns:
            Best child node according to UCB1
        """
        if not self.children:
            return None
        
        best_child = None
        best_value = float('-inf')
        
        for child in self.children:
            if child.visits == 0:
                return child  # Prioritize unvisited children
            
            # UCB1 formula
            exploitation = child.wins / child.visits
            exploration = exploration_constant * ((2 * self.visits / child.visits) ** 0.5)
            ucb_value = exploitation + exploration
            
            if ucb_value > best_value:
                best_value = ucb_value
                best_child = child
        
        return best_child
    
    def expand(self) -> Optional['MCTSNode']:
        """
        Expand the tree by adding a new child node.
        
        Returns:
            New child node, or None if fully expanded
        """
        if not self.untried_moves:
            return None
        
        # Choose a random untried move
        move = self.untried_moves.pop()
        
        # Create new game state and make the move
        new_state = copy.deepcopy(self.game_state)
        
        # Make the move based on current player
        current_player = new_state.game_state.turn
        if current_player == Player.MRX:
            # Mr. X move
            dest, transport, use_double = move
            new_state.make_move(mr_x_moves=[(dest, transport)], use_double_move=use_double)
        else:
            # Detective move - need to construct full move list
            detective_moves = []
            for i in range(new_state.num_detectives):
                if i == 0:  # Simplified: just move first detective
                    detective_moves.append(move)
                else:
                    # Keep other detectives in place
                    pos = new_state.game_state.detective_positions[i]
                    detective_moves.append((pos, TransportType.TAXI))
            new_state.make_move(detective_moves=detective_moves)
        
        # Create and add child node
        child = MCTSNode(new_state, move, self, self.max_depth)
        self.children.append(child)
        
        return child
    
    def simulate(self) -> str:
        """
        Run a random simulation from this node to a terminal state.
        
        Returns:
            Winner of the simulation ('mr_x', 'detectives', or 'timeout')
        """
        simulation_state = copy.deepcopy(self.game_state)
        
        for _ in range(self.max_depth):
            if simulation_state.is_game_over():
                break
            
            # Get valid moves for current player
            current_player = simulation_state.game_state.turn
            if current_player == Player.MRX:
                valid_moves = list(simulation_state.get_valid_moves(Player.MRX))
                if not valid_moves:
                    break
                dest, transport = random.choice(valid_moves)
                simulation_state.make_move(mr_x_moves=[(dest, transport)])
            else:
                # Detective moves - simplified random moves
                detective_moves = []
                for i in range(simulation_state.num_detectives):
                    pos = simulation_state.game_state.detective_positions[i]
                    moves = list(simulation_state.get_valid_moves(Player.DETECTIVES, pos))
                    if moves:
                        detective_moves.append(random.choice(moves))
                    else:
                        detective_moves.append((pos, TransportType.TAXI))
                
                if detective_moves:
                    simulation_state.make_move(detective_moves=detective_moves)
                else:
                    break
        
        # Determine winner
        if simulation_state.is_game_over():
            winner = simulation_state.get_winner()
            return "mr_x" if winner == Player.MRX else "detectives"
        else:
            return "timeout"
    
    def backpropagate(self, result: str, perspective: Player):
        """
        Backpropagate simulation result up the tree.
        
        Args:
            result: Simulation result ('mr_x', 'detectives', or 'timeout')
            perspective: Player perspective for calculating wins
        """
        self.visits += 1
        
        # Update wins based on perspective
        if perspective == Player.MRX and result == "mr_x":
            self.wins += 1
        elif perspective == Player.DETECTIVES and result == "detectives":
            self.wins += 1
        elif result == "timeout":
            self.wins += 0.5  # Neutral result
        
        # Recurse to parent
        if self.parent:
            self.parent.backpropagate(result, perspective)


class MCTSAgent:
    """Base MCTS agent implementation."""
    
    def __init__(self, 
                 simulation_time: Optional[float] = None,
                 max_iterations: Optional[int] = None,
                 exploration_constant: Optional[float] = None,
                 config_path: Optional[str] = None):
        """
        Initialize MCTS agent.
        
        Args:
            simulation_time: Maximum time to spend on MCTS search (overrides config)
            max_iterations: Maximum number of MCTS iterations (overrides config)
            exploration_constant: UCB1 exploration parameter (overrides config)
            config_path: Path to MCTS config file
        """
        # Load configuration from file
        self.config = load_mcts_config(config_path)
        
        # Use provided parameters or fall back to config
        search_params = self.config.get('search_parameters', {})
        self.simulation_time = simulation_time or (search_params.get('num_simulations', 1000) / 1000.0)  # Convert sims to time estimate
        self.max_iterations = max_iterations or search_params.get('num_simulations', 1000)
        self.exploration_constant = exploration_constant or search_params.get('exploration_constant', 1.414)
        
        # Store max depth for simulations
        self.max_simulation_depth = search_params.get('max_depth', 50)
        
        self.statistics = {
            'total_searches': 0,
            'total_iterations': 0,
            'total_search_time': 0.0,
            'avg_iterations_per_search': 0.0,
            'avg_search_time': 0.0,
            'avg_simulations_per_second': 0.0
        }
    
    def mcts_search(self, game_state: ScotlandYardGame, perspective: Player) -> Optional[Tuple]:
        """
        Perform MCTS search to find the best move.
        
        Args:
            game_state: Current game state
            perspective: Player perspective (MrX or detectives)
            
        Returns:
            Best move found by MCTS
        """
        start_time = time.time()
        
        # Create root node with config-based max depth
        root = MCTSNode(game_state, max_depth=self.max_simulation_depth)
        
        iterations = 0
        while (time.time() - start_time < self.simulation_time and 
               iterations < self.max_iterations):
            
            # Selection phase
            node = root
            while not node.is_terminal() and node.is_fully_expanded():
                node = node.select_child(self.exploration_constant)
                if node is None:
                    break
            
            if node is None:
                break
            
            # Expansion phase
            if not node.is_terminal() and not node.is_fully_expanded():
                node = node.expand()
            
            if node is None:
                continue
            
            # Simulation phase
            result = node.simulate()
            
            # Backpropagation phase
            node.backpropagate(result, perspective)
            
            iterations += 1
        
        # Update statistics
        search_time = time.time() - start_time
        self._update_statistics(iterations, search_time)
        
        # Select best move
        if not root.children:
            # Fallback to random move if no children
            if perspective == Player.MRX:
                moves = list(game_state.get_valid_moves(Player.MRX))
                if moves:
                    dest, transport = random.choice(moves)
                    return (dest, transport, False)
            else:
                # For detective, this shouldn't happen in normal use
                return None
        
        # Choose child with highest visit count (most explored)
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.move
    
    def _update_statistics(self, iterations: int, search_time: float):
        """Update search statistics."""
        self.statistics['total_searches'] += 1
        self.statistics['total_iterations'] += iterations
        self.statistics['total_search_time'] += search_time
        
        total_searches = self.statistics['total_searches']
        self.statistics['avg_iterations_per_search'] = self.statistics['total_iterations'] / total_searches
        self.statistics['avg_search_time'] = self.statistics['total_search_time'] / total_searches
        
        if search_time > 0:
            self.statistics['avg_simulations_per_second'] = iterations / search_time
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get search statistics."""
        return self.statistics.copy()
    
    def get_training_parameters(self) -> Dict[str, Any]:
        """Get training parameters from config."""
        return self.config.get('training_parameters', {})
    
    def get_recommended_game_settings(self) -> Dict[str, Any]:
        """Get recommended game settings from config for training/testing."""
        training_params = self.get_training_parameters()
        return {
            'map_size': training_params.get('map_size', 'test'),
            'num_detectives': training_params.get('num_detectives', 2),
            'max_turns_per_game': training_params.get('max_turns_per_game', 24),
            'num_episodes': training_params.get('num_episodes', 1000),
            'evaluation_interval': training_params.get('evaluation_interval', 100)
        }


class MCTSMrXAgent(MrXAgent, MCTSAgent):
    """MCTS agent for Mr. X."""
    
    def __init__(self, 
                 simulation_time: Optional[float] = None,
                 max_iterations: Optional[int] = None,
                 exploration_constant: Optional[float] = None,
                 config_path: Optional[str] = None):
        """Initialize MCTS Mr. X agent."""
        MrXAgent.__init__(self)
        MCTSAgent.__init__(self, simulation_time, max_iterations, exploration_constant, config_path)
    
    def choose_move(self, game: ScotlandYardGame) -> Optional[Tuple[int, TransportType, bool]]:
        """Choose move using MCTS."""
        return self.mcts_search(game, Player.MRX)


class MCTSDetectiveAgent(DetectiveAgent, MCTSAgent):
    """MCTS agent for a single detective."""
    
    def __init__(self, 
                 detective_id: int,
                 simulation_time: Optional[float] = None,
                 max_iterations: Optional[int] = None,
                 exploration_constant: Optional[float] = None,
                 config_path: Optional[str] = None):
        """Initialize MCTS detective agent."""
        DetectiveAgent.__init__(self, detective_id)
        MCTSAgent.__init__(self, simulation_time, max_iterations, exploration_constant, config_path)
    
    def choose_move(self, game: ScotlandYardGame) -> Optional[Tuple[int, TransportType]]:
        """Choose move using MCTS."""
        result = self.mcts_search(game, Player.DETECTIVES)
        if result and len(result) >= 2:
            return (result[0], result[1])  # Return only destination and transport
        return result


class MCTSMultiDetectiveAgent(MultiDetectiveAgent):
    """MCTS agent for multiple detectives."""
    
    def __init__(self, 
                 num_detectives: int,
                 simulation_time: Optional[float] = None,
                 max_iterations: Optional[int] = None,
                 exploration_constant: Optional[float] = None,
                 config_path: Optional[str] = None):
        """Initialize MCTS multi-detective agent."""
        super().__init__(num_detectives)
        
        # Load config to get default parameters
        config = load_mcts_config(config_path)
        search_params = config.get('search_parameters', {})
        
        # Use provided parameters or fall back to config
        self.simulation_time = simulation_time or (search_params.get('num_simulations', 1000) / 1000.0)
        self.max_iterations = max_iterations or search_params.get('num_simulations', 1000)
        self.exploration_constant = exploration_constant or search_params.get('exploration_constant', 1.414)
        
        # Store MCTS parameters for creating temporary agents
        self.mcts_params = {
            'simulation_time': self.simulation_time,
            'max_iterations': self.max_iterations,
            'exploration_constant': self.exploration_constant,
            'config_path': config_path
        }
    
    def choose_all_moves(self, game: ScotlandYardGame) -> List[Tuple[int, TransportType]]:
        """Make MCTS moves for all detectives considering pending moves."""
        detective_moves = []
        pending_moves = []
        
        for i in range(self.num_detectives):
            current_pos = game.game_state.detective_positions[i]
            
            # Get valid moves considering previous detectives' moves
            valid_moves = list(game.get_valid_moves(Player.DETECTIVES, current_pos, pending_moves=pending_moves))
            
            if not valid_moves:
                # Stay in place if no valid moves
                move = (current_pos, None)
            else:
                # Create a temporary MCTS agent for this detective
                temp_agent = MCTSDetectiveAgent(
                    i, 
                    self.mcts_params['simulation_time'],
                    self.mcts_params['max_iterations'],
                    self.mcts_params['exploration_constant'],
                    self.mcts_params['config_path']
                )
                
                # Use MCTS to choose move
                mcts_move = temp_agent.choose_move(game)
                
                # Validate the MCTS move is in valid moves
                if mcts_move and mcts_move in valid_moves:
                    move = mcts_move
                else:
                    # Fallback to random move if MCTS fails
                    move = random.choice(valid_moves)
            
            detective_moves.append(move)
            pending_moves.append(move)
        
        return detective_moves
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics. Note: Individual detective statistics are not tracked in this implementation."""
        return {
            'total_searches': 0,
            'total_iterations': 0,
            'total_search_time': 0.0,
            'avg_iterations_per_search': 0.0,
            'avg_search_time': 0.0,
            'avg_simulations_per_second': 0.0,
            'note': 'Statistics not tracked for multi-detective MCTS agent'
        }
    
    def get_training_parameters(self) -> Dict[str, Any]:
        """Get training parameters from config."""
        config = load_mcts_config()
        return config.get('training_parameters', {})
    
    def get_recommended_game_settings(self) -> Dict[str, Any]:
        """Get recommended game settings from config for training/testing."""
        training_params = self.get_training_parameters()
        return {
            'map_size': training_params.get('map_size', 'test'),
            'num_detectives': training_params.get('num_detectives', 2),
            'max_turns_per_game': training_params.get('max_turns_per_game', 24),
            'num_episodes': training_params.get('num_episodes', 1000),
            'evaluation_interval': training_params.get('evaluation_interval', 100)
        }
