"""
Epsilon-Greedy MCTS (Monte Carlo Tree Search) agents for Shadow Chase with heuristic-guided simulations.

This module provides MCTS-based agents that combine tree search with epsilon-greedy playouts
using heuristics for smarter move selection during simulations. This approach significantly
improves the quality of MCTS search by replacing purely random playouts with guided ones.

Key features:
1. **Epsilon-Greedy Playouts**: Uses heuristics with probability (1-epsilon) and random moves with probability epsilon
2. **Heuristic Integration**: Leverages GameHeuristics class for intelligent move selection
3. **Cache System**: Inherits optimizations from OptimizedMCTSAgent
4. **Smart Mr. X Strategy**: Maximizes distance to closest detective and increases possible locations
5. **Smart Detective Strategy**: Minimizes sum of distances to all possible Mr. X positions

Performance improvements over random MCTS:
- Better move quality through heuristic guidance
- Faster convergence to good strategies
- More realistic game simulations
- Maintains cache benefits from optimized base class
"""

import os
import json
import time
import random
import hashlib
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, NamedTuple
import time
import random
import hashlib

from ShadowChase.core.game import ShadowChaseGame, Player, TransportType
from .base_agent import MrXAgent, MultiDetectiveAgent, DetectiveAgent
from .mcts_agent import load_mcts_config
from .heuristics import GameHeuristics
from ShadowChase.services.cache_system import get_global_cache, CacheNamespace


class GameStateHash(NamedTuple):
    """Hashable representation of game state for caching."""
    MrX_position: Optional[int]
    detective_positions: Tuple[int, ...]
    turn: int
    MrX_tickets: Tuple[int, ...]  # (taxi, bus, underground, black, double)
    detective_tickets: Tuple[Tuple[int, ...], ...]  # Nested tuple for each detective
    reveal_turns: Tuple[int, ...]
    current_player: str


class CachedNodeResult(NamedTuple):
    """Cached result for a node evaluation."""
    wins: int
    visits: int
    best_move: Optional[Tuple]
    timestamp: float


class GameStateCache:
    """Cache for storing MCTS node evaluations."""
    
    def __init__(self, max_size: int = 10000, max_age_seconds: float = 300.0):
        """
        Initialize the cache.
        
        Args:
            max_size: Maximum number of entries to store
            max_age_seconds: Maximum age of entries in seconds
        """
        self.max_size = max_size
        self.max_age_seconds = max_age_seconds
        self._cache: Dict[str, CachedNodeResult] = {}
        self._access_order: List[str] = []
    
    def _create_state_hash(self, game_state: ShadowChaseGame) -> str:
        """Create a hash string for the game state."""
        state = game_state.game_state
        
        # Create hashable representation
        hash_data = GameStateHash(
            MrX_position=state.MrX_position,
            detective_positions=tuple(state.detective_positions),
            turn=state.turn,
            MrX_tickets=tuple(state.MrX_tickets.values()) if hasattr(state, 'MrX_tickets') else (0, 0, 0, 0, 0),
            detective_tickets=tuple(tuple(tickets.values()) if hasattr(tickets, 'values') else (0, 0, 0) 
                                  for tickets in (getattr(state, 'detective_tickets', []) or [])),
            reveal_turns=tuple(getattr(state, 'reveal_turns', [])),
            current_player=state.turn.name if hasattr(state.turn, 'name') else str(state.turn)
        )
        
        # Create hash string
        hash_str = str(hash_data)
        return hashlib.md5(hash_str.encode()).hexdigest()
    
    def get(self, game_state: ShadowChaseGame) -> Optional[CachedNodeResult]:
        """Get cached result for a game state."""
        state_hash = self._create_state_hash(game_state)
        
        if state_hash in self._cache:
            result = self._cache[state_hash]
            
            # Check if entry is too old
            if time.time() - result.timestamp > self.max_age_seconds:
                self._remove_entry(state_hash)
                return None
            
            # Update access order
            if state_hash in self._access_order:
                self._access_order.remove(state_hash)
            self._access_order.append(state_hash)
            
            return result
        
        return None
    
    def put(self, game_state: ShadowChaseGame, wins: int, visits: int, best_move: Optional[Tuple]):
        """Store a result in the cache."""
        state_hash = self._create_state_hash(game_state)
        
        # Create cache entry
        result = CachedNodeResult(
            wins=wins,
            visits=visits,
            best_move=best_move,
            timestamp=time.time()
        )
        
        # Add to cache
        self._cache[state_hash] = result
        
        # Update access order
        if state_hash in self._access_order:
            self._access_order.remove(state_hash)
        self._access_order.append(state_hash)
        
        # Enforce size limit
        while len(self._cache) > self.max_size:
            oldest_key = self._access_order.pop(0)
            if oldest_key in self._cache:
                del self._cache[oldest_key]
    
    def _remove_entry(self, state_hash: str):
        """Remove an entry from the cache."""
        if state_hash in self._cache:
            del self._cache[state_hash]
        if state_hash in self._access_order:
            self._access_order.remove(state_hash)
    
    def clear(self):
        """Clear the cache."""
        self._cache.clear()
        self._access_order.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        current_time = time.time()
        expired_count = sum(1 for result in self._cache.values() 
                          if current_time - result.timestamp > self.max_age_seconds)
        
        return {
            'size': len(self._cache),
            'max_size': self.max_size,
            'expired_entries': expired_count,
            'hit_rate': getattr(self, '_hit_count', 0) / max(getattr(self, '_total_requests', 1), 1)
        }


class EpsilonGreedyMCTSNode:
    """
    An MCTS node with epsilon-greedy heuristic-guided simulations.
    
    Each node represents a game state and tracks statistics
    about wins/losses from this state, using smart playouts
    instead of purely random ones.
    """
    
    def __init__(self, game_state: ShadowChaseGame, move: Optional[Tuple] = None, 
                 parent: Optional['EpsilonGreedyMCTSNode'] = None, max_depth: int = 20, 
                 cache: Optional[GameStateCache] = None):
        """
        Initialize epsilon-greedy MCTS node.
        
        Args:
            game_state: Game state for this node
            move: Move that led to this state
            parent: Parent node in the tree
            max_depth: Maximum simulation depth
            cache: Cache for storing node evaluations
        """
        self.game_state = self._efficient_copy_game(game_state)
        self.move = move
        self.parent = parent
        self.max_depth = max_depth
        self.cache = cache
        self.children: List['EpsilonGreedyMCTSNode'] = []
        self.visits = 0
        self.wins = 0
        self.untried_moves: List[Tuple] = []
        self._initialize_untried_moves()
        
        # Try to load from cache
        self._load_from_cache()
    
    def _efficient_copy_game(self, game: ShadowChaseGame) -> ShadowChaseGame:
        """
        Create an efficient copy of the game state for MCTS simulations.
        This avoids the heavy deepcopy operation by only copying essential data.
        """
        # Create a new game instance with the same graph (shared reference is fine)
        new_game = ShadowChaseGame(game.graph, game.num_detectives)
        
        # Copy essential state data efficiently
        if game.game_state is not None:
            new_game.game_state = game.game_state.copy()  # Uses the efficient copy method
            new_game.game_history = [new_game.game_state.copy()]  # Minimal history for simulation
            new_game.ticket_history = []  # Empty for simulation - saves memory
            new_game.last_visible_position = game.last_visible_position
            new_game.reveal_turns = game.reveal_turns  # Shared reference is fine, immutable
        
        return new_game
    
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
    
    def _load_from_cache(self):
        """Load node statistics from cache if available."""
        if self.cache:
            cached_result = self.cache.get(self.game_state)
            if cached_result:
                self.wins = cached_result.wins
                self.visits = cached_result.visits
                # Note: We don't restore best_move as it affects tree structure
    
    def _save_to_cache(self):
        """Save node statistics to cache."""
        if self.cache and self.visits > 0:
            best_move = None
            if self.children:
                best_child = max(self.children, key=lambda c: c.visits)
                best_move = best_child.move
            
            self.cache.put(self.game_state, self.wins, self.visits, best_move)
    
    def is_fully_expanded(self) -> bool:
        """Check if all moves from this node have been tried."""
        return len(self.untried_moves) == 0
    
    def is_terminal(self) -> bool:
        """Check if this is a terminal node (game over)."""
        return self.game_state.is_game_over()
    
    def select_child(self, exploration_constant: float = 1.414) -> 'EpsilonGreedyMCTSNode':
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
    
    def expand(self) -> Optional['EpsilonGreedyMCTSNode']:
        """
        Expand the tree by adding a new child node.
        
        Returns:
            New child node, or None if fully expanded
        """
        if not self.untried_moves:
            return None
        
        # Choose a random untried move
        move = self.untried_moves.pop()
        
        # Create new game state and make the move using efficient copying
        new_state = self._efficient_copy_game(self.game_state)
        
        # Make the move based on current player
        current_player = new_state.game_state.turn
        if current_player == Player.MRX:
            # Mr. X move
            dest, transport, use_double = move
            new_state.make_move(MrX_moves=[(dest, transport)], use_double_move=use_double)
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
        child = EpsilonGreedyMCTSNode(new_state, move, self, self.max_depth, self.cache)
        self.children.append(child)
        
        return child
    
    def simulate(self, heuristics: GameHeuristics, epsilon: float, possible_MrX_locations: set) -> Tuple[str, Optional[int]]:
        """
        Run an epsilon-greedy simulation from this node to a terminal state.
        Uses heuristics with probability (1-epsilon) and random moves with probability epsilon.
        
        Args:
            heuristics: GameHeuristics instance for intelligent move selection
            epsilon: Probability of making random moves (0.0 = always heuristic, 1.0 = always random)
            possible_MrX_locations: Set of possible Mr. X positions for detective strategy
            
        Returns:
            Tuple of (winner, capturing_detective_id) where winner is 'MrX', 'detectives', or 'timeout'
            and capturing_detective_id is the detective that captured Mr. X (or None)
        """
        simulation_state = self._efficient_copy_game(self.game_state)
        
        # Update heuristics with simulation state
        heuristics.update_game_state(simulation_state)
        
        capturing_detective_id = None
        
        for _ in range(self.max_depth):
            if simulation_state.is_game_over():
                # Check if Mr. X was captured and by which detective
                if simulation_state.get_winner() == Player.DETECTIVES:
                    MrX_pos = simulation_state.game_state.MrX_position
                    for i, det_pos in enumerate(simulation_state.game_state.detective_positions):
                        if det_pos == MrX_pos:
                            capturing_detective_id = i
                            break
                break
            
            # Get valid moves for current player
            current_player = simulation_state.game_state.turn
            
            if current_player == Player.MRX:
                valid_moves = list(simulation_state.get_valid_moves(Player.MRX))
                if not valid_moves:
                    break
                
                # Choose move: epsilon-greedy strategy
                if random.random() < epsilon:
                    # Random move
                    dest, transport = random.choice(valid_moves)
                else:
                    # Heuristic move: maximize distance to closest detective
                    dest, transport = self._choose_MrX_heuristic_move(
                        simulation_state, valid_moves, heuristics, possible_MrX_locations
                    )
                
                simulation_state.make_move(MrX_moves=[(dest, transport)])
                
            else:
                # Detective moves - epsilon-greedy for each detective
                detective_moves = []
                for i in range(simulation_state.num_detectives):
                    pos = simulation_state.game_state.detective_positions[i]
                    moves = list(simulation_state.get_valid_moves(Player.DETECTIVES, pos))
                    
                    if moves:
                        if random.random() < epsilon:
                            # Random move
                            detective_moves.append(random.choice(moves))
                        else:
                            # Heuristic move: minimize sum of distances to possible Mr. X positions
                            best_move = self._choose_detective_heuristic_move(
                                pos, moves, heuristics, possible_MrX_locations
                            )
                            detective_moves.append(best_move)
                    else:
                        detective_moves.append((pos, TransportType.TAXI))
                
                if detective_moves:
                    simulation_state.make_move(detective_moves=detective_moves)
                else:
                    break
            
            # Update heuristics and possible positions after each move
            heuristics.update_game_state(simulation_state)
            if current_player == Player.MRX:
                possible_MrX_locations = heuristics.get_possible_MrX_positions()
        
        # Determine winner
        if simulation_state.is_game_over():
            winner = simulation_state.get_winner()
            return ("MrX" if winner == Player.MRX else "detectives", capturing_detective_id)
        else:
            return ("timeout", None)
    
    def _choose_MrX_heuristic_move(self, game_state: ShadowChaseGame, valid_moves: List[Tuple], 
                                   heuristics: GameHeuristics, possible_MrX_locations: set) -> Tuple[int, TransportType]:
        """
        Choose Mr. X's best move using heuristics:
        1. Maximize distance to closest detective
        2. If multiple moves have same max distance, choose one that increases possible locations
        3. If still tied, choose randomly
        
        Args:
            game_state: Current game state
            valid_moves: List of valid (destination, transport) moves
            heuristics: GameHeuristics instance
            possible_MrX_locations: Current set of possible Mr. X positions
            
        Returns:
            Best (destination, transport) move
        """
        if not valid_moves:
            return None
        
        if len(valid_moves) == 1:
            return valid_moves[0]
        
        best_moves = []
        max_min_distance = -1
        
        for dest, transport in valid_moves:
            # Calculate minimum distance to any detective from this destination
            min_distance_to_detective = float('inf')
            for det_pos in game_state.game_state.detective_positions:
                distance = heuristics.calculate_shortest_distance(dest, det_pos)
                if distance >= 0 and distance < min_distance_to_detective:
                    min_distance_to_detective = distance
            
            # If no valid distance found, skip this move
            if min_distance_to_detective == float('inf'):
                continue
            
            if min_distance_to_detective > max_min_distance:
                max_min_distance = min_distance_to_detective
                best_moves = [(dest, transport)]
            elif min_distance_to_detective == max_min_distance:
                best_moves.append((dest, transport))
        
        if not best_moves:
            return random.choice(valid_moves)
        
        if len(best_moves) == 1:
            return best_moves[0]
        
        # Tie-breaker: choose move that maximizes possible locations
        # This is a simplified heuristic - we could implement more sophisticated logic
        return random.choice(best_moves)
    
    def _choose_detective_heuristic_move(self, detective_pos: int, valid_moves: List[Tuple], 
                                       heuristics: GameHeuristics, possible_MrX_locations: set) -> Tuple[int, TransportType]:
        """
        Choose detective's best move using heuristics:
        Minimize sum of distances to all possible Mr. X positions
        
        Args:
            detective_pos: Current detective position
            valid_moves: List of valid (destination, transport) moves
            heuristics: GameHeuristics instance
            possible_MrX_locations: Set of possible Mr. X positions
            
        Returns:
            Best (destination, transport) move
        """
        if not valid_moves:
            return None
        
        if len(valid_moves) == 1:
            return valid_moves[0]
        
        if not possible_MrX_locations:
            # If no known possible positions, move randomly
            return random.choice(valid_moves)
        
        best_move = None
        min_total_distance = float('inf')
        
        for dest, transport in valid_moves:
            total_distance = 0
            valid_distances = 0
            
            for possible_pos in possible_MrX_locations:
                distance = heuristics.calculate_shortest_distance(dest, possible_pos)
                if distance >= 0:
                    total_distance += distance
                    valid_distances += 1
            
            # Average distance to avoid bias based on number of possible positions
            if valid_distances > 0:
                avg_distance = total_distance / valid_distances
                if avg_distance < min_total_distance:
                    min_total_distance = avg_distance
                    best_move = (dest, transport)
        
        return best_move if best_move else random.choice(valid_moves)
    
    def backpropagate(self, result: Tuple[str, Optional[int]], perspective: Player, 
                     root_detective_id: Optional[int] = None, 
                     coalition_reduction_factor_r: float = 0.0):
        """
        Backpropagate simulation result up the tree with coalition reduction.
        
        Args:
            result: Tuple of (winner, capturing_detective_id) from simulation
            perspective: Player perspective for calculating wins
            root_detective_id: ID of detective who initiated the MCTS search (for coalition reduction)
            coalition_reduction_factor_r: Reduction factor when other detectives capture Mr. X
        """
        self.visits += 1
        
        # Unpack result tuple
        winner, capturing_detective_id = result
        
        # Update wins based on perspective and coalition reduction
        if perspective == Player.MRX and winner == "MrX":
            self.wins += 1
        elif perspective == Player.DETECTIVES and winner == "detectives":
            # Apply coalition reduction if applicable
            if root_detective_id is not None and capturing_detective_id is not None:
                if capturing_detective_id == root_detective_id:
                    # This detective made the capture - full reward
                    self.wins += 1.0
                else:
                    # Another detective made the capture - reduced reward
                    self.wins += (1.0 - coalition_reduction_factor_r)
            else:
                # No coalition reduction (backwards compatibility)
                self.wins += 1
        elif winner == "timeout":
            self.wins += 0.5  # Neutral result
        
        # Save to cache after updating statistics
        self._save_to_cache()
        
        # Recurse to parent
        if self.parent:
            self.parent.backpropagate(result, perspective, root_detective_id, coalition_reduction_factor_r)


class EpsilonGreedyMCTSAgent:
    """Base epsilon-greedy MCTS agent implementation with heuristic-guided simulations."""
    
    def __init__(self, 
                 simulation_time: Optional[float] = None,
                 max_iterations: Optional[int] = None,
                 exploration_constant: Optional[float] = None,
                 epsilon: float = 0.2,
                 config_path: Optional[str] = None,
                 cache_size: Optional[int] = None,
                 cache_max_age: Optional[float] = None):
        """
        Initialize epsilon-greedy MCTS agent.
        
        Args:
            simulation_time: Maximum time to spend on MCTS search (overrides config)
            max_iterations: Maximum number of MCTS iterations (overrides config)
            exploration_constant: UCB1 exploration parameter (overrides config)
            epsilon: Probability of random moves in simulation (0.0 = always heuristic, 1.0 = always random)
            config_path: Path to MCTS config file
            cache_size: Maximum cache size (overrides config)
            cache_max_age: Maximum age of cache entries in seconds (overrides config)
        """
        # Load configuration from file
        self.config = load_mcts_config(config_path)
        
        # Use provided parameters or fall back to config
        search_params = self.config.get('search_parameters', {})
        cache_params = self.config.get('cache_parameters', {})
        epsilon_params = self.config.get('epsilon_greedy_parameters', {})
        
        self.simulation_time = simulation_time or (search_params.get('num_simulations', 1000) / 1000.0)
        self.max_iterations = max_iterations or search_params.get('num_simulations', 1000)
        self.exploration_constant = exploration_constant or search_params.get('exploration_constant', 1.414)
        
        # Load epsilon greedy parameters from config
        self.epsilon = epsilon if epsilon != 0.2 else epsilon_params.get('epsilon', 0.2)
        self.epsilon_decay = epsilon_params.get('epsilon_decay', 0.995)
        self.min_epsilon = epsilon_params.get('min_epsilon', 0.05)
        self.epsilon_decay_steps = epsilon_params.get('epsilon_decay_steps', 1000)
        self.heuristic_weight = epsilon_params.get('heuristic_weight', 0.8)
        self.enable_adaptive_epsilon = epsilon_params.get('enable_adaptive_epsilon', False)
        
        # Track epsilon decay
        self.current_epsilon = self.epsilon
        self.epsilon_step_count = 0
        
        # Store max depth for simulations
        self.max_simulation_depth = search_params.get('max_depth', 50)
        
        # Initialize cache
        cache_size = cache_size or cache_params.get('max_size', 10000)
        cache_max_age = cache_max_age or cache_params.get('max_age_seconds', 300.0)
        self.cache = GameStateCache(cache_size, cache_max_age)
        self.persistent_cache = get_global_cache()  # Shared persistent cache
        
        # Initialize heuristics (will be updated for each game)
        self.heuristics = None
        
        self.statistics = {
            'total_searches': 0,
            'total_iterations': 0,
            'total_search_time': 0.0,
            'avg_iterations_per_search': 0.0,
            'avg_search_time': 0.0,
            'avg_simulations_per_second': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'cache_hit_rate': 0.0,
            'heuristic_moves': 0,
            'random_moves': 0,
            'heuristic_ratio': 0.0
        }
    
    def update_epsilon(self):
        """Update epsilon using decay schedule if enabled."""
        if self.enable_adaptive_epsilon and self.epsilon_step_count < self.epsilon_decay_steps:
            self.current_epsilon = max(
                self.min_epsilon,
                self.epsilon * (self.epsilon_decay ** self.epsilon_step_count)
            )
            self.epsilon_step_count += 1
    
    def get_current_epsilon(self) -> float:
        """Get the current epsilon value."""
        return self.current_epsilon
    
    def reset_epsilon(self):
        """Reset epsilon to initial value."""
        self.current_epsilon = self.epsilon
        self.epsilon_step_count = 0
    
    def mcts_search(self, game_state: ShadowChaseGame, perspective: Player, 
                   root_detective_id: Optional[int] = None, 
                   coalition_reduction_factor_r: float = 0.0) -> Optional[Tuple]:
        """
        Perform epsilon-greedy MCTS search to find the best move.
        
        Args:
            game_state: Current game state
            perspective: Player perspective (MrX or detectives)
            root_detective_id: ID of detective who initiated search (for coalition reduction)
            coalition_reduction_factor_r: Reduction factor when other detectives capture Mr. X
            
        Returns:
            Best move found by MCTS
        """
        start_time = time.time()
        
        # Update epsilon if decay is enabled
        self.update_epsilon()
        current_epsilon = self.get_current_epsilon()
        
        # Initialize or update heuristics for this game state
        if self.heuristics is None:
            self.heuristics = GameHeuristics(game_state)
        else:
            self.heuristics.update_game_state(game_state)
        
        # Get possible Mr. X locations for detective strategy
        possible_MrX_locations = self.heuristics.get_possible_MrX_positions()
        
        # Create persistent cache key for this search
        persistent_cache_key = self.persistent_cache.create_agent_cache_key(
            "epsilon_greedy_mcts",
            str(id(self)),  # Use object id as agent identifier
            perspective=perspective.value,
            detective_positions=tuple(game_state.game_state.detective_positions),
            mrx_position=game_state.game_state.MrX_position,
            turn=game_state.game_state.turn.value,
            turn_count=game_state.game_state.turn_count,
            mrx_turn_count=game_state.game_state.MrX_turn_count,
            detective_tickets=[(k, tuple(v.items())) for k, v in game_state.game_state.detective_tickets.items()],
            MrX_tickets=tuple(game_state.game_state.MrX_tickets.items()),
            epsilon=current_epsilon
        )
        
        # Check persistent cache first for quick lookup
        cached_result = self.persistent_cache.get(persistent_cache_key, CacheNamespace.MCTS_NODES)
        if cached_result and cached_result.get('best_move'):
            self.statistics['cache_hits'] += 1
            self._update_cache_statistics()
            # Convert back from serializable format
            move_data = cached_result['best_move']
            if perspective == Player.MRX:
                return (move_data[0], TransportType(move_data[1]), move_data[2])
            else:
                return (move_data[0], TransportType(move_data[1]))
        
        # Check in-memory cache
        cached_result = self.cache.get(game_state)
        if cached_result and cached_result.best_move:
            self.statistics['cache_hits'] += 1
            self._update_cache_statistics()
            
            # Store in persistent cache for next time
            if perspective == Player.MRX:
                serializable_move = [cached_result.best_move[0], cached_result.best_move[1].value, cached_result.best_move[2]]
            else:
                serializable_move = [cached_result.best_move[0], cached_result.best_move[1].value]
            
            persistent_result = {
                'best_move': serializable_move,
                'wins': cached_result.wins,
                'visits': cached_result.visits,
                'search_time': 0.0  # Cache hit, no search time
            }
            self.persistent_cache.put(persistent_cache_key, persistent_result, CacheNamespace.MCTS_NODES, ttl_seconds=1800)
            return cached_result.best_move
        
        self.statistics['cache_misses'] += 1
        
        # Create root node with config-based max depth and cache
        root = EpsilonGreedyMCTSNode(game_state, max_depth=self.max_simulation_depth, cache=self.cache)
        
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
            
            # Update heuristics for current node's game state
            self.heuristics.update_game_state(node.game_state)
            possible_MrX_locations = self.heuristics.get_possible_MrX_positions()
            
            # Simulation phase with epsilon-greedy heuristics
            result = node.simulate(self.heuristics, current_epsilon, possible_MrX_locations)
            
            # Backpropagation phase with coalition reduction
            node.backpropagate(result, perspective, root_detective_id, coalition_reduction_factor_r)
            
            iterations += 1
        
        # Update statistics
        search_time = time.time() - start_time
        self._update_statistics(iterations, search_time)
        self._update_cache_statistics()
        
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
        best_move = best_child.move
        
        # Store result in persistent cache
        if best_move:
            if perspective == Player.MRX:
                serializable_move = [best_move[0], best_move[1].value, best_move[2]]
            else:
                serializable_move = [best_move[0], best_move[1].value]
            
            persistent_result = {
                'best_move': serializable_move,
                'wins': best_child.wins,
                'visits': best_child.visits,
                'search_time': search_time
            }
            self.persistent_cache.put(persistent_cache_key, persistent_result, CacheNamespace.MCTS_NODES, ttl_seconds=1800)
        
        return best_move
    
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
    
    def _update_cache_statistics(self):
        """Update cache-related statistics."""
        total_requests = self.statistics['cache_hits'] + self.statistics['cache_misses']
        if total_requests > 0:
            self.statistics['cache_hit_rate'] = self.statistics['cache_hits'] / total_requests
    
    def _update_heuristic_statistics(self, used_heuristic: bool):
        """Update heuristic vs random move statistics."""
        if used_heuristic:
            self.statistics['heuristic_moves'] += 1
        else:
            self.statistics['random_moves'] += 1
        
        total_moves = self.statistics['heuristic_moves'] + self.statistics['random_moves']
        if total_moves > 0:
            self.statistics['heuristic_ratio'] = self.statistics['heuristic_moves'] / total_moves
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get search statistics including cache and heuristic performance."""
        stats = self.statistics.copy()
        stats['memory_cache_stats'] = self.cache.get_stats()
        stats['persistent_cache_stats'] = self.persistent_cache.get_global_stats()
        stats['initial_epsilon'] = self.epsilon
        stats['current_epsilon'] = self.current_epsilon
        stats['epsilon_decay'] = self.epsilon_decay
        stats['min_epsilon'] = self.min_epsilon
        stats['epsilon_step_count'] = self.epsilon_step_count
        stats['enable_adaptive_epsilon'] = self.enable_adaptive_epsilon
        stats['heuristic_weight'] = self.heuristic_weight
        return stats
    
    def clear_cache(self):
        """Clear both in-memory and persistent agent caches."""
        self.cache.clear()
        self.persistent_cache.clear_namespace(CacheNamespace.MCTS_NODES)
        self.persistent_cache.clear_namespace(CacheNamespace.AGENT_DECISIONS)
    
    def clear_memory_cache_only(self):
        """Clear only the in-memory cache, keeping persistent cache."""
        self.cache.clear()
    
    def save_persistent_cache(self):
        """Force save persistent cache to disk."""
        self.persistent_cache._save_to_disk()
    
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


class EpsilonGreedyMCTSMrXAgent(MrXAgent, EpsilonGreedyMCTSAgent):
    """Epsilon-greedy MCTS agent for Mr. X with heuristic-guided simulations."""
    
    def __init__(self, 
                 simulation_time: Optional[float] = None,
                 max_iterations: Optional[int] = None,
                 exploration_constant: Optional[float] = None,
                 epsilon: float = 0.2,
                 config_path: Optional[str] = None,
                 cache_size: Optional[int] = None,
                 cache_max_age: Optional[float] = None):
        """Initialize epsilon-greedy MCTS Mr. X agent."""
        MrXAgent.__init__(self)
        EpsilonGreedyMCTSAgent.__init__(self, simulation_time, max_iterations, exploration_constant, 
                                       epsilon, config_path, cache_size, cache_max_age)
    
    def choose_move(self, game: ShadowChaseGame) -> Optional[Tuple[int, TransportType, bool]]:
        """Choose move using epsilon-greedy MCTS with heuristic-guided simulations."""
        return self.mcts_search(game, Player.MRX)


class EpsilonGreedyMCTSDetectiveAgent(DetectiveAgent, EpsilonGreedyMCTSAgent):
    """Epsilon-greedy MCTS agent for a single detective with heuristic-guided simulations."""
    
    def __init__(self, 
                 detective_id: int,
                 simulation_time: Optional[float] = None,
                 max_iterations: Optional[int] = None,
                 exploration_constant: Optional[float] = None,
                 epsilon: float = 0.2,
                 config_path: Optional[str] = None,
                 cache_size: Optional[int] = None,
                 cache_max_age: Optional[float] = None):
        """Initialize epsilon-greedy MCTS detective agent."""
        DetectiveAgent.__init__(self, detective_id)
        EpsilonGreedyMCTSAgent.__init__(self, simulation_time, max_iterations, exploration_constant, 
                                       epsilon, config_path, cache_size, cache_max_age)
    
    def choose_move(self, game: ShadowChaseGame, 
                   root_detective_id: Optional[int] = None, 
                   coalition_reduction_factor_r: float = 0.0) -> Optional[Tuple[int, TransportType]]:
        """Choose move using epsilon-greedy MCTS with heuristic-guided simulations and coalition reduction."""
        result = self.mcts_search(game, Player.DETECTIVES, root_detective_id, coalition_reduction_factor_r)
        if result and len(result) >= 2:
            return (result[0], result[1])  # Return only destination and transport
        return result


class EpsilonGreedyMCTSMultiDetectiveAgent(MultiDetectiveAgent):
    """Epsilon-greedy MCTS agent for multiple detectives with heuristic-guided simulations."""
    
    def __init__(self, 
                 num_detectives: int,
                 simulation_time: Optional[float] = None,
                 max_iterations: Optional[int] = None,
                 exploration_constant: Optional[float] = None,
                 epsilon: float = 0.2,
                 coalition_reduction_factor_r: float = 0.25,
                 config_path: Optional[str] = None,
                 cache_size: Optional[int] = None,
                 cache_max_age: Optional[float] = None):
        """Initialize epsilon-greedy MCTS multi-detective agent with coalition reduction."""
        super().__init__(num_detectives)
        
        # Load config to get default parameters
        config = load_mcts_config(config_path)
        search_params = config.get('search_parameters', {})
        cache_params = config.get('cache_parameters', {})
        epsilon_params = config.get('epsilon_greedy_parameters', {})
        
        # Use provided parameters or fall back to config
        self.simulation_time = simulation_time or (search_params.get('num_simulations', 1000) / 1000.0)
        self.max_iterations = max_iterations or search_params.get('num_simulations', 1000)
        self.exploration_constant = exploration_constant or search_params.get('exploration_constant', 1.414)
        
        # Load epsilon greedy parameters from config
        self.epsilon = epsilon if epsilon != 0.2 else epsilon_params.get('epsilon', 0.2)
        self.coalition_reduction_factor_r = coalition_reduction_factor_r if coalition_reduction_factor_r != 0.25 else epsilon_params.get('coalition_reduction_factor_r', 0.25)
        
        # Cache parameters
        cache_size = cache_size or cache_params.get('max_size', 10000)
        cache_max_age = cache_max_age or cache_params.get('max_age_seconds', 300.0)
        
        # Store epsilon-greedy MCTS parameters for creating temporary agents
        self.mcts_params = {
            'simulation_time': self.simulation_time,
            'max_iterations': self.max_iterations,
            'exploration_constant': self.exploration_constant,
            'epsilon': self.epsilon,
            'config_path': config_path,
            'cache_size': cache_size,
            'cache_max_age': cache_max_age
        }
    
    def choose_all_moves(self, game: ShadowChaseGame) -> List[Tuple[int, TransportType]]:
        """Make epsilon-greedy MCTS moves for all detectives with coalition reduction."""
        detective_moves = []
        pending_moves = []
        
        for i in range(self.num_detectives):
            current_pos = game.game_state.detective_positions[i]
            
            # Get valid moves considering previous detectives' moves
            valid_moves = list(game.get_valid_moves(Player.DETECTIVES, current_pos, pending_moves=pending_moves))
            
            if not valid_moves:
                # No valid moves, stay in place
                detective_moves.append((current_pos, None))
                pending_moves.append((current_pos, None))
            else:
                # Create temporary agent for this detective with coalition reduction
                temp_agent = EpsilonGreedyMCTSDetectiveAgent(i, **self.mcts_params)
                # Pass the detective ID as root_detective_id and coalition reduction factor
                move = temp_agent.choose_move(game, root_detective_id=i, 
                                            coalition_reduction_factor_r=self.coalition_reduction_factor_r)
                
                if move and move in valid_moves:
                    detective_moves.append(move)
                    pending_moves.append(move)
                else:
                    # Fallback to first valid move
                    move = valid_moves[0]
                    detective_moves.append(move)
                    pending_moves.append(move)
        
        return detective_moves
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics including coalition reduction information."""
        return {
            'epsilon': self.epsilon,
            'coalition_reduction_factor_r': self.coalition_reduction_factor_r,
            'num_detectives': self.num_detectives,
            'mcts_params': self.mcts_params,
            'epsilon_greedy_config': {
                'epsilon_decay': getattr(self, 'epsilon_decay', 'N/A'),
                'min_epsilon': getattr(self, 'min_epsilon', 'N/A'),
                'heuristic_weight': getattr(self, 'heuristic_weight', 'N/A'),
                'enable_adaptive_epsilon': getattr(self, 'enable_adaptive_epsilon', 'N/A')
            }
        }
    
    def get_training_parameters(self) -> Dict[str, Any]:
        """Get training parameters from config."""
        config = load_mcts_config(self.mcts_params.get('config_path'))
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
