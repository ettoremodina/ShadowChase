from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
import networkx as nx
from .base_solver import BaseSolver, SolverResult, Strategy
from ..core.game import Game, GameState, Player

class MinimaxSolver(BaseSolver):
    """Minimax solver with memoization for Cops and Robbers"""
    
    def __init__(self, game: Game, max_depth: int = 50):
        super().__init__(game)
        self.max_depth = max_depth
        self.memo = {}  # Memoization cache
        self.visited_states = set()
    
    def solve(self, initial_cop_positions: List[int], 
              initial_robber_position: int) -> SolverResult:
        """Solve game using minimax algorithm"""
        self.memo.clear()
        self.visited_states.clear()
        
        initial_state = GameState(initial_cop_positions, initial_robber_position, Player.COPS)
        self.game.initialize_game(initial_cop_positions, initial_robber_position)
        
        cops_can_win, game_length = self._minimax(initial_state, 0, True)
        
        # Extract strategies from memoization table
        cop_strategy = Strategy(Player.COPS)
        robber_strategy = Strategy(Player.ROBBER)
        
        for state_key, (can_win, depth, best_move) in self.memo.items():
            state = self._key_to_state(state_key)
            if state.turn == Player.COPS and best_move:
                cop_strategy.add_move(state, best_move)
            elif state.turn == Player.ROBBER and best_move:
                robber_strategy.add_move(state, [best_move])
        
        return SolverResult(
            cops_can_win=cops_can_win,
            cop_strategy=cop_strategy if cops_can_win else None,
            robber_strategy=robber_strategy if not cops_can_win else None,
            game_length=game_length if game_length < float('inf') else None
        )
    
    def can_cops_win(self, initial_cop_positions: List[int], 
                     initial_robber_position: int) -> bool:
        """Quick check if cops can win"""
        result = self.solve(initial_cop_positions, initial_robber_position)
        return result.cops_can_win
    
    def _minimax(self, state: GameState, depth: int, maximizing: bool) -> Tuple[bool, int]:
        """Minimax algorithm with memoization"""
        state_key = self._state_to_key(state)
        
        # Check memoization
        if state_key in self.memo:
            return self.memo[state_key][:2]
        
        # Check for cycles (draw)
        if state_key in self.visited_states:
            return False, float('inf')  # Robber wins in cycles
        
        # Check terminal conditions
        if self.game.win_condition.is_game_over(state):
            result = self.game.win_condition.is_cops_win(state), 0
            self.memo[state_key] = (*result, None)
            return result
        
        # Depth limit reached
        if depth >= self.max_depth:
            return False, float('inf')  # Assume robber wins if too deep
        
        self.visited_states.add(state_key)
        
        if state.turn == Player.COPS:  # Maximizing player (cops)
            best_value = False
            best_depth = float('inf')
            best_move = None
            
            # Try all possible cop moves
            for move_combination in self._get_all_cop_moves(state):
                new_state = state.copy()
                new_state.cop_positions = list(move_combination)
                new_state.turn = Player.ROBBER
                
                can_win, win_depth = self._minimax(new_state, depth + 1, False)
                
                if can_win and (not best_value or win_depth < best_depth):
                    best_value = True
                    best_depth = win_depth + 1
                    best_move = list(move_combination)
                
                # Alpha-beta pruning: if we found a winning move, no need to search further
                if best_value:
                    break
        
        else:  # Minimizing player (robber)
            best_value = True
            best_depth = 0
            best_move = None
            
            valid_moves = self.game.get_valid_moves(Player.ROBBER, state.robber_position)
            
            for new_pos in valid_moves:
                new_state = state.copy()
                new_state.robber_position = new_pos
                new_state.turn = Player.COPS
                new_state.turn_count += 1
                
                can_win, win_depth = self._minimax(new_state, depth + 1, True)
                
                if not can_win:
                    best_value = False
                    best_depth = float('inf')
                    best_move = new_pos
                    break  # Robber found an escape
                elif win_depth > best_depth:
                    best_depth = win_depth
                    best_move = new_pos
        
        self.visited_states.remove(state_key)
        result = (best_value, best_depth)
        self.memo[state_key] = (*result, best_move)
        return result
    
    def _get_all_cop_moves(self, state: GameState) -> List[Tuple[int, ...]]:
        """Generate all possible combinations of cop moves"""
        from itertools import product
        
        cop_moves = []
        for i, cop_pos in enumerate(state.cop_positions):
            valid_moves = self.game.get_valid_moves(Player.COPS, cop_pos)
            cop_moves.append(list(valid_moves))
        
        return list(product(*cop_moves))
    
    def _state_to_key(self, state: GameState) -> str:
        """Convert state to hashable key"""
        return f"{tuple(sorted(state.cop_positions))}_{state.robber_position}_{state.turn.value}"
    
    def _key_to_state(self, key: str) -> GameState:
        """Convert key back to state (for strategy extraction)"""
        parts = key.split('_')
        cop_positions = eval(parts[0])  # Convert string tuple back to list
        robber_position = int(parts[1])
        turn = Player.COPS if parts[2] == 'cops' else Player.ROBBER
        return GameState(list(cop_positions), robber_position, turn)
