from collections import deque
from typing import List, Set, Dict, Tuple, Optional
from .base_solver import BaseSolver, SolverResult, Strategy
from ..core.game import Game, GameState, Player

class BFSSolver(BaseSolver):
    """BFS solver for determining reachability"""
    
    def __init__(self, game: Game, max_turns: int = 100):
        super().__init__(game)
        self.max_turns = max_turns
    
    def solve(self, initial_cop_positions: List[int], 
              initial_robber_position: int) -> SolverResult:
        """Solve using BFS to find if cops can win"""
        
        initial_state = GameState(initial_cop_positions, initial_robber_position, Player.COPS)
        self.game.initialize_game(initial_cop_positions, initial_robber_position)
        
        # BFS to find winning states for cops
        queue = deque([(initial_state, 0)])
        visited = set()
        winning_states = set()
        
        while queue:
            state, depth = queue.popleft()
            
            if depth > self.max_turns:
                continue
                
            state_key = self._state_to_key(state)
            if state_key in visited:
                continue
            visited.add(state_key)
            
            # Check if cops win
            if self.game.win_condition.is_cops_win(state):
                winning_states.add(state_key)
                continue
            
            # Generate next states
            if state.turn == Player.COPS:
                for move_combination in self._get_all_cop_moves(state):
                    new_state = state.copy()
                    new_state.cop_positions = list(move_combination)
                    new_state.turn = Player.ROBBER
                    queue.append((new_state, depth))
            else:
                valid_moves = self.game.get_valid_moves(Player.ROBBER, state.robber_position)
                for new_pos in valid_moves:
                    new_state = state.copy()
                    new_state.robber_position = new_pos
                    new_state.turn = Player.COPS
                    new_state.turn_count += 1
                    queue.append((new_state, depth + 1))
        
        cops_can_win = len(winning_states) > 0
        
        return SolverResult(
            cops_can_win=cops_can_win,
            cop_strategy=None,  # BFS doesn't generate strategies
            robber_strategy=None,
            game_length=None
        )
    
    def can_cops_win(self, initial_cop_positions: List[int], 
                     initial_robber_position: int) -> bool:
        """Quick BFS check if cops can win"""
        result = self.solve(initial_cop_positions, initial_robber_position)
        return result.cops_can_win
    
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


# Main application entry point

#
