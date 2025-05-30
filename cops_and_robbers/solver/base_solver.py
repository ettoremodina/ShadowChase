from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Tuple
from ..core.game import Game, GameState, Player

class Strategy:
    """Represents a strategy for a player"""
    
    def __init__(self, player: Player):
        self.player = player
        self.moves = {}  # state -> best_move mapping
    
    def get_move(self, game_state: GameState) -> Optional[List[int]]:
        """Get the best move for current state"""
        state_key = self._state_to_key(game_state)
        return self.moves.get(state_key)
    
    def add_move(self, game_state: GameState, move: List[int]):
        """Add a move to the strategy"""
        state_key = self._state_to_key(game_state)
        self.moves[state_key] = move
    
    def _state_to_key(self, game_state: GameState) -> str:
        """Convert game state to hashable key"""
        return f"{sorted(game_state.cop_positions)}_{game_state.robber_position}_{game_state.turn.value}"

class SolverResult:
    """Result of solver computation"""
    
    def __init__(self, cops_can_win: bool, cop_strategy: Strategy = None, 
                 robber_strategy: Strategy = None, game_length: int = None):
        self.cops_can_win = cops_can_win
        self.cop_strategy = cop_strategy
        self.robber_strategy = robber_strategy
        self.game_length = game_length

class BaseSolver(ABC):
    """Abstract base class for game solvers"""
    
    def __init__(self, game: Game):
        self.game = game
    
    @abstractmethod
    def solve(self, initial_cop_positions: List[int], 
              initial_robber_position: int) -> SolverResult:
        """Solve the game from initial positions"""
        pass
    
    @abstractmethod
    def can_cops_win(self, initial_cop_positions: List[int], 
                     initial_robber_position: int) -> bool:
        """Determine if cops can win from initial positions"""
        pass
