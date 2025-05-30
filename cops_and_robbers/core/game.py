from abc import ABC, abstractmethod
from typing import List, Set, Tuple, Dict, Optional
from enum import Enum
import networkx as nx


class Player(Enum):
    COPS = "cops"
    ROBBER = "robber"

class GameState:
    """Represents the current state of the game"""
    def __init__(self, cop_positions: List[int], robber_position: int, 
                 turn: Player, turn_count: int = 0):
        self.cop_positions = cop_positions.copy()
        self.robber_position = robber_position
        self.turn = turn
        self.turn_count = turn_count
    
    def copy(self):
        return GameState(self.cop_positions, self.robber_position, 
                        self.turn, self.turn_count)
    
    def __eq__(self, other):
        return (self.cop_positions == other.cop_positions and 
                self.robber_position == other.robber_position and
                self.turn == other.turn)
    
    def __hash__(self):
        return hash((tuple(sorted(self.cop_positions)), self.robber_position, self.turn))

class MovementRule(ABC):
    """Abstract base class for movement rules"""
    
    @abstractmethod
    def get_valid_moves(self, graph: nx.Graph, position: int, 
                       game_state: GameState) -> Set[int]:
        """Return set of valid positions to move to"""
        pass
    
    @abstractmethod
    def can_stay(self) -> bool:
        """Whether player can stay in current position"""
        pass

class StandardMovement(MovementRule):
    """Standard movement: move to adjacent vertices or stay"""
    
    def get_valid_moves(self, graph: nx.Graph, position: int, 
                       game_state: GameState) -> Set[int]:
        moves = set(graph.neighbors(position))
        if self.can_stay():
            moves.add(position)
        return moves
    
    def can_stay(self) -> bool:
        return True

class DistanceKMovement(MovementRule):
    """Movement within distance k"""
    
    def __init__(self, k: int):
        self.k = k
    
    def get_valid_moves(self, graph: nx.Graph, position: int, 
                       game_state: GameState) -> Set[int]:
        moves = set()
        for node in graph.nodes():
            try:
                if nx.shortest_path_length(graph, position, node) <= self.k:
                    moves.add(node)
            except nx.NetworkXNoPath:
                continue
        return moves
    
    def can_stay(self) -> bool:
        return True

class WinCondition(ABC):
    """Abstract base class for win conditions"""
    
    @abstractmethod
    def is_cops_win(self, game_state: GameState) -> bool:
        """Check if cops have won"""
        pass
    
    @abstractmethod
    def is_game_over(self, game_state: GameState) -> bool:
        """Check if game is over"""
        pass

class CaptureWinCondition(WinCondition):
    """Standard win condition: cops win by occupying robber's vertex"""
    
    def is_cops_win(self, game_state: GameState) -> bool:
        return game_state.robber_position in game_state.cop_positions
    
    def is_game_over(self, game_state: GameState) -> bool:
        return self.is_cops_win(game_state)

class DistanceKWinCondition(WinCondition):
    """Win condition: cops win by being within distance k of robber"""
    
    def __init__(self, k: int, graph: nx.Graph):
        self.k = k
        self.graph = graph
    
    def is_cops_win(self, game_state: GameState) -> bool:
        for cop_pos in game_state.cop_positions:
            try:
                if nx.shortest_path_length(self.graph, cop_pos, 
                                         game_state.robber_position) <= self.k:
                    return True
            except nx.NetworkXNoPath:
                continue
        return False
    
    def is_game_over(self, game_state: GameState) -> bool:
        return self.is_cops_win(game_state)

class Obstacle(ABC):
    """Abstract base class for obstacles"""
    
    @abstractmethod
    def blocks_movement(self, from_pos: int, to_pos: int, 
                       game_state: GameState, player: Player) -> bool:
        """Check if obstacle blocks movement"""
        pass

class StaticObstacle(Obstacle):
    """Static obstacles that block specific positions"""
    
    def __init__(self, blocked_positions: Set[int]):
        self.blocked_positions = blocked_positions
    
    def blocks_movement(self, from_pos: int, to_pos: int, 
                       game_state: GameState, player: Player) -> bool:
        return to_pos in self.blocked_positions

class Game:
    """Main game class that orchestrates the game"""
    
    def __init__(self, graph: nx.Graph, num_cops: int, 
                 cop_movement: MovementRule = None,
                 robber_movement: MovementRule = None,
                 win_condition: WinCondition = None,
                 obstacles: List[Obstacle] = None):
        self.graph = graph
        self.num_cops = num_cops
        self.cop_movement = cop_movement or StandardMovement()
        self.robber_movement = robber_movement or StandardMovement()
        self.win_condition = win_condition or CaptureWinCondition()
        self.obstacles = obstacles or []
        self.game_state = None
        self.game_history = []
    
    def initialize_game(self, cop_positions: List[int], robber_position: int):
        """Initialize game with starting positions"""
        if len(cop_positions) != self.num_cops:
            raise ValueError(f"Expected {self.num_cops} cops, got {len(cop_positions)}")
        
        self.game_state = GameState(cop_positions, robber_position, Player.COPS)
        self.game_history = [self.game_state.copy()]
    
    def get_valid_moves(self, player: Player, position: int = None) -> Set[int]:
        """Get valid moves for a player from a position"""
        if self.game_state is None:
            raise ValueError("Game not initialized")
        
        if player == Player.COPS:
            movement_rule = self.cop_movement
            if position is None:
                raise ValueError("Must specify position for cops")
        else:
            movement_rule = self.robber_movement
            if position is None:
                position = self.game_state.robber_position
        
        valid_moves = movement_rule.get_valid_moves(self.graph, position, self.game_state)
        
        # Filter out moves blocked by obstacles
        filtered_moves = set()
        for move in valid_moves:
            blocked = False
            for obstacle in self.obstacles:
                if obstacle.blocks_movement(position, move, self.game_state, player):
                    blocked = True
                    break
            if not blocked:
                filtered_moves.add(move)
        
        return filtered_moves
    
    def make_move(self, new_positions: List[int] = None, new_robber_pos: int = None) -> bool:
        """Make a move and return True if valid"""
        if self.game_state is None:
            raise ValueError("Game not initialized")
        
        if self.is_game_over():
            return False
        
        if self.game_state.turn == Player.COPS:
            if new_positions is None or len(new_positions) != self.num_cops:
                return False
            
            # Validate all cop moves
            for i, new_pos in enumerate(new_positions):
                valid_moves = self.get_valid_moves(Player.COPS, self.game_state.cop_positions[i])
                if new_pos not in valid_moves:
                    return False
            
            self.game_state.cop_positions = new_positions
            self.game_state.turn = Player.ROBBER
        
        else:  # Robber's turn
            if new_robber_pos is None:
                return False
            
            valid_moves = self.get_valid_moves(Player.ROBBER)
            if new_robber_pos not in valid_moves:
                return False
            
            self.game_state.robber_position = new_robber_pos
            self.game_state.turn = Player.COPS
            self.game_state.turn_count += 1
        
        self.game_history.append(self.game_state.copy())
        return True
    
    def is_game_over(self) -> bool:
        """Check if game is over"""
        if self.game_state is None:
            return False
        return self.win_condition.is_game_over(self.game_state)
    
    def get_winner(self) -> Optional[Player]:
        """Get winner if game is over"""
        if not self.is_game_over():
            return None
        
        if self.win_condition.is_cops_win(self.game_state):
            return Player.COPS
        else:
            return Player.ROBBER
    
    def reset(self):
        """Reset game to initial state"""
        if self.game_history:
            self.game_state = self.game_history[0].copy()
            self.game_history = [self.game_state.copy()]
    
    def get_state_representation(self) -> Dict:
        """Get serializable representation of current state"""
        if self.game_state is None:
            return {}
        
        return {
            'cop_positions': self.game_state.cop_positions,
            'robber_position': self.game_state.robber_position,
            'turn': self.game_state.turn.value,
            'turn_count': self.game_state.turn_count,
            'game_over': self.is_game_over(),
            'winner': self.get_winner().value if self.get_winner() else None
        }

        
    def save_game(self, loader, game_id: str = None, 
                metadata: Dict = None) -> str:
        """Save this game using the loader"""
        return loader.save_game(self, game_id, metadata)
    
    def export_game(self, loader, game_id: str, 
                format: str = 'json') -> Optional[str]:
        """Export this game using the loader"""
        return loader.export_game(game_id, format)
