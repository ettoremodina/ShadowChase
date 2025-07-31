from abc import ABC, abstractmethod
from typing import List, Set, Tuple, Dict, Optional
from enum import Enum
import networkx as nx


class Player(Enum):
    DETECTIVES = "detectives"
    MRX = "mr_x"  # Added for Scotland Yard

class TransportType(Enum):
    TAXI = 1
    BUS = 2
    UNDERGROUND = 3
    BLACK = 4  # Special for Mr. X
    FERRY = 5  # Special routes

class TicketType(Enum):
    TAXI = "taxi"
    BUS = "bus" 
    UNDERGROUND = "underground"
    BLACK = "black"
    DOUBLE_MOVE = "double_move"
    
    def __lt__(self, other):
        if not isinstance(other, TicketType):
            return NotImplemented
        return self.value < other.value

class GameState:
    """Represents the current state of the game"""
    def __init__(self, detective_positions: List[int], MrX_position: int, 
                 turn: Player, turn_count: int = 0,
                 MrX_turn_count: int = 1,
                 detective_tickets: Dict[int, Dict[TicketType, int]] = None,
                 mr_x_tickets: Dict[TicketType, int] = None,
                 mr_x_visible: bool = True,
                 mr_x_moves_log: List[Tuple[int, TransportType]] = None):
        self.detective_positions = detective_positions.copy()
        self.MrX_position = MrX_position
        self.turn = turn
        self.MrX_turn_count = MrX_turn_count
        self.turn_count = turn_count
        # Scotland Yard specific
        self.detective_tickets = detective_tickets or {}
        self.mr_x_tickets = mr_x_tickets or {}
        self.mr_x_visible = mr_x_visible
        self.mr_x_moves_log = mr_x_moves_log or []
        self.double_move_active = False

    
    def copy(self):
        new_state = GameState(
            self.detective_positions, self.MrX_position, 
            self.turn, self.turn_count,
            self.MrX_turn_count,
            {k: v.copy() for k, v in self.detective_tickets.items()},
            self.mr_x_tickets.copy(),
            self.mr_x_visible,
            self.mr_x_moves_log.copy(),
        )
        new_state.double_move_active = self.double_move_active
        return new_state
    
    def __eq__(self, other):
        return (self.detective_positions == other.detective_positions and 
                self.MrX_position == other.MrX_position and
                self.turn == other.turn)
    
    def __hash__(self):
        return hash((tuple(sorted(self.detective_positions)), self.MrX_position, self.turn))

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
        return False

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
    def is_detectives_win(self, game_state: GameState) -> bool:
        """Check if detectives have won"""
        pass
    
    @abstractmethod
    def is_game_over(self, game_state: GameState) -> bool:
        """Check if game is over"""
        pass

class CaptureWinCondition(WinCondition):
    """Standard win condition: detectives win by occupying MrX's vertex"""
    
    def is_detectives_win(self, game_state: GameState) -> bool:
        # Don't consider it a win if no moves have been made yet
        if game_state.turn_count == 0 and game_state.turn == Player.DETECTIVES:
            return False
        return game_state.MrX_position in game_state.detective_positions
    
    def is_game_over(self, game_state: GameState) -> bool:
        return self.is_detectives_win(game_state)

class DistanceKWinCondition(WinCondition):
    """Win condition: detectives win by being within distance k of MrX"""
    
    def __init__(self, k: int, graph: nx.Graph):
        self.k = k
        self.graph = graph
    
    def is_detectives_win(self, game_state: GameState) -> bool:
        for detective_pos in game_state.detective_positions:
            try:
                if nx.shortest_path_length(self.graph, detective_pos, 
                                         game_state.MrX_position) <= self.k:
                    return True
            except nx.NetworkXNoPath:
                continue
        return False
    
    def is_game_over(self, game_state: GameState) -> bool:
        return self.is_detectives_win(game_state)

class ScotlandYardMovement(MovementRule):
    """Movement rule for Scotland Yard game with transport types"""
    
    def get_valid_moves(self, graph: nx.Graph, position: int, 
                       game_state: GameState) -> Set[Tuple[int, TransportType]]:
        moves = set()
        
        # Handle multiple edges between same nodes
        for neighbor in graph.neighbors(position):
            # Get all edges between position and neighbor
            if graph.is_multigraph():
                # For multigraphs, get all edges
                edge_data_dict = graph.get_edge_data(position, neighbor)
                for edge_key, edge_data in edge_data_dict.items():
                    transport_type_val = edge_data.get('edge_type', 1)
                    transport_type = TransportType(transport_type_val)
                    moves.add((neighbor, transport_type))
            else:
                # For simple graphs, check if there are multiple transport types stored
                edge_data = graph.get_edge_data(position, neighbor)
                # Check if edge_data contains multiple transport types
                if 'transports' in edge_data:
                    # Multiple transport types stored as a list
                    for transport_val in edge_data['transports']:
                        transport_type = TransportType(transport_val)
                        moves.add((neighbor, transport_type))
                else:
                    # Single transport type
                    transport_type_val = edge_data.get('edge_type', 1)
                    transport_type = TransportType(transport_type_val)
                    moves.add((neighbor, transport_type))
        
        return moves
    
    def can_stay(self) -> bool:
        return False

class ScotlandYardWinCondition(WinCondition):
    """Scotland Yard specific win conditions"""
    
    def __init__(self, graph: nx.Graph, max_turns: int = 24):
        self.graph = graph
        self.max_turns = max_turns
        self.movement_rule = ScotlandYardMovement()
    
    def is_detectives_win(self, game_state: GameState) -> bool:
        # Detectives win if any detective is on Mr. X's position
        return game_state.MrX_position in game_state.detective_positions
    
    def is_mr_x_win(self, game_state: GameState) -> bool:
        # Mr. X wins if max turns reached
        if game_state.MrX_turn_count >= self.max_turns:
            return True
        
        # Mr. X wins if all detectives are stuck
        if game_state.turn == Player.DETECTIVES:
            all_stuck = True
            for i, pos in enumerate(game_state.detective_positions):
                if self._detective_can_move(game_state, i, pos):
                    all_stuck = False
                    break
            if all_stuck:
                return True

        return False
    
    def _detective_can_move(self, game_state: GameState, detective_id: int, position: int) -> bool:
        """Check if a detective has any valid move."""
        # This is a simplified check. For a full check, we'd need to consider
        # pending moves of other detectives in the same turn.
        # This check is sufficient for end-of-turn evaluation.
        all_moves = self.movement_rule.get_valid_moves(self.graph, position, game_state)
        detective_tickets = game_state.detective_tickets.get(detective_id, {})
        
        other_detective_positions = {pos for i, pos in enumerate(game_state.detective_positions) if i != detective_id}

        for dest, transport in all_moves:
            if dest in other_detective_positions:
                continue
            
            required_ticket = TicketType[transport.name]
            if detective_tickets.get(required_ticket, 0) > 0:
                return True # Found at least one valid move
        
        return False
    
    def is_game_over(self, game_state: GameState) -> bool:
        return self.is_detectives_win(game_state) or self.is_mr_x_win(game_state)
    
    
class Game:
    """Main game class that orchestrates the game"""
    
    def __init__(self, graph: nx.Graph, num_detectives: int, 
                 detective_movement: MovementRule = None,
                 MrX_movement: MovementRule = None,
                 win_condition: WinCondition = None):
        self.graph = graph
        self.num_detectives = num_detectives
        self.detective_movement = detective_movement or StandardMovement()
        self.MrX_movement = MrX_movement or StandardMovement()
        self.win_condition = win_condition or CaptureWinCondition()
        self.game_state = None
        self.game_history = []
    
    def initialize_game(self, detective_positions: List[int], MrX_position: int):
        """Initialize game with starting positions"""
        if len(detective_positions) != self.num_detectives:
            raise ValueError(f"Expected {self.num_detectives} detectives, got {len(detective_positions)}")
        
        # Check for position conflicts during setup
        if MrX_position in detective_positions:
            raise ValueError("MrX and detective cannot start in the same position")
        
        self.game_state = GameState(detective_positions, MrX_position, Player.DETECTIVES)
        self.game_history = [self.game_state.copy()]
    
    def get_valid_moves(self, player: Player, position: int = None) -> Set[int]:
        """Get valid moves for a player from a position"""
        if self.game_state is None:
            raise ValueError("Game not initialized")
        
        if player == Player.DETECTIVES:
            movement_rule = self.detective_movement
            if position is None:
                raise ValueError("Must specify position for detectives")
        else:
            movement_rule = self.MrX_movement
            if position is None:
                position = self.game_state.MrX_position
        
        valid_moves = movement_rule.get_valid_moves(self.graph, position, self.game_state)
        return valid_moves
    
    def make_move(self, new_positions: List[int] = None, new_MrX_pos: int = None) -> bool:
        """Make a move and return True if valid. Assumes move is pre-validated"""
        if self.game_state is None:
            raise ValueError("Game not initialized")
        
        # if self.is_game_over():
        #     return False
        
        if self.game_state.turn == Player.DETECTIVES:
            if new_positions is None or len(new_positions) != self.num_detectives:
                return False # Basic input check
            
            self.game_state.detective_positions = new_positions
            self.game_state.turn = Player.MRX
        
        else:  # MrX's turn
            if new_MrX_pos is None:
                return False # Basic input check
            
            self.game_state.MrX_position = new_MrX_pos
            self.game_state.turn = Player.DETECTIVES
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
        
        if self.win_condition.is_detectives_win(self.game_state):
            return Player.DETECTIVES
        else:
            return Player.MRX
    
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
            'detective_positions': self.game_state.detective_positions,
            'MrX_position': self.game_state.MrX_position,
            'turn': self.game_state.turn.value,
            'turn_count': self.game_state.turn_count,
            'game_over': self.is_game_over(),
            'winner': self.get_winner().value if self.get_winner() else None
        }
    
    def save_game(self, loader=None, game_id: str = None, 
                  additional_metadata: Dict = None) -> str:
        """
        Save this game using the loader.
        DEPRECATED: Use GameService.save_game() instead for better consistency.
        """
        # Maintain backward compatibility but recommend using GameService
        if loader is None:
            from ..services.game_loader import GameLoader
            loader = GameLoader()
        return loader.save_game(self, game_id, additional_metadata)
    
    def export_game(self, loader, game_id: str, 
                format: str = 'json') -> Optional[str]:
        """Export this game using the loader"""
        return loader.export_game(game_id, format)

    
class ScotlandYardGame(Game):
    """Scotland Yard variant of the game"""
    
    def __init__(self, graph: nx.Graph, num_detectives: int = 3):
        super().__init__(graph, num_detectives, 
                        ScotlandYardMovement(), ScotlandYardMovement(),
                        ScotlandYardWinCondition(graph))
        self.reveal_turns = {3, 8, 13, 18, 24}
        
    def initialize_scotland_yard_game(self, detective_positions: List[int], 
                                    mr_x_position: int):
        """Initialize Scotland Yard game with tickets"""
        # Initialize detective tickets
        detective_tickets = {}
        for i in range(len(detective_positions)):
            detective_tickets[i] = {
                TicketType.TAXI: 10,
                TicketType.BUS: 8, 
                TicketType.UNDERGROUND: 4
            }
            # detective_tickets[i] = {
            #     TicketType.TAXI: 1,
            #     TicketType.BUS: 1, 
            #     TicketType.UNDERGROUND: 1
            # }
        
        # Initialize Mr. X tickets
        mr_x_tickets = {
            TicketType.TAXI: 4,
            TicketType.BUS: 3,
            TicketType.UNDERGROUND: 3,
            TicketType.BLACK: 5,
            TicketType.DOUBLE_MOVE: 2
        }
        # mr_x_tickets = {
        #     TicketType.TAXI: 1,#4,
        #     TicketType.BUS: 3,
        #     TicketType.UNDERGROUND: 3,
        #     TicketType.BLACK: 1,#5,
        #     TicketType.DOUBLE_MOVE: 2
        # }
        
        self.game_state = GameState(
            detective_positions, mr_x_position, Player.MRX, 0,  0,
            detective_tickets, mr_x_tickets, False, []
        )
        self.game_history = [self.game_state.copy()]
        self.ticket_history = []
        self.last_visible_position = None

    def get_valid_moves(self, player: Player, position: int = None, pending_moves: List[Tuple[int, TransportType]] = None) -> Set[Tuple[int, TransportType]]:
        """Get valid moves for a player considering tickets.    
        Returns a set of (destination, transport_type) tuples.
        """
        if self.game_state is None:
            raise ValueError("Game not initialized")
        
        if player == Player.DETECTIVES:
            if position is None:
                raise ValueError("Must specify position for detectives")
            # Find which detective this is
            try:
                detective_id = self.game_state.detective_positions.index(position)
            except ValueError:
                return set() # Position does not match any detective
            
            return self._get_valid_detective_moves(detective_id, position, pending_moves)
        else: # MR_X
            position = self.game_state.MrX_position
            return self._get_valid_mr_x_moves(position)

    def _get_valid_detective_moves(self, detective_id: int, position: int, pending_moves: List[Tuple[int, TransportType]] = None) -> Set[Tuple[int, TransportType]]:
        """Get valid moves for a specific detective."""
        valid_moves = set()
        all_moves = self.detective_movement.get_valid_moves(self.graph, position, self.game_state)
        detective_tickets = self.get_detective_tickets(detective_id)
        
        pending_moves = pending_moves or []
        occupied_positions = set(move[0] for move in pending_moves)
        for i, pos in enumerate(self.game_state.detective_positions):
            if i > detective_id:
                occupied_positions.add(pos)

        for dest, transport in all_moves:
            if dest in occupied_positions:
                continue

            required_ticket = TicketType[transport.name]
            if detective_tickets.get(required_ticket, 0) > 0:
                valid_moves.add((dest, transport))
        return valid_moves

    def _get_valid_mr_x_moves(self, position: int) -> Set[Tuple[int, TransportType]]:
        """Get valid moves for Mr. X."""
        valid_moves = set()
        all_moves = self.MrX_movement.get_valid_moves(self.graph, position, self.game_state)
        mr_x_tickets = self.get_mr_x_tickets()

        for dest, transport in all_moves:
            required_ticket = TicketType[transport.name]
            
            # Mr. X can use specific ticket or black ticket
            if mr_x_tickets.get(required_ticket, 0) > 0:
                valid_moves.add((dest, transport))
            if mr_x_tickets.get(TicketType.BLACK, 0) > 0:
                valid_moves.add((dest, TransportType.BLACK))
        return valid_moves

    def can_use_double_move(self) -> bool:
        """Check if Mr. X can use double move ticket"""
        return (self.game_state.mr_x_tickets.get(TicketType.DOUBLE_MOVE, 0) > 0 and 
                not self.game_state.double_move_active)

    def make_move(self, detective_moves: List[Tuple[int, TransportType]] = None, 
                  mr_x_moves: List[Tuple[int, TransportType]] = None, 
                  use_double_move: bool = False) -> bool:
        """
        Make a move in Scotland Yard. Simplified double move handling.
        """
        if self.game_state is None:
            raise ValueError("Game not initialized")
        
        if self.is_game_over():
            return False
        
        # Prepare turn data for ticket history
        turn_data = {
            'turn_number': self.game_state.turn_count,
            'player': self.game_state.turn.value,
            'detective_moves': [],
            'mr_x_moves': [],
            'double_move_used': False
        }
        
        if self.game_state.turn == Player.DETECTIVES:
            if detective_moves is None or len(detective_moves) != self.num_detectives:
                return False

            new_positions = []
            for i, move in enumerate(detective_moves):
                new_pos, transport = move
                old_pos = self.game_state.detective_positions[i]
                new_positions.append(new_pos)
                
                # Record detective move in ticket history
                detective_move_data = {
                    'detective_id': i,
                    'edge': (old_pos, new_pos),
                    'ticket_used': None,
                    'stayed': new_pos == old_pos
                }
                
                if new_pos != old_pos and transport is not None:
                    required_ticket = TicketType[transport.name]
                    self.game_state.detective_tickets[i][required_ticket] -= 1
                    self.game_state.mr_x_tickets[required_ticket] = self.game_state.mr_x_tickets.get(required_ticket, 0) + 1
                    detective_move_data['ticket_used'] = required_ticket.value
                
                turn_data['detective_moves'].append(detective_move_data)
            
            self.game_state.detective_positions = new_positions
            self.game_state.turn = Player.MRX

        else:  # Mr. X's turn
            if not mr_x_moves or len(mr_x_moves) != 1:
                return False

            old_pos = self.game_state.MrX_position
            new_pos, transport_to_use = mr_x_moves[0]
            
            # Determine ticket to consume
            ticket_to_consume = TicketType[transport_to_use.name]
            self.game_state.mr_x_tickets[ticket_to_consume] -= 1

            # Handle double move ticket consumption
            if use_double_move and not self.game_state.double_move_active:
                if self.game_state.mr_x_tickets.get(TicketType.DOUBLE_MOVE, 0) <= 0:
                    return False
                self.game_state.mr_x_tickets[TicketType.DOUBLE_MOVE] -= 1
                self.game_state.double_move_active = True
                turn_data['double_move_used'] = True

            # Check and consume ticket
            # actual_ticket_used = None
            # if self.game_state.mr_x_tickets.get(ticket_to_consume, 0) > 0:
            #     # self.game_state.mr_x_tickets[ticket_to_consume] -= 1
            #     # actual_ticket_used = ticket_to_consume
            # # elif self.game_state.mr_x_tickets.get(TicketType.BLACK, 0) > 0:
            #     # self.game_state.mr_x_tickets[TicketType.BLACK] -= 1
            #     # actual_ticket_used = TicketType.BLACK
            # else:
            #     return False

            # Execute the move
            self.game_state.mr_x_moves_log.append((new_pos, transport_to_use))
            self.game_state.MrX_position = new_pos
            
            # Record Mr. X move in ticket history
            mr_x_move_data = {
                'edge': (old_pos, new_pos),
                'transport_used': transport_to_use.value,
                'ticket_used': ticket_to_consume.value,
                'double_move_part': 1 if self.game_state.double_move_active else 0
            }
            turn_data['mr_x_moves'].append(mr_x_move_data)
            
            # Handle turn progression for double moves
            if self.game_state.double_move_active:
                # We're in a double move - check if this completes it
                if use_double_move:
                    # This is the first move of a double move - stay on Mr. X's turn
                    mr_x_move_data['double_move_part'] = 1
                    self.game_state.turn = Player.MRX
                else:
                    # This is the second move of a double move - end it
                    mr_x_move_data['double_move_part'] = 2
                    self.game_state.double_move_active = False
                    self.game_state.turn = Player.DETECTIVES
            else:
                # Regular single move
                self.game_state.turn = Player.DETECTIVES
            self.game_state.MrX_turn_count += 1

        self.game_state.turn_count += 1
        self.game_state.mr_x_visible = self.game_state.MrX_turn_count in self.reveal_turns
        if self.game_state.mr_x_visible:
            # If Mr. X is visible, update last visible position
            self.last_visible_position = self.game_state.MrX_position
        # Add turn data to ticket history
        self.ticket_history.append(turn_data)
        self.game_history.append(self.game_state.copy())
        return True
    
    def get_winner(self) -> Optional[Player]:
        """Get winner if game is over"""
        if not self.is_game_over():
            return None
        
        if self.win_condition.is_detectives_win(self.game_state):
            return Player.DETECTIVES
        else:
            return Player.MRX

    def get_mr_x_last_visible_position(self) -> Optional[int]:
        """Get Mr. X's last visible position"""
        return self.last_visible_position
    
    def _all_detectives_moved(self) -> bool:
        """Check if all detectives have moved this turn"""
        # This is a simplified check - in full implementation you'd track who moved
        return True

    def get_detective_tickets(self, detective_id: int) -> Dict[TicketType, int]:
        """Get ticket counts for a detective"""
        return self.game_state.detective_tickets.get(detective_id, {})
    
    def get_mr_x_tickets(self) -> Dict[TicketType, int]:
        """Get Mr. X's ticket counts"""
        return self.game_state.mr_x_tickets.copy()

