from abc import ABC, abstractmethod
from typing import List, Set, Tuple, Dict, Optional
from enum import Enum
import networkx as nx


class Player(Enum):
    COPS = "cops"
    ROBBER = "robber"
    MR_X = "mr_x"  # Added for Scotland Yard

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

class GameState:
    """Represents the current state of the game"""
    def __init__(self, cop_positions: List[int], robber_position: int, 
                 turn: Player, turn_count: int = 0,
                 detective_tickets: Dict[int, Dict[TicketType, int]] = None,
                 mr_x_tickets: Dict[TicketType, int] = None,
                 mr_x_visible: bool = True,
                 mr_x_moves_log: List[Tuple[int, TransportType]] = None):
        self.cop_positions = cop_positions.copy()
        self.robber_position = robber_position
        self.turn = turn
        self.turn_count = turn_count
        # Scotland Yard specific
        self.detective_tickets = detective_tickets or {}
        self.mr_x_tickets = mr_x_tickets or {}
        self.mr_x_visible = mr_x_visible
        self.mr_x_moves_log = mr_x_moves_log or []
        self.double_move_active = False
    
    def copy(self):
        return GameState(
            self.cop_positions, self.robber_position, 
            self.turn, self.turn_count,
            {k: v.copy() for k, v in self.detective_tickets.items()},
            self.mr_x_tickets.copy(),
            self.mr_x_visible,
            self.mr_x_moves_log.copy()
        )
    
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
        # Don't consider it a win if no moves have been made yet
        if game_state.turn_count == 0 and game_state.turn == Player.COPS:
            return False
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
        
        # Check for position conflicts during setup
        if robber_position in cop_positions:
            raise ValueError("Robber and cop cannot start in the same position")
        
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
        
        # For Scotland Yard games, use the transport-aware method
        if hasattr(self, 'is_scotland_yard') and self.is_scotland_yard:
            if player == Player.COPS:
                # Find which detective this is
                detective_id = None
                for i, cop_pos in enumerate(self.game_state.cop_positions):
                    if cop_pos == position:
                        detective_id = i
                        break
                
                if detective_id is not None:
                    return self.get_valid_moves_with_tickets(Player.COPS, detective_id)
            else:
                return self.get_valid_moves_with_tickets(Player.MR_X)
        
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
    
    def get_valid_moves_with_tickets(self, player: Player, detective_id: int = None) -> Set[int]:
        """Get valid moves considering ticket availability (Scotland Yard specific)"""
        if not hasattr(self, 'is_scotland_yard') or not self.is_scotland_yard:
            return set()
        
        valid_moves = set()
        
        if player == Player.COPS and detective_id is not None:
            current_pos = self.game_state.cop_positions[detective_id]
            detective_tickets = self.get_detective_tickets(detective_id)
            
            for neighbor in self.graph.neighbors(current_pos):
                # Check if position is occupied by another detective
                if neighbor in self.game_state.cop_positions:
                    continue
                
                # Check if position is occupied by Mr. X (detectives can't move there)
                if neighbor == self.game_state.robber_position:
                    continue
                
                edge_data = self.graph.get_edge_data(current_pos, neighbor)
                transport_type = edge_data.get('edge_type', 1)
                
                # Check if detective has the required ticket
                ticket_mapping = {
                    1: TicketType.TAXI,
                    2: TicketType.BUS,
                    3: TicketType.UNDERGROUND
                }
                
                required_ticket = ticket_mapping.get(transport_type, TicketType.TAXI)
                if detective_tickets.get(required_ticket, 0) > 0:
                    valid_moves.add(neighbor)
        
        elif player == Player.MR_X:
            current_pos = self.game_state.robber_position
            mr_x_tickets = self.get_mr_x_tickets()
            
            for neighbor in self.graph.neighbors(current_pos):
                # Check if position is occupied by a detective
                if neighbor in self.game_state.cop_positions:
                    continue
                
                edge_data = self.graph.get_edge_data(current_pos, neighbor)
                transport_type = edge_data.get('edge_type', 1)
                
                ticket_mapping = {
                    1: TicketType.TAXI,
                    2: TicketType.BUS,
                    3: TicketType.UNDERGROUND
                }
                
                required_ticket = ticket_mapping.get(transport_type, TicketType.TAXI)
                
                # Mr. X can use specific ticket or black ticket
                if (mr_x_tickets.get(required_ticket, 0) > 0 or 
                    mr_x_tickets.get(TicketType.BLACK, 0) > 0):
                    valid_moves.add(neighbor)
        
        return valid_moves
    
    def make_move(self, new_positions: List[int] = None, new_robber_pos: int = None) -> bool:
        """Make a move and return True if valid"""
        if self.game_state is None:
            raise ValueError("Game not initialized")
        
        if self.is_game_over():
            return False
        
        if self.game_state.turn == Player.COPS:
            if new_positions is None or len(new_positions) != self.num_cops:
                return False
            
            # Check for duplicate positions among cops
            if len(set(new_positions)) != len(new_positions):
                return False
            
            # Validate all cop moves and handle ticket consumption for Scotland Yard
            for i, new_pos in enumerate(new_positions):
                old_pos = self.game_state.cop_positions[i]
                
                # Skip validation if not moving
                if old_pos == new_pos:
                    continue
                
                # Check basic connectivity
                if not self.graph.has_edge(old_pos, new_pos):
                    return False
                
                # For Scotland Yard games, check tickets and consume them
                if hasattr(self, 'is_scotland_yard') and self.is_scotland_yard:
                    # Get edge data to determine transport type
                    edge_data = self.graph.get_edge_data(old_pos, new_pos)
                    transport_type = edge_data.get('edge_type', 1)
                    
                    # Map transport to ticket
                    ticket_mapping = {1: TicketType.TAXI, 2: TicketType.BUS, 3: TicketType.UNDERGROUND}
                    required_ticket = ticket_mapping.get(transport_type, TicketType.TAXI)
                    
                    # Check if detective has the ticket
                    detective_tickets = self.game_state.detective_tickets.get(i, {})
                    if detective_tickets.get(required_ticket, 0) <= 0:
                        return False
                    
                    # Check if position conflicts with other new positions (already handled by duplicate check)
                    # Check if moving to Mr. X's current position
                    if new_pos == self.game_state.robber_position:
                        return False
                else:
                    # Basic game validation
                    valid_moves = self.get_valid_moves(Player.COPS, old_pos)
                    if new_pos not in valid_moves:
                        return False
                    
                    # Check if new position conflicts with robber
                    if new_pos == self.game_state.robber_position:
                        return False
            
            # If we get here, all moves are valid - now consume tickets for Scotland Yard
            if hasattr(self, 'is_scotland_yard') and self.is_scotland_yard:
                for i, new_pos in enumerate(new_positions):
                    old_pos = self.game_state.cop_positions[i]
                    
                    if old_pos != new_pos:  # Only consume tickets if actually moving
                        edge_data = self.graph.get_edge_data(old_pos, new_pos)
                        transport_type = edge_data.get('edge_type', 1)
                        ticket_mapping = {1: TicketType.TAXI, 2: TicketType.BUS, 3: TicketType.UNDERGROUND}
                        required_ticket = ticket_mapping.get(transport_type, TicketType.TAXI)
                        
                        # Consume the ticket and give it to Mr. X
                        self.game_state.detective_tickets[i][required_ticket] -= 1
                        if required_ticket not in self.game_state.mr_x_tickets:
                            self.game_state.mr_x_tickets[required_ticket] = 0
                        self.game_state.mr_x_tickets[required_ticket] += 1
            
            self.game_state.cop_positions = new_positions
            self.game_state.turn = Player.ROBBER if not hasattr(self, 'is_scotland_yard') else Player.MR_X
        
        else:  # Robber's/Mr. X's turn
            if new_robber_pos is None:
                return False
            
            old_pos = self.game_state.robber_position
            
            # Skip validation if not moving
            if old_pos == new_robber_pos:
                self.game_state.turn = Player.COPS
                self.game_state.turn_count += 1
                self.game_history.append(self.game_state.copy())
                return True
            
            # Check basic connectivity
            if not self.graph.has_edge(old_pos, new_robber_pos):
                return False
            
            # For Scotland Yard games, handle ticket consumption
            if hasattr(self, 'is_scotland_yard') and self.is_scotland_yard:
                edge_data = self.graph.get_edge_data(old_pos, new_robber_pos)
                transport_type = edge_data.get('edge_type', 1)
                
                # Map transport to ticket
                ticket_mapping = {1: TicketType.TAXI, 2: TicketType.BUS, 3: TicketType.UNDERGROUND}
                required_ticket = ticket_mapping.get(transport_type, TicketType.TAXI)
                
                # Check if Mr. X has the specific ticket or black ticket
                mr_x_tickets = self.game_state.mr_x_tickets
                can_use_specific = mr_x_tickets.get(required_ticket, 0) > 0
                can_use_black = mr_x_tickets.get(TicketType.BLACK, 0) > 0
                
                if not (can_use_specific or can_use_black):
                    return False
                
                # Check if moving to a cop position
                if new_robber_pos in self.game_state.cop_positions:
                    return False
                
                # Consume appropriate ticket (prefer specific ticket over black)
                if can_use_specific:
                    self.game_state.mr_x_tickets[required_ticket] -= 1
                else:
                    self.game_state.mr_x_tickets[TicketType.BLACK] -= 1
                
                # Update visibility based on reveal turns
                if hasattr(self, 'reveal_turns'):
                    self.game_state.mr_x_visible = (self.game_state.turn_count + 1) in self.reveal_turns
            else:
                # Basic game validation
                valid_moves = self.get_valid_moves(Player.ROBBER)
                if new_robber_pos not in valid_moves:
                    return False
                
                # Check if robber is moving to a cop position
                if new_robber_pos in self.game_state.cop_positions:
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

class ScotlandYardMovement(MovementRule):
    """Movement rule for Scotland Yard game with transport types"""
    
    def get_valid_moves(self, graph: nx.Graph, position: int, 
                       game_state: GameState) -> Set[Tuple[int, TransportType]]:
        moves = set()
        
        for neighbor in graph.neighbors(position):
            edge_data = graph.get_edge_data(position, neighbor)
            if 'edge_type' in edge_data:
                transport_type = TransportType(edge_data['edge_type'])
                moves.add((neighbor, transport_type))
        
        return moves
    
    def can_stay(self) -> bool:
        return False

class ScotlandYardWinCondition(WinCondition):
    """Scotland Yard specific win conditions"""
    
    def __init__(self, max_turns: int = 22):
        self.max_turns = max_turns
        # self.reveal_turns = {3, 8, 13, 18, 24}
    
    def is_cops_win(self, game_state: GameState) -> bool:
        # Detectives win if any detective is on Mr. X's position
        return game_state.robber_position in game_state.cop_positions
    
    def is_mr_x_win(self, game_state: GameState) -> bool:
        # Mr. X wins if max turns reached or detectives can't move
        if game_state.turn_count >= self.max_turns:
            return True
        
        # Check if all detectives are stuck
        for i, detective_pos in enumerate(game_state.cop_positions):
            if self._detective_can_move(game_state, i, detective_pos):
                return False
        return True
    
    def _detective_can_move(self, game_state: GameState, detective_id: int, position: int) -> bool:
        # Check if detective has tickets and valid moves
        if detective_id not in game_state.detective_tickets:
            return False
        
        tickets = game_state.detective_tickets[detective_id]
        return any(count > 0 for count in tickets.values())
    
    def is_game_over(self, game_state: GameState) -> bool:
        return self.is_cops_win(game_state) or self.is_mr_x_win(game_state)

class ScotlandYardGame(Game):
    """Scotland Yard variant of the game"""
    
    def __init__(self, graph: nx.Graph, num_detectives: int = 3):
        super().__init__(graph, num_detectives, 
                        ScotlandYardMovement(), ScotlandYardMovement(),
                        ScotlandYardWinCondition())
        self.is_scotland_yard = True
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
        
        # Initialize Mr. X tickets
        mr_x_tickets = {
            TicketType.TAXI: 4,
            TicketType.BUS: 3,
            TicketType.UNDERGROUND: 3,
            TicketType.BLACK: 5,
            TicketType.DOUBLE_MOVE: 2
        }
        
        self.game_state = GameState(
            detective_positions, mr_x_position, Player.MR_X, 0,
            detective_tickets, mr_x_tickets, False, []
        )
        self.game_history = [self.game_state.copy()]
    
    def make_scotland_yard_move(self, player: Player, new_position: int = None,
                              transport_type: TransportType = None,
                              detective_id: int = None,
                              use_double_move: bool = False) -> bool:
        """Make a move in Scotland Yard game"""
        if self.game_state is None:
            raise ValueError("Game not initialized")
        
        if self.is_game_over():
            return False
        
        if player == Player.MR_X:
            return self._make_mr_x_move(new_position, transport_type, use_double_move)
        elif player == Player.COPS:
            return self._make_detective_move(detective_id, new_position, transport_type)
        
        return False
    
    def _make_mr_x_move(self, new_position: int, transport_type: TransportType,
                       use_double_move: bool = False) -> bool:
        """Handle Mr. X's move"""
        if not self._validate_mr_x_move(new_position, transport_type):
            return False
        
        # Use ticket
        ticket_type = self._transport_to_ticket(transport_type)
        if use_double_move:
            self.game_state.mr_x_tickets[TicketType.DOUBLE_MOVE] -= 1
            self.game_state.double_move_active = True
        
        self.game_state.mr_x_tickets[ticket_type] -= 1
        
        # Record move
        self.game_state.robber_position = new_position
        self.game_state.mr_x_moves_log.append((new_position, transport_type))
        
        # Check if position should be revealed
        self.game_state.mr_x_visible = (self.game_state.turn_count + 1) in self.reveal_turns
        
        if not self.game_state.double_move_active:
            self.game_state.turn = Player.COPS
        else:
            self.game_state.double_move_active = False
        
        return True
    
    def _make_detective_move(self, detective_id: int, new_position: int,
                           transport_type: TransportType) -> bool:
        """Handle detective's move"""
        if not self._validate_detective_move(detective_id, new_position, transport_type):
            return False
        
        # Use ticket and give to Mr. X
        ticket_type = self._transport_to_ticket(transport_type)
        self.game_state.detective_tickets[detective_id][ticket_type] -= 1
        self.game_state.mr_x_tickets[ticket_type] += 1
        
        # Move detective
        self.game_state.cop_positions[detective_id] = new_position
        
        # Check if all detectives moved
        if self._all_detectives_moved():
            self.game_state.turn = Player.MR_X
            self.game_state.turn_count += 1
        
        return True
    
    def _validate_mr_x_move(self, new_position: int, transport_type: TransportType) -> bool:
        """Validate Mr. X's move"""
        current_pos = self.game_state.robber_position
        
        # Check if move is valid on graph
        if not self.graph.has_edge(current_pos, new_position):
            return False
        
        edge_data = self.graph.get_edge_data(current_pos, new_position)
        valid_transports = [TransportType(edge_data.get('edge_type', 1))]
        
        # Black tickets can use any transport
        if transport_type == TransportType.BLACK:
            valid_transports.extend([TransportType.TAXI, TransportType.BUS, TransportType.UNDERGROUND])
        
        if transport_type not in valid_transports:
            return False
        
        # Check if Mr. X has the required ticket
        ticket_type = self._transport_to_ticket(transport_type)
        return self.game_state.mr_x_tickets.get(ticket_type, 0) > 0
    
    def _validate_detective_move(self, detective_id: int, new_position: int,
                               transport_type: TransportType) -> bool:
        """Validate detective's move"""
        if detective_id >= len(self.game_state.cop_positions):
            return False
        
        current_pos = self.game_state.cop_positions[detective_id]
        
        # Check graph connectivity
        if not self.graph.has_edge(current_pos, new_position):
            return False
        
        # Check transport type
        edge_data = self.graph.get_edge_data(current_pos, new_position)
        if TransportType(edge_data.get('edge_type', 1)) != transport_type:
            return False
        # Check if position is occupied by another detective
        if new_position in self.game_state.cop_positions:
            return False
        
        # Check if detective has the ticket
        ticket_type = self._transport_to_ticket(transport_type)
        return self.game_state.detective_tickets[detective_id].get(ticket_type, 0) > 0
    
    def _transport_to_ticket(self, transport_type: TransportType) -> TicketType:
        """Convert transport type to ticket type"""
        mapping = {
            TransportType.TAXI: TicketType.TAXI,
            TransportType.BUS: TicketType.BUS,
            TransportType.UNDERGROUND: TicketType.UNDERGROUND,
            TransportType.BLACK: TicketType.BLACK
        }
        return mapping.get(transport_type, TicketType.TAXI)
    
    def _all_detectives_moved(self) -> bool:
        """Check if all detectives have moved this turn"""
        # This is a simplified check - in full implementation you'd track who moved
        return True
    
    def get_mr_x_visible_position(self) -> Optional[int]:
        """Get Mr. X's position if visible"""
        if self.game_state.mr_x_visible:
            return self.game_state.robber_position
        return None
    
    def get_detective_tickets(self, detective_id: int) -> Dict[TicketType, int]:
        """Get ticket counts for a detective"""
        return self.game_state.detective_tickets.get(detective_id, {})
    
    def get_mr_x_tickets(self) -> Dict[TicketType, int]:
        """Get Mr. X's ticket counts"""
        return self.game_state.mr_x_tickets.copy()
