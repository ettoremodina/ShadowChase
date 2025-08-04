"""
Heuristic-based agent implementations for Shadow Chase game.

This module contains AI agents that use heuristic calculations to make strategic moves.
Detectives try to minimize distance to Mr. X's last known position, while Mr. X tries to 
maximize distance from the closest detective.
"""

import random
from typing import List, Tuple, Optional
from ShadowChase.core.game import ShadowChaseGame, Player, TransportType, TicketType
from .base_agent import DetectiveAgent, MrXAgent, MultiDetectiveAgent
from .heuristics import GameHeuristics


class HeuristicMrXAgent(MrXAgent):
    """Mr. X agent that tries to maximize distance from the closest detective"""
    
    def __init__(self):
        super().__init__()
        self.heuristics = None
    
    def choose_move(self, game: ShadowChaseGame) -> Optional[Tuple[int, TransportType, bool]]:
        """Make a move that maximizes distance from closest detective"""
        if self.heuristics is None:
            self.heuristics = GameHeuristics(game)
        else:
            self.heuristics.update_game_state(game)
        
        valid_moves = list(game.get_valid_moves(Player.MRX))
        if not valid_moves:
            return (None, None, False)  # No valid moves
        
        # Evaluate each move and choose the one that maximizes distance from closest detective
        best_moves = []
        best_score = -1
        
        detective_positions = game.game_state.detective_positions
        
        for destination, transport in valid_moves:
            # Calculate minimum distance from this destination to any detective
            min_distance = float('inf')
            for detective_pos in detective_positions:
                distance = self.heuristics.calculate_shortest_distance(destination, detective_pos)
                if distance >= 0 and distance < min_distance:
                    min_distance = distance
            
            if min_distance == float('inf'):
                min_distance = -1
            
            # Prefer moves that maximize distance from closest detective
            if min_distance > best_score:
                best_score = min_distance
                best_moves = [(destination, transport)]
            elif min_distance == best_score and min_distance >= 0:
                best_moves.append((destination, transport))
        
        if not best_moves:
            # Fallback to random move if no good heuristic move found
            destination, transport = random.choice(valid_moves)
        else:
            destination, transport = random.choice(best_moves)
        
        # Decide on special moves
        use_double = self.should_use_double_move(game) and random.random() < 0.3
        
        # Consider using black ticket strategically
        if self._should_use_black_ticket(game, transport):
            transport = TransportType.BLACK
        
        return (destination, transport, use_double)
    
    def should_use_double_move(self, game: ShadowChaseGame) -> bool:
        """Use double move strategically when detectives are close"""
        if not self.can_use_double_move(game):
            return False
        
        if self.heuristics is None:
            return False
        
        # Use double move if closest detective is within 3 steps
        min_distance = self.heuristics.get_minimum_distance_to_MrX()
        return min_distance >= 0 and min_distance <= 3
    
    def _should_use_black_ticket(self, game: ShadowChaseGame, required_transport: TransportType) -> bool:
        """Use black ticket when detectives are close or we're low on regular tickets"""
        tickets = self.get_available_tickets(game)
        
        # If we don't have the required ticket, use black if available
        required_ticket = TicketType[required_transport.name]
        if tickets.get(required_ticket, 0) == 0:
            return tickets.get(TicketType.BLACK, 0) > 0
        
        # Use black ticket if detectives are close (within 2 steps) and we have black tickets
        if tickets.get(TicketType.BLACK, 0) > 0:
            if self.heuristics:
                min_distance = self.heuristics.get_minimum_distance_to_MrX()
                if min_distance >= 0 and min_distance <= 2:
                    return random.random() < 0.7  # 70% chance when close
        
        return False


class HeuristicMultiDetectiveAgent(MultiDetectiveAgent):
    """Agent that controls all detectives to minimize distance to Mr. X's last known position"""
    
    def __init__(self, num_detectives: int):
        super().__init__(num_detectives)
        self.heuristics = None
    
    def choose_all_moves(self, game: ShadowChaseGame) -> List[Tuple[int, TransportType]]:
        """Make moves for all detectives to minimize distance to Mr. X's last known position"""
        if self.heuristics is None:
            self.heuristics = GameHeuristics(game)
        else:
            self.heuristics.update_game_state(game)
        
        detective_moves = []
        pending_moves = []
        
        # Get Mr. X's last known position or possible positions
        target_position = None
        if game.game_state.MrX_visible:
            target_position = game.game_state.MrX_position
        elif hasattr(game, 'last_visible_position') and game.last_visible_position is not None:
            target_position = game.last_visible_position
        
        for i in range(self.num_detectives):
            current_pos = game.game_state.detective_positions[i]
            
            # Get valid moves considering previous detectives' moves
            valid_moves = list(game.get_valid_moves(Player.DETECTIVES, current_pos, pending_moves=pending_moves))
            
            if not valid_moves:
                # Stay in place if no valid moves
                move = (current_pos, None)
            else:
                move = self._choose_best_detective_move(current_pos, valid_moves, target_position)
            
            detective_moves.append(move)
            pending_moves.append(move)
        
        return detective_moves
    
    def _choose_best_detective_move(self, current_pos: int, valid_moves: List[Tuple[int, TransportType]], 
                                  target_position: Optional[int]) -> Tuple[int, TransportType]:
        """Choose the best move for a detective to get closer to the target"""
        if target_position is None:
            # If no target known, make random move
            return random.choice(valid_moves)
        
        best_moves = []
        best_distance = float('inf')
        
        for destination, transport in valid_moves:
            # Calculate distance from this destination to target
            distance = self.heuristics.calculate_shortest_distance(destination, target_position)
            
            if distance >= 0:  # Valid path exists
                if distance < best_distance:
                    best_distance = distance
                    best_moves = [(destination, transport)]
                elif distance == best_distance:
                    best_moves.append((destination, transport))
        
        if best_moves:
            # Choose randomly among equally good moves
            return random.choice(best_moves)
        else:
            # Fallback to random move if no good heuristic move found
            return random.choice(valid_moves)


