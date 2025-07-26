"""
Feature extraction for Scotland Yard game states.

This module converts game states into feature vectors that can be used by
machine learning algorithms like MCTS and Deep Q-Learning.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from cops_and_robbers.core.game import ScotlandYardGame, Player, TransportType, TicketType
from agents.heuristics import GameHeuristics


@dataclass
class FeatureConfig:
    """Configuration for feature extraction."""
    include_distances: bool = True
    include_tickets: bool = True
    include_board_state: bool = True
    include_game_phase: bool = True
    include_transport_connectivity: bool = True
    include_possible_positions: bool = True
    max_nodes: int = 200  # Maximum number of nodes for fixed-size vectors
    distance_normalization: float = 20.0  # Max distance for normalization


class GameFeatureExtractor:
    """
    Extracts numerical feature vectors from Scotland Yard game states.
    
    This class converts complex game states into fixed-size numerical vectors
    that can be used as input to machine learning algorithms.
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        """
        Initialize the feature extractor.
        
        Args:
            config: Configuration for feature extraction
        """
        self.config = config or FeatureConfig()
        self.heuristics = None
        self._feature_size = None
        
    def get_feature_size(self, game: ScotlandYardGame) -> int:
        """
        Get the size of the feature vector for this game.
        
        Args:
            game: Scotland Yard game instance
            
        Returns:
            Size of the feature vector
        """
        if self._feature_size is None:
            # Calculate feature size based on configuration
            size = 0
            
            # Basic game state features
            if self.config.include_game_phase:
                size += 4  # turn, total_turns, mr_x_visible, reveal_turn, game_phase
            
            # Position features
            if self.config.include_board_state:
                size += self.config.max_nodes  # One-hot encoding for all positions
                
            # Distance features  
            if self.config.include_distances:
                max_detectives = 5  # Assume max 5 detectives
                size += max_detectives * 3  # min_dist, max_dist, avg_dist for each detective
                size += 3  # Overall min, max, avg distances
                
            # Ticket features
            if self.config.include_tickets:
                size += 5  # taxi, bus, underground, black, double tickets for Mr. X
                size += 15  # taxi, bus, underground for each of 5 detectives
                
            # Transport connectivity features
            if self.config.include_transport_connectivity:
                size += 12  # 3 transport types * 4 connectivity metrics
                
            # Possible positions features (for Mr. X when hidden)
            if self.config.include_possible_positions:
                size += 3  # num_possible_positions, min_dist_to_possible, connectivity_score
                
            self._feature_size = size
            
        return self._feature_size
    
    def extract_features(self, game: ScotlandYardGame, player: Player) -> np.ndarray:
        """
        Extract feature vector from the current game state.
        
        Args:
            game: Scotland Yard game instance
            player: Player perspective (ROBBER for Mr. X, COPS for detectives)
            
        Returns:
            Numpy array representing the game state features
        """
        if self.heuristics is None:
            self.heuristics = GameHeuristics(game)
        else:
            self.heuristics.update_game_state(game)
        
        features = []
        
        # Game phase features
        if self.config.include_game_phase:
            features.extend(self._extract_game_phase_features(game))
        
        # Board state features
        if self.config.include_board_state:
            features.extend(self._extract_board_state_features(game, player))
        
        # Distance features
        if self.config.include_distances:
            features.extend(self._extract_distance_features(game, player))
        
        # Ticket features
        if self.config.include_tickets:
            features.extend(self._extract_ticket_features(game, player))
        
        # Transport connectivity features
        if self.config.include_transport_connectivity:
            features.extend(self._extract_connectivity_features(game, player))
        
        # Possible positions features
        if self.config.include_possible_positions:
            features.extend(self._extract_possible_positions_features(game, player))
        
        return np.array(features, dtype=np.float32)
    
    def _extract_game_phase_features(self, game: ScotlandYardGame) -> List[float]:
        """Extract features related to game phase and timing."""
        features = []
        
        # Current turn normalized
        max_turns = getattr(game, 'max_turns', 24)
        current_turn = len(game.game_history) if hasattr(game, 'game_history') else 0
        features.append(current_turn / max_turns)
        
        # Total turns completed (same as above, but not normalized)
        features.append(float(current_turn))
        
        # Mr. X visibility
        features.append(1.0 if game.game_state.mr_x_visible else 0.0)
        
        # Game phase (early/mid/late game)
        if current_turn / max_turns < 0.3:
            game_phase = 0.0  # Early game
        elif current_turn / max_turns < 0.7:
            game_phase = 0.5  # Mid game
        else:
            game_phase = 1.0  # Late game
        features.append(game_phase)
        
        return features
    
    def _extract_board_state_features(self, game: ScotlandYardGame, player: Player) -> List[float]:
        """Extract features representing board positions."""
        features = [0.0] * self.config.max_nodes
        
        # Mr. X position (if visible or if we're Mr. X)
        if game.game_state.mr_x_visible or player == Player.ROBBER:
            mr_x_pos = game.game_state.robber_position
            if mr_x_pos < self.config.max_nodes:
                features[mr_x_pos] = 1.0
        
        # Detective positions (negative values to distinguish from Mr. X)
        for i, detective_pos in enumerate(game.game_state.cop_positions):
            if detective_pos < self.config.max_nodes:
                features[detective_pos] = -1.0 - (i * 0.1)  # Different values for different detectives
        
        return features
    
    def _extract_distance_features(self, game: ScotlandYardGame, player: Player) -> List[float]:
        """Extract distance-based features."""
        features = []
        
        # Get distances between Mr. X and detectives
        distances = self.heuristics.get_mr_x_distances_from_detectives()
        
        # Normalize distances
        normalized_distances = [d / self.config.distance_normalization if d >= 0 else -1.0 
                              for d in distances]
        
        # Pad or truncate to ensure consistent size (max 5 detectives)
        while len(normalized_distances) < 5:
            normalized_distances.append(-1.0)
        normalized_distances = normalized_distances[:5]
        
        # Individual detective distances with additional metrics
        for dist in normalized_distances:
            features.append(dist)  # Raw distance
            features.append(1.0 if dist > 0 else 0.0)  # Reachable flag
            features.append(min(dist * 2, 1.0) if dist > 0 else 0.0)  # Threat level (closer = higher)
        
        # Overall distance statistics
        valid_distances = [d for d in distances if d >= 0]
        if valid_distances:
            features.append(min(valid_distances) / self.config.distance_normalization)  # Min distance
            features.append(max(valid_distances) / self.config.distance_normalization)  # Max distance
            features.append(sum(valid_distances) / len(valid_distances) / self.config.distance_normalization)  # Avg distance
        else:
            features.extend([0.0, 0.0, 0.0])
        
        return features
    
    def _extract_ticket_features(self, game: ScotlandYardGame, player: Player) -> List[float]:
        """Extract ticket-related features."""
        features = []
        
        # Mr. X tickets
        mr_x_tickets = game.get_mr_x_tickets()
        max_tickets = 50  # Normalize by this value
        
        features.append(mr_x_tickets.get(TicketType.TAXI, 0) / max_tickets)
        features.append(mr_x_tickets.get(TicketType.BUS, 0) / max_tickets)
        features.append(mr_x_tickets.get(TicketType.UNDERGROUND, 0) / max_tickets)
        features.append(mr_x_tickets.get(TicketType.BLACK, 0) / 5)  # Black tickets are rarer
        features.append(mr_x_tickets.get(TicketType.DOUBLE_MOVE, 0) / 2)  # Double move availability

        # Detective tickets (for each detective, up to 5)
        for i in range(len(game.game_state.cop_positions)):
            detective_tickets = game.get_detective_tickets(i)
            features.append(detective_tickets.get(TicketType.TAXI, 0) / max_tickets)
            features.append(detective_tickets.get(TicketType.BUS, 0) / max_tickets)
            features.append(detective_tickets.get(TicketType.UNDERGROUND, 0) / max_tickets)
        
        return features
    
    def _extract_connectivity_features(self, game: ScotlandYardGame, player: Player) -> List[float]:
        """Extract transport connectivity features."""
        features = []
        
        # Current position of the player
        if player == Player.ROBBER:
            current_pos = game.game_state.robber_position
        else:
            # For detectives, use average connectivity of all detectives
            detective_positions = game.game_state.cop_positions
            current_pos = detective_positions[0] if detective_positions else 1
        
        # Count available moves by transport type
        for transport_type in [TransportType.TAXI, TransportType.BUS, TransportType.UNDERGROUND]:
            valid_moves = list(game.get_valid_moves(player, current_pos))
            transport_moves = [move for move in valid_moves if move[1] == transport_type]
            
            # Number of moves available with this transport
            features.append(len(transport_moves) / 10.0)  # Normalize by max expected moves
            
            # Average distance of reachable positions
            if transport_moves:
                distances = []
                for dest, _ in transport_moves:
                    # Calculate distance to closest detective/Mr. X
                    if player == Player.ROBBER:
                        # For Mr. X, distance to closest detective
                        min_dist = float('inf')
                        for detective_pos in game.game_state.cop_positions:
                            dist = self.heuristics.calculate_shortest_distance(dest, detective_pos)
                            if dist >= 0 and dist < min_dist:
                                min_dist = dist
                        if min_dist != float('inf'):
                            distances.append(min_dist)
                    else:
                        # For detectives, distance to Mr. X (if known)
                        if game.game_state.mr_x_visible:
                            dist = self.heuristics.calculate_shortest_distance(dest, game.game_state.robber_position)
                            if dist >= 0:
                                distances.append(dist)
                
                avg_dist = sum(distances) / len(distances) if distances else 0
                features.append(avg_dist / self.config.distance_normalization)
            else:
                features.append(0.0)
            
            # Connectivity diversity (number of unique destinations)
            unique_destinations = len(set(dest for dest, _ in transport_moves))
            features.append(unique_destinations / 10.0)
            
            # Transport efficiency (ratio of this transport to total moves)
            total_moves = len(valid_moves)
            efficiency = len(transport_moves) / total_moves if total_moves > 0 else 0
            features.append(efficiency)
        
        return features
    
    def _extract_possible_positions_features(self, game: ScotlandYardGame, player: Player) -> List[float]:
        """Extract features related to Mr. X's possible positions (when hidden)."""
        features = []
        
        if player == Player.COPS and not game.game_state.mr_x_visible:
            # For detectives when Mr. X is hidden
            possible_positions = self.heuristics.get_possible_mr_x_positions()
            
            # Number of possible positions (normalized)
            features.append(len(possible_positions) / 50.0)  # Normalize by reasonable max
            
            # Minimum distance to any possible position
            if possible_positions:
                detective_distances = self.heuristics.get_detective_distances_to_possible_mr_x_positions()
                all_distances = []
                for distances in detective_distances.values():
                    all_distances.extend([d for d in distances if d >= 0])
                
                if all_distances:
                    min_dist = min(all_distances)
                    features.append(min_dist / self.config.distance_normalization)
                else:
                    features.append(1.0)  # Max distance if no valid paths
            else:
                features.append(1.0)
            
            # Connectivity score (how well-connected possible positions are)
            connectivity_score = 0.0
            if possible_positions:
                total_connections = 0
                for pos in possible_positions:
                    connections = len(list(game.graph.neighbors(pos)))
                    total_connections += connections
                connectivity_score = total_connections / (len(possible_positions) * 10.0)  # Normalize
            features.append(connectivity_score)
            
        else:
            # When Mr. X is visible or we're playing as Mr. X
            features.extend([0.0, 0.0, 0.0])  # Placeholder values
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """
        Get human-readable names for all features.
        
        Returns:
            List of feature names in the same order as the feature vector
        """
        names = []
        
        if self.config.include_game_phase:
            names.extend(['turn_progress', 'current_turn', 'mr_x_visible', 'is_reveal_turn', 'game_phase'])
        
        if self.config.include_board_state:
            for i in range(self.config.max_nodes):
                names.append(f'node_{i}_occupation')
        
        if self.config.include_distances:
            for i in range(5):
                names.extend([f'detective_{i}_distance', f'detective_{i}_reachable', f'detective_{i}_threat'])
            names.extend(['min_distance', 'max_distance', 'avg_distance'])
        
        if self.config.include_tickets:
            names.extend(['mrx_taxi', 'mrx_bus', 'mrx_underground', 'mrx_black', 'mrx_double'])
            for i in range(5):
                names.extend([f'det_{i}_taxi', f'det_{i}_bus', f'det_{i}_underground'])
        
        if self.config.include_transport_connectivity:
            for transport in ['taxi', 'bus', 'underground']:
                names.extend([f'{transport}_moves', f'{transport}_avg_dist', f'{transport}_diversity', f'{transport}_efficiency'])
        
        if self.config.include_possible_positions:
            names.extend(['num_possible_positions', 'min_dist_to_possible', 'connectivity_score'])
        
        return names
