"""
Feature extraction for Scotland Yard game states.

This module converts game states into feature vectors that can be used by
machine learning algorithms like MCTS and Deep Q-Learning.
"""
import numpy as np
from typing import List, Optional
from dataclasses import dataclass

from ScotlandYard.core.game import ScotlandYardGame, Player, TicketType


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
        self._feature_size = 0
        
    def get_feature_size(self, game: ScotlandYardGame, player: Player = Player.MRX) -> int:
        """
        Get the size of the feature vector for this game.
        
        Args:
            game: Scotland Yard game instance
            
        Returns:
            Size of the feature vector
        """
        if self._feature_size > 0:
            return self._feature_size
        else:
            _ = self.extract_features(game, player)  # Use any player to initialize
        return self._feature_size
    
    def extract_features(self, game: ScotlandYardGame, player: Player) -> np.ndarray:
        """
        Extract feature vector from the current game state.
        
        Args:
            game: Scotland Yard game instance
            player: Player perspective (MrX for Mr. X, detectives for detectives)
            
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
            temp_features, feature_size = self._extract_game_phase_features(game)
            features.extend(temp_features)
            self._feature_size += feature_size

        # Board state features
        if self.config.include_board_state:
            temp_features, feature_size = self._extract_board_state_features(game, player)
            features.extend(temp_features)
            self._feature_size += feature_size

        # Distance features
        if self.config.include_distances:
            temp_features, feature_size = self._extract_distance_features(game, player)
            features.extend(temp_features)
            self._feature_size += feature_size
        # Ticket features
        if self.config.include_tickets:
            temp_features, feature_size = self._extract_ticket_features(game, player)
            features.extend(temp_features)
            self._feature_size += feature_size
        # Transport connectivity features
        if self.config.include_transport_connectivity:
            temp_features, feature_size = self._extract_connectivity_features(game, player)
            features.extend(temp_features)
            self._feature_size += feature_size
        # Possible positions features
        if self.config.include_possible_positions:
            temp_features, feature_size = self._extract_possible_positions_features(game, player)
            features.extend(temp_features)
            self._feature_size += feature_size
        return np.array(features, dtype=np.float32)
    
    def _extract_game_phase_features(self, game: ScotlandYardGame) -> List[float]:
        """Extract features related to game phase and timing."""
        features = []
        
        # Current turn normalized
        max_turns = 24  # Standard Scotland Yard game length
        current_turn = game.game_state.MrX_turn_count if hasattr(game.game_state, 'MrX_turn_count') else 0
        features.append(min(current_turn / max_turns, 1.0))
        
        return features, len(features)
    
    def _extract_board_state_features(self, game: ScotlandYardGame, player: Player) -> List[float]:
        """Extract features representing board positions."""
        features = [0.0] * self.config.max_nodes
        
        # Mr. X position (if visible or if we're Mr. X)
        if hasattr(game.game_state, 'mr_x_visible') and game.game_state.mr_x_visible:
            mr_x_pos = game.game_state.MrX_position
            if 0 <= mr_x_pos < self.config.max_nodes:
                features[mr_x_pos] = 1.0
        elif player == Player.MRX:
            # If we are Mr. X, we always know our position
            mr_x_pos = game.game_state.MrX_position
            if 0 <= mr_x_pos < self.config.max_nodes:
                features[mr_x_pos] = 1.0
        
        # Detective positions (negative values to distinguish from Mr. X)
        for i, detective_pos in enumerate(game.game_state.detective_positions):
            if 0 <= detective_pos < self.config.max_nodes:
                features[detective_pos] = -1.0  # Negative for detectives
        
        return features, len(features)
    
    def _extract_distance_features(self, game: ScotlandYardGame, player: Player) -> List[float]:
        """Extract distance-based features."""
        features = []
        
        # Get positions
        mr_x_pos = game.game_state.MrX_position
        detective_positions = game.game_state.detective_positions
        
        # Distance from Mr. X to each detective (padded to 5 detectives)
        distances = []
        for i in range(5):  # Support up to 5 detectives
            if i < len(detective_positions):
                det_pos = detective_positions[i]
                try:
                    dist = self.heuristics.calculate_shortest_distance(mr_x_pos, det_pos)
                    distances.append(min(dist / self.config.distance_normalization, 1.0))
                except:
                    distances.append(0.5)  # Default if distance calculation fails
            else:
                distances.append(0.0)  # No detective at this index
        
        features.extend(distances)
        
        return features, len(features)
    
    def _extract_ticket_features(self, game: ScotlandYardGame, player: Player) -> List[float]:
        """Extract ticket-based features."""
        features = []
        
        # Mr. X tickets (normalized by typical starting amounts)
        mr_x_tickets = getattr(game.game_state, 'mr_x_tickets', {})
        features.append(mr_x_tickets.get(TicketType.TAXI, 0) / 4.0)
        features.append(mr_x_tickets.get(TicketType.BUS, 0) / 3.0)
        features.append(mr_x_tickets.get(TicketType.UNDERGROUND, 0) / 3.0)
        features.append(mr_x_tickets.get(TicketType.BLACK, 0) / 5.0)
        features.append(mr_x_tickets.get(TicketType.DOUBLE_MOVE, 0) / 2.0)
        
        # Detective tickets (padded to 5 detectives)
        detective_tickets = getattr(game.game_state, 'detective_tickets', {})
        for i in range(5):
            if i in detective_tickets:
                tickets = detective_tickets[i]
                features.append(tickets.get(TicketType.TAXI, 0) / 10.0)
                features.append(tickets.get(TicketType.BUS, 0) / 8.0)
                features.append(tickets.get(TicketType.UNDERGROUND, 0) / 4.0)
            else:
                features.extend([0.0, 0.0, 0.0])
        
        return features, len(features)  
    
    def _extract_connectivity_features(self, game: ScotlandYardGame, player: Player) -> List[float]:
        """Extract transport connectivity features."""
        features = []
        
        try:
            # Get current position based on player
            if player == Player.MRX:
                position = game.game_state.MrX_position
            else:
                # For detectives, use first detective position as representative
                position = game.game_state.detective_positions[0] if game.game_state.detective_positions else 1
            
            # Count connections by transport type
            graph = game.graph
            taxi_connections = 0
            bus_connections = 0
            underground_connections = 0
            
            if hasattr(graph, 'neighbors') and position in graph:
                for neighbor in graph.neighbors(position):
                    edge_data = graph.get_edge_data(position, neighbor)
                    if edge_data:
                        # Handle different edge data formats
                        if isinstance(edge_data, dict):
                            if 'transports' in edge_data:
                                transports = edge_data['transports']
                            elif 'edge_type' in edge_data:
                                transports = [edge_data['edge_type']]
                            else:
                                transports = [1]  # Default to taxi
                        else:
                            transports = [1]  # Default to taxi
                        
                        for transport in transports:
                            if transport == 1:  # Taxi
                                taxi_connections += 1
                            elif transport == 2:  # Bus
                                bus_connections += 1
                            elif transport == 3:  # Underground
                                underground_connections += 1
            
            # Normalize by max expected connections (rough estimates)
            features.append(min(taxi_connections / 5.0, 1.0))
            features.append(min(bus_connections / 3.0, 1.0))
            features.append(min(underground_connections / 2.0, 1.0))
            
            # Connectivity ratios
            total_connections = taxi_connections + bus_connections + underground_connections
            if total_connections > 0:
                features.append(taxi_connections / total_connections)
                features.append(bus_connections / total_connections)
                features.append(underground_connections / total_connections)
            else:
                features.extend([0.0, 0.0, 0.0])
            
            # Overall connectivity score
            features.append(min(total_connections / 10.0, 1.0))
            
            # Connectivity diversity (Shannon entropy)
            if total_connections > 0:
                p_taxi = taxi_connections / total_connections
                p_bus = bus_connections / total_connections  
                p_underground = underground_connections / total_connections
                
                entropy = 0.0
                for p in [p_taxi, p_bus, p_underground]:
                    if p > 0:
                        entropy -= p * np.log2(p)
                
                features.append(entropy / np.log2(3))  # Normalized entropy
            else:
                features.append(0.0)
        
        except Exception:
            # Fallback if connectivity calculation fails
            features = [0.0] * 9
        
        return features, len(features)
    
    def _extract_possible_positions_features(self, game: ScotlandYardGame, player: Player) -> List[float]:
        """Extract features related to possible Mr. X positions."""
        features = []
        
        try:
            # Get possible Mr. X positions from heuristics
            possible_positions = self.heuristics.get_possible_mr_x_positions()
            
            # Number of possible positions (normalized)
            features.append(min(len(possible_positions) / 50.0, 1.0))
            
            # Average distance from detectives to possible positions
            if possible_positions and game.game_state.detective_positions:
                total_dist = 0.0
                count = 0
                
                for pos in possible_positions:
                    for det_pos in game.game_state.detective_positions:
                        try:
                            dist = self.heuristics.get_shortest_distance(pos, det_pos)
                            total_dist += dist
                            count += 1
                        except:
                            continue
                
                if count > 0:
                    avg_dist = total_dist / count
                    features.append(min(avg_dist / self.config.distance_normalization, 1.0))
                else:
                    features.append(0.5)
            else:
                features.append(0.5)
            
            # Connectivity score of possible positions
            if possible_positions:
                total_connectivity = 0
                for pos in possible_positions:
                    if hasattr(game.graph, 'neighbors') and pos in game.graph:
                        total_connectivity += len(list(game.graph.neighbors(pos)))
                
                avg_connectivity = total_connectivity / len(possible_positions)
                features.append(min(avg_connectivity / 10.0, 1.0))
            else:
                features.append(0.0)
        
        except Exception:
            # Fallback values
            features = [0.5, 0.5, 0.5]
        
        return features, len(features)
    
    def get_feature_names(self) -> List[str]:
        """Get names of all features for debugging."""
        names = []
        
        if self.config.include_game_phase:
            names.extend(['turn_normalized', 'mr_x_visible', 'early_game', 'mid_game', 'late_game'])
        
        if self.config.include_board_state:
            names.extend([f'position_{i}' for i in range(self.config.max_nodes)])
        
        if self.config.include_distances:
            names.extend([f'dist_to_det_{i}' for i in range(5)])
            names.extend(['min_dist', 'max_dist', 'avg_dist'])
        
        if self.config.include_tickets:
            names.extend(['mrx_taxi', 'mrx_bus', 'mrx_underground', 'mrx_black', 'mrx_double'])
            for i in range(5):
                names.extend([f'det_{i}_taxi', f'det_{i}_bus', f'det_{i}_underground'])
        
        if self.config.include_transport_connectivity:
            names.extend(['taxi_conn', 'bus_conn', 'underground_conn', 
                         'taxi_ratio', 'bus_ratio', 'underground_ratio',
                         'total_conn', 'conn_diversity'])
        
        if self.config.include_possible_positions:
            names.extend(['num_possible_pos', 'avg_dist_to_possible', 'possible_conn_score'])
        
        return names
