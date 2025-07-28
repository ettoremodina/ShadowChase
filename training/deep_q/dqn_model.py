"""
Deep Q-Network model for Scotland Yard.

This module implements a simple neural network for Q-learning with variable action spaces.
The key innovation is handling the variable number of valid moves per game state.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from ScotlandYard.core.game import TransportType


class DQNModel(nn.Module):
    """
    Simple Deep Q-Network for Scotland Yard.
    
    Handles variable action spaces by:
    1. Using a fixed-size output layer for all possible destination-transport combinations
    2. Masking invalid actions during training and inference
    3. Using separate heads for Mr. X and Detective agents
    """
    
    def __init__(self, 
                 input_size: int,
                 max_nodes: int = 200,
                 hidden_layers: List[int] = [512, 256, 128],
                 dropout_rate: float = 0.1):
        """
        Initialize the DQN model.
        
        Args:
            input_size: Size of the feature vector
            max_nodes: Maximum number of nodes in the graph (for action space)
            hidden_layers: List of hidden layer sizes
            dropout_rate: Dropout rate for regularization
        """
        super(DQNModel, self).__init__()
        
        self.input_size = input_size
        self.max_nodes = max_nodes
        self.dropout_rate = dropout_rate
        
        # Number of transport types (taxi, bus, underground, black, double for Mr. X)
        self.num_transports = len(TransportType)
        
        # Total possible actions: each node x each transport type
        # We'll use action masking to handle invalid combinations
        self.max_actions = max_nodes * self.num_transports
        
        # Build the network
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_layers:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Remove the last dropout
        if layers:
            layers = layers[:-1]
        
        self.feature_network = nn.Sequential(*layers)
        
        # Output layer for Q-values
        self.q_network = nn.Linear(prev_size, self.max_actions)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, state_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.
        
        Args:
            state_features: Batch of state feature vectors [batch_size, input_size]
            
        Returns:
            Q-values for all possible actions [batch_size, max_actions]
        """
        features = self.feature_network(state_features)
        q_values = self.q_network(features)
        return q_values
    
    def get_action_index(self, destination: int, transport: TransportType) -> int:
        """
        Convert (destination, transport) to a single action index.
        
        Args:
            destination: Destination node (0 to max_nodes-1)
            transport: Transport type
            
        Returns:
            Action index for the neural network output
        """
        return destination * self.num_transports + transport.value
    
    def get_action_from_index(self, action_index: int) -> Tuple[int, TransportType]:
        """
        Convert action index back to (destination, transport).
        
        Args:
            action_index: Index from neural network output
            
        Returns:
            Tuple of (destination, transport_type)
        """
        destination = action_index // self.num_transports
        transport_value = action_index % self.num_transports
        transport = TransportType(transport_value)
        return destination, transport
    
    def get_masked_q_values(self, 
                           state_features: torch.Tensor, 
                           valid_moves: List[Set[Tuple[int, TransportType]]]) -> torch.Tensor:
        """
        Get Q-values with invalid actions masked to negative infinity.
        
        Args:
            state_features: Batch of state features [batch_size, input_size]
            valid_moves: List of valid moves for each state in the batch
            
        Returns:
            Masked Q-values [batch_size, max_actions]
        """
        q_values = self.forward(state_features)
        batch_size = q_values.size(0)
        
        # Create mask for valid actions
        mask = torch.full_like(q_values, float('-inf'))
        
        for i, moves in enumerate(valid_moves):
            for dest, transport in moves:
                if dest < self.max_nodes:  # Safety check
                    action_idx = self.get_action_index(dest, transport)
                    if action_idx < self.max_actions:  # Safety check
                        mask[i, action_idx] = 0.0
        
        # Apply mask (add 0 for valid actions, -inf for invalid ones)
        masked_q_values = q_values + mask
        return masked_q_values
    
    def select_action(self, 
                     state_features: torch.Tensor, 
                     valid_moves: Set[Tuple[int, TransportType]], 
                     epsilon: float = 0.0) -> Tuple[int, TransportType]:
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state_features: Single state feature vector [1, input_size]
            valid_moves: Set of valid (destination, transport) pairs
            epsilon: Exploration rate
            
        Returns:
            Selected (destination, transport) pair
        """
        if np.random.random() < epsilon:
            # Random action from valid moves
            return np.random.choice(list(valid_moves))
        else:
            # Greedy action
            with torch.no_grad():
                masked_q_values = self.get_masked_q_values(state_features, [valid_moves])
                action_idx = masked_q_values.argmax().item()
                return self.get_action_from_index(action_idx)


class DoubleDQNModel(DQNModel):
    """
    Double DQN variant that uses separate networks for action selection and evaluation.
    This helps reduce overestimation bias in Q-learning.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward_target(self, state_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through target network (same as main network for now).
        In practice, this would be a separate target network.
        """
        return self.forward(state_features)


def create_dqn_model(config: Dict) -> DQNModel:
    """
    Factory function to create a DQN model from configuration.
    
    Args:
        config: Configuration dictionary with network parameters
        
    Returns:
        Initialized DQN model
    """
    network_params = config.get('network_parameters', {})
    feature_params = config.get('feature_extraction', {})
    
    # Use actual input size if available (from saved model), otherwise calculate estimate
    if 'input_size' in feature_params:
        input_size = feature_params['input_size']
    else:
        # Calculate input size based on feature extraction config
        # This is a rough estimate - should match GameFeatureExtractor output
        input_size = (
            feature_params.get('max_nodes', 200) +  # Board state features
            10 +  # Ticket features (rough estimate)
            5 +   # Game phase features
            20 +  # Distance features (rough estimate)
            20    # Additional features
        )
    
    return DQNModel(
        input_size=input_size,
        max_nodes=feature_params.get('max_nodes', 200),
        hidden_layers=network_params.get('hidden_layers', [512, 256, 128]),
        dropout_rate=network_params.get('dropout_rate', 0.1)
    )
