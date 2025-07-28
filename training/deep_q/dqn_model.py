"""
Deep Q-Network model for Scotland Yard using action querying.

This module implements a DQN that takes [state, action] pairs as input and outputs
a single Q-value. This approach efficiently handles variable action spaces without
requiring fixed-size output layers or action masking.

Key features:
- Action querying: model takes [state, action] concatenated input
- Variable action space support without masking
- Efficient action selection by querying only valid actions
- Simple action encoding: [destination_normalized, transport_type_normalized]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from ScotlandYard.core.game import TransportType


class DQNModel(nn.Module):
    """
    Deep Q-Network for Scotland Yard using action querying.
    
    This model takes [state, action] pairs as input and outputs a single Q-value.
    This approach handles variable action spaces efficiently without needing
    a fixed-size output layer or action masking.
    """
    
    def __init__(self, 
                 state_size: int,
                 action_size: int = 2,  # (destination, transport_type)
                 hidden_layers: List[int] = [512, 256, 128],
                 dropout_rate: float = 0.1):
        """
        Initialize the DQN model.
        
        Args:
            state_size: Size of the state feature vector
            action_size: Size of the action encoding (destination + transport)
            hidden_layers: List of hidden layer sizes
            dropout_rate: Dropout rate for regularization
        """
        super(DQNModel, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        self.dropout_rate = dropout_rate
        
        # Input size is state + action concatenated
        input_size = state_size + action_size
        
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
        
        # Output layer for single Q-value
        self.q_network = nn.Linear(prev_size, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def encode_action(self, destination: int, transport: TransportType) -> torch.Tensor:
        """
        Encode an action as a feature vector.
        
        Args:
            destination: Destination node ID
            transport: Transport type
            
        Returns:
            Action encoding tensor [action_size]
        """
        # Simple encoding: [destination_normalized, transport_type_normalized]
        # Normalize destination to [0, 1] range (assuming max 200 nodes)
        dest_normalized = destination / 200.0
        
        # Normalize transport type to [0, 1] range
        transport_normalized = transport.value / len(TransportType)
        
        return torch.tensor([dest_normalized, transport_normalized], dtype=torch.float32)
    
    def forward(self, state_action_pairs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.
        
        Args:
            state_action_pairs: Batch of [state, action] concatenated vectors 
                               [batch_size, state_size + action_size]
            
        Returns:
            Q-values for each state-action pair [batch_size, 1]
        """
        features = self.feature_network(state_action_pairs)
        q_values = self.q_network(features)
        return q_values.squeeze(-1)  # Remove last dimension to get [batch_size]
    
    def query_action(self, 
                    state_features: torch.Tensor, 
                    destination: int, 
                    transport: TransportType) -> torch.Tensor:
        """
        Query Q-value for a specific state-action pair.
        
        Args:
            state_features: State feature vector [state_size] or [1, state_size]
            destination: Destination node
            transport: Transport type
            
        Returns:
            Q-value for this state-action pair [1] or scalar
        """
        # Ensure state_features is 2D
        if state_features.dim() == 1:
            state_features = state_features.unsqueeze(0)
        
        # Encode the action
        action_encoding = self.encode_action(destination, transport)
        action_encoding = action_encoding.unsqueeze(0).to(state_features.device)
        
        # Concatenate state and action
        state_action = torch.cat([state_features, action_encoding], dim=1)
        
        # Get Q-value
        q_value = self.forward(state_action)
        return q_value
    
    def query_batch_actions(self,
                           states_batch: torch.Tensor,
                           actions_batch: List[Tuple[int, TransportType]]) -> torch.Tensor:
        """
        Query Q-values for a batch of state-action pairs efficiently.
        
        Args:
            states_batch: Batch of state features [batch_size, state_size]
            actions_batch: List of actions, one per state [(dest, transport), ...]
            
        Returns:
            Q-values for each state-action pair [batch_size]
        """
        if len(actions_batch) == 0:
            return torch.tensor([])
        
        # Ensure states_batch is 2D
        if states_batch.dim() == 1:
            states_batch = states_batch.unsqueeze(0)
        
        batch_size = states_batch.size(0)
        
        # Encode all actions at once
        action_encodings = []
        for dest, transport in actions_batch:
            action_encoding = self.encode_action(dest, transport)
            action_encodings.append(action_encoding)
        
        # Stack action encodings into batch [batch_size, action_size]
        actions_tensor = torch.stack(action_encodings).to(states_batch.device)
        
        # Concatenate states and actions [batch_size, state_size + action_size]
        state_action_batch = torch.cat([states_batch, actions_tensor], dim=1)
        
        # Forward pass through network
        q_values = self.forward(state_action_batch)
        
        return q_values

    def query_batch_max_q_values(self,
                                 states_batch: torch.Tensor,
                                 valid_moves_batch: List[Set[Tuple[int, TransportType]]]) -> torch.Tensor:
        """
        Query maximum Q-values for a batch of states, each with their own valid moves.
        
        Args:
            states_batch: Batch of state features [batch_size, state_size]
            valid_moves_batch: List of valid moves for each state
            
        Returns:
            Maximum Q-values for each state [batch_size]
        """
        batch_size = states_batch.size(0)
        max_q_values = []
        
        for i in range(batch_size):
            state = states_batch[i]
            valid_moves = valid_moves_batch[i]
            
            if not valid_moves:
                max_q_values.append(torch.tensor(0.0, device=states_batch.device))
            else:
                q_vals, _ = self.query_multiple_actions(state, valid_moves)
                if len(q_vals) > 0:
                    max_q_values.append(q_vals.max())
                else:
                    max_q_values.append(torch.tensor(0.0, device=states_batch.device))
        
        return torch.stack(max_q_values)

    def query_multiple_actions(self,
                              state_features: torch.Tensor,
                              valid_moves: Set[Tuple[int, TransportType]]) -> Tuple[torch.Tensor, List[Tuple[int, TransportType]]]:
        """
        Query Q-values for multiple actions from the same state.
        
        Args:
            state_features: State feature vector [state_size] or [1, state_size]
            valid_moves: Set of valid (destination, transport) pairs
            
        Returns:
            Tuple of (q_values [num_actions], actions_list)
        """
        if not valid_moves:
            return torch.tensor([]), []
        
        # Ensure state_features is 1D for this function
        if state_features.dim() == 2:
            state_features = state_features.squeeze(0)
        
        # Convert valid moves to list for consistent ordering
        actions_list = list(valid_moves)
        
        # Create batch of state-action pairs
        state_action_pairs = []
        for dest, transport in actions_list:
            action_encoding = self.encode_action(dest, transport)
            state_action = torch.cat([state_features, action_encoding.to(state_features.device)], dim=0)
            state_action_pairs.append(state_action)
        
        # Stack into batch
        if state_action_pairs:
            batch = torch.stack(state_action_pairs).to(state_features.device)
            q_values = self.forward(batch)
        else:
            q_values = torch.tensor([])
        
        return q_values, actions_list
    
    def select_action(self, 
                     state_features: torch.Tensor, 
                     valid_moves: Set[Tuple[int, TransportType]], 
                     epsilon: float = 0.0) -> Tuple[int, TransportType]:
        """
        Select an action using epsilon-greedy policy with action querying.
        
        Args:
            state_features: Single state feature vector [state_size] or [1, state_size]
            valid_moves: Set of valid (destination, transport) pairs
            epsilon: Exploration rate
            
        Returns:
            Selected (destination, transport) pair
        """
        if np.random.random() < epsilon:
            # Random action from valid moves
            chosen_move = np.random.choice(len(valid_moves))
            return list(valid_moves)[chosen_move]
        else:
            # Greedy action: query all valid actions and select best
            with torch.no_grad():
                q_values, actions_list = self.query_multiple_actions(state_features, valid_moves)
                
                if len(q_values) == 0:
                    # Fallback to random if no valid moves
                    chosen_move = np.random.choice(len(valid_moves))
                    return list(valid_moves)[chosen_move]
                
                # Select action with highest Q-value
                best_action_idx = q_values.argmax().item()
                return actions_list[best_action_idx]
    
    def get_action_index(self, destination: int, transport: TransportType) -> int:
        """
        Legacy method for compatibility. 
        Note: This is less meaningful with action querying architecture.
        """
        # Keep for backward compatibility with existing code
        return destination * 5 + transport.value  # 5 transport types
    
    def get_action_from_index(self, action_index: int) -> Tuple[int, TransportType]:
        """
        Legacy method for compatibility.
        Note: This is less meaningful with action querying architecture.
        """
        # Keep for backward compatibility
        destination = action_index // 5
        transport_value = action_index % 5
        transport = TransportType(transport_value)
        return destination, transport


# class DoubleDQNModel(DQNModel):
#     """
#     Double DQN variant that uses separate networks for action selection and evaluation.
#     This helps reduce overestimation bias in Q-learning.
#     """
    
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
        
#     def forward_target(self, state_features: torch.Tensor) -> torch.Tensor:
#         """
#         Forward pass through target network (same as main network for now).
#         In practice, this would be a separate target network.
#         """
#         return self.forward(state_features)


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
    
    # Use actual state size if available (from saved model), otherwise calculate estimate
    if 'input_size' in feature_params:
        state_size = feature_params['input_size']
    else:
        # Calculate state size based on feature extraction config
        # This is a rough estimate - should match GameFeatureExtractor output
        state_size = (
            feature_params.get('max_nodes', 200) +  # Board state features
            10 +  # Ticket features (rough estimate)
            5 +   # Game phase features
            20 +  # Distance features (rough estimate)
            20    # Additional features
        )
    
    return DQNModel(
        state_size=state_size,
        action_size=network_params.get('action_size', 2),  # (destination, transport)
        hidden_layers=network_params.get('hidden_layers', [512, 256, 128]),
        dropout_rate=network_params.get('dropout_rate', 0.1)
    )


def test_action_querying():
    """
    Simple test to verify action querying functionality works correctly.
    """
    print("🧪 Testing Action Querying DQN Model...")
    
    # Create a simple model for testing
    config = {
        'network_parameters': {
            'hidden_layers': [64, 32],
            'dropout_rate': 0.1,
            'action_size': 2
        },
        'feature_extraction': {
            'input_size': 50  # Small state size for testing
        }
    }
    
    model = create_dqn_model(config)
    model.eval()
    
    # Create sample state
    state = torch.randn(50)  # Random state features
    
    # Test single action query
    dest, transport = 42, TransportType.TAXI
    q_value = model.query_action(state, dest, transport)
    print(f"✓ Single action query: destination={dest}, transport={transport.name}, Q={q_value.item():.4f}")
    
    # Test multiple action queries
    valid_moves = {
        (42, TransportType.TAXI),
        (35, TransportType.BUS),
        (67, TransportType.UNDERGROUND),
        (15, TransportType.BLACK)
    }
    
    q_values, actions_list = model.query_multiple_actions(state, valid_moves)
    print(f"✓ Multiple action queries: {len(actions_list)} actions")
    for i, ((dest, transport), q_val) in enumerate(zip(actions_list, q_values)):
        print(f"   Action {i}: dest={dest}, transport={transport.name}, Q={q_val.item():.4f}")
    
    # Test action selection
    selected_action = model.select_action(state, valid_moves, epsilon=0.0)  # Greedy
    print(f"✓ Greedy action selection: {selected_action}")
    
    # Test with random exploration
    selected_action_random = model.select_action(state, valid_moves, epsilon=1.0)  # Random
    print(f"✓ Random action selection: {selected_action_random}")
    
    print("✅ Action querying tests passed!")


if __name__ == "__main__":
    test_action_querying()
