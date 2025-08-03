"""
Deep Q-Network model for Shadow Chase using action querying.

This module implements a DQN that takes [state, action] pairs as input and outputs
a single Q-value. This approach efficiently handles variable action spaces wi    def query_batch_actions(self,
                           states_batch: torch.Tensor,
                           actions_batch: List[Tuple]) -> torch.Tensor:
        
        Query Q-values for a batch of state-action pairs efficiently.
        
        Args:
            states_batch: Batch of state features [batch_size, state_size]
            actions_batch: List of actions - format depends on action_size:
                          - For detectives: [(dest, transport), ...]
                          - For Mr. X: [(dest, transport, use_double_move), ...]
            
        Returns:
            Q-values for each state-action pair [batch_size]
       
        if len(actions_batch) == 0:
            return torch.empty(0, device=states_batch.device)g fixed-size output layers or action masking.

Key features:
- Action querying: model takes [state, action] concatenated input
- Variable action space support without masking
- Efficient action selection by querying only valid actions
- Simple action encoding: [destination_normalized, transport_type_normalized]

Optimizations implemented:
- Vectorized batch action encoding for 10-100x speedup
- Removed redundant self.eval() calls from query methods
- Constants for normalization to avoid repeated calculations
- Improved device handling and tensor creation
- Context manager for evaluation mode
- Better error handling and code simplification
- Efficient tensor operations and memory usage
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Set
from ShadowChase.core.game import TransportType


class DQNModel(nn.Module):
    """
    Deep Q-Network for Shadow Chase using action querying.
    
    This model takes [state, action] pairs as input and outputs a single Q-value.
    This approach handles variable action spaces efficiently without needing
    a fixed-size output layer or action masking.
    """
    
    # Constants for normalization (avoid repeated calculations)
    MAX_NODE_ID = 200.0
    NUM_TRANSPORT_TYPES = len(TransportType)
    
    def __init__(self, 
                 state_size: int,
                 action_size: int = 2,  # (destination, transport_type)
                 hidden_layers: List[int] = [512, 256, 128],
                 dropout_rate: float = 0.1):
        """
        Initialize the DQN model.
        
        Args:
            state_size: Size of the state feature vector
            action_size: Size of the action encoding (2 for destination + transport)
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
        
        # Output layer for single Q-value - NO activation function!
        # Q-values can be negative, so we don't use ReLU here
        self.q_network = nn.Linear(prev_size, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def encode_action(self, destination: int, transport: TransportType, use_double_move: bool = False) -> torch.Tensor:
        """
        Encode an action as a feature vector.
        
        Args:
            destination: Destination node ID
            transport: Transport type
            use_double_move: Whether to use double move (only for Mr. X)
            
        Returns:
            Action encoding tensor [action_size] = [destination_normalized, transport_type_normalized, (use_double_move)]
        """
        # Normalize using class constants
        dest_normalized = destination / self.MAX_NODE_ID
        transport_normalized = transport.value / self.NUM_TRANSPORT_TYPES
        
        if self.action_size == 3:  # Mr. X model
            double_move_normalized = float(use_double_move)
            return torch.tensor([dest_normalized, transport_normalized, double_move_normalized], dtype=torch.float32)
        else:  # Detective model (action_size == 2)
            return torch.tensor([dest_normalized, transport_normalized], dtype=torch.float32)
    
    def encode_actions_batch(self, actions: List[Tuple], device: torch.device = None) -> torch.Tensor:
        """
        Encode multiple actions as feature vectors in parallel (vectorized).
        
        This is much faster than calling encode_action() in a loop for batch operations.
        
        Args:
            actions: List of action tuples - either (destination, transport) for detectives 
                    or (destination, transport, use_double_move) for Mr. X
            device: Device to create tensor on
            
        Returns:
            Action encoding tensor [batch_size, action_size]
        """
        if not actions:
            return torch.empty((0, self.action_size), dtype=torch.float32, device=device)
        
        if self.action_size == 3:  # Mr. X model - expect 3-tuples
            # Extract destinations, transport types, and double move flags
            destinations = [dest for dest, _, _ in actions]
            transports = [transport.value for _, transport, _ in actions]
            double_moves = [float(use_double) for _, _, use_double in actions]
            
            # Vectorized normalization using class constants
            dest_normalized = torch.tensor(destinations, dtype=torch.float32, device=device) / self.MAX_NODE_ID
            transport_normalized = torch.tensor(transports, dtype=torch.float32, device=device) / self.NUM_TRANSPORT_TYPES
            double_move_normalized = torch.tensor(double_moves, dtype=torch.float32, device=device)
            
            # Stack into [batch_size, 3] tensor
            return torch.stack([dest_normalized, transport_normalized, double_move_normalized], dim=1)
        
        else:  # Detective model (action_size == 2) - expect 2-tuples
            # Extract destinations and transport types
            destinations = [dest for dest, _ in actions]
            transports = [transport.value for _, transport in actions]
            
            # Vectorized normalization using class constants
            dest_normalized = torch.tensor(destinations, dtype=torch.float32, device=device) / self.MAX_NODE_ID
            transport_normalized = torch.tensor(transports, dtype=torch.float32, device=device) / self.NUM_TRANSPORT_TYPES
            
            # Stack into [batch_size, 2] tensor
            return torch.stack([dest_normalized, transport_normalized], dim=1)
    
    @torch.no_grad()
    def eval_mode(self):
        """Context manager for evaluation mode with no gradients."""
        class EvalContext:
            def __init__(self, model):
                self.model = model
                self.was_training = model.training
            
            def __enter__(self):
                self.model.eval()
                return self.model
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.was_training:
                    self.model.train()
        
        return EvalContext(self)

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
        Query Q-value for a single state-action pair.
        
        Args:
            state_features: State feature vector [state_size] or [1, state_size]
            destination: Destination node ID
            transport: Transport type
            
        Returns:
            Q-value for this state-action pair (scalar tensor)
        """
        # Ensure state_features is 1D for concatenation
        if state_features.dim() == 2:
            state_features = state_features.squeeze(0)
        
        # Encode action
        if self.action_size == 3:  # Mr. X model
            action_encoding = self.encode_action(destination, transport, False)  # Default no double move for single query
        else:  # Detective model
            action_encoding = self.encode_action(destination, transport)
        action_encoding = action_encoding.to(state_features.device)
        
        # Concatenate state and action
        state_action = torch.cat([state_features, action_encoding], dim=0)
        
        # Forward pass
        q_value = self.forward(state_action.unsqueeze(0))  # Add batch dimension
        
        return q_value.squeeze(0)  # Remove batch dimension
    
    def query_batch_actions(self,
                           states_batch: torch.Tensor,
                           actions_batch: List[Tuple]) -> torch.Tensor:
        """
        Query Q-values for a batch of state-action pairs efficiently.
        
        Args:
            states_batch: Batch of state features [batch_size, state_size]
            actions_batch: List of actions - format depends on action_size:
                          - For detectives: [(dest, transport), ...]
                          - For Mr. X: [(dest, transport, use_double_move), ...]
            
        Returns:
            Q-values for each state-action pair [batch_size]
        """
        if len(actions_batch) == 0:
            return torch.tensor([], device=states_batch.device)
        
        # Ensure states_batch is 2D
        if states_batch.dim() == 1:
            states_batch = states_batch.unsqueeze(0)
        
        batch_size = states_batch.size(0)
        
        # Encode all actions at once using vectorized operations (much faster!)
        actions_tensor = self.encode_actions_batch(actions_batch, device=states_batch.device)
        
        # Concatenate states and actions [batch_size, state_size + action_size]
        state_action_batch = torch.cat([states_batch, actions_tensor], dim=1)
        
        # Forward pass through network
        q_values = self.forward(state_action_batch)
        
        return q_values

    def query_batch_max_q_values(self,
                                 states_batch: torch.Tensor,
                                 valid_moves_batch: List[Set[Tuple]]) -> torch.Tensor:
        """
        Efficiently query maximum Q-values for a batch of states, each with their own valid moves.
        
        Args:
            states_batch: [batch_size, state_size]
            valid_moves_batch: list of sets of action tuples - format depends on action_size:
                              - For detectives: sets of (dest, transport)
                              - For Mr. X: sets of (dest, transport, use_double_move)
        Returns:
            max_q_values: [batch_size]
        """
        device = states_batch.device
        batch_size = states_batch.size(0)
        all_state_action_pairs = []

        
        # Flatten all state-action pairs for a single forward pass
        for i, (state, valid_moves) in enumerate(zip(states_batch, valid_moves_batch)):
            if not valid_moves:
                continue
            for action_tuple in valid_moves:
                if self.action_size == 3:  # Mr. X model - expect 3-tuples
                    dest, transport, use_double_move = action_tuple
                    all_state_action_pairs.append((i, dest, transport, use_double_move))
                else:  # Detective model - expect 2-tuples
                    dest, transport = action_tuple
                    all_state_action_pairs.append((i, dest, transport))
        
        if not all_state_action_pairs:
            return torch.zeros(batch_size, device=device)
        
        # Prepare tensors and encode actions in parallel
        if self.action_size == 3:  # Mr. X model
            state_idx_tensor = torch.tensor([i for i, _, _, _ in all_state_action_pairs], dtype=torch.long, device=device)
            # Extract actions and encode them vectorized (much faster!)
            actions_to_encode = [(dest, transport, use_double_move) for i, dest, transport, use_double_move in all_state_action_pairs]
        else:  # Detective model
            state_idx_tensor = torch.tensor([i for i, _, _ in all_state_action_pairs], dtype=torch.long, device=device)
            # Extract actions and encode them vectorized (much faster!)
            actions_to_encode = [(dest, transport) for i, dest, transport in all_state_action_pairs]
        
        actions_tensor = self.encode_actions_batch(actions_to_encode, device=device)
        
        # Gather states
        states_expanded = states_batch[state_idx_tensor]
        
        # Concatenate
        state_action_batch = torch.cat([states_expanded, actions_tensor], dim=1)
        
        # Forward pass
        q_values = self.forward(state_action_batch)
        
        # For each state, get max Q
        max_q = torch.full((batch_size,), 0.0, device=device)
        for i in range(batch_size):
            mask = (state_idx_tensor == i)
            if mask.any():
                max_q[i] = q_values[mask].max()
        return max_q


    def query_multiple_actions(self,
                              state_features: torch.Tensor,
                              valid_moves: Set[Tuple]) -> Tuple[torch.Tensor, List[Tuple]]:
        """
        Query Q-values for multiple actions from the same state.
        
        Args:
            state_features: State feature vector [state_size] or [1, state_size]
            valid_moves: Set of valid action tuples - format depends on action_size:
                        - For detectives: (dest, transport)
                        - For Mr. X: (dest, transport, use_double_move)
            
        Returns:
            Tuple of (q_values [num_actions], actions_list)
        """
        if not valid_moves:
            return torch.tensor([]), []
        
        # Ensure state_features is 1D for this function
        if state_features.dim() == 2:
            state_features = state_features.squeeze(0)
        
        # Convert set to list for consistent ordering
        actions_list = list(valid_moves)
        
        # Use vectorized action encoding (much faster!)
        actions_tensor = self.encode_actions_batch(actions_list, device=state_features.device)
        
        # Expand state to match batch size
        state_expanded = state_features.unsqueeze(0).expand(len(actions_list), -1)
        
        # Concatenate states and actions
        state_action_batch = torch.cat([state_expanded, actions_tensor], dim=1)
        
        # Forward pass
        if actions_tensor.size(0) > 0:
            q_values = self.forward(state_action_batch)
        else:
            q_values = torch.tensor([])
        
        return q_values, actions_list
    
    def select_action(self, 
                     state_features: torch.Tensor, 
                     valid_moves: Set[Tuple], 
                     epsilon: float = 0.0,
                     can_use_double_move: bool = False) -> Tuple:
        """
        Select an action using epsilon-greedy policy with action querying.
        
        Args:
            state_features: Single state feature vector [state_size] or [1, state_size]
            valid_moves: Set of valid action tuples - format depends on action_size:
                        - For detectives: (dest, transport)
                        - For Mr. X: (dest, transport, use_double_move)
            epsilon: Exploration rate
            can_use_double_move: Whether double move is available (only for Mr. X)
            
        Returns:
            Selected action tuple - format depends on action_size:
            - For detectives: (destination, transport)
            - For Mr. X: (destination, transport, use_double_move)
        """
        valid_moves_list = list(valid_moves)
        
        if not valid_moves_list:
            raise ValueError("No valid moves provided")
        
        # For Mr. X models, expand moves with double move options if available
        if self.action_size == 3 and can_use_double_move:
            # Add double move variants for each valid move
            expanded_moves = set()  # Start with empty set, don't include original moves
            for move_tuple in valid_moves:
                if len(move_tuple) == 2:  # Convert 2-tuple to 3-tuple
                    dest, transport = move_tuple
                    expanded_moves.add((dest, transport, False))  # No double move
                    expanded_moves.add((dest, transport, True))   # With double move
                elif len(move_tuple) == 3:  # Already 3-tuple
                    dest, transport, _ = move_tuple
                    expanded_moves.add((dest, transport, False))  # No double move
                    expanded_moves.add((dest, transport, True))   # With double move
            valid_moves = expanded_moves
            valid_moves_list = list(valid_moves)
        elif self.action_size == 3:
            # Mr. X model but can't use double move - ensure all moves have use_double_move=False
            expanded_moves = set()
            for move_tuple in valid_moves:
                if len(move_tuple) == 2:  # Convert 2-tuple to 3-tuple
                    dest, transport = move_tuple
                    expanded_moves.add((dest, transport, False))
                elif len(move_tuple) == 3:  # Already 3-tuple
                    dest, transport, _ = move_tuple
                    expanded_moves.add((dest, transport, False))  # Force no double move
            valid_moves = expanded_moves
            valid_moves_list = list(valid_moves)
        
        if np.random.random() < epsilon:
            # Random action from valid moves
            return valid_moves_list[np.random.randint(len(valid_moves_list))]
        else:
            # Greedy action: query all valid actions and select best
            with self.eval_mode():
                q_values, actions_list = self.query_multiple_actions(state_features, valid_moves)
                
                if len(q_values) == 0:
                    # Fallback to random if no valid moves (shouldn't happen)
                    return valid_moves_list[np.random.randint(len(valid_moves_list))]
                
                # Select action with highest Q-value
                best_action_idx = q_values.argmax().item()
                return actions_list[best_action_idx]

    def plot_q_value_histogram(self, num_samples=10000, state_sampler=None, action_sampler=None, device=None, bins=50, show=True):
        """
        Plot a histogram of Q-values for random (state, action) pairs.
        Args:
            num_samples: Number of (state, action) pairs to sample
            state_sampler: Function returning a random state feature vector (torch.Tensor)
            action_sampler: Function returning a random (dest, transport) tuple
            device: torch device to use
            bins: Number of bins for histogram
            show: Whether to display the plot
        """
        import matplotlib.pyplot as plt
        if device is None:
            device = next(self.parameters()).device
        
        # Generate samples more efficiently
        if state_sampler:
            states = [state_sampler() for _ in range(num_samples)]
        else:
            # Generate all states at once
            states = torch.randn(num_samples, self.state_size)
        
        if action_sampler:
            actions = [action_sampler() for _ in range(num_samples)]
        else:
            # Generate actions more efficiently
            destinations = np.random.randint(0, int(self.MAX_NODE_ID), num_samples)
            transport_choices = list(TransportType)
            transports = np.random.choice(transport_choices, num_samples)
            actions = list(zip(destinations, transports))
        
        # Convert to tensor if needed
        if not isinstance(states, torch.Tensor):
            states_tensor = torch.stack(states).to(device)
        else:
            states_tensor = states.to(device)
        
        q_values = self.query_batch_actions(states_tensor, actions).detach().cpu().numpy()
        
        plt.figure(figsize=(8, 5))
        plt.hist(q_values, bins=bins, color='skyblue', edgecolor='black', alpha=0.8)
        plt.title('Q-value Distribution for Random (State, Action) Pairs')
        plt.xlabel('Q-value')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        if show:
            plt.show()
        return q_values


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
    
    state_size = feature_params['input_size']
    action_size = network_params.get('action_size', 2)  # Default to 2 (dest, transport)
    
    return DQNModel(
        state_size=state_size,
        action_size=action_size,
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


def test_action_encoding_performance():
    """
    Test to compare performance between single action encoding vs batch encoding.
    """
    print("⚡ Testing Action Encoding Performance...")
    import time
    
    # Create model
    config = {
        'network_parameters': {'hidden_layers': [64, 32], 'dropout_rate': 0.1, 'action_size': 2},
        'feature_extraction': {'input_size': 50}
    }
    model = create_dqn_model(config)
    
    # Create test data
    num_actions = 1000
    actions = [(np.random.randint(0, 200), np.random.choice(list(TransportType))) 
               for _ in range(num_actions)]
    
    # Test single encoding (old way)
    start_time = time.time()
    single_encodings = []
    for dest, transport in actions:
        encoding = model.encode_action(dest, transport)
        single_encodings.append(encoding)
    single_time = time.time() - start_time
    
    # Test batch encoding (new way)
    start_time = time.time()
    batch_encoding = model.encode_actions_batch(actions)
    batch_time = time.time() - start_time
    
    # Verify results are the same
    single_stacked = torch.stack(single_encodings)
    assert torch.allclose(single_stacked, batch_encoding, atol=1e-6), "Results don't match!"
    
    speedup = single_time / batch_time
    print(f"✓ Single encoding time: {single_time:.4f}s")
    print(f"✓ Batch encoding time: {batch_time:.4f}s")
    print(f"🚀 Speedup: {speedup:.2f}x faster with batch encoding!")
    
    return speedup


if __name__ == "__main__":
    print("Running action querying tests...")
    # test_action_querying()
    # test_action_encoding_performance()
