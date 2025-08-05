"""
PettingZoo environment for Shadow Chase game.

This environment wraps the Shadow Chase game to be compatible with the PettingZoo API,
enabling multi-agent reinforcement learning experiments.
"""

from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import agent_selector
from pettingzoo.utils import wrappers
import os

from ShadowChase.core.game import ShadowChaseGame, Player, TransportType, TicketType
from ShadowChase.services.game_service import GameService
from training.feature_extractor_simple import GameFeatureExtractor, FeatureConfig
from game_controls.game_utils import create_and_initialize_game


class ShadowChaseEnv(AECEnv):
    """
    PettingZoo environment for Shadow Chase game.
    
    This environment treats the game as a 2-player asymmetric game:
    - Player "mrx": Controls Mr. X
    - Player "detectives": Controls all detectives as a single agent
    
    The environment handles the complex turn structure including double moves
    and sequential detective movements internally.
    """
    
    metadata = {
        "render_modes": ["human", "ansi"],
        "name": "shadow_chase_v1",
        "is_parallelizable": False,  # Sequential turns
    }
    
    def __init__(
        self,
        map_size: str = "test",
        num_detectives: int = 2,
        max_turns: int = 24,
        render_mode: Optional[str] = None,
        feature_config: Optional[FeatureConfig] = None,
    ):
        """
        Initialize Shadow Chase PettingZoo environment.
        
        Args:
            map_size: Size of the map ("test" or "full")
            num_detectives: Number of detectives (2-5)
            max_turns: Maximum number of turns before draw
            render_mode: Rendering mode ("human" or "ansi")
            feature_config: Configuration for feature extraction
        """
        super().__init__()
        
        # Game parameters
        self.map_size = map_size
        self.num_detectives = num_detectives
        self.max_turns = max_turns
        self.render_mode = render_mode
        
        # Initialize feature extractor
        if feature_config is None:
            feature_config = FeatureConfig()
        self.feature_extractor = GameFeatureExtractor(feature_config)
        
        # Agent names - mrx and individual detectives for sequential movement
        self.possible_agents = ["mrx"] + [f"detective_{i}" for i in range(num_detectives)]
        
        # Initialize game
        self._init_game()
        
        # Action and observation spaces
        self._setup_spaces()
        
        # Sequential movement state
        self.pending_detective_moves = []  # Store moves made by detectives in current turn
        self.current_detective_turn = 0    # Which detective is currently moving
        
        # Agent selector for turn management
        self.agent_selector = agent_selector(self.possible_agents)
        
        # Initialize GameService for saving completed games
        self._setup_game_service()
        
        # Environment state
        self.reset()
    
    def _init_game(self):
        """Initialize the Shadow Chase game instance."""
        self.game = create_and_initialize_game(self.map_size, self.num_detectives)
        
        # Get graph info for action space sizing
        self.num_nodes = len(self.game.graph.nodes())
        self.transport_types = [TransportType.TAXI, TransportType.BUS, TransportType.UNDERGROUND]  # Only basic transports for detectives
        self.mrx_transport_types = [t for t in TransportType]  # All transports for MrX
        
        # Calculate maximum number of neighbors for reduced action space
        self.max_neighbors = max(len(list(self.game.graph.neighbors(node))) for node in self.game.graph.nodes())
        print(f"Graph has {self.num_nodes} nodes, max neighbors: {self.max_neighbors}")
        
    def _setup_game_service(self):
        """Setup GameService for saving completed games."""
        # Create PZgames directory if it doesn't exist
        
        # Initialize GameService with custom save directory
        from ShadowChase.services.game_loader import GameLoader
        loader = GameLoader(base_directory="PZgames")
        self.game_service = GameService(loader=loader)
        self.game_saved = False  # Track if current game has been saved
        
    def _setup_spaces(self):
        """Setup action and observation spaces for both agents."""
        # Get feature dimensions
        dummy_features = self.feature_extractor.extract_features(self.game, Player.MRX)
        feature_dim = len(dummy_features)
        
        # Observation space - same for all agents (full game state features)
        self.observation_spaces = {
            agent: spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=(feature_dim,), 
                dtype=np.float32
            )
            for agent in self.possible_agents
        }
        
        # Action spaces using relative/local encoding
        # Mr. X: (neighbor_index, transport_type, use_double_move)
        # neighbor_index: 0=stay, 1-max_neighbors=move to neighbor
        mrx_action_space = spaces.MultiDiscrete([
            self.max_neighbors + 1,           # neighbor choice (0=stay, 1-N=neighbors)
            len(self.mrx_transport_types),    # transport type
            2                                 # use_double_move (0=False, 1=True)
        ])
        
        # Detective: (neighbor_index, transport_type)
        # Only basic transports (taxi, bus, underground) for detectives
        detective_action_space = spaces.MultiDiscrete([
            self.max_neighbors + 1,           # neighbor choice (0=stay, 1-N=neighbors)
            len(self.transport_types)         # transport type (taxi, bus, underground)
        ])
        
        # Build action spaces dict
        self.action_spaces = {"mrx": mrx_action_space}
        for i in range(self.num_detectives):
            self.action_spaces[f"detective_{i}"] = detective_action_space
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset the environment to initial state."""
        if seed is not None:
            np.random.seed(seed)
        
        # Reset game
        self._init_game()
        
        # Initialize agent states
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        
        # Initialize PettingZoo required attributes
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        
        # Reset sequential movement state
        self.pending_detective_moves = []
        self.current_detective_turn = 0
        
        # Reset game saving state
        self.game_saved = False
        
        # Reset agent selector
        self.agent_selector.reinit(self.agents)
        
        # Set initial agent based on who starts (should be MrX in Shadow Chase)
        if self.game.game_state.turn == Player.MRX:
            self.agent_selection = "mrx"
        else:
            # Start with first detective
            self.agent_selection = "detective_0"
        
        self.agent_selector.reset()
        
        # Initialize step count
        self.step_count = 0
        
        return self._get_obs(), self._get_info()
    
    def _get_neighbors_list(self, position: int) -> List[int]:
        """Get ordered list of neighbors for a position."""
        return sorted(list(self.game.graph.neighbors(position)))
    
    def _relative_action_to_absolute(self, position: int, neighbor_idx: int, transport_idx: int, is_mrx: bool = False) -> Tuple[int, TransportType]:
        """Convert relative action to absolute position and transport."""
        if neighbor_idx == 0:
            # Stay in place (though this might not be valid in Shadow Chase)
            return position, (self.mrx_transport_types if is_mrx else self.transport_types)[transport_idx]
        
        neighbors = self._get_neighbors_list(position)
        if neighbor_idx > len(neighbors):
            # Invalid neighbor index, stay in place
            return position, (self.mrx_transport_types if is_mrx else self.transport_types)[transport_idx]
        
        dest = neighbors[neighbor_idx - 1]  # neighbor_idx is 1-based
        transport = (self.mrx_transport_types if is_mrx else self.transport_types)[transport_idx]
        return dest, transport
    
    def _get_valid_relative_actions(self, position: int, is_mrx: bool = False, pending_moves: List[Tuple[int, TransportType]] = None) -> List[Tuple[int, int]]:
        """Get list of valid (neighbor_idx, transport_idx) pairs for relative actions."""
        valid_actions = []
        
        if is_mrx:
            # Get valid moves for MrX
            valid_moves = self.game.get_valid_moves(Player.MRX)
            transport_types = self.mrx_transport_types
        else:
            # Get valid moves for detective
            valid_moves = self.game.get_valid_moves(Player.DETECTIVES, position, pending_moves=pending_moves)
            transport_types = self.transport_types
        
        neighbors = self._get_neighbors_list(position)
        
        for dest, transport in valid_moves:
            if dest == position:
                # Staying in place
                neighbor_idx = 0
            else:
                # Moving to neighbor
                if dest in neighbors:
                    neighbor_idx = neighbors.index(dest) + 1  # 1-based indexing
                else:
                    continue  # Destination not in neighbors list (shouldn't happen)
            
            # Find transport index
            if transport in transport_types:
                transport_idx = transport_types.index(transport)
                valid_actions.append((neighbor_idx, transport_idx))
        
        return valid_actions
    
    def step(self, action):
        """Execute one step of the environment."""
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            return self._was_dead_step(action)
        
        agent = self.agent_selection
        
        # Execute action
        success = self._execute_action(agent, action)
        
        if not success:
            # Invalid action - penalize and continue
            self.rewards[agent] = -1.0
            self.infos[agent] = {"error": "invalid_action"}
        else:
            # Calculate rewards
            self._calculate_rewards()
            
            # Update cumulative rewards
            for agent in self.agents:
                self._cumulative_rewards[agent] += self.rewards[agent]
        
        # Check if game is over
        self._check_termination()
        
        # Update step count
        self.step_count += 1
              
        # Select next agent
        self._select_next_agent()
        
        return self._get_obs(), self._get_rewards(), self._get_terminations(), self._get_truncations(), self._get_infos()
    
    def _was_dead_step(self, action):
        """Handle step when agent is already dead/terminated."""
        # PettingZoo expects this method for handling dead agents
        if action is not None:
            raise ValueError("when an agent is dead, the only valid action is None")
        
        # Select next agent
        self._select_next_agent()
        
        return self._get_obs(), self._get_rewards(), self._get_terminations(), self._get_truncations(), self._get_infos()
    
    def _execute_action(self, agent: str, action) -> bool:
        """Execute the action for the given agent."""
        try:
            if agent == "mrx":
                return self._execute_mrx_action(action)
            else:  # detective
                return self._execute_detective_action(agent, action)
        except Exception as e:
            print(f"Error executing action for {agent}: {e}")
            return False
    
    def _execute_mrx_action(self, action) -> bool:
        """Execute Mr. X action using relative encoding."""
        if self.game.game_state.turn != Player.MRX:
            return False
        
        # Decode relative action
        neighbor_idx, transport_idx, use_double_move = action
        
        # Convert to absolute position and transport
        mrx_pos = self.game.game_state.MrX_position
        dest, transport = self._relative_action_to_absolute(mrx_pos, neighbor_idx, transport_idx, is_mrx=True)
        
        use_double = bool(use_double_move)
        # Disable double moves for now for simplicity
        
        # Check if action is valid
        valid_moves = self.game.get_valid_moves(Player.MRX)
        if (dest, transport) not in valid_moves:
            return False
        
        # Check double move validity
        if use_double and not self.game.can_use_double_move():
            return False
        
        # Execute move
        success = self.game.make_move(
            MrX_moves=[(dest, transport)],
            use_double_move=use_double
        )
        
        if success:
            # Reset detective turn state for next detective phase
            self.pending_detective_moves = []
            self.current_detective_turn = 0
        
        return success
    
    def _execute_detective_action(self, agent: str, action) -> bool:
        """Execute detective action using relative encoding and sequential movement."""
        if self.game.game_state.turn != Player.DETECTIVES:
            return False
        
        # Extract detective ID from agent name (e.g., "detective_0" -> 0)
        detective_id = int(agent.split("_")[1])
        
        # Decode relative action
        neighbor_idx, transport_idx = action
        
        # Get detective position
        detective_pos = self.game.game_state.detective_positions[detective_id]
        
        # Convert to absolute position and transport
        dest, transport = self._relative_action_to_absolute(detective_pos, neighbor_idx, transport_idx, is_mrx=False)
        
        # Check if action is valid (considering pending moves)
        valid_moves = self.game.get_valid_moves(
            Player.DETECTIVES, 
            detective_pos, 
            pending_moves=self.pending_detective_moves
        )
        
        if (dest, transport) not in valid_moves:
            # Invalid move - detective stays in place
            move = (detective_pos, None)
        else:
            move = (dest, transport)
        
        # Add move to pending moves
        self.pending_detective_moves.append(move)
        
        # Check if all detectives have moved
        if len(self.pending_detective_moves) == self.num_detectives:
            # Execute all detective moves at once
            success = self.game.make_move(detective_moves=self.pending_detective_moves)
            # Reset for next turn
            self.pending_detective_moves = []
            self.current_detective_turn = 0
            return success
        else:
            # More detectives need to move
            self.current_detective_turn += 1
            return True  # Successful partial turn
    
    def _calculate_rewards(self):
        """Calculate rewards for all agents."""
        # Reset rewards
        self.rewards = {agent: 0.0 for agent in self.agents}
        
        if self.game.is_game_over():
            # Terminal rewards
            winner = self.game.get_winner()
            if winner == Player.MRX:
                self.rewards["mrx"] = 1.0
                # Distribute negative reward among detectives
                for i in range(self.num_detectives):
                    self.rewards[f"detective_{i}"] = -1.0
            elif winner == Player.DETECTIVES:
                self.rewards["mrx"] = -1.0
                # Distribute positive reward among detectives
                for i in range(self.num_detectives):
                    self.rewards[f"detective_{i}"] = 1.0
            else:
                # Draw
                for agent in self.agents:
                    self.rewards[agent] = 0.0
        else:
            # Step rewards - small incentives for good play
            # Mr. X gets small reward for staying far from detectives
            # Detectives get small reward for getting close to Mr. X
            
            # Calculate minimum distance between Mr. X and detectives
            mrx_pos = self.game.game_state.MrX_position
            detective_positions = self.game.game_state.detective_positions
            
            min_distance = float('inf')
            for det_pos in detective_positions:
                # Simple Manhattan distance (could be improved with graph distance)
                distance = abs(mrx_pos - det_pos)
                min_distance = min(min_distance, distance)
            
            # Small shaped rewards
            distance_factor = min(min_distance / 10.0, 1.0)  # Normalize
            self.rewards["mrx"] = 0.01 * distance_factor  # Reward for being far
            
            # Distribute rewards among detectives
            for i in range(self.num_detectives):
                self.rewards[f"detective_{i}"] = 0.01 * (1.0 - distance_factor)  # Reward for being close
    
    def _check_termination(self):
        """Check if the game has terminated."""
        if self.game.is_game_over():
            print(f"DEBUG: Game is over! Winner: {self.game.get_winner()}")
            self.terminations = {agent: True for agent in self.agents}
            
            # Save the completed game
            self._save_completed_game()
        else:
            self.terminations = {agent: False for agent in self.agents}
    
    def _save_completed_game(self):
        """Save the completed game using GameService."""
        if self.game_saved:
            return  # Already saved this game
        
        try:
            # Determine winner for metadata
            winner = self.game.get_winner()
            winner_str = "Mr. X" if winner == Player.MRX else "Detectives" if winner == Player.DETECTIVES else "Draw"
            
            # Create metadata for PettingZoo training session
            additional_metadata = {
                'source': 'pettingzoo_training',
                'map_size': self.map_size,
                'num_detectives': self.num_detectives,
                'max_turns': self.max_turns,
                'final_turn_count': self.step_count,
                'winner': winner_str,
                'training_session': True,
                'env_type': 'sequential_detectives'
            }
            
            # Save the game
            game_id = self.game_service.save_game(
                game=self.game,
                game_mode="pz_training",
                player_types={
                    'detectives': 'AI (PettingZoo Training)',
                    'MrX': 'AI (PettingZoo Training)'
                },
                additional_metadata=additional_metadata
            )
            
            print(f"DEBUG: Game saved with ID: {game_id}")
            self.game_saved = True
            
        except Exception as e:
            print(f"WARNING: Failed to save game: {e}")
            # Don't raise the exception to avoid breaking the training
    
    def save_current_game(self, force: bool = False) -> Optional[str]:
        """
        Manually save the current game state.
        
        Args:
            force: If True, save even if game is not complete or already saved
            
        Returns:
            Game ID if saved successfully, None otherwise
        """
        if not force and (self.game_saved or not self.game.is_game_over()):
            return None
            
        try:
            # Temporarily reset saved flag if forcing
            if force:
                self.game_saved = False
                
            self._save_completed_game()
            return "Game saved successfully"
        except Exception as e:
            print(f"ERROR: Failed to manually save game: {e}")
            return None
    
    def _select_next_agent(self):
        """Select the next agent to act with sequential detective movement."""
        # If game is over or agents are terminated/truncated, clear agents list
        if (self.game.is_game_over() or 
            any(self.terminations.values()) or 
            any(self.truncations.values())):
            print(f"DEBUG: Game ended, clearing agents")
            self.agents = []
            return
            
        # Game continues - use game state to determine next agent
        if self.game.game_state.turn == Player.MRX:
            self.agent_selection = "mrx"
            print(f"DEBUG: Set agent_selection to mrx")
        else:
            # Detective turn - select next detective in sequence
            next_detective_id = len(self.pending_detective_moves)
            self.agent_selection = f"detective_{next_detective_id}"
            print(f"DEBUG: Set agent_selection to detective_{next_detective_id}")
        
    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Get observations for all agents."""
        obs = {}
        for agent in self.agents:
            if agent == "mrx":
                features = self.feature_extractor.extract_features(self.game, Player.MRX)
            else:
                features = self.feature_extractor.extract_features(self.game, Player.DETECTIVES)
            obs[agent] = np.array(features, dtype=np.float32)
        return obs
    
    def _get_rewards(self) -> Dict[str, float]:
        """Get rewards for all agents."""
        return self.rewards.copy()
    
    def _get_terminations(self) -> Dict[str, bool]:
        """Get termination status for all agents."""
        return self.terminations.copy()
    
    def _get_truncations(self) -> Dict[str, bool]:
        """Get truncation status for all agents."""
        return self.truncations.copy()
    
    def _get_infos(self) -> Dict[str, Dict]:
        """Get info dictionaries for all agents."""
        return self.infos.copy()
    
    def _get_info(self) -> Dict[str, Dict]:
        """Get info for reset."""
        return {agent: {} for agent in self.agents}
    
    def observe(self, agent: str) -> np.ndarray:
        """Get observation for a specific agent."""
        if agent == "mrx":
            features = self.feature_extractor.extract_features(self.game, Player.MRX)
        else:
            features = self.feature_extractor.extract_features(self.game, Player.DETECTIVES)
        return np.array(features, dtype=np.float32)
    
    def action_mask(self, agent: str) -> np.ndarray:
        """Get action mask for valid actions using relative encoding."""
        if agent == "mrx":
            return self._get_mrx_action_mask()
        else:
            return self._get_detective_action_mask(agent)
    
    def _get_mrx_action_mask(self) -> np.ndarray:
        """Get action mask for Mr. X using relative encoding."""
        # Action space: [max_neighbors + 1, num_transports, 2]
        mask_shape = tuple(self.action_spaces["mrx"].nvec)
        mask = np.zeros(mask_shape, dtype=bool)
        
        if self.game.game_state.turn == Player.MRX:
            mrx_pos = self.game.game_state.MrX_position
            valid_relative_actions = self._get_valid_relative_actions(mrx_pos, is_mrx=True)
            can_double_move = self.game.can_use_double_move()
            
            for neighbor_idx, transport_idx in valid_relative_actions:
                if neighbor_idx < mask_shape[0] and transport_idx < mask_shape[1]:
                    mask[neighbor_idx, transport_idx, 0] = True  # No double move
                    if can_double_move:
                        mask[neighbor_idx, transport_idx, 1] = True  # With double move
        
        return mask.flatten()
    
    def _get_detective_action_mask(self, agent: str) -> np.ndarray:
        """Get action mask for detective using relative encoding."""
        # Action space: [max_neighbors + 1, num_transports]
        mask_shape = tuple(self.action_spaces[agent].nvec)
        mask = np.zeros(mask_shape, dtype=bool)
        
        if self.game.game_state.turn == Player.DETECTIVES:
            # Extract detective ID from agent name
            detective_id = int(agent.split("_")[1])
            detective_pos = self.game.game_state.detective_positions[detective_id]
            
            # Get valid actions considering pending moves
            valid_relative_actions = self._get_valid_relative_actions(
                detective_pos, 
                is_mrx=False, 
                pending_moves=self.pending_detective_moves
            )
            
            for neighbor_idx, transport_idx in valid_relative_actions:
                if neighbor_idx < mask_shape[0] and transport_idx < mask_shape[1]:
                    mask[neighbor_idx, transport_idx] = True
        
        return mask.flatten()
    
    def render(self, mode: str = "human") -> Optional[str]:
        """Render the environment."""
        if mode == "ansi":
            return self._render_ansi()
        elif mode == "human":
            print(self._render_ansi())
            return None
        else:
            raise ValueError(f"Unsupported render mode: {mode}")
    
    def _render_ansi(self) -> str:
        """Render the environment as ASCII text."""
        lines = []
        lines.append("=" * 50)
        lines.append(f"SHADOW CHASE - Turn {self.step_count}")
        lines.append("=" * 50)
        
        # Current turn
        current_player = "Mr. X" if self.game.game_state.turn == Player.MRX else "Detectives"
        lines.append(f"Current Player: {current_player}")
        lines.append("")
        
        # Positions
        lines.append("Positions:")
        if self.game.game_state.MrX_visible:
            lines.append(f"  Mr. X: {self.game.game_state.MrX_position}")
        else:
            lines.append(f"  Mr. X: Hidden {self.game.game_state.MrX_position}")
        
        for i, pos in enumerate(self.game.game_state.detective_positions):
            lines.append(f"  Detective {i+1}: {pos}")
        lines.append("")
        
        # Tickets
        # lines.append("Tickets:")
        # mrx_tickets = self.game.get_MrX_tickets()
        # lines.append(f"  Mr. X: {mrx_tickets}")
        
        # for i in range(self.num_detectives):
        #     det_tickets = self.game.get_detective_tickets(i)
        #     lines.append(f"  Detective {i+1}: {det_tickets}")
        
        return "\n".join(lines)
    
    def close(self):
        """Close the environment."""
        pass


# Wrapper for easier usage
def shadow_chase_v1(**kwargs):
    """Create a Shadow Chase environment."""
    env = ShadowChaseEnv(**kwargs)
    # Add standard PettingZoo wrappers
    env = wrappers.CaptureStdoutWrapper(env)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env
