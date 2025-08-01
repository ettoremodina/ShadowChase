"""
Deep Q-Network trainer for Scotland Yard.

This module implements the DQN training algorithm using the existing training infrastructure.
"""

import json
import time
import numpy as np
from typing import Optional


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from ScotlandYard.core.game import ScotlandYardGame
from training.base_trainer import BaseTrainer, TrainingResult
from training.feature_extractor_simple import GameFeatureExtractor, FeatureConfig
from training.training_environment import TrainingEnvironment
from ScotlandYard.core.game import TransportType

from .dqn_model import create_dqn_model
from .replay_buffer import create_replay_buffer

from game_controls.game_utils import create_and_initialize_game
from game_controls.game_utils import create_and_initialize_game


from agents import AgentType, get_agent_registry
from agents.dqn_agent import DQNMrXAgent, DQNMultiDetectiveAgent

class DQNTrainer(BaseTrainer):
    """
    Deep Q-Network trainer for Scotland Yard agents.
    
    Implements DQN with experience replay and target networks.
    """
    
    def __init__(self, 
                 player_role: str = "mr_x",  # "mr_x" or "detectives"
                 config_path: str = "training/configs/dqn_config.json",
                 save_dir: str = "training_results"):
        """
        Initialize DQN trainer.
        
        Args:
            player_role: Which player the agent will control ("mr_x" or "detectives")
            config_path: Path to DQN configuration file
            save_dir: Directory to save training results
        """
        super().__init__("dqn", save_dir)
    
        
        self.player_role = player_role
        self.config_path = config_path
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Initialize training parameters
        training_params = self.config.get('training_parameters', {})
        self.batch_size = training_params.get('batch_size', 32)
        self.learning_rate = training_params.get('learning_rate', 0.001)
        self.gamma = training_params.get('gamma', 0.95)
        self.epsilon_start = training_params.get('epsilon_start', 1.0)
        self.epsilon_end = training_params.get('epsilon_end', 0.01)
        self.epsilon_decay = training_params.get('epsilon_decay', 0.995)
        self.target_update_frequency = training_params.get('target_update_frequency', 100)
        self.min_replay_size = training_params.get('min_replay_buffer_size', 100)
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = "cpu"  # Force CPU fsor now
        print(f"Using device: {self.device}")
        
        # Initialize feature extractor
        feature_config = FeatureConfig(**self.config.get('feature_extraction', {}))
        self.feature_extractor = GameFeatureExtractor(feature_config)
        
        # Initialize components (will be created in train())
        self.main_network = None
        self.target_network = None
        self.optimizer = None
        self.replay_buffer = None
        self.current_epsilon = self.epsilon_start
        self.epsilon_decay_rate = (self.epsilon_start - self.epsilon_end) / training_params.get('num_episodes', 10000)
        
        # Training state
        self.step_count = 0
        self.episode_rewards = []
        self.losses = []
        self.q_value_samples = []  # For monitoring Q-value distributions
    
    def _initialize_networks(self, sample_game: ScotlandYardGame):
        """Initialize the neural networks based on a sample game."""
        # Get feature size from a sample state
        feature_size = self.feature_extractor.get_feature_size(sample_game)
        print(f"Feature vector size: {feature_size}")
        
        # Store feature size for monitoring
        self._feature_size = feature_size
        
        # Update config with actual feature size
        self.config['feature_extraction']['input_size'] = feature_size
        
        # Set action size - all agents use (destination, transport)
        action_size = 2  # (destination, transport)
        
        self.config['network_parameters']['action_size'] = action_size
        print(f"Action size for {self.player_role}: {action_size}")
        
        # Create networks
        self.main_network = create_dqn_model(self.config).to(self.device)
        self.target_network = create_dqn_model(self.config).to(self.device)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.main_network.state_dict())
        self.target_network.eval()  # Target network is always in eval mode
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=self.learning_rate)
        
        # Initialize replay buffer
        self.replay_buffer = create_replay_buffer(self.config)
        
        print(f"Initialized DQN with {sum(p.numel() for p in self.main_network.parameters())} parameters")
    
    def train(self, 
              num_episodes: int,
              map_size: str = "test",
              num_detectives: int = 2,
              max_turns_per_game: int = 24,
              plotting_every: int = 5000,
              **kwargs) -> TrainingResult:
        """
        Train the DQN agent.
        
        Args:
            num_episodes: Number of training episodes
            map_size: Game map size
            num_detectives: Number of detective agents
            max_turns_per_game: Maximum turns per game
            
        Returns:
            TrainingResult with training metrics
        """
        start_time = time.time()
        
        # Create training environment
        env = TrainingEnvironment(
            map_size=map_size,
            num_detectives=num_detectives,
            max_turns=max_turns_per_game
        )
        
        # Create a sample game to initialize networks
        sample_game = create_and_initialize_game(map_size, num_detectives)
        # Initialize networks if not already done
        if self.main_network is None:
            self._initialize_networks(sample_game)
        
        # Create baseline opponent
        registry = get_agent_registry()
        
        if self.player_role == "mr_x":
            opponent_agent = registry.create_multi_detective_agent(AgentType.RANDOM, num_detectives)
        else:
            opponent_agent = registry.create_mr_x_agent(AgentType.RANDOM)
        
        print(f"\n🚀 Starting DQN Training for {self.player_role.upper()}")
        print("=" * 80)
        print(f"Episodes: {num_episodes:,} │ Map: {map_size} │ Detectives: {num_detectives}")
        print(f"Device: {self.device} │ Network: {sum(p.numel() for p in self.main_network.parameters()):,} params")
        print(f"Learning Rate: {self.learning_rate} │ Batch Size: {self.batch_size}")
        print("=" * 80)
        
        # Training loop
        for episode in range(num_episodes):
            # if we are after X% of episodes the opponent becomes the heuristic agent
            # if episode >= int(num_episodes * 0.9):  
            #     opponent_agent = registry.create_multi_detective_agent(AgentType.HEURISTIC, num_detectives) if self.player_role == "mr_x" else registry.create_mr_x_agent(AgentType.HEURISTIC)

            episode_reward = self._train_episode(env, opponent_agent)
            self.episode_rewards.append(episode_reward)
            
            # Monitor Q-values periodically
            self.monitor_q_values(episode, plotting_every)
            
            # Update target network
            if episode % self.target_update_frequency == 0:
                self.target_network.load_state_dict(self.main_network.state_dict())
            
            # Decay epsilon
            # self.current_epsilon = max(
            #     self.epsilon_end, 
            #     self.current_epsilon * self.epsilon_decay
            # )
            self.current_epsilon = max(
                self.epsilon_end,
                self.epsilon_start - self.epsilon_decay_rate * episode
            )
            
            # Logging
            if episode % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                avg_loss = np.mean(self.losses[-100:]) if self.losses else 0
                buffer_size = len(self.replay_buffer)
                win_rate = sum(1 for r in self.episode_rewards[-100:] if r > 0) / min(100, len(self.episode_rewards))
                
                print(f"Episode {episode:5d} │ Reward: {avg_reward:7.2f} │ "
                      f"ε: {self.current_epsilon:.3f} │ Loss: {avg_loss:8.4f} │ "
                      f"Buffer: {buffer_size:5d} │ Win%: {win_rate:5.1%}")
                
            
            # Log training step
            self._log_training_step(episode, {
                'episode_reward': episode_reward,
                'epsilon': self.current_epsilon,
                'buffer_size': len(self.replay_buffer),
                'avg_loss': np.mean(self.losses[-10:]) if self.losses else 0
            })
        
        training_duration = time.time() - start_time
        
        # Calculate final performance
        final_performance = {
            'avg_episode_reward': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0,
            'final_epsilon': self.current_epsilon,
            'total_steps': self.step_count,
            'final_buffer_size': len(self.replay_buffer)
        }
        
        print(f"\n✅ Training Completed!")
        print("=" * 80)
        print(f"Duration: {training_duration:.1f}s │ Final Reward: {final_performance['avg_episode_reward']:.2f}")
        print(f"Total Steps: {final_performance['total_steps']:,} │ Buffer Size: {final_performance['final_buffer_size']:,}")
        print("=" * 80)
        
        # Save model
        model_path = self.save_model()
        
        return TrainingResult(
            algorithm="dqn",
            total_episodes=num_episodes,
            training_duration=training_duration,
            final_performance=final_performance,
            training_history=self.training_history,
            model_path=model_path
        )
    
    def _train_episode(self, env: TrainingEnvironment, opponent_agent) -> float:
        """Train for a single episode."""
        episode_reward = 0.0
        
        # Create our training agent
        if self.player_role == "mr_x":
            our_agent = DQNMrXAgent(trainer=self)
            # Run episode without collecting environment experiences
            result, _ = env.run_episode(our_agent, opponent_agent, collect_experience=False)
        else:
            # Get number of detectives from environment
            num_detectives = env.num_detectives
            our_agent = DQNMultiDetectiveAgent(num_detectives=num_detectives, trainer=self)
            result, _ = env.run_episode(opponent_agent, our_agent, collect_experience=False)
        
        # Calculate episode reward based on game outcome
        episode_reward = self._calculate_episode_reward(result)
        

        
        # Train on batch if we have enough experiences
        if len(self.replay_buffer) >= self.min_replay_size:
            loss = self._train_step()
            if loss is not None:
                self.losses.append(loss)
    
        
        return episode_reward
    
    def _calculate_episode_reward(self, result) -> float:
        """Calculate episode reward based on game outcome and distance-based shaping for both roles."""
        # Final outcome reward - scale based on game length
        base_win_reward = 10.0
        base_loss_reward = -10.0
        
        # Adjust reward based on game efficiency
        # Shorter games for detectives are better, longer games for Mr. X are better
        turn_factor = max(0.5, min(2.0, result.total_turns / 12.0))  # Normalize around 12 turns
        
        if result.winner == self.player_role.replace("_", ""):
            if self.player_role == "mr_x":
                # Mr. X: bonus for surviving longer
                outcome = base_win_reward * turn_factor
            else:
                # Detectives: bonus for catching Mr. X quickly
                outcome = base_win_reward / turn_factor
        else:
            if self.player_role == "mr_x":
                # Mr. X: less penalty if survived longer
                outcome = base_loss_reward / turn_factor
            else:
                # Detectives: more penalty for taking longer
                outcome = base_loss_reward * turn_factor
        
        # Distance-based shaping
        shaping = 0.0
        if hasattr(result, 'mr_x_min_distances') and result.mr_x_min_distances:
            valid_distances = [d for d in result.mr_x_min_distances if d >= 0]
            if valid_distances:
                avg_min_dist = np.mean(valid_distances)
                # Mr. X: reward being far from detectives; Detectives: reward being close
                if self.player_role == "mr_x":
                    shaping = 0.3 * avg_min_dist  # Reward staying far
                else:
                    shaping = -0.3 * avg_min_dist  # Reward staying close

        return outcome + shaping 
    
    def _train_step(self) -> Optional[float]:
        """Perform one training step using batch from replay buffer."""
        if not self.replay_buffer.can_sample(self.batch_size):
            return None
        
        # Sample batch
        experiences = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors (efficiently)
        states = torch.FloatTensor(np.array([exp.state for exp in experiences])).to(self.device)
        actions = [exp.action for exp in experiences]  # (dest, transport) pairs
        rewards = torch.FloatTensor(np.array([exp.reward for exp in experiences])).to(self.device)
        next_states = torch.FloatTensor(np.array([exp.next_state for exp in experiences])).to(self.device)
        dones = torch.BoolTensor(np.array([exp.done for exp in experiences])).to(self.device)
        next_valid_moves = [exp.next_valid_moves for exp in experiences]
        
        # Get current Q-values using efficient batching
        current_q_values = self.main_network.query_batch_actions(states, actions)
        
        # Get next Q-values (max over valid actions for each next state) - using batching
        with torch.no_grad():
            # Filter out terminal states for batch processing
            non_terminal_indices = [i for i, done in enumerate(dones) if not done and next_valid_moves[i]]
            
            if non_terminal_indices:
                # Process non-terminal states in batch
                non_terminal_states = next_states[non_terminal_indices]
                non_terminal_valid_moves = [next_valid_moves[i] for i in non_terminal_indices]
                
                batch_max_q_values = self.target_network.query_batch_max_q_values(
                    non_terminal_states, non_terminal_valid_moves
                )
                
                # Create full tensor with zeros for terminal states
                max_next_q_values = torch.zeros(len(experiences), device=self.device)
                max_next_q_values[non_terminal_indices] = batch_max_q_values
            else:
                # All states are terminal or have no valid moves
                max_next_q_values = torch.zeros(len(experiences), device=self.device)
            target_q_values = rewards + (self.gamma * max_next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        # Add gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.main_network.parameters(), max_norm=1.0)
        self.optimizer.step() 
        self.step_count += 1
        return loss.item()
    
    def monitor_q_values(self, episode: int, plotting_every: int = 5000):
        """Monitor Q-value distribution periodically."""
        if episode % plotting_every == 0:  # Sample based on plotting_every parameter
            try:
                self.main_network.eval()
                
                # Create random state-action pairs
                num_samples = 500
                # Use the stored feature size from network initialization
                if hasattr(self, '_feature_size'):
                    feature_size = self._feature_size
                else:
                    # Fallback: create a sample game to get feature size
                    sample_game = create_and_initialize_game("extended", 5)
                    feature_size = self.feature_extractor.get_feature_size(sample_game)
                    self._feature_size = feature_size
                
                q_values = []
                for _ in range(num_samples):
                    state = torch.randn(feature_size).to(self.device)
                    dest = np.random.randint(1, 200)
                    transport = np.random.choice(list(TransportType))
                    
                    # All agents use (destination, transport) actions
                    action = (dest, transport)
                    
                    with torch.no_grad():
                        q_val = self.main_network.query_batch_actions(state.unsqueeze(0), [action])
                        q_values.append(q_val.item())
                
                self.q_value_samples.append(q_values)
                
                q_array = np.array(q_values)
                print(f"\n📊 Q-Value Analysis (Episode {episode}):")
                print(f"   Mean: {q_array.mean():8.3f} │ Std: {q_array.std():7.3f}")
                print(f"   Range: [{q_array.min():7.3f}, {q_array.max():7.3f}]")
                print(f"   Negative: {(q_array < 0).sum():3d}/{len(q_array)} ({100*(q_array < 0).mean():4.1f}%)")
                print("─" * 60)
                
                # Put network back in training mode
                self.main_network.train()
            
            except Exception as e:
                print(f"Warning: Could not monitor Q-values: {e}")
    
    def save_model(self, model_name: Optional[str] = None) -> str:
        """Save the trained model."""
        if model_name is None:
            model_name = f"dqn_{self.player_role}_{int(time.time())}"
        
        model_path = self.save_dir / f"{model_name}.pth"
        
        torch.save({
            'main_network': self.main_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config,
            'player_role': self.player_role,
            'training_stats': {
                'episode_rewards': self.episode_rewards,
                'losses': self.losses,
                'step_count': self.step_count,
                'final_epsilon': self.current_epsilon,
                'q_value_samples': self.q_value_samples
            }
        }, model_path)
        
        print(f"Model saved to: {model_path}")
        return str(model_path)

