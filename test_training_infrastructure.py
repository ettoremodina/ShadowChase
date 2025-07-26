#!/usr/bin/env python3
"""
Test script for the Scotland Yard training infrastructure.

This script demonstrates how to use the base training classes and feature extraction
to set up training for advanced AI agents.
"""

import json
import numpy as np
from pathlib import Path

# Import training infrastructure
from training import BaseTrainer, TrainingResult, GameFeatureExtractor, FeatureConfig
from training.utils import TrainingEnvironment
from simple_play.game_utils import create_and_initialize_game
from agents import AgentType, get_agent_registry
from ScotlandYard.core.game import Player


class MockTrainer(BaseTrainer):
    """
    Mock trainer implementation for testing the base infrastructure.
    
    This class demonstrates how to implement the BaseTrainer interface
    without actually doing any sophisticated training.
    """
    
    def __init__(self, save_dir: str = "test_training_results"):
        config = {
            "mock_parameter": 42,
            "learning_rate": 0.01
        }
        super().__init__("mock_algorithm", save_dir, config)
        self.feature_extractor = None
        
    def train(self, 
              num_episodes: int,
              map_size: str = "test",
              num_detectives: int = 2,
              max_turns_per_game: int = 24,
              **kwargs) -> TrainingResult:
        """Mock training implementation."""
        print(f"Starting mock training for {num_episodes} episodes...")
        
        # Initialize feature extractor
        feature_config = FeatureConfig()
        self.feature_extractor = GameFeatureExtractor(feature_config)
        
        # Set up training environment
        env = TrainingEnvironment(map_size, num_detectives, max_turns_per_game)
        
        # Get baseline agents for training
        registry = get_agent_registry()
        mr_x_agent = registry.create_mr_x_agent(AgentType.RANDOM)
        detective_agent = registry.create_multi_detective_agent(AgentType.RANDOM, num_detectives)
        
        # Mock training loop
        results = []
        
        for episode in range(num_episodes):
            # Run one episode
            result, experience = env.run_episode(mr_x_agent, detective_agent, collect_experience=True)
            results.append(result)
            
            # Mock learning from experience
            if experience:
                self._process_experience(experience)
            
            # Log progress
            if episode % 10 == 0:
                metrics = {
                    'win_rate': sum(1 for r in results[-10:] if r.winner == "mr_x") / min(len(results), 10),
                    'avg_game_length': sum(r.total_turns for r in results[-10:]) / min(len(results), 10)
                }
                self._log_training_step(episode, metrics)
        
        # Mark as trained
        self.is_trained = True
        
        # Calculate final performance
        final_performance = {
            'mr_x_win_rate': sum(1 for r in results if r.winner == "mr_x") / len(results),
            'detective_win_rate': sum(1 for r in results if r.winner == "detectives") / len(results),
            'timeout_rate': sum(1 for r in results if r.winner == "timeout") / len(results),
            'avg_game_length': sum(r.total_turns for r in results) / len(results)
        }
        
        # print_training_statistics(results, "Mock Algorithm")
        
        return TrainingResult(
            algorithm=self.algorithm_name,
            total_episodes=num_episodes,
            training_duration=10.0,  # Mock duration
            final_performance=final_performance,
            training_history=self.training_history
        )
    
    def get_trained_agent(self, player: Player):
        """Return a mock trained agent (just returns a random agent)."""
        if not self.is_trained:
            raise RuntimeError("Agent must be trained first")
        
        registry = get_agent_registry()
        if player == Player.MRX:
            return registry.create_mr_x_agent(AgentType.RANDOM)
        else:
            return registry.create_multi_detective_agent(AgentType.RANDOM, 2)
    
    def _process_experience(self, experience):
        """Mock experience processing."""
        # In a real implementation, this would update the agent's knowledge
        pass


def test_feature_extraction():
    """Test the feature extraction system."""
    print("Testing feature extraction...")
    
    # Create a test game
    game = create_and_initialize_game("test", 2)
    
    # Create feature extractor
    config = FeatureConfig()
    extractor = GameFeatureExtractor(config)
    
    # Extract features for both players
    mr_x_features = extractor.extract_features(game, Player.MRX)
    detective_features = extractor.extract_features(game, Player.DETECTIVES)
    
    print(f"Feature vector size: {len(mr_x_features)}")
    print(f"Expected size: {extractor.get_feature_size(game)}")
    print(f"Mr. X features shape: {mr_x_features.shape}")
    print(f"Detective features shape: {detective_features.shape}")
    
    # Print some sample features
    feature_names = extractor.get_feature_names()
    print(f"\nFirst 10 features for Mr. X:")
    for i in range(min(10, len(mr_x_features))):
        print(f"  {feature_names[i]}: {mr_x_features[i]:.3f}")
    
    print("✅ Feature extraction test passed!")


def test_training_environment():
    """Test the training environment."""
    print("\nTesting training environment...")
    
    # Create environment
    env = TrainingEnvironment("test", 2, 10)  # Short games for testing
    
    # Get agents
    registry = get_agent_registry()
    mr_x_agent = registry.create_mr_x_agent(AgentType.RANDOM)
    detective_agent = registry.create_multi_detective_agent(AgentType.RANDOM, 2)
    
    # Run a few test episodes
    results = []
    for i in range(3):
        print(f"  Running test episode {i+1}/3...")
        result, experience = env.run_episode(mr_x_agent, detective_agent, collect_experience=True)
        results.append(result)
        print(f"    Winner: {result.winner}, Turns: {result.total_turns}, Experience points: {len(experience)}")
    
    print("✅ Training environment test passed!")


def test_mock_trainer():
    """Test the mock trainer implementation."""
    print("\nTesting mock trainer...")
    
    # Create trainer
    trainer = MockTrainer()
    
    # Run short training
    training_result = trainer.train(num_episodes=20, map_size="test", num_detectives=2)
    
    print(f"Training completed: {training_result.total_episodes} episodes")
    print(f"Final performance: {training_result.final_performance}")
    
    # Test saving and loading
    model_path = trainer.save_model("test_model")
    print(f"Model saved to: {model_path}")
    
    # Test evaluation
    evaluation_result = trainer.evaluate(num_games=5)
    print(f"Evaluation win rate: {evaluation_result.win_rate:.2%}")
    
    print("✅ Mock trainer test passed!")




def main():
    """Run all tests."""
    print("🧪 Testing Scotland Yard Training Infrastructure")
    print("=" * 60)
    
    try:
        test_feature_extraction()
        test_training_environment()
        test_mock_trainer()
        print("\n🎉 All tests passed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
