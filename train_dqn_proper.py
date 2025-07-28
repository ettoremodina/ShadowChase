"""
Proper DQN training script for Scotland Yard.

This script trains a DQN agent from scratch with proper configuration
and monitoring of training progress.
"""

import sys
import os
from pathlib import Path
import time
import torch
import traceback
print(f"✅ PyTorch available: {torch.__version__}")
print(f"   Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
from training.deep_q.dqn_trainer import DQNTrainer
print("✅ DQN components loaded successfully")
from agents.dqn_agent import DQNMrXAgent

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def main():
    """Main training function."""
    print("🎯 Scotland Yard DQN Training")
    print("=" * 60)
    
    
    # Training configuration - proper values for real training
    training_config = {
        'player_role': 'mr_x',    
        'num_episodes': 1000,       
        'map_size': 'extended',        
        'num_detectives': 5,        
        'max_turns_per_game': 24   
    }
    
    print(f"\n🚀 Training Configuration:")
    for key, value in training_config.items():
        print(f"   {key}: {value}")
    
    # Create trainer
    print(f"\n🔧 Initializing trainer...")

    trainer = DQNTrainer(
        player_role=training_config['player_role'],
        config_path="training/configs/dqn_config.json",
        save_dir="training_results"
    )
    print("✅ DQN trainer created successfully")
    
    # Start training
    print(f"\n🏋️ Starting training for {training_config['num_episodes']} episodes...")
    print("   This will take some time. Progress will be shown during training.")
    print("-" * 60)
    
    start_time = time.time()
    

    result = trainer.train(
        num_episodes=training_config['num_episodes'],
        map_size=training_config['map_size'],
        num_detectives=training_config['num_detectives'],
        max_turns_per_game=training_config['max_turns_per_game']
    )
    
    training_time = time.time() - start_time
    
    print(f"\n🎉 Training completed successfully!")
    print("=" * 60)
    print(f"   Algorithm: {result.algorithm}")
    print(f"   Episodes: {result.total_episodes}")
    print(f"   Training duration: {result.training_duration:.2f} seconds")
    print(f"   Total time: {training_time:.2f} seconds")
    print(f"   Model saved to: {result.model_path}")
    
    # Print final performance metrics
    if hasattr(result, 'final_performance') and result.final_performance:
        print(f"\n📊 Final Performance Metrics:")
        for key, value in result.final_performance.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.4f}")
            else:
                print(f"   {key}: {value}")
    
    # Print training history summary
    if hasattr(result, 'training_history') and result.training_history:
        history = result.training_history
        if 'episode_rewards' in history and history['episode_rewards']:
            rewards = history['episode_rewards']
            print(f"\n📈 Training Progress Summary:")
            print(f"   Episodes trained: {len(rewards)}")
            print(f"   Average reward (last 100): {sum(rewards[-100:]) / min(100, len(rewards)):.4f}")
            print(f"   Best episode reward: {max(rewards):.4f}")
            print(f"   Worst episode reward: {min(rewards):.4f}")
        
    
    # Test the trained agent
    print(f"\n🧪 Testing trained agent...")

        
    # Create agent with the trained model
    agent = DQNMrXAgent(model_path=result.model_path)
    
    if agent.model is not None:
        print("✅ DQN agent loaded successfully and ready for use")
        print(f"   Model parameters: {sum(p.numel() for p in agent.model.parameters())}")
    else:
        print("⚠️  Agent created but model not loaded properly")
    
    print(f"\n✅ Training script completed!")
    print("   You can now use the trained agent in games.")


if __name__ == "__main__":
    main()
