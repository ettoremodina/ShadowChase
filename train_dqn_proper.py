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
import matplotlib.pyplot as plt
import numpy as np
import json
print(f"✅ PyTorch available: {torch.__version__}")
print(f"   Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
from training.deep_q.dqn_trainer import DQNTrainer
print("✅ DQN components loaded successfully")
from agents.dqn_agent import DQNMrXAgent

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def plot_training_metrics(model_path: str, save_plots: bool = True):
    """
    Plot training metrics (loss and epsilon) from saved model.
    
    Args:
        model_path: Path to the saved model (.pth file)
        save_plots: Whether to save plots to files
    """
    try:
        # Load the saved model to get training stats
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        training_stats = checkpoint.get('training_stats', {})
        
        episode_rewards = training_stats.get('episode_rewards', [])
        losses = training_stats.get('losses', [])
        final_epsilon = training_stats.get('final_epsilon', None)
        
        if not episode_rewards and not losses:
            print("⚠️  No training metrics found in saved model")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('DQN Training Metrics', fontsize=16, fontweight='bold')
        
        # Plot 1: Episode Rewards
        if episode_rewards:
            axes[0, 0].plot(episode_rewards, alpha=0.7, linewidth=1)
            # Add moving average
            if len(episode_rewards) > 10:
                window_size = min(50, len(episode_rewards) // 10)
                moving_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
                axes[0, 0].plot(range(window_size-1, len(episode_rewards)), moving_avg, 
                               color='red', linewidth=2, label=f'Moving Average ({window_size})')
                axes[0, 0].legend()
            
            axes[0, 0].set_title('Episode Rewards')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].grid(True, alpha=0.3)
            
        # Plot 2: Training Loss
        if losses:
            axes[0, 1].plot(losses, alpha=0.7, linewidth=1, color='orange')
            # Add moving average for loss
            if len(losses) > 10:
                window_size = min(100, len(losses) // 10)
                moving_avg = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
                axes[0, 1].plot(range(window_size-1, len(losses)), moving_avg, 
                               color='red', linewidth=2, label=f'Moving Average ({window_size})')
                axes[0, 1].set_yscale('log')
                axes[0, 1].legend()
            
            axes[0, 1].set_title('Training Loss')
            axes[0, 1].set_xlabel('Training Step')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].grid(True, alpha=0.3)
            
        # Plot 3: Epsilon Decay (approximate reconstruction)
        if episode_rewards and final_epsilon is not None:
            # Reconstruct epsilon values based on config
            try:
                config = checkpoint.get('config', {})
                training_params = config.get('training_parameters', {})
                epsilon_start = training_params.get('epsilon_start', 1.0)
                epsilon_decay = training_params.get('epsilon_decay', 0.995)
                epsilon_end = training_params.get('epsilon_end', 0.01)
                
                # Reconstruct epsilon values
                epsilons = []
                current_eps = epsilon_start
                for _ in range(len(episode_rewards)):
                    epsilons.append(current_eps)
                    current_eps = max(epsilon_end, current_eps * epsilon_decay)
                
                axes[1, 0].plot(epsilons, color='green', linewidth=2)
                axes[1, 0].set_title('Epsilon Decay')
                axes[1, 0].set_xlabel('Episode')
                axes[1, 0].set_ylabel('Epsilon')
                axes[1, 0].grid(True, alpha=0.3)
                axes[1, 0].set_ylim(0, epsilon_start * 1.1)
                
            except Exception as e:
                axes[1, 0].text(0.5, 0.5, f'Epsilon plot unavailable\n({str(e)})', 
                               ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Epsilon Decay (Unavailable)')
        
        # Plot 4: Performance Summary
        if episode_rewards:
            # Calculate performance metrics over time
            window_size = 100
            performance_windows = []
            window_centers = []
            
            for i in range(window_size, len(episode_rewards) + 1, window_size // 4):
                window_rewards = episode_rewards[max(0, i-window_size):i]
                performance_windows.append(np.mean(window_rewards))
                window_centers.append(i - window_size // 2)
            
            if performance_windows:
                axes[1, 1].plot(window_centers, performance_windows, 'bo-', linewidth=2, markersize=4)
                axes[1, 1].set_title(f'Performance Over Time (Window: {window_size})')
                axes[1, 1].set_xlabel('Episode')
                axes[1, 1].set_ylabel('Average Reward')
                axes[1, 1].grid(True, alpha=0.3)
        
        # Remove empty subplots
        for i, ax in enumerate(axes.flat):
            if not ax.has_data():
                ax.remove()
        
        plt.tight_layout()
        
        if save_plots:
            # Save the plot
            plot_path = Path(model_path).parent / f"training_plots_{Path(model_path).stem}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"📊 Training plots saved to: {plot_path}")
        
        # Show the plot
        plt.show()
        
        # Print summary statistics
        print(f"\n📈 Training Metrics Summary:")
        if episode_rewards:
            print(f"   Episodes: {len(episode_rewards)}")
            print(f"   Final reward: {episode_rewards[-1]:.3f}")
            print(f"   Best reward: {max(episode_rewards):.3f}")
            print(f"   Average reward (last 100): {np.mean(episode_rewards[-100:]):.3f}")
        
        if losses:
            print(f"   Training steps: {len(losses)}")
            print(f"   Final loss: {losses[-1]:.6f}")
            print(f"   Average loss (last 100): {np.mean(losses[-100:]):.6f}")
        
        if final_epsilon is not None:
            print(f"   Final epsilon: {final_epsilon:.4f}")
            
    except Exception as e:
        print(f"❌ Error plotting training metrics: {e}")
        traceback.print_exc()

def main():
    """Main training function."""
    print("🎯 Scotland Yard DQN Training")
    print("=" * 60)
    
    
    # Training configuration - proper values for real training
    with open('training/configs/dqn_config.json', 'r') as f:
        training_config = json.load(f)
    print(f"   Using configuration: {training_config}")


    # Create trainer
    print(f"\n🔧 Initializing trainer...")

    trainer = DQNTrainer(
        player_role=training_config["training_parameters"]['player_role'],
        save_dir="training_results"
    )
    print("✅ DQN trainer created successfully")
    
    # Start training

    print("   This will take some time. Progress will be shown during training.")
    print("-" * 60)
    
    start_time = time.time()
    

    result = trainer.train(
        num_episodes=training_config["training_parameters"]['num_episodes'],
        map_size=training_config["game_parameters"]['map_size'],
        num_detectives=training_config["game_parameters"]['num_detectives'],
        max_turns_per_game=training_config["game_parameters"]['max_turns_per_game']
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
    
    # Plot training metrics
    print(f"\n📊 Generating training plots...")
    plot_training_metrics(result.model_path, save_plots=True)

    # Plot Q-value histogram for random (state, action) pairs
    if agent.model is not None:
        print("\n📊 Plotting Q-value histogram for random (state, action) pairs...")
        agent.model.plot_q_value_histogram(num_samples=10000)
    else:
        print("⚠️  Cannot plot Q-value histogram: model not loaded.")

    print(f"\n✅ Training script completed!")
    print("   You can now use the trained agent in games.")


if __name__ == "__main__":
    main()
