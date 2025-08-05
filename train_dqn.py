"""
Improved DQN training script with better monitoring and diagnostics.
"""

import torch
import numpy as np

from training.training_environment import TrainingEnvironment
from agents.dqn_agent import DQNMrXAgent, DQNMultiDetectiveAgent
import argparse
from training.deep_q.dqn_trainer import DQNTrainer
from agents import AgentType, get_agent_registry
from training.plot_utils import plot_training_metrics


def train_with_monitoring(player_role="MrX", num_episodes=5000, plotting_every=1000, device=None):
    """Train DQN with enhanced monitoring and diagnostics."""
    print(f"🚀 Starting enhanced DQN training for {player_role}")
    print("=" * 60)
    
    # Set device if not provided
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device = "cpu"  # Uncomment to force CPU
    
    print(f"🖥️  Using device: {device}")
    
    # Create trainer
    trainer = DQNTrainer(
        player_role=player_role,
        config_path="training/configs/dqn_config.json",
        device=device
    )
    
    # Override the training loop to add monitoring
    original_train_episode = trainer._train_episode
    
    def monitored_train_episode(env, opponent_agent):
        episode_reward = original_train_episode(env, opponent_agent)
        
        # Monitor Q-values periodically using trainer's method
        current_episode = len(trainer.episode_rewards)

        # Plot progress periodically
        if current_episode % plotting_every == 0 and current_episode > 0:
            print(f"📈 Generating training progress plot...")
            plot_training_metrics(trainer, 
                                f"training_results/training_progress_{player_role}_{current_episode}.png", 
                                plotting_every, show_plot=True)
        
        return episode_reward
    
    trainer._train_episode = monitored_train_episode
    
    # Start training
    result = trainer.train(
        map_size="extended",  # Start with smaller map
        num_detectives=5,
        max_turns_per_game=24,
        plotting_every=plotting_every
    )
    
    # Final plots
    print(f"📈 Generating final training plots...")
    plot_training_metrics(trainer, 
                         f"training_results/final_training_metrics_{player_role}.png", 
                         plotting_every, show_plot=True)
    
    print(f"\n🎉 Training Complete!")
    print("=" * 60)
    print(f"Model: {result.model_path}")
    print(f"Performance: {result.final_performance}")
    print("=" * 60)
    
    return result, trainer


def evaluate_trained_agent(model_path, player_role, num_games=100, device=None):
    """Evaluate a trained agent against random opponents."""
    print(f"\n🎯 Evaluating trained {player_role} agent...")
    
    # Set device if not provided
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    env = TrainingEnvironment("test", 2, 24)
    registry = get_agent_registry()
    
    # Create trained agent
    if player_role == "MrX":
        trained_agent = DQNMrXAgent(model_path=model_path, epsilon=0.0, device=device)  # No exploration
        opponent = registry.create_multi_detective_agent(AgentType.RANDOM, 2)
    else:
        trained_agent = DQNMultiDetectiveAgent(2, model_path=model_path, epsilon=0.0, device=device)
        opponent = registry.create_MrX_agent(AgentType.RANDOM)
    
    # Run evaluation games
    wins = 0
    total_rewards = []
    
    for i in range(num_games):
        if player_role == "MrX":
            result, _ = env.run_episode(trained_agent, opponent)
            wins += (result.winner == "MrX")
        else:
            result, _ = env.run_episode(opponent, trained_agent)
            wins += (result.winner == "detectives")
        
        if i % 20 == 0 and i > 0:
            current_win_rate = wins / (i + 1)
            print(f"   Game {i+1:3d}/{num_games} │ Current Win Rate: {current_win_rate:5.1%}")
    
    win_rate = wins / num_games
    print(f"\n🎯 Final Results:")
    print(f"   Win Rate: {win_rate:5.1%} ({wins}/{num_games})")
    print(f"   {'✅ Better than random!' if win_rate > 0.5 else '❌ Needs improvement'}")
    print("─" * 50)
    
    return win_rate


def main():
    """Main training and evaluation pipeline."""
    
    parser = argparse.ArgumentParser(description='Enhanced DQN Training')
    parser.add_argument('--role', choices=['MrX', 'detectives'], default='MrX',
                       help='Player role to train')
    parser.add_argument('--episodes', type=int, default=5000,
                       help='Number of training episodes')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate existing model instead of training')
    parser.add_argument('--model_path', type=str,
                       help='Path to model for evaluation')
    
    args = parser.parse_args()
    
    if args.evaluate:
        if not args.model_path:
            print("Error: --model_path required for evaluation")
            return
        
        # Set device choice for evaluation
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device = "cpu"  # Uncomment to force CPU
        print(f"🖥️  Using device: {device}")
        
        evaluate_trained_agent(args.model_path, args.role, device=device)
    else:
        # Training
        plotting_every = args.episodes // 3
        
        # Set device choice here
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device = "cpu"  # Uncomment to force CPU
        print(f"🖥️  Using device: {device}")
        
        result, trainer = train_with_monitoring(args.role, args.episodes, plotting_every, device)

        # Quick evaluation
        print("\n" + "="*60)
        evaluate_trained_agent(result.model_path, args.role, num_games=50, device=device)


if __name__ == "__main__":
    # For now, just run Mr. X training
    print("Starting DQN training with improved monitoring...")
    
    # Set device choice here
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"  # Uncomment to force CPU
    print(f"🖥️  Using device: {device}")
    
    # detective_result, detective_trainer = train_with_monitoring("detectives", 9000, plotting_every=100, device=device)
    MrX_result, MrX_trainer = train_with_monitoring("MrX", 9000, plotting_every=3000, device=device)
    # MrX_result, MrX_trainer = train_with_monitoring("MrX", 400, plotting_every=200, device=device)
    
    # # Evaluate the trained agents
    # print("\n=== Detective Evaluation ===")
    # evaluate_trained_agent(detective_result.model_path, "detectives", num_games=30, device=device)
    # print("\n=== Mr. X Evaluation ===") 
    # evaluate_trained_agent(MrX_result.model_path, "MrX", num_games=30, device=device)
