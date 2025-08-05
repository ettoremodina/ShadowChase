"""
Training visualization utilities with enhanced aesthetics and functionality.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Dict, Any
import os
from pathlib import Path

# Set modern plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Color scheme for consistent plotting
COLORS = {
    'primary': '#2E86C1',
    'secondary': '#E74C3C',
    'accent': '#F39C12',
    'success': '#27AE60',
    'warning': '#F1C40F',
    'muted': '#95A5A6',
    'background': '#F8F9FA'
}

def setup_plot_style():
    """Configure matplotlib for high-quality plots."""
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': '#CCCCCC',
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.labelsize': 11,
        'axes.titlesize': 13,
        'axes.titleweight': 'bold',
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'legend.frameon': True,
        'legend.fancybox': True,
        'legend.shadow': True,
        'legend.framealpha': 0.9,
        'grid.alpha': 0.3,
        'lines.linewidth': 2,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans']
    })

def smooth_curve(data: List[float], window_size: Optional[int] = None) -> tuple:
    """Apply smoothing to data with adaptive window size."""
    if len(data) < 10:
        return range(len(data)), data
    
    if window_size is None:
        window_size = max(10, min(50, len(data) // 10))
    
    smoothed = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    episodes = range(window_size-1, len(data))
    
    return episodes, smoothed

def plot_training_metrics(trainer, save_path: str = "training_results/training_metrics.png", 
                         plotting_every: int = 1000, show_plot: bool = True,
                         figsize: tuple = (16, 12)) -> None:
    """
    Plot comprehensive training metrics with enhanced aesthetics.
    
    Args:
        trainer: DQN trainer object with training history
        save_path: Path to save the plot
        plotting_every: Interval for Q-value sampling
        show_plot: Whether to display the plot
        figsize: Figure size (width, height)
    """
    if len(trainer.episode_rewards) < 10:
        print("⚠️  Not enough data to generate meaningful plots (need at least 10 episodes)")
        return
    
    # Setup plot style
    setup_plot_style()
    
    # Create figure with improved layout
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.25)
    
    # Main episode rewards plot (spans top row)
    ax_main = fig.add_subplot(gs[0, :])
    _plot_episode_rewards(ax_main, trainer.episode_rewards)
    
    # Training loss
    ax_loss = fig.add_subplot(gs[1, 0])
    _plot_training_loss(ax_loss, trainer.losses)
    
    # Epsilon decay
    ax_epsilon = fig.add_subplot(gs[1, 1])
    _plot_epsilon_decay(ax_epsilon, trainer.training_history)
    
    # Q-value distributions
    ax_q_vals = fig.add_subplot(gs[2, 0])
    _plot_q_value_distributions(ax_q_vals, trainer, plotting_every)
    
    # Performance statistics
    ax_stats = fig.add_subplot(gs[2, 1])
    _plot_performance_stats(ax_stats, trainer.episode_rewards)
    
    # Add overall title
    fig.suptitle('DQN Training Progress Dashboard', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Add training info text
    episodes_completed = len(trainer.episode_rewards)
    avg_reward = np.mean(trainer.episode_rewards[-100:]) if len(trainer.episode_rewards) >= 100 else np.mean(trainer.episode_rewards)
    
    info_text = f"Episodes: {episodes_completed:,} | Avg Reward (last 100): {avg_reward:.2f}"
    fig.text(0.5, 0.02, info_text, ha='center', fontsize=11, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS['background'], alpha=0.8))
    
    # Save plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Training metrics saved to: {save_path}")
    
    if show_plot:
        plt.show(block=False)
        plt.pause(2.0)
        plt.close('all')
    else:
        plt.close('all')

def _plot_episode_rewards(ax, episode_rewards: List[float]) -> None:
    """Plot episode rewards with raw and smoothed curves."""
    episodes = range(len(episode_rewards))
    
    # Raw rewards (more transparent)
    ax.plot(episodes, episode_rewards, alpha=0.25, color=COLORS['muted'], 
           linewidth=1, label='Raw Rewards')
    
    # Smoothed curve
    if len(episode_rewards) > 50:
        smooth_episodes, smoothed = smooth_curve(episode_rewards)
        ax.plot(smooth_episodes, smoothed, color=COLORS['primary'], 
               linewidth=2.5, label=f'Smoothed (window={len(episodes)-len(smooth_episodes)+1})')
    
    # Add trend line for recent episodes
    if len(episode_rewards) > 100:
        recent_episodes = episodes[-100:]
        recent_rewards = episode_rewards[-100:]
        z = np.polyfit(recent_episodes, recent_rewards, 1)
        trend_line = np.poly1d(z)
        ax.plot(recent_episodes, trend_line(recent_episodes), 
               color=COLORS['accent'], linestyle='--', alpha=0.8,
               linewidth=2, label='Recent Trend')
    
    ax.set_title('Episode Rewards Over Time')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Cumulative Reward')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Add performance zones
    if len(episode_rewards) > 10:
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        ax.axhline(mean_reward, color=COLORS['success'], alpha=0.5, linestyle=':')
        ax.fill_between(episodes, mean_reward - std_reward, mean_reward + std_reward,
                       alpha=0.1, color=COLORS['success'])

def _plot_training_loss(ax, losses: List[float]) -> None:
    """Plot training loss with logarithmic scale."""
    if not losses:
        ax.text(0.5, 0.5, 'No loss data available', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Training Loss')
        return
    
    # Plot with gradient color based on loss magnitude
    training_steps = range(len(losses))
    ax.plot(training_steps, losses, color=COLORS['secondary'], alpha=0.7, linewidth=1.5)
    
    # Add smoothed loss if enough data
    if len(losses) > 50:
        smooth_steps, smoothed_losses = smooth_curve(losses, window_size=min(100, len(losses)//5))
        ax.plot(smooth_steps, smoothed_losses, color=COLORS['primary'], 
               linewidth=2.5, label='Smoothed')
        ax.legend()
    
    ax.set_title('Training Loss (Log Scale)')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('MSE Loss')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

def _plot_epsilon_decay(ax, training_history: List[Dict[str, Any]]) -> None:
    """Plot epsilon decay schedule."""
    epsilons = [step.get('epsilon', 0) for step in training_history if 'epsilon' in step]
    
    if not epsilons:
        ax.text(0.5, 0.5, 'No epsilon data available', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Epsilon Decay')
        return
    
    episodes = range(len(epsilons))
    ax.plot(episodes, epsilons, color=COLORS['accent'], linewidth=2.5)
    
    # Add exploration phases
    ax.axhline(0.1, color=COLORS['warning'], alpha=0.6, linestyle='--', 
              label='Low Exploration (ε=0.1)')
    ax.axhline(0.01, color=COLORS['success'], alpha=0.6, linestyle='--', 
              label='Minimal Exploration (ε=0.01)')
    
    ax.set_title('Epsilon Decay Schedule')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Epsilon (Exploration Rate)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, max(epsilons) * 1.1)

def _plot_q_value_distributions(ax, trainer, plotting_every: int) -> None:
    """Plot Q-value distributions over time."""
    if not hasattr(trainer, 'q_value_samples') or not trainer.q_value_samples:
        ax.text(0.5, 0.5, 'No Q-value samples available', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Q-value Distributions')
        return
    
    colors = sns.color_palette("viridis", len(trainer.q_value_samples))
    
    for i, (q_vals, color) in enumerate(zip(trainer.q_value_samples, colors)):
        episode = i * plotting_every
        ax.hist(q_vals, bins=25, alpha=0.6, color=color, 
               label=f'Episode {episode}', density=True)
    
    ax.set_title('Q-value Distributions')
    ax.set_xlabel('Q-value')
    ax.set_ylabel('Density')
    if len(trainer.q_value_samples) <= 5:
        ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

def _plot_performance_stats(ax, episode_rewards: List[float]) -> None:
    """Plot performance statistics summary."""
    if len(episode_rewards) < 10:
        ax.text(0.5, 0.5, 'Insufficient data', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Performance Statistics')
        return
    
    # Calculate rolling statistics
    window_sizes = [10, 50, 100]
    stats_data = []
    
    for window in window_sizes:
        if len(episode_rewards) >= window:
            rolling_means = [np.mean(episode_rewards[max(0, i-window):i+1]) 
                           for i in range(len(episode_rewards))]
            stats_data.append((f'{window}-episode avg', rolling_means[-1]))
    
    # Overall statistics
    stats_data.extend([
        ('Overall mean', np.mean(episode_rewards)),
        ('Best episode', np.max(episode_rewards)),
        ('Worst episode', np.min(episode_rewards)),
        ('Std deviation', np.std(episode_rewards))
    ])
    
    # Create bar plot
    labels, values = zip(*stats_data)
    colors_list = [COLORS['primary'], COLORS['secondary'], COLORS['accent'], 
                  COLORS['success'], COLORS['warning'], COLORS['muted']]
    
    bars = ax.bar(range(len(labels)), values, 
                 color=colors_list[:len(labels)], alpha=0.8)
    
    ax.set_title('Performance Statistics')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Reward')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{value:.1f}', ha='center', va='bottom', fontsize=9)

def create_comparison_plot(results_dict: Dict[str, List[float]], 
                          save_path: str = "training_results/comparison.png",
                          title: str = "Training Comparison") -> None:
    """
    Create a comparison plot for multiple training runs.
    
    Args:
        results_dict: Dictionary mapping run names to reward lists
        save_path: Path to save the comparison plot
        title: Plot title
    """
    setup_plot_style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    colors = sns.color_palette("husl", len(results_dict))
    
    # Plot 1: Smoothed reward curves
    for (name, rewards), color in zip(results_dict.items(), colors):
        episodes = range(len(rewards))
        ax1.plot(episodes, rewards, alpha=0.3, color=color, linewidth=1)
        
        if len(rewards) > 50:
            smooth_episodes, smoothed = smooth_curve(rewards)
            ax1.plot(smooth_episodes, smoothed, color=color, 
                    linewidth=2.5, label=name)
    
    ax1.set_title('Reward Curves Comparison')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Final performance comparison
    final_performances = []
    names = []
    for name, rewards in results_dict.items():
        if len(rewards) >= 100:
            final_perf = np.mean(rewards[-100:])
        else:
            final_perf = np.mean(rewards)
        final_performances.append(final_perf)
        names.append(name)
    
    bars = ax2.bar(names, final_performances, color=colors, alpha=0.8)
    ax2.set_title('Final Performance Comparison')
    ax2.set_ylabel('Average Reward (last 100 episodes)')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, value in zip(bars, final_performances):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2f}', ha='center', va='bottom')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Comparison plot saved to: {save_path}")
    
    plt.show(block=False)
    plt.pause(2.0)
    plt.close('all')
