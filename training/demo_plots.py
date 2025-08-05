"""
Example usage of the enhanced plotting utilities.
"""

import numpy as np
from training.plot_utils import create_comparison_plot, setup_plot_style

def generate_sample_data():
    """Generate sample training data for demonstration."""
    episodes = 1000
    
    # Simulate different agent performances
    agents = {
        'DQN Agent': {
            'base_reward': 50,
            'improvement_rate': 0.001,
            'noise_level': 15
        },
        'MCTS Agent': {
            'base_reward': 30,
            'improvement_rate': 0.0008,
            'noise_level': 20
        },
        'Heuristic Agent': {
            'base_reward': 20,
            'improvement_rate': 0.0003,
            'noise_level': 8
        },
        'Random Agent': {
            'base_reward': 10,
            'improvement_rate': 0.0001,
            'noise_level': 5
        }
    }
    
    results = {}
    
    for agent_name, params in agents.items():
        # Generate synthetic training curve
        episode_range = np.arange(episodes)
        
        # Base trend (learning curve)
        trend = params['base_reward'] + params['improvement_rate'] * episode_range**1.5
        
        # Add realistic noise
        noise = np.random.normal(0, params['noise_level'], episodes)
        
        # Add some occasional dips and spikes
        spikes = np.random.choice([0, 1], episodes, p=[0.95, 0.05])
        spike_magnitude = np.random.normal(0, params['noise_level'] * 2, episodes)
        
        rewards = trend + noise + spikes * spike_magnitude
        
        # Ensure rewards don't go negative
        rewards = np.maximum(rewards, 0)
        
        results[agent_name] = rewards.tolist()
    
    return results

def main():
    """Demonstrate the plotting utilities."""
    print("Generating sample training comparison...")
    
    # Setup the plotting style
    setup_plot_style()
    
    # Generate sample data
    sample_results = generate_sample_data()
    
    # Create comparison plot
    create_comparison_plot(
        sample_results, 
        save_path="training_results/example_comparison.png",
        title="Sample Agent Training Comparison"
    )
    
    print("Example plots generated successfully!")
    print("   Check 'training_results/example_comparison.png' to see the enhanced aesthetics")

if __name__ == "__main__":
    main()
