"""
Example usage of Shadow Chase PettingZoo environment.

This script demonstrates how to use the Shadow Chase environment with PettingZoo
for multi-agent reinforcement learning experiments.
"""

import sys
import os
import numpy as np
from typing import Dict, Any

# Add the parent directory to Python path to enable imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from pettingzoo_integration import  create_test_env, create_training_env


def random_policy(observation: np.ndarray, action_mask: np.ndarray, action_space) -> int:
    """
    Simple random policy that respects action masks.
    
    Args:
        observation: Current observation (not used in random policy)
        action_mask: Boolean mask of valid actions
        action_space: The action space definition
        
    Returns:
        Random valid action
    """
    if hasattr(action_space, 'nvec'):
        # MultiDiscrete action space - need to respect action mask
        mask = action_mask.reshape(action_space.nvec)
        
        # Find valid action combinations
        valid_combinations = np.where(mask)
        if len(valid_combinations[0]) > 0:
            # Choose random valid combination
            idx = np.random.randint(len(valid_combinations[0]))
            action = np.array([valid_combinations[i][idx] for i in range(len(valid_combinations))])
            return action
        else:
            # Fallback: return zeros if no valid actions (shouldn't happen)
            return np.zeros(len(action_space.nvec), dtype=int)
    else:
        # Discrete action space
        valid_actions = np.where(action_mask)[0]
        if len(valid_actions) > 0:
            return np.random.choice(valid_actions)
        else:
            return 0  # Fallback


def run_random_game(render: bool = True) -> Dict[str, Any]:
    """
    Run a complete game with random policies for both agents.
    
    Args:
        render: Whether to render the game state
        
    Returns:
        Dictionary with game statistics
    """  
    # Create environment
    # env = create_test_env()
    env = create_training_env()
    
    # Initialize statistics
    stats = {
        "total_steps": 0,
        "winner": None,
        "final_rewards": {},
        "game_length": 0
    }
    
    # Reset environment
    observations, infos = env.reset()
    
    if render:
        print("Starting Shadow Chase game with random agents...")
        print(env.render())
    
    # Game loop
    while env.agents:
        # Get current agent
        agent = env.agent_selection
        
        # Check if agent is terminated or truncated
        if env.terminations[agent] or env.truncations[agent]:
            # Agent is dead - pass None action
            env.step(None)
            continue
        
        # Get observation and action mask
        observation = env.observe(agent)
        action_mask = env.action_mask(agent)
        action_space = env.action_spaces[agent]
        
        # Choose random action
        action = random_policy(observation, action_mask, action_space)
        
        if render:
            print(f"\n{agent.upper()} takes action: {action}")
        
        # Step environment
        env.step(action)
        
        stats["total_steps"] += 1
        
        if render:
            print(env.render())
        
        # Check if game ended
        if not env.agents:
            break
    
    # Collect final statistics
    stats["final_rewards"] = env.rewards.copy()
    stats["game_length"] = stats["total_steps"]
    
    # Determine winner
    if env.rewards["mrx"] > 0:
        stats["winner"] = "mrx"
    elif any(env.rewards[f"detective_{i}"] > 0 for i in range(env.num_detectives)):
        stats["winner"] = "detectives"
    else:
        stats["winner"] = "draw"
    
    if render:
        print(f"\nGame Over!")
        print(f"Winner: {stats['winner']}")
        print(f"Final rewards: {stats['final_rewards']}")
        print(f"Game length: {stats['game_length']} steps")
    
    env.close()
    return stats


def run_multiple_games(num_games: int = 10) -> Dict[str, Any]:
    """
    Run multiple games and collect statistics.
    
    Args:
        num_games: Number of games to run
        
    Returns:
        Aggregated statistics
    """    
    print(f"Running {num_games} random games...")
    
    all_stats = []
    win_counts = {"mrx": 0, "detectives": 0, "draw": 0}
    total_steps = 0
    
    for i in range(num_games):
        print(f"Game {i+1}/{num_games}", end="... ")
        stats = run_random_game(render=False)
        all_stats.append(stats)
        
        win_counts[stats["winner"]] += 1
        total_steps += stats["total_steps"]
        
        print(f"Winner: {stats['winner']}")
    
    # Calculate aggregate statistics
    avg_game_length = total_steps / num_games
    mrx_win_rate = win_counts["mrx"] / num_games
    detective_win_rate = win_counts["detectives"] / num_games
    draw_rate = win_counts["draw"] / num_games
    
    aggregate_stats = {
        "num_games": num_games,
        "win_counts": win_counts,
        "win_rates": {
            "mrx": mrx_win_rate,
            "detectives": detective_win_rate,
            "draw": draw_rate
        },
        "avg_game_length": avg_game_length,
        "total_steps": total_steps
    }
    
    print(f"\nAggregate Statistics:")
    print(f"Mr. X wins: {win_counts['mrx']} ({mrx_win_rate:.1%})")
    print(f"Detective wins: {win_counts['detectives']} ({detective_win_rate:.1%})")
    print(f"Draws: {win_counts['draw']} ({draw_rate:.1%})")
    print(f"Average game length: {avg_game_length:.1f} steps")
    
    return aggregate_stats


def test_environment_api():
    """Test the basic PettingZoo API compliance."""    
    env = create_test_env()
    
    # Test reset
    observations, infos = env.reset()
    print(f"✓ Reset successful")
    print(f"  Agents: {env.agents}")
    print(f"  Observation shapes: {[obs.shape for obs in observations.values()]}")
    
    # Test action spaces
    print(f"✓ Action spaces defined")
    for agent in env.possible_agents:
        print(f"  {agent}: {env.action_spaces[agent]}")
    
    # Test observation spaces  
    print(f"✓ Observation spaces defined")
    for agent in env.possible_agents:
        print(f"  {agent}: {env.observation_spaces[agent]}")
    
    # Test action masks
    print(f"✓ Action masks work")
    for agent in env.agents:
        mask = env.action_mask(agent)
        print(f"  {agent}: mask shape {mask.shape}")
    
    # Test one step
    agent = env.agent_selection
    action_space = env.action_spaces[agent]
    
    if hasattr(action_space, 'nvec'):
        action = np.random.randint(0, action_space.nvec)
    else:
        action = action_space.sample()
    
    env.step(action)
    print(f"✓ Step successful")
    
    env.close()
    print("✓ Environment API test complete")


if __name__ == "__main__":
    # Test the API
    test_environment_api()
    
    print("\n" + "="*50)
    
    # Run a single game with rendering
    print("Running single game with random agents:")
    run_random_game(render=True)
    
    print("\n" + "="*50)
    
    # Run multiple games for statistics
    # run_multiple_games(num_games=5)
