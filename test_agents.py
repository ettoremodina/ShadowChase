#!/usr/bin/env python3
"""
Simple agent testing script for Scotland Yard.
Runs all agent combinations and generates analysis.
"""

import subprocess
import os


def play_combination(test_name, mr_x_agent, detective_agent, games_per_combo, map_size, num_detectives, max_turns):
    """Helper function to run a single agent combination"""
    combo = f"{mr_x_agent}_vs_{detective_agent}"
    save_dir = f"{test_name}/{combo}"

    print(f"\nRunning: {combo}")
    cmd = [
        "python", "simple_play/simple_game.py",
        "--batch", str(games_per_combo),
        "--map-size", map_size,
        "--detectives", str(num_detectives),
        "--max-turns", str(max_turns),
        "--save-dir", save_dir,
        "--mr-x-agent", mr_x_agent,
        "--detective-agent", detective_agent,
        "--verbosity", "0"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"✓ Completed: {combo}")
    except subprocess.CalledProcessError:
        print(f"✗ Failed: {combo}")


def main():
    # Import AgentSelector inside the function to avoid circular imports
    from agents import AgentSelector, AgentType
    
    # Configuration
    test_name = "test_optimization" 
    games_per_combo = 3
    num_detectives = 5
    map_size = "extracted"

    
    # Get agent types dynamically
    agent_types = [agent_name[0] for agent_name in AgentSelector.get_agent_choices_for_ui()]
    # agent_types = ["mcts",  "random"]
    print(f"Testing agents: {agent_types}")
    print(f"Games per combination: {games_per_combo}")
    print(f"Test directory: {test_name}")
    # play_combination(test_name, "mcts", "mcts", games_per_combo, map_size, num_detectives, 24)
    play_combination(test_name, "optimized_mcts", "optimized_mcts", games_per_combo, map_size, num_detectives, 24)

    # RUN ALL COMBINATIONS
    # for mr_x in agent_types:
    #     for detective in agent_types:
    #         play_combination(test_name, mr_x, detective, games_per_combo, map_size, num_detectives, 24)




    # Generate analysis
    print(f"\nGenerating analysis...")
    try:
        subprocess.run(["python", "analyze_games.py", test_name], check=True)
        print(f"✓ Analysis complete: {test_name}/analysis_graphs/")
    except subprocess.CalledProcessError:
        print("✗ Analysis failed")

if __name__ == "__main__":
    main()
