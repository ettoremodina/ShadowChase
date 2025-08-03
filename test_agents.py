#!/usr/bin/env python3
"""
Simple agent testing script for Shadow Chase.
Runs all agent combinations and generates analysis.
"""

import subprocess
import os
import sys
from ShadowChase.services.cache_system import (
    enable_cache, disable_cache, is_cache_enabled, get_global_cache,
    enable_namespace_cache, disable_namespace_cache, is_namespace_cache_enabled,
    reset_namespace_cache_settings, get_cache_status, CacheNamespace
)

from agents import AgentSelector

def play_combination(test_name, mr_x_agent, detective_agent, games_per_combo, map_size, num_detectives, max_turns):
    """Helper function to run a single agent combination"""
    combo = f"{mr_x_agent}_vs_{detective_agent}"
    save_dir = f"{test_name}/{combo}"

    print(f"\nRunning: {combo}")
    # Use the same Python executable that's running this script
    cmd = [
        sys.executable, "game_controls/simple_game.py",
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

def analyze_games(test_name):
    """Analyze the games played in the test"""
    print(f"\nAnalyzing games in {test_name}...")
    try:
        subprocess.run([sys.executable, "ShadowChase/services/analyze_games.py", test_name], check=True)
        print(f"✓ Analysis complete: {test_name}/analysis_graphs/")
    except subprocess.CalledProcessError:
        print("✗ Analysis failed")


def main():
    # Import AgentSelector inside the function to avoid circular imports
    
    # Configuration
    test_name = "video_exporting_test" 
    games_per_combo = 10
    num_detectives = 5
    map_size = "extracted"
    enable_cache()
    reset_namespace_cache_settings()
    disable_namespace_cache(CacheNamespace.GAME_METHODS)
    enable_namespace_cache(CacheNamespace.MCTS_NODES)
    enable_namespace_cache(CacheNamespace.AGENT_DECISIONS)
        

    
    # Get agent types dynamically
    agent_types = [agent_name[0] for agent_name in AgentSelector.get_agent_choices_for_ui()]
    # agent_types = ["deep_q", "random"]
    print(f"Testing agents: {agent_types}")
    print(f"Games per combination: {games_per_combo}")
    print(f"Test directory: {test_name}")

    play_combination(test_name, "random", "random", games_per_combo, map_size, num_detectives, 24)


    # RUN ALL COMBINATIONS
    run_all_combinations = False
    
    if run_all_combinations:
        for mr_x in agent_types:
            for detective in agent_types:
                if mr_x != detective:
                    play_combination(test_name, mr_x, detective, games_per_combo, map_size, num_detectives, 24)



    analyze_games(test_name)

if __name__ == "__main__":
    main()
