#!/usr/bin/env python3
"""
Simple agent testing script for Scotland Yard.
Runs all agent combinations and generates analysis.
"""

import subprocess
import os

def main():
    # Import AgentSelector inside the function to avoid circular imports
    from agents import AgentSelector, AgentType
    
    # Configuration
    test_name = "test0" 
    games_per_combo = 30
    num_detectives = 5
    map_size = "extracted"

    
    # Get agent types dynamically
    agent_types = [agent_name[0] for agent_name in AgentSelector.get_agent_choices_for_ui()]
    print(f"Testing agents: {agent_types}")
    print(f"Games per combination: {games_per_combo}")
    print(f"Test directory: {test_name}")

    # Run all combinations
    for mr_x in agent_types:
        for detective in agent_types:
            combo = f"{mr_x}_vs_{detective}"
            save_dir = f"{test_name}/{combo}"

            print(f"\nRunning: {combo}")
            cmd = [
                "python", "simple_play/simple_game.py",
                "--batch", str(games_per_combo),
                "--map-size", map_size,
                "--detectives", str(num_detectives),
                "--max-turns", "24",
                "--save-dir", save_dir,
                "--mr-x-agent", mr_x,
                "--detective-agent", detective,
                "--verbosity", "0"
            ]
            
            try:
                subprocess.run(cmd, check=True)
                print(f"✓ Completed: {combo}")
            except subprocess.CalledProcessError:
                print(f"✗ Failed: {combo}")
    
    # Generate analysis
    print(f"\nGenerating analysis...")
    try:
        subprocess.run(["python", "analyze_games.py", test_name], check=True)
        print(f"✓ Analysis complete: {test_name}/analysis_graphs/")
    except subprocess.CalledProcessError:
        print("✗ Analysis failed")

if __name__ == "__main__":
    main()
