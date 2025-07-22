#!/usr/bin/env python3
"""
Quick test for agent import
"""
import importlib

print("Testing dynamic import of agents...")

try:
    agents_module = importlib.import_module('agents')
    print("✓ Successfully imported agents module")
    
    AgentSelector = agents_module.AgentSelector  
    print("✓ Successfully accessed AgentSelector")
    
    agent_types = AgentSelector.get_agent_choices_for_ui()
    print(f"✓ Successfully got agent types: {agent_types}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
