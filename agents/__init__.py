"""
Agents package for Scotland Yard game.

This package contains agent implementations for both detectives and Mr. X,
including basic random agents and more sophisticated AI strategies.
"""

from .base_agent import Agent, DetectiveAgent, MrXAgent, MultiDetectiveAgent
from .random_agent import RandomMrXAgent, RandomMultiDetectiveAgent
from .heuristic_agent import HeuristicMrXAgent, HeuristicMultiDetectiveAgent, HeuristicDetectiveAgent
from .agent_registry import (
    AgentType, AgentRegistry, agent_registry, get_agent_registry,
    AgentSelector, create_agents_from_types, create_agents_from_strings
)
from .heuristics import GameHeuristics

__all__ = [
    'Agent',
    'DetectiveAgent', 
    'MrXAgent',
    'MultiDetectiveAgent',
    'RandomDetectiveAgent',
    'RandomMrXAgent',
    'RandomMultiDetectiveAgent',
    'SmartRandomMrXAgent',
]
