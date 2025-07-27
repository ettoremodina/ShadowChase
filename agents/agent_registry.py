"""
Agent Registry for Scotland Yard game.

This module provides a registry system for managing different AI agent implementations.
It allows easy selection and instantiation of agents for both terminal and UI interfaces.
"""

from typing import Dict, Tuple, List, Type, Callable
from enum import Enum
from ScotlandYard.core.game import Player
from .base_agent import MrXAgent, MultiDetectiveAgent, DetectiveAgent
from .random_agent import RandomMrXAgent, RandomMultiDetectiveAgent
from .heuristic_agent import HeuristicMrXAgent, HeuristicMultiDetectiveAgent
from .mcts_agent import MCTSMrXAgent, MCTSMultiDetectiveAgent
from .optimized_mcts_agent import OptimizedMCTSMrXAgent, OptimizedMCTSMultiDetectiveAgent

class AgentType(Enum):
    """Available agent types"""
    RANDOM = "random"
    HEURISTIC = "heuristic"
    MCTS = "mcts"
    OPTIMIZED_MCTS = "optimized_mcts"
    # Future agent types can be added here
    # MINIMAX = "minimax"
    # DEEP_Q = "deep_q"


class AgentRegistry:
    """Registry for managing different AI agent implementations"""
    
    def __init__(self):
        """Initialize the agent registry with available agents"""
        self._mr_x_agents: Dict[AgentType, Tuple[Type[MrXAgent], str]] = {
            AgentType.RANDOM: (RandomMrXAgent, "Random Mr. X - Makes random valid moves"),
            AgentType.HEURISTIC: (HeuristicMrXAgent, "Heuristic Mr. X - Maximizes distance from closest detective"),
            AgentType.MCTS: (MCTSMrXAgent, "MCTS Mr. X - Uses Monte Carlo Tree Search with random simulations"),
            AgentType.OPTIMIZED_MCTS: (OptimizedMCTSMrXAgent, "Optimized MCTS Mr. X - Fast MCTS with caching and minimal deep copying")
        }
        
        self._multi_detective_agents: Dict[AgentType, Tuple[Type[MultiDetectiveAgent], str]] = {
            AgentType.RANDOM: (RandomMultiDetectiveAgent, "Random Detectives - Make random valid moves"),
            AgentType.HEURISTIC: (HeuristicMultiDetectiveAgent, "Heuristic Detectives - Minimize distance to Mr. X's last known position"),
            AgentType.MCTS: (MCTSMultiDetectiveAgent, "MCTS Detectives - Use Monte Carlo Tree Search with random simulations"),
            AgentType.OPTIMIZED_MCTS: (OptimizedMCTSMultiDetectiveAgent, "Optimized MCTS Detectives - Fast MCTS with caching and minimal deep copying")
        }
    
    def get_available_agent_types(self) -> List[AgentType]:
        """Get list of available agent types"""
        return list(self._mr_x_agents.keys())
    
    def get_agent_description(self, agent_type: AgentType, player: Player) -> str:
        """Get description of an agent type for a specific player"""
        if player == Player.MRX:
            return self._mr_x_agents[agent_type][1]
        else:
            return self._multi_detective_agents[agent_type][1]
    
    def get_agent_display_name(self, agent_type: AgentType) -> str:
        """Get display name for an agent type"""
        display_names = {
            AgentType.RANDOM: "Random AI",
            AgentType.HEURISTIC: "Heuristic AI",
            AgentType.MCTS: "MCTS AI",
            AgentType.OPTIMIZED_MCTS: "Optimized MCTS AI"
        }
        return display_names.get(agent_type, str(agent_type.value).title())
    
    def create_mr_x_agent(self, agent_type: AgentType) -> MrXAgent:
        """Create a Mr. X agent of the specified type"""
        if agent_type not in self._mr_x_agents:
            raise ValueError(f"Unknown Mr. X agent type: {agent_type}")
        
        agent_class = self._mr_x_agents[agent_type][0]
        
        return agent_class()
    
    def create_multi_detective_agent(self, agent_type: AgentType, num_detectives: int) -> MultiDetectiveAgent:
        """Create a multi-detective agent of the specified type"""
        if agent_type not in self._multi_detective_agents:
            raise ValueError(f"Unknown multi-detective agent type: {agent_type}")
        
        agent_class = self._multi_detective_agents[agent_type][0]
    
        return agent_class(num_detectives)
    
    def register_mr_x_agent(self, agent_type: AgentType, agent_class: Type[MrXAgent], description: str):
        """Register a new Mr. X agent type"""
        self._mr_x_agents[agent_type] = (agent_class, description)
    
    def register_multi_detective_agent(self, agent_type: AgentType, agent_class: Type[MultiDetectiveAgent], description: str):
        """Register a new multi-detective agent type"""
        self._multi_detective_agents[agent_type] = (agent_class, description)


# Global registry instance
agent_registry = AgentRegistry()


def get_agent_registry() -> AgentRegistry:
    """Get the global agent registry instance"""
    return agent_registry


class AgentSelector:
    """Helper class for selecting agents in terminal and UI interfaces"""
    
    @staticmethod
    def display_agent_menu(title: str = "Select AI Agent") -> None:
        """Display the agent selection menu"""
        print(f"\n{title}")
        print("=" * len(title))
        
        registry = get_agent_registry()
        agent_types = registry.get_available_agent_types()
        
        for i, agent_type in enumerate(agent_types, 1):
            display_name = registry.get_agent_display_name(agent_type)
            mr_x_desc = registry.get_agent_description(agent_type, Player.MRX)
            detective_desc = registry.get_agent_description(agent_type, Player.DETECTIVES)
            
            print(f"{i}. {display_name}")
            print(f"   Mr. X: {mr_x_desc}")
            print(f"   Detectives: {detective_desc}")
            print()
    
    @staticmethod
    def get_user_agent_choice(prompt: str = "Choose an agent type") -> AgentType:
        """Get user's agent choice from terminal input"""
        registry = get_agent_registry()
        agent_types = registry.get_available_agent_types()
        
        while True:
            try:
                AgentSelector.display_agent_menu()
                choice = input(f"{prompt} (1-{len(agent_types)}): ").strip()
                
                if not choice.isdigit():
                    print("Please enter a valid number.")
                    continue
                
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(agent_types):
                    return agent_types[choice_idx]
                else:
                    print(f"Please enter a number between 1 and {len(agent_types)}.")
            
            except (ValueError, KeyboardInterrupt):
                print("Invalid input. Please try again.")
                continue
    
    @staticmethod
    def get_agent_choices_for_ui() -> List[Tuple[str, str]]:
        """Get agent choices formatted for UI dropdown/selection"""
        registry = get_agent_registry()
        agent_types = registry.get_available_agent_types()
        
        choices = []
        for agent_type in agent_types:
            value = agent_type.value
            display_name = registry.get_agent_display_name(agent_type)
            choices.append((value, display_name))
        
        return choices


def create_agents_from_types(mr_x_type: AgentType, detective_type: AgentType, 
                           num_detectives: int) -> Tuple[MrXAgent, MultiDetectiveAgent]:
    """Create agent instances from agent types"""
    registry = get_agent_registry()
    
    mr_x_agent = registry.create_mr_x_agent(mr_x_type)
    detective_agent = registry.create_multi_detective_agent(detective_type, num_detectives)
    
    return mr_x_agent, detective_agent


def create_agents_from_strings(mr_x_type_str: str, detective_type_str: str, 
                             num_detectives: int) -> Tuple[MrXAgent, MultiDetectiveAgent]:
    """Create agent instances from string representations of agent types"""
    mr_x_type = AgentType(mr_x_type_str)
    detective_type = AgentType(detective_type_str)
    
    return create_agents_from_types(mr_x_type, detective_type, num_detectives)
