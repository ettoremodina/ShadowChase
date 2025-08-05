"""
Utility functions for creating and configuring Shadow Chase PettingZoo environments.
"""

from typing import Dict, Any, Optional
from .shadow_chase_env import ShadowChaseEnv
from training.feature_extractor_simple import FeatureConfig


def create_shadow_chase_env(
    map_size: str = "extended",
    num_detectives: int = 5,
    max_turns: int = 24,
    render_mode: Optional[str] = None,
    feature_config: Optional[Dict[str, Any]] = None,
) -> ShadowChaseEnv:
    """
    Create a Shadow Chase PettingZoo environment with specified configuration.
    
    Args:
        map_size: Size of the map ("test" for 10 nodes, "full" for 199 nodes)
        num_detectives: Number of detectives (2-5)
        max_turns: Maximum number of turns before game is truncated
        render_mode: Rendering mode ("human" or "ansi")
        feature_config: Configuration dictionary for feature extraction
        
    Returns:
        Configured Shadow Chase environment
    """
    # Create feature config if provided
    feat_config = None
    if feature_config is not None:
        feat_config = FeatureConfig(**feature_config)
    
    return ShadowChaseEnv(
        map_size=map_size,
        num_detectives=num_detectives,
        max_turns=max_turns,
        render_mode=render_mode,
        feature_config=feat_config,
    )


def get_default_feature_config() -> Dict[str, Any]:
    """
    Get default feature extraction configuration optimized for RL training.
    
    Returns:
        Dictionary with default feature extraction settings
    """
    return {
        "include_distances": True,
        "include_tickets": True,
        "include_board_state": True,
        "include_game_phase": True,
        "include_transport_connectivity": True,
        "include_possible_positions": True,
        "max_nodes": 200,
        "distance_normalization": 20.0,
    }


def get_minimal_feature_config() -> Dict[str, Any]:
    """
    Get minimal feature extraction configuration for faster training.
    
    Returns:
        Dictionary with minimal feature extraction settings
    """
    return {
        "include_distances": False,
        "include_tickets": True,
        "include_board_state": True,
        "include_game_phase": True,
        "include_transport_connectivity": False,
        "include_possible_positions": False,
        "max_nodes": 50,
        "distance_normalization": 20.0,
    }


def get_rich_feature_config() -> Dict[str, Any]:
    """
    Get rich feature extraction configuration for complex RL experiments.
    
    Returns:
        Dictionary with comprehensive feature extraction settings
    """
    return {
        "include_distances": True,
        "include_tickets": True,
        "include_board_state": True,
        "include_game_phase": True,
        "include_transport_connectivity": True,
        "include_possible_positions": True,
        "max_nodes": 200,
        "distance_normalization": 30.0,
    }


def create_test_env() -> ShadowChaseEnv:
    """
    Create a test environment with minimal configuration for quick experiments.
    
    Returns:
        Test Shadow Chase environment
    """
    return create_shadow_chase_env(
        map_size="test",
        num_detectives=2,
        max_turns=50,
        render_mode="ansi",
        feature_config=get_minimal_feature_config()
    )


def create_training_env() -> ShadowChaseEnv:
    """
    Create an environment optimized for training RL agents.
    
    Returns:
        Training-optimized Shadow Chase environment
    """
    return create_shadow_chase_env(
        map_size="extended",
        num_detectives=5,
        max_turns=55,
        render_mode=None,  # No rendering for faster training
        feature_config=get_default_feature_config()
    )


def create_evaluation_env() -> ShadowChaseEnv:
    """
    Create an environment for evaluating trained agents.
    
    Returns:
        Evaluation-optimized Shadow Chase environment
    """
    return create_shadow_chase_env(
        map_size="extended",
        num_detectives=2,
        max_turns=24,
        render_mode="human",
        feature_config=get_rich_feature_config()
    )
