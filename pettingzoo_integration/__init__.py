"""
PettingZoo integration for Shadow Chase / Scotland Yard game.

This module provides PettingZoo-compatible environments for the Shadow Chase game,
allowing for multi-agent reinforcement learning experiments.
"""

# Conditional imports to handle missing dependencies gracefully
try:
    from .shadow_chase_env import ShadowChaseEnv, shadow_chase_v1
    from .env_utils import (
        create_shadow_chase_env, 
        create_test_env, 
        create_training_env, 
        create_evaluation_env,
        get_default_feature_config,
        get_minimal_feature_config,
        get_rich_feature_config
    )
    
    __all__ = [
        "ShadowChaseEnv", 
        "shadow_chase_v1",
        "create_shadow_chase_env", 
        "create_test_env",
        "create_training_env",
        "create_evaluation_env",
        "get_default_feature_config",
        "get_minimal_feature_config", 
        "get_rich_feature_config"
    ]
    
    PETTINGZOO_AVAILABLE = True
    
except ImportError as e:
    print(f"PettingZoo integration not available: {e}")
    print("Install with: pip install pettingzoo gymnasium")
    
    # Provide dummy implementations
    __all__ = []
    PETTINGZOO_AVAILABLE = False
