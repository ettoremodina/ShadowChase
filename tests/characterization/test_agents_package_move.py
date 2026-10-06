"""Characterize the ``agents`` -> ``shadow_chase.agents`` compatibility shim.

``agents/`` moved to ``src/shadow_chase/agents/`` in the physical
reorganization (checkpoint 5). The old import path is kept alive by
``agents/__init__.py`` so existing call sites (``train_dqn.py``,
``game_controls/``, ``training/``) keep working unedited. These tests pin the
two properties that make the shim safe: identical class identity between the
old and new paths, and no new circular-import failure from eagerly loading
``dqn_agent``, which the original package only ever loaded lazily.
"""

import subprocess
import sys

import pytest


def test_shim_and_new_path_resolve_to_the_same_objects():
    """Verify the legacy path is an alias, not a copy."""
    import agents
    import shadow_chase.agents as new_agents

    assert agents.AgentType is new_agents.AgentType
    assert agents.RandomMrXAgent is new_agents.RandomMrXAgent
    assert agents.GameHeuristics is new_agents.GameHeuristics


def test_shim_exposes_submodules_for_absolute_imports():
    """Verify `from agents.<submodule> import X`, used across the repo, still works."""
    from agents.heuristics import GameHeuristics
    from agents.random_agent import RandomMrXAgent
    from agents.dqn_agent import DQNMrXAgent, DQNMultiDetectiveAgent

    import shadow_chase.agents.heuristics as new_heuristics
    import shadow_chase.agents.dqn_agent as new_dqn_agent

    assert GameHeuristics is new_heuristics.GameHeuristics
    assert RandomMrXAgent is not None
    assert DQNMrXAgent is new_dqn_agent.DQNMrXAgent
    assert DQNMultiDetectiveAgent is new_dqn_agent.DQNMultiDetectiveAgent


@pytest.mark.integration
def test_dqn_agent_import_succeeds_as_the_first_touch_in_a_fresh_process(project_root):
    """Verify importing agents.dqn_agent first, before anything else, still works.

    agents.dqn_agent and training.deep_q.dqn_trainer reference each other by
    absolute path. Whichever one a process imports first pulls in the other
    mid-initialization, and this exists in the original code too — the shim
    keeps dqn_agent lazy specifically so this order-dependent tangle isn't
    forced to resolve while the shim package is still mid-init.
    """
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from agents.dqn_agent import DQNMrXAgent, DQNMultiDetectiveAgent; "
            "print(DQNMrXAgent.__name__)",
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "DQNMrXAgent" in completed.stdout
