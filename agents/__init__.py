"""Compatibility shim: ``agents`` moved to ``shadow_chase.agents``.

Nothing under ``agents/`` is pickled directly (DQN checkpoints store only
``state_dict`` tensors, plain strings, and JSON-loaded config), so this shim
exists for import compatibility, not for unpickling: existing call sites using
``from agents import ...`` or ``agents.dqn_agent.DQNMrXAgent`` keep resolving
without edits. New code should import ``shadow_chase.agents`` directly.
"""

import importlib
import importlib.util
import sys

# Re-exporting the submodules, not just the package, is what makes
# `from agents.heuristics import ...` keep working: a plain
# `from shadow_chase.agents import *` only aliases the names already imported
# into shadow_chase/agents/__init__.py, not the submodules themselves.
for _name in (
    "agent_registry",
    "base_agent",
    "epsilon_greedy_mcts_agent",
    "heuristic_agent",
    "heuristics",
    "mcts_agent",
    "optimized_mcts_agent",
    "random_agent",
):
    _module = importlib.import_module(f"shadow_chase.agents.{_name}")
    sys.modules[f"{__name__}.{_name}"] = _module
    globals()[_name] = _module

# dqn_agent stays lazy, matching the original package: it is the only
# submodule that pulls in the training package, and training.deep_q.dqn_trainer
# does `from agents import AgentType` at module level. Eagerly importing
# dqn_agent here would execute that line while this shim is still mid-init and
# AgentType is not bound yet, raising a circular-import error that never
# happened before the move, since agent_registry only imports dqn_agent lazily,
# inside a method, on first request for a DQN agent.
_dqn_spec = importlib.util.find_spec("shadow_chase.agents.dqn_agent")
_dqn_loader = importlib.util.LazyLoader(_dqn_spec.loader)
_dqn_spec.loader = _dqn_loader
dqn_agent = importlib.util.module_from_spec(_dqn_spec)
# Both names must point at this one module object, or a later plain
# `import shadow_chase.agents.dqn_agent` finds nothing in sys.modules under
# its real name, imports the file a second time, and produces a second,
# distinct DQNMrXAgent class that fails identity checks against this one.
sys.modules[f"{__name__}.dqn_agent"] = dqn_agent
sys.modules[_dqn_spec.name] = dqn_agent
setattr(sys.modules["shadow_chase.agents"], "dqn_agent", dqn_agent)
_dqn_loader.exec_module(dqn_agent)

from shadow_chase.agents import *  # noqa: E402,F401,F403
from shadow_chase.agents import __all__  # noqa: E402
