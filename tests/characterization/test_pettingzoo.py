"""Characterize the current PettingZoo adapter surface."""

import pytest

from ml_logger import get_logger
from pettingzoo_integration import ShadowChaseEnv, create_test_env


logger = get_logger(__name__)


@pytest.mark.integration
def test_pettingzoo_environment_resets_with_expected_shapes(
    project_root,
    monkeypatch,
):
    """Freeze agent selection and observation/action-mask dimensions."""
    monkeypatch.chdir(project_root)
    monkeypatch.setattr(ShadowChaseEnv, "_setup_game_service", lambda self: None)
    environment = create_test_env()
    try:
        agent = environment.agent_selection
        observation = environment.observe(agent)
        action_mask = environment.action_mask(agent)

        assert agent == "mrx"
        assert observation.shape == (71,)
        assert action_mask.shape == (60,)
        assert int(action_mask.sum()) == 18
    finally:
        environment.close()
