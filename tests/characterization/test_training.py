"""Characterize feature extraction and DQN initialization contracts."""

import pytest
import torch

from ml_logger import get_logger
from ShadowChase.core.game import Player
from game_controls.game_utils import create_and_initialize_game
from training.deep_q.dqn_trainer import DQNTrainer
from training.feature_extractor_simple import GameFeatureExtractor


logger = get_logger(__name__)


def _build_trainer(project_root, save_dir, device):
    """Construct a trainer without starting an episode or writing a checkpoint."""
    return DQNTrainer(
        player_role="MrX",
        config_path=str(project_root / "training" / "configs" / "dqn_config.json"),
        save_dir=str(save_dir),
        device=device,
    )


def test_default_feature_vector_shape_is_stable():
    """Freeze the current default Mr. X observation shape and dtype."""
    game = create_and_initialize_game("test", 2)
    features = GameFeatureExtractor().extract_features(game, Player.MRX)

    assert features.shape == (237,)
    assert str(features.dtype) == "float32"


def test_dqn_network_initializes_on_cpu(project_root, tmp_path, shadow_game):
    """Freeze the current DQN topology independently of CUDA availability."""
    trainer = _build_trainer(project_root, tmp_path / "models", torch.device("cpu"))
    trainer._initialize_networks(shadow_game)

    assert trainer._feature_size == 226
    assert trainer.config["network_parameters"]["action_size"] == 3
    assert sum(parameter.numel() for parameter in trainer.main_network.parameters()) == 54273
    assert next(trainer.main_network.parameters()).device.type == "cpu"


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
def test_dqn_network_initializes_on_cuda(project_root, tmp_path, shadow_game):
    """Protect the required CUDA-enabled local DQN workflow."""
    trainer = _build_trainer(project_root, tmp_path / "models", torch.device("cuda"))
    trainer._initialize_networks(shadow_game)

    parameter = next(trainer.main_network.parameters())
    assert parameter.is_cuda
    assert torch.version.cuda == "11.8"
