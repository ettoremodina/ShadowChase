"""Lock the module and class names recorded inside historical saved games.

A pickle stores the import path of every class it references and resolves it at
load time. Moving or renaming those classes makes existing ``saved_games/``
files unreadable, and a round trip through freshly created objects cannot
detect it: a new save records whatever path is current. These tests read the
real corpus, which is the only thing that catches a rename after the fact.
"""

import pickle

import pytest

# Importing the package installs the aliases for the historical import paths.
import ShadowChase  # noqa: F401
from ml_logger import get_logger
from ShadowChase.compat import current_module_name
from ShadowChase.core.game import (
    Player,
    ShadowChaseGame,
    ShadowChaseMovement,
    ShadowChaseWinCondition,
)
from ShadowChase.services.game_loader import GameRecord


logger = get_logger(__name__)

SAMPLE_SIZE = 60


def _historical_pickles(project_root):
    """Sample the corpus evenly so every save format era is represented."""
    saves_root = project_root / "saved_games"
    if not saves_root.is_dir():
        return []
    # by_date, by_graph_type and by_outcome are hard links to the same files.
    paths = sorted(
        path
        for path in saves_root.rglob("*.pkl")
        if "/by_" not in path.as_posix()
    )
    if len(paths) <= SAMPLE_SIZE:
        return paths
    stride = len(paths) // SAMPLE_SIZE
    return paths[::stride][:SAMPLE_SIZE]


@pytest.fixture
def historical_pickles(project_root):
    """Skip the corpus tests on a checkout that carries no saved games."""
    paths = _historical_pickles(project_root)
    if not paths:
        pytest.skip("No historical saved games are present in this checkout")
    logger.info("Verifying %d historical saved games", len(paths))
    return paths


@pytest.mark.parametrize(
    "legacy_name, expected",
    [
        ("ScotlandYard.core.game", "ShadowChase.core.game"),
        ("ScotlandYard.services.game_loader", "ShadowChase.services.game_loader"),
        # storage/ was renamed to services/ before the package itself moved.
        ("ScotlandYard.storage.game_loader", "ShadowChase.services.game_loader"),
        ("cops_and_robbers.storage.game_loader", "ShadowChase.services.game_loader"),
        ("cops_and_robbers.core.game", "ShadowChase.core.game"),
    ],
)
def test_legacy_module_names_map_to_their_replacements(legacy_name, expected):
    """Specify every historical package path a saved game can reference."""
    assert current_module_name(legacy_name) == expected


def test_legacy_module_names_import_the_current_classes():
    """Verify the aliases resolve to the same objects, not to copies."""
    import cops_and_robbers.storage.game_loader as legacy_loader
    import ScotlandYard.core.game as legacy_game

    assert legacy_loader.GameRecord is GameRecord
    assert legacy_game.ScotlandYardMovement is ShadowChaseMovement
    assert legacy_game.ScotlandYardWinCondition is ShadowChaseWinCondition
    assert legacy_game.ScotlandYardGame is ShadowChaseGame


def test_legacy_player_values_resolve_to_current_members():
    """Verify a renamed enum value still restores the player it named."""
    assert Player("mr_x") is Player.MRX
    assert Player("MrX") is Player.MRX
    assert Player("detectives") is Player.DETECTIVES
    with pytest.raises(ValueError):
        Player("not-a-player")


@pytest.mark.integration
def test_historical_saves_still_load_and_keep_their_outcomes(historical_pickles):
    """Verify every sampled game unpickles into a usable record.

    This is the test that fails when a class moves without a compatibility
    alias, when an enum member is renamed, or when a restored attribute
    disappears, because unpickling skips ``__init__`` and restores state
    directly.
    """
    for path in historical_pickles:
        with path.open("rb") as stream:
            record = pickle.load(stream)

        assert record.game_id, path
        assert record.game_history, path
        assert record.metadata.get("created_at"), path

        final_state = record.game_history[-1]
        assert isinstance(final_state.turn_count, int), path
        assert final_state.MrX_position is not None, path
        assert final_state.detective_positions, path
        assert final_state.turn in (Player.MRX, Player.DETECTIVES), path
