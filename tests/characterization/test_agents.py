"""Characterize baseline random-agent contracts."""

import random

from agents.random_agent import RandomMrXAgent, RandomMultiDetectiveAgent
from ml_logger import get_logger
from ShadowChase.core.game import Player


logger = get_logger(__name__)


def test_random_mrx_returns_a_currently_valid_move(shadow_game):
    """Ensure the random fugitive selects a move exposed by the game engine."""
    random.seed(20260801)
    move = RandomMrXAgent().choose_move(shadow_game)

    destination, transport, use_double = move
    assert (destination, transport) in shadow_game.get_valid_moves(Player.MRX)
    assert isinstance(use_double, bool)


def test_random_detectives_return_one_non_conflicting_move_each(shadow_game):
    """Ensure the team agent respects sequential pending detective moves."""
    random.seed(20260801)
    moves = RandomMultiDetectiveAgent(2).choose_all_moves(shadow_game)

    assert len(moves) == 2
    pending = []
    for position, move in zip([1, 4], moves):
        assert move in shadow_game.get_valid_moves(
            Player.DETECTIVES,
            position,
            pending_moves=pending,
        )
        pending.append(move)
    assert len({destination for destination, _ in moves}) == 2

