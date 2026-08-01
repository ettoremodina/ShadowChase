"""Characterize current Shadow Chase state and movement behavior."""

from ml_logger import get_logger
from ShadowChase.core.game import Player, TicketType, TransportType


logger = get_logger(__name__)


def test_shadow_game_initializes_current_ticket_and_turn_state(shadow_game):
    """Freeze the initial player, visibility, positions, and ticket allocation."""
    state = shadow_game.game_state

    assert state.turn is Player.MRX
    assert state.turn_count == 0
    assert state.MrX_turn_count == 0
    assert state.detective_positions == [1, 4]
    assert state.MrX_position == 3
    assert state.MrX_visible is False
    assert state.detective_tickets[0] == {
        TicketType.TAXI: 10,
        TicketType.BUS: 8,
        TicketType.UNDERGROUND: 4,
    }
    assert state.MrX_tickets[TicketType.BLACK] == 5
    assert state.MrX_tickets[TicketType.DOUBLE_MOVE] == 2
    assert len(shadow_game.game_history) == 1
    assert shadow_game.ticket_history == []


def test_valid_moves_include_matching_and_black_tickets(shadow_game):
    """Characterize the destination and ticket variants exposed to Mr. X."""
    moves = shadow_game.get_valid_moves(Player.MRX)

    assert moves == {
        (2, TransportType.BUS),
        (2, TransportType.BLACK),
        (4, TransportType.UNDERGROUND),
        (4, TransportType.BLACK),
    }


def test_regular_turns_transfer_tickets_and_reveal_mrx(shadow_game):
    """Freeze turn counters, ticket transfer, history, and reveal timing."""
    assert shadow_game.make_move(
        MrX_moves=[(2, TransportType.BUS)]
    )
    assert shadow_game.game_state.MrX_tickets[TicketType.BUS] == 2
    assert shadow_game.game_state.turn is Player.DETECTIVES

    assert shadow_game.make_move(
        detective_moves=[
            (6, TransportType.UNDERGROUND),
            (5, TransportType.TAXI),
        ]
    )
    assert shadow_game.game_state.detective_tickets[0][TicketType.UNDERGROUND] == 3
    assert shadow_game.game_state.detective_tickets[1][TicketType.TAXI] == 9
    assert shadow_game.game_state.MrX_tickets[TicketType.UNDERGROUND] == 4
    assert shadow_game.game_state.MrX_tickets[TicketType.TAXI] == 5

    assert shadow_game.make_move(
        MrX_moves=[(1, TransportType.TAXI)]
    )
    assert shadow_game.make_move(
        detective_moves=[(6, None), (5, None)]
    )
    assert shadow_game.make_move(
        MrX_moves=[(2, TransportType.TAXI)]
    )

    state = shadow_game.game_state
    assert state.MrX_turn_count == 3
    assert state.turn_count == 5
    assert state.MrX_visible is True
    assert shadow_game.get_MrX_last_visible_position() == 2
    assert len(shadow_game.game_history) == 6
    assert len(shadow_game.ticket_history) == 5


def test_double_move_keeps_then_releases_mrx_turn(shadow_game):
    """Freeze the two-call double-move protocol and ticket consumption."""
    assert shadow_game.make_move(
        MrX_moves=[(2, TransportType.BUS)],
        use_double_move=True,
    )
    state = shadow_game.game_state
    assert state.turn is Player.MRX
    assert state.double_move_active is True
    assert state.MrX_tickets[TicketType.DOUBLE_MOVE] == 1

    assert shadow_game.make_move(
        MrX_moves=[(3, TransportType.BUS)],
        use_double_move=False,
    )
    state = shadow_game.game_state
    assert state.turn is Player.DETECTIVES
    assert state.double_move_active is False
    assert state.MrX_turn_count == 2
    assert state.turn_count == 2
    assert state.MrX_tickets[TicketType.BUS] == 1
    assert shadow_game.ticket_history[0]["double_move_used"] is True
    assert shadow_game.ticket_history[1]["MrX_moves"][0]["double_move_part"] == 2

