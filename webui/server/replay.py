"""Turning a finished game into something the replay window can scrub through.

Every step is precomputed once and sent as a single payload, so dragging the
timeline is instant and needs no further server calls.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .layout import TICKET_STYLES, build_layout
from .session import TICKET_LABELS, TICKET_ORDER, _ticket_count

TICKET_COLORS = {style["key"]: style["color"] for style in TICKET_STYLES.values()}
TICKET_COLORS["double_move"] = "#B07CF0"


def _normalize_turn(state) -> str:
    turn = getattr(state, "turn", None)
    value = getattr(turn, "value", None) or getattr(turn, "name", None) or str(turn)
    value = str(value).lower()
    if "detective" in value:
        return "detectives"
    if "mrx" in value or value == "x":
        return "MrX"
    return "detectives"


def _outcome_at(game, state) -> Dict[str, Optional[str]]:
    """Ask the engine about a historical state without disturbing the live one."""
    original = getattr(game, "game_state", None)
    try:
        game.game_state = state
        over = bool(game.is_game_over()) if hasattr(game, "is_game_over") else False
        winner = None
        if over and hasattr(game, "get_winner"):
            winner_player = game.get_winner()
            winner = getattr(winner_player, "value", None)
        return {"gameOver": over, "winner": winner}
    except Exception:  # noqa: BLE001 - a broken step must not break the replay
        return {"gameOver": False, "winner": None}
    finally:
        game.game_state = original


def _ticket_rows(game, state) -> List[Dict[str, object]]:
    rows = []
    detective_tickets = getattr(state, "detective_tickets", {}) or {}
    positions = list(getattr(state, "detective_positions", []))

    for index in range(getattr(game, "num_detectives", len(positions))):
        if isinstance(detective_tickets, dict):
            tickets = detective_tickets.get(index, {})
        elif index < len(detective_tickets):
            tickets = detective_tickets[index]
        else:
            tickets = {}
        rows.append(
            {
                "id": f"detective-{index}",
                "name": f"Detective {index + 1}",
                "short": f"D{index + 1}",
                "side": "detectives",
                "position": positions[index] if index < len(positions) else None,
                "counts": {
                    name: (
                        _ticket_count(tickets, name)
                        if name not in ("black", "double_move")
                        else None
                    )
                    for name in TICKET_ORDER
                },
            }
        )

    mrx_tickets = getattr(state, "MrX_tickets", {}) or {}
    rows.append(
        {
            "id": "mrx",
            "name": "Mr. X",
            "short": "X",
            "side": "mrx",
            "position": getattr(state, "MrX_position", None),
            "counts": {name: _ticket_count(mrx_tickets, name) for name in TICKET_ORDER},
        }
    )
    return rows


def _move_log(game) -> List[Dict[str, object]]:
    """Flatten ``ticket_history`` into one card per turn."""
    history = getattr(game, "ticket_history", None) or []
    log: List[Dict[str, object]] = []

    for index, turn in enumerate(history):
        player = turn.get("player", "")
        turn_number = turn.get("turn_number", index)
        entries: List[Dict[str, object]] = []

        if player == "MrX":
            for move in turn.get("MrX_moves", []):
                edge = move.get("edge") or (None, None)
                ticket = move.get("ticket_used") or "unknown"
                entries.append(
                    {
                        "label": "Mr. X",
                        "side": "mrx",
                        "from": edge[0],
                        "to": edge[1],
                        "ticket": TICKET_LABELS.get(ticket, str(ticket).title()),
                        "ticketKey": ticket,
                        "color": TICKET_COLORS.get(ticket, "#C9D1DE"),
                        "stayed": False,
                        "doubleMovePart": move.get("double_move_part") or 0,
                    }
                )
        elif player == "detectives":
            for move in turn.get("detective_moves", []):
                edge = move.get("edge") or (None, None)
                ticket = move.get("ticket_used")
                stayed = bool(move.get("stayed"))
                detective_id = move.get("detective_id")
                entries.append(
                    {
                        "label": f"Detective {int(detective_id) + 1}"
                        if detective_id is not None
                        else "Detective",
                        "side": "detectives",
                        "from": edge[0],
                        "to": edge[1],
                        "ticket": TICKET_LABELS.get(ticket, "Stayed" if stayed else "—"),
                        "ticketKey": ticket or "none",
                        "color": TICKET_COLORS.get(ticket, "#6B7688"),
                        "stayed": stayed,
                        "doubleMovePart": 0,
                    }
                )

        log.append(
            {
                "index": index,
                "turnNumber": turn_number,
                "player": player or "none",
                "doubleMove": bool(turn.get("double_move_used")),
                "entries": entries,
            }
        )

    return log


def build_replay(game, game_id: str) -> Dict[str, object]:
    """Everything the replay window needs, in one payload."""
    layout = build_layout(game)
    history = list(getattr(game, "game_history", []) or [])
    reveal_turns = sorted(getattr(game, "reveal_turns", []) or [])

    steps: List[Dict[str, object]] = []
    for index, state in enumerate(history):
        outcome = _outcome_at(game, state)
        steps.append(
            {
                "index": index,
                "turn": _normalize_turn(state),
                "turnCount": int(getattr(state, "turn_count", index)),
                "mrxTurnCount": int(getattr(state, "MrX_turn_count", 0)),
                "detectivePositions": list(getattr(state, "detective_positions", [])),
                # The replay always shows where Mr. X actually was; `mrxVisible`
                # records whether the detectives could see him at the time.
                "mrxPosition": getattr(state, "MrX_position", None),
                "mrxVisible": bool(getattr(state, "MrX_visible", True)),
                "doubleMoveActive": bool(getattr(state, "double_move_active", False)),
                "tickets": _ticket_rows(game, state),
                **outcome,
            }
        )

    final = steps[-1] if steps else None
    return {
        "gameId": game_id,
        "board": layout.to_dict(),
        "steps": steps,
        "moves": _move_log(game),
        "revealTurns": reveal_turns,
        "ticketOrder": TICKET_ORDER,
        "ticketLabels": TICKET_LABELS,
        "ticketColors": TICKET_COLORS,
        "summary": {
            "steps": len(steps),
            "numDetectives": int(getattr(game, "num_detectives", 0)),
            "winner": final["winner"] if final else None,
            "gameOver": bool(final["gameOver"]) if final else False,
        },
    }
