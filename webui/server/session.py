"""Interactive game session.

This is the state machine the Tkinter `GameVisualizer`, `GameControls` and
`SetupControls` classes used to hold between them, lifted out of the widget tree
so it can be driven over HTTP. The rules still live in `ShadowChase.core`; this
class only decides what the player is allowed to click next and what the panel
should say.
"""
from __future__ import annotations

import threading
from typing import Dict, List, Optional, Tuple

from ShadowChase.core.game import (
    Player,
    ShadowChaseGame,
    TicketType,
    TransportType,
)
from ShadowChase.services.game_loader import GameLoader
from ShadowChase.services.game_service import GameService

from .layout import TICKET_STYLES, BoardLayout, build_layout

try:  # Heuristics are optional, exactly as in the Tkinter board.
    from agents.heuristics import GameHeuristics

    HEURISTICS_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on optional deps
    GameHeuristics = None
    HEURISTICS_AVAILABLE = False


GAME_MODES = [
    {
        "value": "human_vs_human",
        "label": "Human vs human",
        "detail": "You play both sides",
    },
    {
        "value": "human_det_vs_ai_mrx",
        "label": "Human detectives vs AI Mr. X",
        "detail": "You hunt, the machine hides",
    },
    {
        "value": "ai_det_vs_human_mrx",
        "label": "AI detectives vs human Mr. X",
        "detail": "You hide, the machine hunts",
    },
    {"value": "ai_vs_ai", "label": "AI vs AI", "detail": "Watch two agents play"},
]

TICKET_ORDER = ["taxi", "bus", "underground", "black", "double_move"]

TICKET_LABELS = {
    "taxi": "Taxi",
    "bus": "Bus",
    "underground": "Tube",
    "black": "Black",
    "double_move": "Double",
}


def _ticket_count(tickets, name: str) -> int:
    """Read a ticket count whatever key flavour the state happens to use."""
    if not tickets:
        return 0
    for key, value in tickets.items():
        key_name = getattr(key, "value", None) or getattr(key, "name", None) or str(key)
        if str(key_name).lower() == name.lower():
            return int(value)
    return 0


class Notice:
    """A message that used to be a `messagebox` call."""

    __slots__ = ("level", "title", "text")

    def __init__(self, level: str, title: str, text: str):
        self.level = level
        self.title = title
        self.text = text

    def to_dict(self) -> Dict[str, str]:
        return {"level": self.level, "title": self.title, "text": self.text}


class GameSession:
    """One game, one board, one player at the keyboard."""

    def __init__(self, game, *, auto_positions: Optional[List[int]] = None,
                 loader: Optional[GameLoader] = None):
        self.lock = threading.RLock()
        self.game = game
        self.loader = loader or GameLoader()
        self.game_service = GameService(self.loader)
        self.layout: BoardLayout = build_layout(game)

        # Setup phase
        self.setup_mode = True
        self.selected_positions: List[int] = list(auto_positions or [])
        self.game_mode = "human_vs_human"
        self.mrx_agent_type = "random"
        self.detective_agent_type = "random"
        self.heuristics_enabled = False

        # Play phase
        self.current_detective_index = 0
        self.detective_selections: List[Tuple[int, Optional[TransportType]]] = []
        self.mrx_selections: List[Tuple[int, TransportType]] = []
        self.selected_nodes: List[int] = []
        self.pending_transport: Optional[Dict] = None
        self.double_move_requested = False

        # Derived each time the board is recomputed
        self.current_player_moves: Dict[int, Dict[int, List[int]]] = {}
        self.active_player_positions: List[int] = []
        self.highlighted_edges: List[Tuple[int, int, int]] = []

        # Agents and analysis
        self.detective_agent = None
        self.mrx_agent = None
        self.heuristics = None

        # Presentation
        self.show_board_image = self.layout.image is not None
        self._notices: List[Notice] = []
        self._game_over_announced = False
        self.saved_game_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------
    @property
    def is_shadow_chase(self) -> bool:
        return isinstance(self.game, ShadowChaseGame)

    @property
    def state(self):
        return getattr(self.game, "game_state", None)

    def notify(self, level: str, title: str, text: str) -> None:
        self._notices.append(Notice(level, title, text))

    def _drain_notices(self) -> List[Dict[str, str]]:
        notices = [notice.to_dict() for notice in self._notices]
        self._notices = []
        return notices

    def _clear_selections(self) -> None:
        self.selected_positions = []
        self.detective_selections = []
        self.mrx_selections = []
        self.current_detective_index = 0
        self.selected_nodes = []
        self.pending_transport = None

    # ------------------------------------------------------------------
    # Setup phase
    # ------------------------------------------------------------------
    def set_mode(self, mode: str) -> None:
        if any(entry["value"] == mode for entry in GAME_MODES):
            self.game_mode = mode

    def set_agent(self, side: str, agent_type: str) -> None:
        if side == "mrx":
            self.mrx_agent_type = agent_type
        elif side == "detectives":
            self.detective_agent_type = agent_type

    def set_heuristics(self, enabled: bool) -> None:
        self.heuristics_enabled = bool(enabled)
        if self.heuristics_enabled:
            self._initialize_heuristics()

    def toggle_board_image(self) -> None:
        if self.layout.image is None:
            self.notify("info", "No board photo", "This board has no scanned image.")
            return
        self.show_board_image = not self.show_board_image

    def select_setup_node(self, node: int) -> None:
        if node in self.selected_positions:
            self.selected_positions.remove(node)
        elif len(self.selected_positions) < self.game.num_detectives + 1:
            self.selected_positions.append(node)

    def can_start(self) -> bool:
        needed = self.game.num_detectives + 1
        if len(self.selected_positions) != needed:
            return False
        detectives = self.selected_positions[: self.game.num_detectives]
        return self.selected_positions[self.game.num_detectives] not in detectives

    def reset_setup(self) -> None:
        self.setup_mode = True
        self._clear_selections()
        self.current_player_moves = {}
        self.active_player_positions = []
        self.highlighted_edges = []
        self.double_move_requested = False
        self.heuristics = None
        self.game.game_state = None
        self.game.game_history = []
        self._game_over_announced = False
        self.saved_game_id = None

    def start_game(self) -> bool:
        needed = self.game.num_detectives + 1
        if len(self.selected_positions) != needed:
            self.notify(
                "error",
                "Not enough stations",
                f"Pick {self.game.num_detectives} detective stations and one for Mr. X.",
            )
            return False

        detectives = self.selected_positions[: self.game.num_detectives]
        mrx = self.selected_positions[self.game.num_detectives]
        if mrx in detectives:
            self.notify(
                "error",
                "Stations overlap",
                "Mr. X cannot start on a station a detective occupies.",
            )
            return False

        self.game.game_state = None
        self.game.game_history = []
        try:
            if self.is_shadow_chase:
                self.game.initialize_shadow_chase_game(detectives, mrx)
            else:
                self.game.initialize_game(detectives, mrx)
        except ValueError as error:
            self.notify("error", "Cannot start", str(error))
            return False

        self.setup_mode = False
        self._clear_selections()
        self.double_move_requested = False
        self._game_over_announced = False
        self.saved_game_id = None
        self._setup_agents()
        if self.heuristics_enabled:
            self._initialize_heuristics()
        self._refresh()
        return True

    # ------------------------------------------------------------------
    # Agents
    # ------------------------------------------------------------------
    def _setup_agents(self) -> None:
        from agents import AgentType, get_agent_registry

        registry = get_agent_registry()
        self.detective_agent = None
        self.mrx_agent = None
        if not self.is_shadow_chase:
            return

        needs_mrx = self.game_mode in ("human_det_vs_ai_mrx", "ai_vs_ai")
        needs_detectives = self.game_mode in ("ai_det_vs_human_mrx", "ai_vs_ai")

        try:
            if needs_mrx:
                self.mrx_agent = registry.create_MrX_agent(
                    AgentType(self.mrx_agent_type)
                )
            if needs_detectives:
                self.detective_agent = registry.create_multi_detective_agent(
                    AgentType(self.detective_agent_type), self.game.num_detectives
                )
        except Exception as error:  # noqa: BLE001 - surfaced to the player
            self.notify("error", "Agent unavailable", str(error))

    def _initialize_heuristics(self) -> None:
        if not HEURISTICS_AVAILABLE or not self.is_shadow_chase or not self.state:
            return
        try:
            self.heuristics = GameHeuristics(self.game)
        except Exception:  # noqa: BLE001 - analysis is best-effort
            self.heuristics = None

    def is_current_player_ai(self) -> bool:
        if not self.state:
            return False
        if self.state.turn == Player.DETECTIVES:
            return self.detective_agent is not None
        return self.mrx_agent is not None

    # ------------------------------------------------------------------
    # Move computation
    # ------------------------------------------------------------------
    def _refresh(self) -> None:
        """Recompute what the current player may do."""
        self.current_player_moves = {}
        self.active_player_positions = []
        self.highlighted_edges = []

        if self.setup_mode or not self.state or self.game.is_game_over():
            return

        try:
            if self.state.turn == Player.DETECTIVES:
                self._refresh_detective_moves()
            else:
                self._refresh_mrx_moves()
        except Exception as error:  # noqa: BLE001 - bad board data, not a crash
            self.notify(
                "error",
                "This board cannot be played",
                f"The engine rejected its route data: {error}",
            )
            return

        if self.is_current_player_ai():
            self._stage_ai_selections()

    def _refresh_detective_moves(self) -> None:
        if self.is_shadow_chase:
            # The engine resets this at the start of a detective turn.
            self.state.double_move_active = False
        self.mrx_selections = []

        if self.current_detective_index >= self.game.num_detectives:
            return

        position = self.state.detective_positions[self.current_detective_index]
        self.active_player_positions = [position]
        destinations: Dict[int, List[int]] = {}

        if self.is_shadow_chase:
            valid = self.game.get_valid_moves(
                Player.DETECTIVES, position, pending_moves=self.detective_selections
            )
            for destination, transport in valid:
                destinations.setdefault(destination, [])
                if transport.value not in destinations[destination]:
                    destinations[destination].append(transport.value)
                    self.highlighted_edges.append(
                        (position, destination, transport.value)
                    )
        else:
            for destination in self.game.get_valid_moves(Player.DETECTIVES, position):
                destinations[destination] = [1]
                self.highlighted_edges.append((position, destination, 1))

        self.current_player_moves[position] = destinations

    def _refresh_mrx_moves(self) -> None:
        position = self.state.MrX_position
        self.active_player_positions = [position]
        destinations: Dict[int, List[int]] = {}

        if self.is_shadow_chase:
            valid = self.game.get_valid_moves(Player.MRX, position)
            tickets = self.game.get_MrX_tickets()
            has_black = tickets.get(TicketType.BLACK, 0) > 0

            for destination, transport in valid:
                bucket = destinations.setdefault(destination, [])
                if tickets.get(TicketType[transport.name], 0) > 0:
                    if transport.value not in bucket:
                        bucket.append(transport.value)
                        self.highlighted_edges.append(
                            (position, destination, transport.value)
                        )
                if has_black and TransportType.BLACK.value not in bucket:
                    bucket.append(TransportType.BLACK.value)
                    self.highlighted_edges.append(
                        (position, destination, TransportType.BLACK.value)
                    )
            # A destination only black tickets can reach still counts.
            destinations = {k: v for k, v in destinations.items() if v}
        else:
            for destination in self.game.get_valid_moves(Player.MRX, position):
                destinations[destination] = [1]
                self.highlighted_edges.append((position, destination, 1))

        self.current_player_moves[position] = destinations

    # ------------------------------------------------------------------
    # Play phase
    # ------------------------------------------------------------------
    def click_node(self, node: int) -> None:
        if self.setup_mode:
            self.select_setup_node(node)
            return
        if not self.state or self.game.is_game_over():
            return
        if self.is_current_player_ai():
            self.notify(
                "info", "Agent's turn", "Press Continue to play the agent's move."
            )
            return
        if not self.active_player_positions:
            return

        source = self.active_player_positions[0]
        available = self.current_player_moves.get(source, {})
        if node not in available:
            self.notify(
                "warning",
                "Out of reach",
                f"Station {node} is not reachable this turn.",
            )
            return

        if self.state.turn == Player.DETECTIVES:
            self._click_detective_destination(source, node)
        else:
            self._click_mrx_destination(source, node)

    def _transport_options(self, values: List[int]) -> List[Dict[str, object]]:
        options = []
        for value in values:
            style = TICKET_STYLES.get(value, {})
            options.append(
                {
                    "transport": value,
                    "key": style.get("key", str(value)),
                    "name": style.get("name", str(value)),
                    "color": style.get("color", "#C9D1DE"),
                }
            )
        return options

    def _click_detective_destination(self, source: int, node: int) -> None:
        if not self.is_shadow_chase:
            self.detective_selections.append(node)
            self.selected_nodes.append(node)
            self.current_detective_index += 1
            self._refresh()
            return

        valid = self.game.get_valid_moves(
            Player.DETECTIVES, source, pending_moves=self.detective_selections
        )
        transports = [t for destination, t in valid if destination == node]
        if not transports:
            self.notify("error", "No route", f"No usable ticket reaches station {node}.")
            return

        if len(transports) > 1:
            self.pending_transport = {
                "node": node,
                "options": self._transport_options([t.value for t in transports]),
            }
            return

        self._commit_detective_choice(node, transports[0])

    def _commit_detective_choice(self, node: int, transport: TransportType) -> None:
        self.detective_selections.append((node, transport))
        self.selected_nodes.append(node)
        self.current_detective_index += 1
        self.pending_transport = None
        self._refresh()

    def _click_mrx_destination(self, source: int, node: int) -> None:
        if not self.is_shadow_chase:
            self.selected_positions = [node]
            self.selected_nodes = [node]
            self._refresh()
            return

        valid = self.game.get_valid_moves(Player.MRX, source)
        transports = [t for destination, t in valid if destination == node]
        if not transports:
            self.notify("error", "No route", f"No usable ticket reaches station {node}.")
            return

        values: List[int] = []
        for transport in transports:
            if transport.value not in values:
                values.append(transport.value)

        tickets = self.game.get_MrX_tickets()
        if tickets.get(TicketType.BLACK, 0) > 0 and TransportType.BLACK.value not in values:
            values.append(TransportType.BLACK.value)

        if len(values) > 1:
            self.pending_transport = {
                "node": node,
                "options": self._transport_options(values),
            }
            return

        self._commit_mrx_choice(node, TransportType(values[0]))

    def _commit_mrx_choice(self, node: int, transport: TransportType) -> None:
        # Mr. X commits one destination per leg; a new pick replaces the old one.
        self.mrx_selections = [(node, transport)]
        self.selected_nodes = [node]
        self.pending_transport = None
        self._refresh()

    def choose_transport(self, transport_value: int) -> None:
        if not self.pending_transport or not self.state:
            return
        node = self.pending_transport["node"]
        allowed = [
            option["transport"] for option in self.pending_transport["options"]
        ]
        if transport_value not in allowed:
            return

        transport = TransportType(transport_value)
        if self.state.turn == Player.DETECTIVES:
            self._commit_detective_choice(node, transport)
        else:
            self._commit_mrx_choice(node, transport)

    def cancel_transport(self) -> None:
        self.pending_transport = None

    def clear_selection(self) -> None:
        """Take back this turn's staged picks without touching the game state."""
        if self.setup_mode or not self.state:
            return
        self.detective_selections = []
        self.mrx_selections = []
        self.current_detective_index = 0
        self.selected_nodes = []
        self.pending_transport = None
        self._refresh()

    def toggle_double_move(self) -> None:
        self.double_move_requested = not self.double_move_requested

    def can_use_double_move(self) -> bool:
        if not self.is_shadow_chase or not self.state:
            return False
        if self.state.turn != Player.MRX or self.is_current_player_ai():
            return False
        tickets = self.game.get_MrX_tickets()
        return (
            tickets.get(TicketType.DOUBLE_MOVE, 0) > 0
            and not self.state.double_move_active
        )

    def can_confirm_move(self) -> bool:
        if self.setup_mode or not self.state or self.game.is_game_over():
            return False
        if self.pending_transport:
            return False
        if self.state.turn == Player.DETECTIVES:
            return len(self.detective_selections) == self.game.num_detectives
        return bool(self.mrx_selections or self.selected_positions)

    def can_skip(self) -> bool:
        """A detective with nowhere to go may stand still."""
        if self.setup_mode or not self.state or self.game.is_game_over():
            return False
        if self.is_current_player_ai() or self.state.turn != Player.DETECTIVES:
            return False
        if self.current_detective_index >= self.game.num_detectives:
            return False
        position = self.state.detective_positions[self.current_detective_index]
        return not self.current_player_moves.get(position)

    def confirm_move(self) -> None:
        if not self.can_confirm_move():
            self.notify(
                "warning", "Nothing to confirm", "Choose a destination first."
            )
            return

        try:
            if self.state.turn == Player.DETECTIVES:
                self._confirm_detective_move()
            else:
                if self._confirm_mrx_move():
                    return  # First leg of a double move: Mr. X stays on turn.
        except Exception as error:  # noqa: BLE001 - surfaced to the player
            self.notify("error", "Move failed", str(error))
            self._clear_selections()
            self.double_move_requested = False
            self._refresh()
            return

        self._finish_turn()

    def _confirm_detective_move(self) -> None:
        if self.is_shadow_chase:
            success = self.game.make_move(detective_moves=self.detective_selections)
        else:
            success = self.game.make_move(new_positions=self.detective_selections)
        if not success:
            self.notify("error", "Move rejected", "The engine refused that move.")

    def _confirm_mrx_move(self) -> bool:
        """Return True when Mr. X keeps the turn for a second leg."""
        if not self.is_shadow_chase:
            success = self.game.make_move(new_MrX_pos=self.selected_positions[0])
            if not success:
                self.notify("error", "Move rejected", "The engine refused that move.")
            return False

        starting_double = (
            self.double_move_requested and not self.state.double_move_active
        )
        success = self.game.make_move(
            MrX_moves=self.mrx_selections, use_double_move=starting_double
        )
        if not success:
            self.notify("error", "Move rejected", "The engine refused that move.")
            return False

        if starting_double:
            self.mrx_selections = []
            self.selected_nodes = []
            self.pending_transport = None
            self._refresh()
            return True
        return False

    def _finish_turn(self) -> None:
        self._clear_selections()
        if not self.state or not self.state.double_move_active:
            self.double_move_requested = False
        self._update_heuristics()
        self._refresh()
        self._check_game_over()

    def skip_turn(self) -> None:
        if not self.can_skip():
            self.notify(
                "warning",
                "Not stuck",
                "Standing still is only for a detective with no route.",
            )
            return
        position = self.state.detective_positions[self.current_detective_index]
        self.detective_selections.append((position, None))
        self.current_detective_index += 1
        self._refresh()

    def ai_continue(self) -> None:
        if not self.state or self.game.is_game_over():
            return
        if not self.is_current_player_ai():
            return

        success = self._play_ai_move()
        if not success:
            self.notify("error", "Agent stalled", "The agent could not find a move.")
            return

        self._clear_selections()
        self._update_heuristics()
        self._refresh()
        self._check_game_over()

    def _play_ai_move(self) -> bool:
        try:
            if self.state.turn == Player.DETECTIVES and self.detective_agent:
                moves = self.detective_agent.choose_all_moves(self.game)
                return bool(moves) and self.game.make_move(detective_moves=moves)
            if self.state.turn == Player.MRX and self.mrx_agent:
                result = self.mrx_agent.choose_move(self.game)
                if result and len(result) == 3:
                    destination, transport, use_double = result
                    return self.game.make_move(
                        MrX_moves=[(destination, transport)],
                        use_double_move=use_double,
                    )
        except Exception as error:  # noqa: BLE001 - surfaced to the player
            self.notify("error", "Agent error", str(error))
        return False

    def _stage_ai_selections(self) -> None:
        """Show what the agent intends before the player presses Continue."""
        try:
            if self.state.turn == Player.DETECTIVES and self.detective_agent:
                moves = self.detective_agent.choose_all_moves(self.game)
                if moves:
                    self.detective_selections = list(moves)
                    self.selected_nodes = [move[0] for move in moves]
                    self.current_detective_index = len(moves)
            elif self.state.turn == Player.MRX and self.mrx_agent:
                result = self.mrx_agent.choose_move(self.game)
                if result and len(result) == 3:
                    destination, transport, use_double = result
                    self.mrx_selections = [(destination, transport)]
                    self.selected_nodes = [destination]
                    self.double_move_requested = bool(use_double)
        except Exception as error:  # noqa: BLE001 - surfaced to the player
            self.notify("error", "Agent error", str(error))

    def _update_heuristics(self) -> None:
        if self.heuristics_enabled and self.heuristics is None:
            self._initialize_heuristics()
        if self.heuristics is not None:
            try:
                self.heuristics.update_game_state(self.game)
            except Exception:  # noqa: BLE001 - analysis is best-effort
                self.heuristics = None

    def _suspect_positions(self) -> List[int]:
        """Stations Mr. X could plausibly be on, while he is under cover."""
        if not (self.heuristics_enabled and self.is_shadow_chase and self.state):
            return []
        if self.setup_mode or self.game.is_game_over():
            return []
        if self.state.turn != Player.DETECTIVES or self.state.MrX_visible:
            return []
        if self.heuristics is None:
            self._initialize_heuristics()
        if self.heuristics is None:
            return []
        try:
            return sorted(int(node) for node in self.heuristics.get_possible_MrX_positions())
        except Exception:  # noqa: BLE001 - analysis is best-effort
            self.heuristics = None
            return []

    def _check_game_over(self) -> None:
        if not self.state or not self.game.is_game_over():
            return
        if self._game_over_announced:
            return
        self._game_over_announced = True
        if not self.is_shadow_chase:
            # The save format records ticket history, which basic games never
            # keep, so there is nothing to write for them.
            return
        try:
            self.saved_game_id = self.game_service.save_ui_game(
                self.game,
                self.game_mode,
                self.detective_agent,
                self.mrx_agent,
                self.detective_agent_type,
                self.mrx_agent_type,
            )
        except Exception as error:  # noqa: BLE001 - saving must not break the game
            self.notify("error", "Could not save", str(error))

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def snapshot(self) -> Dict[str, object]:
        state = self.state
        game_over = bool(state and self.game.is_game_over())
        winner = None
        if game_over:
            winner_player = self.game.get_winner()
            winner = winner_player.value if winner_player else None

        payload: Dict[str, object] = {
            "phase": "setup" if self.setup_mode else "play",
            "kind": "shadow_chase" if self.is_shadow_chase else "basic",
            "numDetectives": self.game.num_detectives,
            "showBoardImage": self.show_board_image,
            "hasBoardImage": self.layout.image is not None,
            "setup": self._setup_payload(),
            "game": self._game_payload(state, game_over, winner),
            "turnPanel": self._turn_payload(state),
            "moves": self._moves_payload(),
            "selections": self._selection_payload(),
            "pendingTransport": self.pending_transport,
            "highlight": {
                "active": list(self.active_player_positions),
                "destinations": {
                    str(destination): transports
                    for source in self.active_player_positions
                    for destination, transports in self.current_player_moves.get(
                        source, {}
                    ).items()
                },
                "edges": [
                    {"u": u, "v": v, "transport": t} for u, v, t in self.highlighted_edges
                ],
            },
            "suspects": self._suspect_positions(),
            "tickets": self._tickets_payload(state),
            "actions": {
                "confirmMove": self.can_confirm_move(),
                "skip": self.can_skip(),
                "aiContinue": bool(
                    state and not game_over and self.is_current_player_ai()
                ),
                "doubleMove": self.can_use_double_move(),
                "clearSelection": bool(self.selected_nodes) and not self.is_current_player_ai(),
                "start": self.can_start(),
            },
            "notices": self._drain_notices(),
            "savedGameId": self.saved_game_id,
        }
        return payload

    def _setup_payload(self) -> Dict[str, object]:
        from agents import AgentSelector

        selected = list(self.selected_positions)
        needed = self.game.num_detectives + 1
        detectives = selected[: self.game.num_detectives]
        mrx = selected[self.game.num_detectives] if len(selected) > needed - 1 else None

        if len(selected) < self.game.num_detectives:
            hint = f"Pick a station for detective {len(selected) + 1}."
        elif len(selected) == self.game.num_detectives:
            hint = "Pick the station where Mr. X goes under cover."
        elif not self.can_start():
            hint = "Mr. X cannot share a station with a detective."
        else:
            hint = "Every station is set. Start the chase."

        return {
            "selected": selected,
            "detectives": detectives,
            "mrx": mrx,
            "needed": needed,
            "hint": hint,
            "modes": GAME_MODES,
            "mode": self.game_mode,
            "agents": [
                {"value": value, "label": label}
                for value, label in AgentSelector.get_agent_choices_for_ui()
            ],
            "mrxAgent": self.mrx_agent_type,
            "detectiveAgent": self.detective_agent_type,
            "heuristics": self.heuristics_enabled,
            "heuristicsAvailable": HEURISTICS_AVAILABLE and self.is_shadow_chase,
            "supportsAgents": self.is_shadow_chase,
        }

    def _game_payload(self, state, game_over: bool, winner: Optional[str]) -> Dict:
        if not state:
            return {
                "started": False,
                "turn": None,
                "turnCount": 0,
                "mrxTurnCount": 0,
                "detectivePositions": [],
                "mrxPosition": None,
                "mrxVisible": False,
                "revealTurns": sorted(getattr(self.game, "reveal_turns", []) or []),
                "gameOver": False,
                "winner": None,
            }

        mrx_visible = bool(getattr(state, "MrX_visible", True))
        if not self.is_shadow_chase:
            mrx_visible = True
        # Once the game is over the fugitive is on the table either way.
        expose = mrx_visible or game_over

        return {
            "started": True,
            "turn": state.turn.value,
            "turnCount": int(getattr(state, "turn_count", 0)),
            "mrxTurnCount": int(getattr(state, "MrX_turn_count", 0)),
            "detectivePositions": list(state.detective_positions),
            "mrxPosition": state.MrX_position if expose else None,
            "mrxVisible": mrx_visible,
            "lastVisiblePosition": getattr(self.game, "last_visible_position", None),
            "revealTurns": sorted(getattr(self.game, "reveal_turns", []) or []),
            "doubleMoveActive": bool(getattr(state, "double_move_active", False)),
            "gameOver": game_over,
            "winner": winner,
        }

    def _turn_payload(self, state) -> Dict[str, object]:
        if self.setup_mode or not state:
            return {
                "actor": "setup",
                "label": "Setting up",
                "sublabel": "Choose starting stations",
                "controller": "human",
                "progress": None,
                "instruction": "Click stations on the map to place the players.",
                "doubleMove": {
                    "available": False,
                    "requested": False,
                    "active": False,
                },
            }

        is_ai = self.is_current_player_ai()
        controller = "ai" if is_ai else "human"

        if state.turn == Player.DETECTIVES:
            index = min(self.current_detective_index, self.game.num_detectives - 1)
            positions = list(state.detective_positions)
            position = positions[index] if index < len(positions) else None
            done = len(self.detective_selections)

            if is_ai:
                instruction = "The agent has picked its moves. Press Continue."
                sublabel = "Agent controlled"
                label = "Detectives"
            elif done >= self.game.num_detectives:
                instruction = "All detectives are set. Confirm the move."
                sublabel = "Ready"
                label = "Detectives"
            elif self.pending_transport:
                instruction = (
                    f"Choose the ticket for station {self.pending_transport['node']}."
                )
                sublabel = f"At station {position}"
                label = f"Detective {index + 1}"
            elif not self.current_player_moves.get(position):
                instruction = "No route left. This detective stands still."
                sublabel = f"At station {position}"
                label = f"Detective {index + 1}"
            else:
                instruction = "Click a highlighted station to move."
                sublabel = f"At station {position}"
                label = f"Detective {index + 1}"

            return {
                "actor": "detectives",
                "label": label,
                "sublabel": sublabel,
                "controller": controller,
                "progress": {"done": done, "total": self.game.num_detectives},
                "instruction": instruction,
                "doubleMove": {
                    "available": False,
                    "requested": False,
                    "active": False,
                },
            }

        double_active = bool(getattr(state, "double_move_active", False))
        if is_ai:
            instruction = "The agent has picked its move. Press Continue."
        elif self.pending_transport:
            instruction = f"Choose the ticket for station {self.pending_transport['node']}."
        elif self.mrx_selections:
            instruction = "Destination set. Confirm the move."
        else:
            instruction = "Click a highlighted station to move."

        if double_active:
            sublabel = "Second leg of a double move"
        elif self.double_move_requested:
            sublabel = "Double move armed"
        elif getattr(state, "MrX_visible", False):
            sublabel = f"Surfaced at station {state.MrX_position}"
        else:
            sublabel = "Under cover"

        return {
            "actor": "mrx",
            "label": "Mr. X",
            "sublabel": sublabel,
            "controller": controller,
            "progress": None,
            "instruction": instruction,
            "doubleMove": {
                "available": self.can_use_double_move(),
                "requested": self.double_move_requested,
                "active": double_active,
            },
        }

    def _moves_payload(self) -> Dict[str, object]:
        source = self.active_player_positions[0] if self.active_player_positions else None
        # Station 0 is a real station on the grid and path boards, so this has
        # to test for None rather than truthiness.
        destinations = (
            self.current_player_moves.get(source, {}) if source is not None else {}
        )

        grouped: Dict[int, List[int]] = {}
        for destination, transports in destinations.items():
            for transport in transports:
                grouped.setdefault(transport, []).append(destination)

        groups = []
        for transport in sorted(grouped):
            style = TICKET_STYLES.get(transport, {})
            groups.append(
                {
                    "transport": transport,
                    "key": style.get("key", str(transport)),
                    "name": style.get("name", str(transport)),
                    "color": style.get("color", "#C9D1DE"),
                    "destinations": sorted(set(grouped[transport])),
                }
            )

        return {"source": source, "groups": groups, "empty": not groups}

    def _selection_payload(self) -> Dict[str, object]:
        detectives = []
        positions = list(self.state.detective_positions) if self.state else []
        for index, selection in enumerate(self.detective_selections):
            if isinstance(selection, tuple):
                destination, transport = selection
            else:
                destination, transport = selection, None
            transport_value = getattr(transport, "value", None)
            style = TICKET_STYLES.get(transport_value, {}) if transport_value else {}
            detectives.append(
                {
                    "index": index,
                    "from": positions[index] if index < len(positions) else None,
                    "to": destination,
                    "transport": transport_value,
                    "ticket": style.get("name", "Stayed" if transport is None else "?"),
                    "color": style.get("color", "#6B7688"),
                }
            )

        mrx = []
        for destination, transport in self.mrx_selections:
            transport_value = getattr(transport, "value", transport)
            style = TICKET_STYLES.get(transport_value, {})
            mrx.append(
                {
                    "to": destination,
                    "transport": transport_value,
                    "ticket": style.get("name", "?"),
                    "color": style.get("color", "#C9D1DE"),
                }
            )

        return {
            "nodes": list(self.selected_nodes),
            "detectives": detectives,
            "mrx": mrx,
        }

    def _tickets_payload(self, state) -> Dict[str, object]:
        if not self.is_shadow_chase or not state:
            return {"enabled": False, "order": TICKET_ORDER, "rows": []}

        rows = []
        for index in range(self.game.num_detectives):
            tickets = self.game.get_detective_tickets(index)
            rows.append(
                {
                    "id": f"detective-{index}",
                    "name": f"Detective {index + 1}",
                    "short": f"D{index + 1}",
                    "side": "detectives",
                    "position": state.detective_positions[index]
                    if index < len(state.detective_positions)
                    else None,
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

        mrx_tickets = self.game.get_MrX_tickets()
        rows.append(
            {
                "id": "mrx",
                "name": "Mr. X",
                "short": "X",
                "side": "mrx",
                "position": state.MrX_position
                if (getattr(state, "MrX_visible", False) or self.game.is_game_over())
                else None,
                "counts": {name: _ticket_count(mrx_tickets, name) for name in TICKET_ORDER},
            }
        )

        return {
            "enabled": True,
            "order": TICKET_ORDER,
            "labels": TICKET_LABELS,
            "colors": {
                style["key"]: style["color"]
                for style in TICKET_STYLES.values()
            }
            | {"double_move": "#B07CF0"},
            "rows": rows,
        }
