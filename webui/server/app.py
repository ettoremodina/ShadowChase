"""HTTP surface for the Shadow Chase board.

Every mutating route returns the full snapshot, so the browser never has to
guess what changed. The engine stays authoritative: the client draws what it is
told and nothing more.
"""
from __future__ import annotations

import random
import threading
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ShadowChase.examples.example_games import (
    create_cycle_graph_game,
    create_extracted_board_game,
    create_grid_graph_game,
    create_path_graph_game,
    create_shadowChase_game,
    create_simple_shadow_chase_game,
    create_simple_test_shadow_chase_game,
    create_test_shadow_chase_game,
)
from ShadowChase.services.game_loader import GameLoader

from . import saves
from .layout import BOARD_IMAGE_PATH
from .replay import build_replay
from .session import GameSession
from .video import export_manager

CLIENT_DIR = Path(__file__).resolve().parent.parent / "client"

# The starting cards the physical game deals from, as used by main.py.
STARTING_CARDS = [
    13, 26, 29, 34, 50, 53, 91, 103, 112, 132,
    138, 141, 155, 174, 197, 94, 117, 198,
]


class BoardOption:
    """One playable board, with the detective counts it accepts."""

    def __init__(self, key: str, name: str, blurb: str, factory,
                 detectives: List[int], default_detectives: int,
                 deals_cards: bool = False, photo: bool = False,
                 tickets: bool = True):
        self.key = key
        self.name = name
        self.blurb = blurb
        self.factory = factory
        self.detectives = detectives
        self.default_detectives = default_detectives
        self.deals_cards = deals_cards
        self.photo = photo
        self.tickets = tickets

    def to_dict(self) -> Dict[str, object]:
        return {
            "key": self.key,
            "name": self.name,
            "blurb": self.blurb,
            "detectives": self.detectives,
            "defaultDetectives": self.default_detectives,
            "dealsCards": self.deals_cards,
            "photo": self.photo,
            "tickets": self.tickets,
        }


BOARDS: Dict[str, BoardOption] = {
    board.key: board
    for board in [
        BoardOption(
            "extracted",
            "London",
            "The scanned board, 200 stations, full ticket rules",
            create_extracted_board_game,
            [1, 2, 3, 4, 5],
            5,
            deals_cards=True,
            photo=True,
        ),
        # "London, schematic" (create_shadowChase_game) is deliberately absent:
        # data/edgelist.csv stores 330 of its 447 edges as edge_type 0, which
        # TransportType rejects, so ticket rules cannot run on it. The same
        # crash happens today with `main.py --demo board`.
        BoardOption(
            "london-simple",
            "London, no tickets",
            "The full map with basic movement and no ticket economy",
            lambda n: create_simple_shadow_chase_game(n, show_MrX=True, use_tickets=False),
            [2],
            2,
            tickets=False,
        ),
        BoardOption(
            "test-board",
            "Test board",
            "Ten stations with full ticket rules, for quick checks",
            create_test_shadow_chase_game,
            [2],
            2,
        ),
        BoardOption(
            "test-simple",
            "Test board, no tickets",
            "Ten stations with basic movement",
            lambda n: create_simple_test_shadow_chase_game(
                n, show_MrX=True, use_tickets=False
            ),
            [2],
            2,
            tickets=False,
        ),
        BoardOption(
            "grid",
            "Grid",
            "A 3 by 3 lattice",
            lambda n: create_grid_graph_game(3, 3, n),
            [1, 2, 3, 4],
            2,
            tickets=False,
        ),
        BoardOption(
            "path",
            "Path",
            "Five stations in a line",
            lambda n: create_path_graph_game(5, n),
            [1, 2],
            1,
            tickets=False,
        ),
        BoardOption(
            "cycle",
            "Cycle",
            "Six stations in a ring",
            lambda n: create_cycle_graph_game(6, n),
            [1, 2],
            1,
            tickets=False,
        ),
    ]
}

DEFAULT_BOARD = "extracted"


class AppState:
    """The one game the window is showing, plus the board it came from."""

    def __init__(self):
        self.lock = threading.RLock()
        self.loader = GameLoader()
        self.board_key = DEFAULT_BOARD
        self.detectives = BOARDS[DEFAULT_BOARD].default_detectives
        self.session: Optional[GameSession] = None

    def ensure(self) -> GameSession:
        if self.session is None:
            self.new_game(self.board_key, self.detectives, deal=True)
        return self.session

    def new_game(self, board_key: str, detectives: int, *, deal: bool) -> GameSession:
        board = BOARDS.get(board_key)
        if board is None:
            raise HTTPException(status_code=400, detail=f"Unknown board: {board_key}")
        if detectives not in board.detectives:
            detectives = board.default_detectives

        game = board.factory(detectives)
        auto_positions = None
        if deal and board.deals_cards:
            auto_positions = _deal_starting_stations(game, detectives)

        self.board_key = board_key
        self.detectives = detectives
        self.session = GameSession(
            game, auto_positions=auto_positions, loader=self.loader
        )
        return self.session

    def board_payload(self) -> Dict[str, object]:
        return {
            "options": [board.to_dict() for board in BOARDS.values()],
            "selected": self.board_key,
            "detectives": self.detectives,
        }


def _deal_starting_stations(game, detectives: int) -> Optional[List[int]]:
    """Deal detective and Mr. X stations from the game's starting cards."""
    available = set(game.graph.nodes())
    cards = [card for card in STARTING_CARDS if card in available]
    if len(cards) < detectives + 1:
        extra = [node for node in available if node not in cards]
        needed = (detectives + 1) - len(cards)
        cards.extend(random.sample(extra, min(needed, len(extra))))
    if len(cards) < detectives + 1:
        return None
    sample = random.sample(cards, detectives + 1)
    # main.py puts Mr. X first in the sample and the detectives after it; the
    # session expects detectives first, Mr. X last.
    return sample[1:] + [sample[0]]


app_state = AppState()
app = FastAPI(title="Shadow Chase", docs_url=None, redoc_url=None)


# ----------------------------------------------------------------------
# Request bodies
# ----------------------------------------------------------------------
class NewGameBody(BaseModel):
    board: str = DEFAULT_BOARD
    detectives: int = 5
    deal: bool = True


class NodeBody(BaseModel):
    node: int


class ModeBody(BaseModel):
    mode: str


class AgentBody(BaseModel):
    side: str
    agent: str


class ToggleBody(BaseModel):
    enabled: bool


class TransportBody(BaseModel):
    transport: int


class OpenSaveBody(BaseModel):
    path: str


class ExportBody(BaseModel):
    path: str
    filename: str = ""
    frameDuration: float = Field(default=1.0, gt=0.05, le=10.0)
    endDelay: float = Field(default=3.0, ge=0.0, le=30.0)


# ----------------------------------------------------------------------
# Pages and assets
# ----------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((CLIENT_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/replay", response_class=HTMLResponse)
def replay_page() -> HTMLResponse:
    return HTMLResponse((CLIENT_DIR / "replay.html").read_text(encoding="utf-8"))


@app.get("/api/board/image")
def board_image() -> FileResponse:
    path = Path(BOARD_IMAGE_PATH)
    if not path.exists():
        raise HTTPException(status_code=404, detail="No board image on disk")
    return FileResponse(path, media_type="image/jpeg")


# ----------------------------------------------------------------------
# Game state
# ----------------------------------------------------------------------
def _envelope(session: GameSession) -> Dict[str, object]:
    return {"state": session.snapshot(), "board": app_state.board_payload()}


@app.get("/api/bootstrap")
def bootstrap() -> Dict[str, object]:
    with app_state.lock:
        session = app_state.ensure()
        return {
            "layout": session.layout.to_dict(),
            **_envelope(session),
        }


@app.get("/api/state")
def get_state() -> Dict[str, object]:
    with app_state.lock:
        return _envelope(app_state.ensure())


@app.post("/api/game/new")
def new_game(body: NewGameBody) -> Dict[str, object]:
    with app_state.lock:
        session = app_state.new_game(body.board, body.detectives, deal=body.deal)
        return {"layout": session.layout.to_dict(), **_envelope(session)}


def _mutate(action) -> Dict[str, object]:
    with app_state.lock:
        session = app_state.ensure()
        with session.lock:
            action(session)
        return _envelope(session)


@app.post("/api/board/node")
def click_node(body: NodeBody) -> Dict[str, object]:
    return _mutate(lambda session: session.click_node(body.node))


@app.post("/api/board/image/toggle")
def toggle_image() -> Dict[str, object]:
    return _mutate(lambda session: session.toggle_board_image())


@app.post("/api/setup/mode")
def set_mode(body: ModeBody) -> Dict[str, object]:
    return _mutate(lambda session: session.set_mode(body.mode))


@app.post("/api/setup/agent")
def set_agent(body: AgentBody) -> Dict[str, object]:
    return _mutate(lambda session: session.set_agent(body.side, body.agent))


@app.post("/api/setup/heuristics")
def set_heuristics(body: ToggleBody) -> Dict[str, object]:
    return _mutate(lambda session: session.set_heuristics(body.enabled))


@app.post("/api/setup/clear")
def clear_setup() -> Dict[str, object]:
    return _mutate(lambda session: session.reset_setup())


@app.post("/api/setup/deal")
def deal_setup() -> Dict[str, object]:
    with app_state.lock:
        session = app_state.ensure()
        with session.lock:
            positions = _deal_starting_stations(session.game, session.game.num_detectives)
            if positions is None:
                session.notify(
                    "warning",
                    "Cannot deal",
                    "This board has too few stations to deal from.",
                )
            else:
                session.reset_setup()
                session.selected_positions = positions
        return _envelope(session)


@app.post("/api/setup/start")
def start_game() -> Dict[str, object]:
    return _mutate(lambda session: session.start_game())


@app.post("/api/play/transport")
def choose_transport(body: TransportBody) -> Dict[str, object]:
    return _mutate(lambda session: session.choose_transport(body.transport))


@app.post("/api/play/transport/cancel")
def cancel_transport() -> Dict[str, object]:
    return _mutate(lambda session: session.cancel_transport())


@app.post("/api/play/confirm")
def confirm_move() -> Dict[str, object]:
    return _mutate(lambda session: session.confirm_move())


@app.post("/api/play/skip")
def skip_turn() -> Dict[str, object]:
    return _mutate(lambda session: session.skip_turn())


@app.post("/api/play/continue")
def ai_continue() -> Dict[str, object]:
    return _mutate(lambda session: session.ai_continue())


@app.post("/api/play/double")
def toggle_double() -> Dict[str, object]:
    return _mutate(lambda session: session.toggle_double_move())


@app.post("/api/play/clear")
def clear_selection() -> Dict[str, object]:
    return _mutate(lambda session: session.clear_selection())


# ----------------------------------------------------------------------
# Saved games, replay and video
# ----------------------------------------------------------------------
@app.get("/api/saves")
def list_saves(query: str = "", limit: int = 400) -> Dict[str, object]:
    return {"saves": saves.list_saves(limit=limit, query=query)}


@app.post("/api/replay/open")
def open_replay(body: OpenSaveBody) -> Dict[str, object]:
    try:
        game, game_id = saves.load_game_file(body.path, app_state.loader)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:  # noqa: BLE001 - bad pickles are user data
        raise HTTPException(status_code=400, detail=f"Could not open: {error}") from error

    if not getattr(game, "game_history", None):
        raise HTTPException(status_code=400, detail="That save has no recorded moves.")
    return build_replay(game, game_id)


@app.post("/api/video/export")
def export_video(body: ExportBody) -> Dict[str, object]:
    try:
        game, game_id = saves.load_game_file(body.path, app_state.loader)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:  # noqa: BLE001 - bad pickles are user data
        raise HTTPException(status_code=400, detail=f"Could not open: {error}") from error

    job = export_manager.start(
        game,
        game_id,
        filename=body.filename or f"{game_id}.mp4",
        frame_duration=body.frameDuration,
        end_delay=body.endDelay,
    )
    return job.to_dict()


@app.get("/api/video/job/{job_id}")
def video_job(job_id: str) -> Dict[str, object]:
    job = export_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="No such export")
    return job.to_dict()


app.mount("/static", StaticFiles(directory=str(CLIENT_DIR)), name="static")
