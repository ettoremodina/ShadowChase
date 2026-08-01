"""Characterize current pickle and JSON persistence behavior."""

import json
import pickle

import pytest

from ml_logger import get_logger
from ShadowChase.core.game import TransportType
from ShadowChase.services.game_loader import GameLoader, GameRecord


logger = get_logger(__name__)


@pytest.mark.integration
def test_game_loader_round_trip_preserves_current_record(tmp_path, monkeypatch, shadow_game):
    """Freeze the current GameRecord module path and reconstruction behavior."""
    monkeypatch.chdir(tmp_path)
    assert shadow_game.make_move(
        MrX_moves=[(2, TransportType.BUS)]
    )
    loader = GameLoader("characterization")

    game_id = loader.save_game(
        shadow_game,
        game_id="fixture-game",
        additional_metadata={"source": "characterization"},
    )
    record_path = tmp_path / "saved_games" / "characterization" / "games" / f"{game_id}.pkl"
    with record_path.open("rb") as stream:
        record = pickle.load(stream)

    assert isinstance(record, GameRecord)
    assert type(record).__module__ == "ShadowChase.services.game_loader"
    assert record.metadata["source"] == "characterization"
    assert len(record.game_history) == 2

    restored = loader.load_game(game_id)
    assert restored.game_state.MrX_position == 2
    assert restored.game_state.detective_positions == [1, 4]


@pytest.mark.integration
@pytest.mark.xfail(
    reason="Legacy JSON export includes a non-serializable graph in win_condition",
    strict=False,
)
def test_game_loader_json_export_contains_history(tmp_path, monkeypatch, shadow_game):
    """Specify the legacy JSON export envelope expected by analysis tools."""
    monkeypatch.chdir(tmp_path)
    assert shadow_game.make_move(
        MrX_moves=[(2, TransportType.BUS)]
    )
    loader = GameLoader("characterization")
    game_id = loader.save_game(shadow_game, game_id="fixture-export")

    export_path = loader.export_game(game_id, "json")
    payload = json.loads((tmp_path / export_path).read_text(encoding="utf-8"))

    assert set(payload) == {
        "metadata",
        "game_config",
        "game_history",
        "ticket_history",
    }
    assert len(payload["game_history"]) == 2
    assert payload["metadata"]["game_id"] == game_id
