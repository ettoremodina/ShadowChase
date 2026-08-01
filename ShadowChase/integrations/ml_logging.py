"""Translate Shadow Chase experiments into generic ``ml_logger`` records.

The adapter owns domain naming and serialization while ``ml_logger`` remains
independent of games, agents, and reinforcement-learning algorithms.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import networkx as nx

from ml_logger import RunContext, get_logger
from ml_logger.serialization import json_safe
from ShadowChase.core.game import GameState, Player, ShadowChaseGame


logger = get_logger(__name__)

REPLAY_SCHEMA_VERSION = 1
RECORDING_LEVELS = {"summary", "actions", "full"}
_UNSAFE_IDENTIFIER = re.compile(r"[^A-Za-z0-9_.-]+")
_UNSAFE_METRIC_SEGMENT = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class RecordedGame:
    """Result returned after one game has been sent to an experiment run."""

    metrics: dict[str, Any]
    replay_path: Path | None


@dataclass
class _EvaluationAggregate:
    """Accumulate batch totals without exposing analysis policy to callers."""

    games: int = 0
    mrx_wins: int = 0
    detective_wins: int = 0
    incomplete_games: int = 0
    turns: int = 0
    duration_seconds: float = 0.0
    timed_games: int = 0

    def update(
        self,
        summary: Mapping[str, Any],
        duration_seconds: float | None,
    ) -> None:
        """Merge one completed or incomplete game into the aggregate."""
        self.games += 1
        self.turns += int(summary["turns"])
        winner = summary.get("winner")
        if winner == Player.MRX.value:
            self.mrx_wins += 1
        elif winner == Player.DETECTIVES.value:
            self.detective_wins += 1
        else:
            self.incomplete_games += 1
        if duration_seconds is not None:
            self.duration_seconds += duration_seconds
            self.timed_games += 1

    def as_summary(self) -> dict[str, Any]:
        """Return stable aggregate fields suitable for a run manifest."""
        games = self.games
        timed_games = self.timed_games
        return {
            "evaluation/games": games,
            "evaluation/mrx_wins": self.mrx_wins,
            "evaluation/detective_wins": self.detective_wins,
            "evaluation/incomplete_games": self.incomplete_games,
            "evaluation/mrx_win_rate": self.mrx_wins / games if games else 0.0,
            "evaluation/detective_win_rate": (
                self.detective_wins / games if games else 0.0
            ),
            "evaluation/average_turns": self.turns / games if games else 0.0,
            "evaluation/average_duration_seconds": (
                self.duration_seconds / timed_games if timed_games else 0.0
            ),
        }


class GameRunRecorder:
    """Record gameplay or evaluation batches through one ``RunContext``."""

    def __init__(
        self,
        context: RunContext,
        recording_level: str | None = None,
        save_replays: bool | None = None,
    ) -> None:
        """Configure domain recording from explicit values or logger settings."""
        settings = (
            context.settings.get("integrations", {})
            .get("shadow_chase", {})
            .get("recording", {})
        )
        level = recording_level or settings.get("level", "actions")
        if level not in RECORDING_LEVELS:
            raise ValueError(f"Unsupported Shadow Chase recording level: {level}")
        configured_replays = settings.get("save_replays", True)

        self.context = context
        self.recording_level = level
        self.save_replays = (
            bool(configured_replays)
            if save_replays is None
            else bool(save_replays)
        )
        self._aggregate = _EvaluationAggregate()
        self.context.log_params(
            {
                "logging/recording_level": self.recording_level,
                "logging/save_replays": self.save_replays,
            }
        )

    def record_run_parameters(self, parameters: Mapping[str, Any]) -> None:
        """Add effective experiment parameters to the canonical run manifest."""
        self.context.log_params(dict(parameters))

    def record_progress(
        self,
        completed: int,
        total: int,
        description: str = "Evaluating games",
    ) -> None:
        """Publish transient batch progress without persisting extra events."""
        self.context.log_progress(completed, total, description)

    def record_game(
        self,
        game_index: int,
        game: ShadowChaseGame,
        *,
        execution_time_seconds: float | None = None,
        game_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        extra_metrics: Mapping[str, Any] | None = None,
    ) -> RecordedGame:
        """Persist one game's numeric metrics and optional versioned replay."""
        if game_index < 0:
            raise ValueError("game_index must be non-negative")
        if execution_time_seconds is not None and execution_time_seconds < 0:
            raise ValueError("execution_time_seconds must be non-negative")

        summary = _game_summary(game)
        metrics = _game_metrics(summary, execution_time_seconds)
        if extra_metrics:
            metrics.update(_prefix_metrics("game", extra_metrics))
        self.context.log_metrics(game_index, metrics)
        self._aggregate.update(summary, execution_time_seconds)

        replay_path = None
        if self.save_replays and self.recording_level != "summary":
            replay = serialize_game_replay(
                game,
                game_index=game_index,
                game_id=game_id,
                metadata=metadata,
                recording_level=self.recording_level,
            )
            identifier = _safe_identifier(game_id or f"game-{game_index:06d}")
            filename = f"{game_index:06d}-{identifier}.json"
            replay_path = self.context.save_result(
                filename,
                replay,
                category="replays",
            )

        logger.debug(
            "Recorded game %d with %d metrics and replay=%s",
            game_index,
            len(metrics),
            bool(replay_path),
        )
        return RecordedGame(metrics=metrics, replay_path=replay_path)

    def finalize(self, summary: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Persist aggregate evaluation results and return the merged summary."""
        merged = self._aggregate.as_summary()
        if summary:
            merged.update(_prefix_metrics("evaluation", summary))
        self.context.log_summary(merged)
        return merged


class TrainingRunRecorder:
    """Record training metrics and artifacts with stable hierarchical names."""

    def __init__(self, context: RunContext) -> None:
        """Bind the recorder to the single run owned by a training command."""
        self.context = context

    def record_run_parameters(self, parameters: Mapping[str, Any]) -> None:
        """Store the effective training configuration in the run manifest."""
        self.context.log_params(dict(parameters))

    def record_metrics(
        self,
        step: int,
        metrics: Mapping[str, Any],
        phase: str = "train",
    ) -> None:
        """Record one same-step metric batch below a phase namespace."""
        if step < 0:
            raise ValueError("step must be non-negative")
        self.context.log_metrics(step, _prefix_metrics(phase, metrics))

    def record_progress(
        self,
        completed: int,
        total: int,
        description: str = "Training",
    ) -> None:
        """Publish transient progress without adding high-frequency storage."""
        self.context.log_progress(completed, total, description)

    def record_artifact(
        self,
        source: str | Path,
        *,
        kind: str,
        name: str | None = None,
    ) -> Path:
        """Copy and register a model, plot, replay, or related training file."""
        return self.context.log_artifact(source, kind=kind, name=name)

    def record_checkpoint(
        self,
        source: str | Path,
        name: str | None = None,
    ) -> Path:
        """Register a model checkpoint using the canonical artifact API."""
        return self.record_artifact(source, kind="model", name=name)

    def finalize(
        self,
        summary: Mapping[str, Any],
        namespace: str = "training",
    ) -> dict[str, Any]:
        """Persist final results below the training or another namespace.

        A standalone evaluation command reuses this recorder but owns an
        ``evaluation`` run, so it overrides the summary namespace instead of
        mislabeling its results as training output.
        """
        namespaced = _prefix_metrics(namespace, summary)
        self.context.log_summary(namespaced)
        return namespaced


def serialize_game_state(state: GameState) -> dict[str, Any]:
    """Convert a game-state snapshot into a stable JSON-compatible record."""
    return {
        "detective_positions": list(state.detective_positions),
        "mrx_position": state.MrX_position,
        "turn": state.turn.value,
        "turn_count": state.turn_count,
        "mrx_turn_count": state.MrX_turn_count,
        "mrx_visible": state.MrX_visible,
        "double_move_active": state.double_move_active,
        "detective_tickets": {
            str(detective_id): _serialize_ticket_mapping(tickets)
            for detective_id, tickets in state.detective_tickets.items()
        },
        "mrx_tickets": _serialize_ticket_mapping(state.MrX_tickets),
        "mrx_moves": [
            {
                "destination": destination,
                "transport": transport.name.lower(),
            }
            for destination, transport in state.MrX_moves_log
        ],
    }


def serialize_game_replay(
    game: ShadowChaseGame,
    *,
    game_index: int,
    game_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    recording_level: str = "actions",
) -> dict[str, Any]:
    """Create a versioned replay at summary, action, or full detail level."""
    if recording_level not in RECORDING_LEVELS:
        raise ValueError(
            f"Unsupported Shadow Chase recording level: {recording_level}"
        )
    replay = {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "game_index": game_index,
        "game_id": game_id,
        "recording_level": recording_level,
        "metadata": json_safe(dict(metadata or {})),
        "summary": _game_summary(game),
    }
    if recording_level in {"actions", "full"}:
        replay["states"] = [
            serialize_game_state(state) for state in game.game_history
        ]
        replay["ticket_history"] = json_safe(game.ticket_history)
    if recording_level == "full":
        replay["configuration"] = {
            "num_detectives": game.num_detectives,
            "reveal_turns": sorted(game.reveal_turns),
            "graph": json_safe(nx.node_link_data(game.graph, edges="links")),
        }
    return replay


def _game_summary(game: ShadowChaseGame) -> dict[str, Any]:
    """Extract stable scalar outcome fields from an initialized game."""
    if game.game_state is None:
        raise ValueError("Cannot record an uninitialized game")
    winner = game.get_winner()
    return {
        "completed": game.is_game_over(),
        "winner": winner.value if winner else None,
        "turns": game.game_state.turn_count,
        "mrx_turns": game.game_state.MrX_turn_count,
        "history_states": len(game.game_history),
        "ticket_events": len(game.ticket_history),
        "mrx_final_position": game.game_state.MrX_position,
        "detective_final_positions": list(
            game.game_state.detective_positions
        ),
    }


def _game_metrics(
    summary: Mapping[str, Any],
    execution_time_seconds: float | None,
) -> dict[str, Any]:
    """Convert a game summary into report-friendly numeric metrics."""
    winner = summary.get("winner")
    metrics = {
        "game/completed": int(bool(summary["completed"])),
        "game/mrx_win": int(winner == Player.MRX.value),
        "game/detective_win": int(winner == Player.DETECTIVES.value),
        "game/incomplete": int(winner is None),
        "game/turns": int(summary["turns"]),
        "game/mrx_turns": int(summary["mrx_turns"]),
        "game/history_states": int(summary["history_states"]),
        "game/ticket_events": int(summary["ticket_events"]),
    }
    if execution_time_seconds is not None:
        metrics["game/duration_seconds"] = execution_time_seconds
    return metrics


def _serialize_ticket_mapping(tickets: Mapping[Any, Any]) -> dict[str, Any]:
    """Normalize ticket enum keys without losing their domain names."""
    return {
        getattr(ticket, "value", str(ticket)): count
        for ticket, count in tickets.items()
    }


def _prefix_metrics(
    namespace: str,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize and prefix metric names while preserving nested suffixes."""
    safe_namespace = _safe_metric_segment(namespace)
    prefixed = {}
    for name, value in metrics.items():
        raw_name = str(name).strip().strip("/")
        if not raw_name:
            raise ValueError("Metric names cannot be empty")
        if raw_name.startswith(f"{safe_namespace}/"):
            key = raw_name
        else:
            key = f"{safe_namespace}/{raw_name}"
        prefixed[key] = value
    return prefixed


def _safe_identifier(value: str) -> str:
    """Convert external game identifiers into one safe filename segment."""
    identifier = _UNSAFE_IDENTIFIER.sub("-", str(value)).strip("-._")
    return identifier or "game"


def _safe_metric_segment(value: str) -> str:
    """Validate and normalize a single metric namespace segment."""
    segment = _UNSAFE_METRIC_SEGMENT.sub("_", str(value).strip()).strip("_.")
    if not segment:
        raise ValueError("Metric namespace cannot be empty")
    return segment
