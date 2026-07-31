"""Typed run events shared by storage, views, and optional integrations."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import Any, Protocol

from .serialization import json_safe


class EventKind(str, Enum):
    """Stable event names persisted in the run catalog."""

    RUN_STARTED = "run.started"
    LOG = "log"
    PARAMS = "params"
    METRICS = "metrics"
    TELEMETRY = "telemetry"
    PROGRESS = "progress"
    ARTIFACT = "artifact"
    RESULT = "result"
    RUN_COMPLETED = "run.completed"
    RUN_FAILED = "run.failed"


@dataclass(frozen=True)
class RunEvent:
    """One immutable observation produced during a run."""

    kind: EventKind
    run_id: str
    timestamp: str
    payload: dict[str, Any] = field(default_factory=dict)
    step: int | None = None

    def as_record(self) -> dict[str, Any]:
        """Return the event in a JSON-compatible representation."""
        record = {
            "timestamp": self.timestamp,
            "kind": self.kind.value,
            **json_safe(self.payload),
        }
        if self.step is not None:
            record["step"] = self.step
        return record


class EventListener(Protocol):
    """Consumer notified synchronously when a run event is published."""

    def on_event(self, event: RunEvent) -> None:
        """Handle one run event."""


class EventBus:
    """Thread-safe in-process fan-out for optional event consumers."""

    def __init__(self) -> None:
        self._listeners: list[EventListener] = []
        self._lock = RLock()
        self._errors: list[tuple[EventListener, Exception]] = []

    @property
    def listener_count(self) -> int:
        """Return the current number of subscribed consumers."""
        with self._lock:
            return len(self._listeners)

    @property
    def errors(self) -> tuple[tuple[EventListener, Exception], ...]:
        """Return non-fatal listener errors collected by the bus."""
        with self._lock:
            return tuple(self._errors)

    def subscribe(self, listener: EventListener) -> None:
        """Subscribe a listener once."""
        with self._lock:
            if listener not in self._listeners:
                self._listeners.append(listener)

    def unsubscribe(self, listener: EventListener) -> None:
        """Remove a listener when currently subscribed."""
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)

    def publish(self, event: RunEvent) -> None:
        """Notify a snapshot of listeners without breaking the training run."""
        with self._lock:
            listeners = tuple(self._listeners)
        for listener in listeners:
            try:
                listener.on_event(event)
            except Exception as error:  # Views must never terminate a run.
                with self._lock:
                    self._errors.append((listener, error))
