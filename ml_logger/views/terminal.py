"""Configurable Rich live view for the shared run event stream."""

from __future__ import annotations

import multiprocessing
from collections import deque
from threading import Lock
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.panel import Panel

from ..core.layout import (
    create_command_center_layout,
    render_hardware_table,
    render_logs_panel,
    render_metrics_table,
)
from ..core.progress import ProgressManager
from ..events import EventKind, RunEvent
from ..utils.formatting import Highlighter
from .selection import flatten_values, matches_metric


def resolve_dashboard_mode(settings: dict[str, Any]) -> str:
    """Resolve ``auto`` into either a live or compact terminal view."""
    mode = settings.get("mode", "auto")
    if mode != "auto":
        return mode
    interactive = Console().is_terminal
    is_parent = multiprocessing.current_process().name == "MainProcess"
    return "live" if interactive and is_parent else "compact"


class LiveDashboard:
    """Render logs, selected metrics, telemetry, and progress from events."""

    def __init__(
        self,
        dashboard_settings: dict[str, Any],
        highlight_rules: list[dict[str, str]],
    ):
        self.settings = dashboard_settings
        self.refresh_rate = dashboard_settings.get("refresh_rate", 4)
        self.max_metrics = dashboard_settings.get("max_metrics", 16)
        self.metric_filters = dashboard_settings.get("metrics", {})
        self.logs = deque(maxlen=dashboard_settings.get("max_log_lines", 50))
        self.metrics: dict[str, Any] = {}
        self.telemetry: dict[str, Any] = {}
        self.highlighter = Highlighter(highlight_rules)
        self.progress = ProgressManager()
        self._layout = create_command_center_layout()
        self._lock = Lock()
        self._live: Live | None = None

    def start(self) -> None:
        """Start the Rich live renderer once."""
        if self._live is not None:
            return
        self._live = Live(
            get_renderable=self._render,
            refresh_per_second=self.refresh_rate,
            screen=self.settings.get("screen", True),
        )
        self._live.start()

    def stop(self) -> None:
        """Stop the Rich renderer when active."""
        if self._live is not None:
            self._live.stop()
            self._live = None

    def on_event(self, event: RunEvent) -> None:
        """Update cached presentation state from one event."""
        if event.kind == EventKind.LOG:
            self._handle_log(event)
        elif event.kind == EventKind.METRICS:
            self._handle_metrics(event)
        elif event.kind == EventKind.TELEMETRY:
            with self._lock:
                self.telemetry = flatten_values(event.payload)
        elif event.kind == EventKind.PROGRESS:
            self._handle_progress(event.payload)

    def _handle_log(self, event: RunEvent) -> None:
        message = event.payload.get("formatted") or event.payload.get("message", "")
        with self._lock:
            self.logs.append(self.highlighter.apply(str(message)))

    def _handle_metrics(self, event: RunEvent) -> None:
        """Merge selected latest values while respecting the display limit."""
        selected = {
            name: value
            for name, value in flatten_values(event.payload).items()
            if matches_metric(name, self.metric_filters)
        }
        with self._lock:
            self.metrics.update(selected)
            while len(self.metrics) > self.max_metrics:
                self.metrics.pop(next(iter(self.metrics)))

    def _handle_progress(self, payload: dict[str, Any]) -> None:
        self.progress.update(
            completed=int(payload["completed"]),
            total=int(payload["total"]),
            description=str(payload["description"]),
        )
        self._layout["footer"].visible = self.progress.is_active

    def _render(self):
        """Refresh Rich panels from short lock-protected snapshots."""
        with self._lock:
            logs = list(self.logs)
            metrics = dict(self.metrics)
            telemetry = dict(self.telemetry)
        self._layout["main_split"]["logs"].update(render_logs_panel(logs))
        self._layout["main_split"]["sidebar"]["metrics"].update(
            Panel(
                render_metrics_table(metrics),
                title="Metrics",
                border_style="magenta",
            )
        )
        self._layout["main_split"]["sidebar"]["hardware"].update(
            Panel(
                render_hardware_table(telemetry),
                title="System",
                border_style="cyan",
            )
        )
        if self.progress.is_active:
            self._layout["footer"].update(self.progress.render())
        return self._layout
