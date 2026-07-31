"""Backward-compatible dashboard facade over the canonical event stream."""

from __future__ import annotations

import logging
import time
from typing import Any

from ..configs.config_loader import load_config
from ..runtime import configure_logging
from ..storage import RunContext
from ..telemetry import TelemetrySampler
from ..utils.export import export_run_to_csv
from ..views.terminal import LiveDashboard


class DashboardLogger:
    """Legacy facade retained while callers migrate to ``run_scope``."""

    def __init__(
        self,
        config_path: str | None = None,
        run_context: RunContext | None = None,
    ):
        self.config = load_config(config_path, run_type="dashboard")
        self.run_context = run_context or RunContext.create(
            "dashboard",
            settings=self.config,
        )
        self._owns_context = run_context is None
        self._metric_step = 0
        self._progress_total = 0
        self._progress_completed = 0
        self._plugins: list[Any] = []
        self._view = LiveDashboard(
            self.config.get("dashboard", {}),
            self.config.get("highlights", []),
        )
        self.run_context.subscribe(self._view)
        telemetry_settings = self.config.get("telemetry", {})
        self._telemetry = (
            TelemetrySampler(self.run_context, telemetry_settings)
            if telemetry_settings.get("enabled", True)
            else None
        )
        configure_logging(self.run_context, console=False)
        self.root_logger = logging.getLogger()

    def add_plugin(self, plugin) -> None:
        """Register a legacy callback plugin."""
        self._plugins.append(plugin)

    def start_progress(
        self,
        total_steps: int,
        description: str = "Training",
    ) -> None:
        """Start a progress sequence."""
        self._progress_total = total_steps
        self._progress_completed = 0
        self.run_context.log_progress(0, total_steps, description)

    def step_progress(
        self,
        advance: int = 1,
        description: str | None = None,
    ) -> None:
        """Advance the active progress sequence."""
        self._progress_completed += advance
        self.run_context.log_progress(
            self._progress_completed,
            self._progress_total,
            description or "Training",
        )

    def log(self, level, message, *args, **kwargs) -> None:
        """Emit a standard logging record."""
        self.root_logger.log(level, message, *args, **kwargs)

    def info(self, message, *args, **kwargs) -> None:
        """Emit an informational record."""
        self.root_logger.info(message, *args, **kwargs)

    def warning(self, message, *args, **kwargs) -> None:
        """Emit a warning record."""
        self.root_logger.warning(message, *args, **kwargs)

    def error(self, message, *args, **kwargs) -> None:
        """Emit an error record."""
        self.root_logger.error(message, *args, **kwargs)

    def update_metrics(self, category: str, name: str, value: Any) -> None:
        """Send one legacy metric through the canonical context."""
        self._metric_step += 1
        self.run_context.log_metric(
            f"{category}/{name}",
            value,
            self._metric_step,
        )
        for plugin in self._plugins:
            plugin.on_metric_update(category, name, value)

    def log_metadata(self, info_dict: dict[str, Any]) -> None:
        """Persist dashboard metadata as a generic result."""
        self.run_context.save_result(
            "dashboard_metadata.json",
            info_dict,
            category="analysis",
        )

    def log_game_run(self, run_data: dict[str, Any]) -> None:
        """Persist a legacy caller-owned record without game assumptions."""
        self.run_context.save_result(
            f"legacy_record_{time.time_ns()}.json",
            run_data,
            category="artifacts",
        )

    def export_to_csv(self) -> None:
        """Export compatibility JSONL streams to CSV."""
        export_run_to_csv(self.run_context.run_dir / "metrics")

    def start(self) -> None:
        """Start the live view, telemetry, and legacy plugin callbacks."""
        self._view.start()
        if self._telemetry is not None:
            self._telemetry.start()
        for plugin in self._plugins:
            plugin.on_startup(self.config)

    def stop(self) -> None:
        """Stop owned services and finalize an internally created run."""
        if self._telemetry is not None:
            self._telemetry.stop()
        self._view.stop()
        for plugin in self._plugins:
            plugin.on_shutdown()
        if self._owns_context:
            self.run_context.complete()
