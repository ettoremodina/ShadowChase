"""Public lifecycle and standard logging configuration for ML runs."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .configs.config_loader import load_config
from .core.handler import EventLogHandler
from .storage import RunContext
from .telemetry import TelemetrySampler
from .terminal import ConfiguredRichHandler, UtcOffsetFormatter, parse_utc_offset
from .views import HtmlReportView, LiveDashboard, resolve_dashboard_mode

FILE_LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s | %(message)s"
PLAIN_CONSOLE_FORMAT = "%(asctime)s %(levelname)-8s %(message)s"
PLAIN_CONSOLE_DATE_FORMAT = "%H:%M:%S %Z"
EVENT_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_MANAGED_HANDLER_ATTRIBUTE = "_ml_logger_managed"


def get_logger(name: str) -> logging.Logger:
    """Return a standard logger routed through the active run configuration."""
    return logging.getLogger(name)


def configure_logging(
    context: RunContext | None = None,
    level: int | str | None = None,
    console: bool | None = None,
) -> logging.Logger:
    """Install owned handlers without modifying unrelated handlers."""
    root_logger = logging.getLogger()
    _remove_managed_handlers(root_logger)
    settings = context.settings if context else load_config()
    logging_settings = settings.get("logging", {})
    if not logging_settings.get("enabled", True):
        return root_logger
    effective_level = level or logging_settings.get("level", "INFO")
    root_logger.setLevel(effective_level)
    terminal_settings = settings.get("terminal", {})
    utc_offset = parse_utc_offset(terminal_settings.get("timezone", "+00:00"))
    console_enabled = _console_enabled(logging_settings, console)
    if console_enabled:
        handler = _create_console_handler(
            terminal_settings,
            settings.get("highlights", []),
            utc_offset,
        )
        _install_handler(root_logger, handler)
    if context and logging_settings.get("file", True):
        _install_handler(root_logger, _create_file_handler(context, utc_offset))
    if context and context.event_bus.listener_count:
        _install_handler(root_logger, _create_event_handler(context))
    return root_logger


def start_run(
    run_type: str,
    name: str | None = None,
    config: dict[str, Any] | None = None,
    root_dir: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
    logger_config_path: str | Path | None = None,
) -> RunContext:
    """Create, configure, and start one observable run."""
    settings = load_config(logger_config_path, run_type=run_type)
    context = RunContext.create(
        run_type=run_type,
        name=name,
        config=config,
        root_dir=root_dir,
        metadata=metadata,
        settings=settings,
    )
    runtime = RunRuntime(context)
    context.install_runtime_closer(runtime.stop)
    configure_logging(context, console=runtime.console_enabled)
    runtime.start_view()
    context.emit_started()
    runtime.start_background_services()
    _log_run_start(context)
    return context


@contextmanager
def run_scope(
    run_type: str,
    name: str | None = None,
    config: dict[str, Any] | None = None,
    root_dir: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
    logger_config_path: str | Path | None = None,
) -> Iterator[RunContext]:
    """Guarantee successful or failed finalization around a run body."""
    context = start_run(
        run_type,
        name,
        config,
        root_dir,
        metadata,
        logger_config_path,
    )
    try:
        yield context
    except BaseException as error:
        context.fail(error)
        raise
    else:
        context.complete()


run = run_scope


class RunRuntime:
    """Own optional views and background services for one context."""

    def __init__(self, context: RunContext):
        self.context = context
        dashboard_settings = context.settings.get("dashboard", {})
        self.dashboard_mode = resolve_dashboard_mode(dashboard_settings)
        self.dashboard = self._create_dashboard(dashboard_settings)
        self.report = self._create_report(context.settings.get("report", {}))
        self.telemetry = self._create_telemetry(
            context.settings.get("telemetry", {})
        )
        self._stopped = False

    @property
    def console_enabled(self) -> bool:
        """Return whether standard compact logging remains visible."""
        configured = self.context.settings.get("logging", {}).get(
            "console",
            True,
        )
        return bool(configured and self.dashboard_mode != "live")

    def start_view(self) -> None:
        """Start the interactive view before the first log message."""
        if self.dashboard is not None:
            self.dashboard.start()

    def start_background_services(self) -> None:
        """Start telemetry after the run-start event has been emitted."""
        if self.telemetry is not None:
            self.telemetry.start()

    def stop(self) -> None:
        """Stop services, views, and handlers exactly once."""
        if self._stopped:
            return
        self._stopped = True
        if self.telemetry is not None:
            self.telemetry.stop()
        if self.dashboard is not None:
            self.dashboard.stop()
        _remove_managed_handlers(logging.getLogger())

    def _create_dashboard(
        self,
        settings: dict[str, Any],
    ) -> LiveDashboard | None:
        if self.dashboard_mode != "live":
            return None
        dashboard = LiveDashboard(
            settings,
            self.context.settings.get("highlights", []),
        )
        self.context.subscribe(dashboard)
        return dashboard

    def _create_report(
        self,
        settings: dict[str, Any],
    ) -> HtmlReportView | None:
        if not settings.get("enabled", True):
            return None
        report = HtmlReportView(self.context, settings)
        self.context.subscribe(report)
        return report

    def _create_telemetry(
        self,
        settings: dict[str, Any],
    ) -> TelemetrySampler | None:
        enabled = settings.get("enabled", True)
        if not enabled or not self.context.saving_enabled("telemetry"):
            return None
        return TelemetrySampler(self.context, settings)


class RunLogger:
    """Compatibility facade for existing solver loops."""

    def __init__(
        self,
        context: RunContext | None = None,
        run_type: str = "solver",
        run_name: str | None = None,
        config: dict[str, Any] | None = None,
    ):
        self.context = context or start_run(
            run_type=run_type,
            name=run_name,
            config=config,
        )
        self.logger = get_logger(run_name or run_type)

    def log(self, message: str, *args: Any) -> None:
        """Log an informational message."""
        self.logger.info(message, *args)

    def info(self, message: str, *args: Any) -> None:
        """Log an informational message."""
        self.logger.info(message, *args)

    def log_metrics(self, step: int, metrics_dict: dict[str, Any]) -> None:
        """Record metrics through the canonical event stream."""
        self.context.log_metrics(step, metrics_dict)

    def get_run_dir(self) -> str:
        """Return the active run directory as a legacy string."""
        return str(self.context.run_dir)

    @property
    def models_dir(self) -> str:
        """Return and create the legacy models directory."""
        models_dir = self.context.run_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        return str(models_dir)


def _create_console_handler(terminal_settings, highlights, utc_offset):
    """Build either the Rich console handler or its plain fallback."""
    if terminal_settings.get("colors", True):
        handler = ConfiguredRichHandler(terminal_settings, highlights)
        handler.setFormatter(logging.Formatter("%(message)s"))
        return handler
    handler = logging.StreamHandler()
    handler.setFormatter(
        UtcOffsetFormatter(
            PLAIN_CONSOLE_FORMAT,
            PLAIN_CONSOLE_DATE_FORMAT,
            utc_offset,
        )
    )
    return handler


def _create_file_handler(
    context: RunContext,
    utc_offset,
) -> logging.FileHandler:
    log_path = context.run_dir / "logs" / "run.log"
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(
        UtcOffsetFormatter(FILE_LOG_FORMAT, None, utc_offset)
    )
    return file_handler


def _create_event_handler(context: RunContext) -> EventLogHandler:
    handler = EventLogHandler(context.run_id, context.event_bus)
    handler.setFormatter(logging.Formatter(EVENT_LOG_FORMAT, datefmt="%H:%M:%S"))
    return handler


def _install_handler(logger: logging.Logger, handler: logging.Handler) -> None:
    setattr(handler, _MANAGED_HANDLER_ATTRIBUTE, True)
    logger.addHandler(handler)


def _remove_managed_handlers(logger: logging.Logger) -> None:
    """Close only handlers installed by this package."""
    for handler in list(logger.handlers):
        if getattr(handler, _MANAGED_HANDLER_ATTRIBUTE, False):
            logger.removeHandler(handler)
            handler.close()


def _console_enabled(settings: dict[str, Any], override: bool | None) -> bool:
    return settings.get("console", True) if override is None else override


def _log_run_start(context: RunContext) -> None:
    """Emit a compact summary of a newly configured run."""
    logger = get_logger(__name__)
    logger.info(
        "Run started  %-22s %s",
        context.run_type,
        context.run_id.rsplit("-", maxsplit=1)[-1],
    )
    logger.info(
        "Output       log:%s  metrics:%s  results:%s  telemetry:%s",
        _switch(context.settings["logging"]["file"]),
        _switch(context.saving_enabled("metrics")),
        _switch(context.saving_enabled("results")),
        _switch(context.saving_enabled("telemetry")),
    )
    logger.debug("Run directory: %s", context.run_dir)


def _switch(enabled: bool) -> str:
    return "on" if enabled else "off"
