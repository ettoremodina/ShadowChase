"""Rich terminal rendering and fixed-offset timestamp formatting."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from rich.console import Console, ConsoleRenderable
from rich.logging import RichHandler
from rich.text import Text
from rich.traceback import Traceback

from .utils.formatting import Highlighter

UTC_OFFSET_PATTERN = re.compile(
    r"^(?:UTC)?(?P<sign>[+-])(?P<hours>\d{1,2})(?::?(?P<minutes>\d{2}))?$",
    re.IGNORECASE,
)


def parse_utc_offset(value: str) -> timezone:
    """Parse offsets such as ``+02:00``, ``+0200``, or ``UTC+2``."""
    match = UTC_OFFSET_PATTERN.fullmatch(value.strip())
    if not match:
        raise ValueError(f"Invalid UTC offset: {value}")
    hours = int(match.group("hours"))
    minutes = int(match.group("minutes") or 0)
    if hours > 23 or minutes > 59:
        raise ValueError(f"Invalid UTC offset: {value}")
    direction = 1 if match.group("sign") == "+" else -1
    return timezone(direction * timedelta(hours=hours, minutes=minutes))


def utc_offset_label(offset: timezone) -> str:
    """Return a compact UTC offset label."""
    total_minutes = int(offset.utcoffset(None).total_seconds() // 60)
    sign = "+" if total_minutes >= 0 else "-"
    hours, minutes = divmod(abs(total_minutes), 60)
    suffix = f":{minutes:02d}" if minutes else ""
    return f"UTC{sign}{hours}{suffix}"


class UtcOffsetFormatter(logging.Formatter):
    """Logging formatter that never depends on the host timezone."""

    def __init__(self, format_string, date_format, utc_offset):
        super().__init__(format_string, datefmt=date_format)
        self.utc_offset = utc_offset

    def formatTime(self, record, datefmt=None):
        """Format a record timestamp in the configured fixed UTC offset."""
        timestamp = datetime.fromtimestamp(record.created, tz=self.utc_offset)
        if datefmt:
            return timestamp.strftime(datefmt)
        return timestamp.isoformat(timespec="seconds")


class ConfiguredRichHandler(RichHandler):
    """Rich handler with project highlighting and explicit timezone."""

    def __init__(
        self,
        terminal_settings,
        highlight_rules,
        console: Console | None = None,
    ):
        self.utc_offset = parse_utc_offset(
            terminal_settings.get("timezone", "+02:00")
        )
        self.offset_label = utc_offset_label(self.utc_offset)
        self.project_highlighter = Highlighter(highlight_rules)
        super().__init__(
            console=console,
            show_time=terminal_settings.get("show_time", True),
            omit_repeated_times=terminal_settings.get(
                "omit_repeated_times", False
            ),
            show_level=terminal_settings.get("show_level", True),
            show_path=terminal_settings.get("show_source", False),
            enable_link_path=False,
            highlighter=self.project_highlighter,
            markup=False,
            rich_tracebacks=terminal_settings.get("rich_tracebacks", True),
            log_time_format=self._format_log_time,
        )

    def render(
        self,
        *,
        record: logging.LogRecord,
        traceback: Traceback | None,
        message_renderable: ConsoleRenderable,
    ) -> ConsoleRenderable:
        """Render a record using its creation time converted to UTC offset."""
        log_time = datetime.fromtimestamp(record.created, tz=self.utc_offset)
        path = Path(record.pathname).name
        return self._log_render(
            self.console,
            [message_renderable] if not traceback else [message_renderable, traceback],
            log_time=log_time,
            time_format=self._format_log_time,
            level=self.get_level_text(record),
            path=path,
            line_no=record.lineno,
            link_path=None,
        )

    def _format_log_time(self, timestamp: datetime) -> Text:
        time_text = f"{timestamp:%H:%M:%S} {self.offset_label}"
        return Text(time_text, style="dim cyan")
