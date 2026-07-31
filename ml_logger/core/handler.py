"""Logging handler that feeds formatted records to the live dashboard."""

import logging
from collections import deque
from datetime import datetime, timezone
from threading import Lock

from ..events import EventBus, EventKind, RunEvent


class EventLogHandler(logging.Handler):
    """Convert standard logging records into non-persistent run events."""

    def __init__(self, run_id: str, event_bus: EventBus):
        super().__init__()
        self.run_id = run_id
        self.event_bus = event_bus

    def emit(self, record: logging.LogRecord) -> None:
        """Publish a formatted log event without recursing into logging."""
        try:
            event = RunEvent(
                kind=EventKind.LOG,
                run_id=self.run_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                payload={
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                    "formatted": self.format(record),
                },
            )
            self.event_bus.publish(event)
        except Exception:
            self.handleError(record)


class DashboardLogHandler(logging.Handler):
    """Custom logging handler that routes formatted logs to a memory deque."""
    def __init__(self, log_queue: deque, queue_lock: Lock, highlighter):
        super().__init__()
        self.log_queue = log_queue
        self.queue_lock = queue_lock
        self.highlighter = highlighter

    def emit(self, record):
        """Format, highlight, and append one record under the queue lock."""
        try:
            msg = self.format(record)
            
            # Apply dynamic rich highlighting
            text_obj = self.highlighter.apply(msg)
            
            # Append to live dashboard memory
            with self.queue_lock:
                self.log_queue.append(text_obj)
        except Exception:
            self.handleError(record)
