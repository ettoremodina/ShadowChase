"""Canonical storage for runs, metrics, results, and their catalog."""

from __future__ import annotations

import fnmatch
import json
import os
import shlex
import shutil
import sqlite3
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .configs.config_loader import load_config
from .events import EventBus, EventKind, EventListener, RunEvent
from .serialization import json_safe

SCHEMA_VERSION = 2
DEFAULT_ARTIFACTS_DIR = "artifacts"
ARTIFACTS_ENVIRONMENT_VARIABLE = "ML_LOGGER_ARTIFACTS_DIR"


def utc_now() -> str:
    """Return an ISO 8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _atomic_json_write(path: Path, data: dict[str, Any]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(json_safe(data), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _git_metadata() -> dict[str, Any]:
    """Capture the current commit and dirty flag without requiring Git."""
    metadata: dict[str, Any] = {"commit": None, "dirty": None}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        )
        metadata.update(commit=commit.stdout.strip(), dirty=bool(status.stdout))
    except (OSError, subprocess.CalledProcessError):
        pass
    return metadata


class RunCatalog:
    """SQLite index and canonical append-only event store."""

    def __init__(self, database_path: str | Path):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> None:
        """Create idempotent run and event tables with lookup indexes."""
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    run_type TEXT NOT NULL,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    path TEXT NOT NULL,
                    manifest_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    step INTEGER,
                    payload_json TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES runs(run_id)
                );
                CREATE INDEX IF NOT EXISTS idx_runs_started_at ON runs(started_at);
                CREATE INDEX IF NOT EXISTS idx_events_run_kind
                    ON events(run_id, kind, sequence);
                """
            )

    def upsert_run(self, manifest: dict[str, Any], path: Path) -> None:
        """Insert a run or refresh its mutable status and manifest fields."""
        payload = json.dumps(json_safe(manifest), ensure_ascii=False)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO runs (
                    run_id, run_type, name, status, started_at, ended_at,
                    path, manifest_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    status=excluded.status,
                    ended_at=excluded.ended_at,
                    manifest_json=excluded.manifest_json
                """,
                (
                    manifest["run_id"],
                    manifest["run_type"],
                    manifest["name"],
                    manifest["status"],
                    manifest["started_at"],
                    manifest.get("ended_at"),
                    str(path),
                    payload,
                ),
            )

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return the newest cataloged runs up to ``limit``."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Return one catalog row by run identifier, if present."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        return dict(row) if row else None

    def append_event(self, event: RunEvent) -> None:
        """Persist one immutable event."""
        payload = json.dumps(json_safe(event.payload), ensure_ascii=False)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO events (run_id, kind, timestamp, step, payload_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    event.run_id,
                    event.kind.value,
                    event.timestamp,
                    event.step,
                    payload,
                ),
            )

    def list_events(
        self,
        run_id: str,
        kind: EventKind | str | None = None,
    ) -> list[RunEvent]:
        """Return persisted events for a run in insertion order."""
        query = "SELECT * FROM events WHERE run_id = ?"
        parameters: tuple[Any, ...] = (run_id,)
        if kind is not None:
            kind_value = kind.value if isinstance(kind, EventKind) else kind
            query += " AND kind = ?"
            parameters = (run_id, kind_value)
        query += " ORDER BY sequence"
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [_event_from_row(row) for row in rows]


@dataclass
class RunContext:
    """Lifecycle, persistence, and event context for one executable run."""

    run_id: str
    run_type: str
    name: str
    root_dir: Path
    run_dir: Path
    manifest: dict[str, Any]
    catalog: RunCatalog
    settings: dict[str, Any]
    _event_bus: EventBus = field(default_factory=EventBus, repr=False)
    _stream_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _finish_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _runtime_closer: Callable[[], None] | None = field(default=None, repr=False)
    _owns_streams: bool = field(default=True, repr=False)

    @classmethod
    def create(
        cls,
        run_type: str,
        name: str | None = None,
        config: dict[str, Any] | None = None,
        root_dir: str | Path | None = None,
        metadata: dict[str, Any] | None = None,
        settings: dict[str, Any] | None = None,
    ) -> "RunContext":
        """Create a running context with versioned metadata and event storage."""
        effective_settings = settings or load_config(run_type=run_type)
        started_at = utc_now()
        run_name = name or run_type
        run_id = _new_run_id(run_type)
        artifacts_root, run_dir = _create_run_directories(
            run_id,
            root_dir,
            effective_settings,
        )
        identity = {
            "run_id": run_id,
            "run_type": run_type,
            "name": run_name,
            "started_at": started_at,
        }
        manifest = _build_manifest(
            identity,
            config,
            metadata,
            effective_settings,
        )
        _write_initial_run_files(run_dir, manifest, config)
        catalog = RunCatalog(artifacts_root / "catalog.sqlite")
        context = cls(
            run_id,
            run_type,
            run_name,
            artifacts_root,
            run_dir,
            manifest,
            catalog,
            effective_settings,
        )
        catalog.upsert_run(manifest, run_dir)
        return context

    @classmethod
    def attach(
        cls,
        run_id: str,
        run_dir: str | Path,
        root_dir: str | Path,
        writer: bool = False,
    ) -> "RunContext":
        """Attach to an existing run, optionally owning compatibility streams."""
        path = Path(run_dir)
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        return cls(
            run_id=run_id,
            run_type=manifest["run_type"],
            name=manifest["name"],
            root_dir=Path(root_dir),
            run_dir=path,
            manifest=manifest,
            catalog=RunCatalog(Path(root_dir) / "catalog.sqlite"),
            settings=manifest.get("logger_settings") or load_config(
                run_type=manifest["run_type"]
            ),
            _owns_streams=writer,
        )

    def descriptor(self) -> dict[str, str]:
        """Return a process-safe descriptor for worker attachment."""
        return {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "root_dir": str(self.root_dir),
        }

    def subscribe(self, listener: EventListener) -> None:
        """Attach an optional view or integration to this run."""
        self._event_bus.subscribe(listener)

    @property
    def event_bus(self) -> EventBus:
        """Expose the bus for standard logging bridges and advanced plugins."""
        return self._event_bus

    @property
    def listener_errors(self) -> tuple[tuple[EventListener, Exception], ...]:
        """Return non-fatal failures raised by optional event consumers."""
        return self._event_bus.errors

    def unsubscribe(self, listener: EventListener) -> None:
        """Detach a previously registered view or integration."""
        self._event_bus.unsubscribe(listener)

    def install_runtime_closer(self, closer: Callable[[], None]) -> None:
        """Register runtime cleanup owned by the run lifecycle."""
        self._runtime_closer = closer

    def emit_started(self) -> None:
        """Publish the start event after runtime consumers are installed."""
        self._publish(
            EventKind.RUN_STARTED,
            {"run_type": self.run_type, "name": self.name},
        )

    def log_params(self, params: dict[str, Any]) -> None:
        """Persist run parameters and broadcast their new values."""
        if not self.saving_enabled("params"):
            return
        safe_params = json_safe(params)
        self.manifest.setdefault("params", {}).update(safe_params)
        _atomic_json_write(self.run_dir / "manifest.json", self.manifest)
        self.catalog.upsert_run(self.manifest, self.run_dir)
        self._publish(EventKind.PARAMS, safe_params)

    def log_summary(self, summary: dict[str, Any]) -> None:
        """Merge final or intermediate summary values into the manifest."""
        safe_summary = json_safe(summary)
        self.manifest.setdefault("result", {}).update(safe_summary)
        _atomic_json_write(self.run_dir / "manifest.json", self.manifest)
        self.catalog.upsert_run(self.manifest, self.run_dir)
        self._publish(EventKind.RESULT, {"summary": safe_summary})

    def log_metric(self, name: str, value: Any, step: int) -> None:
        """Record one metric through the batch-oriented metric API."""
        self.log_metrics(step, {name: value})

    def log_metrics(self, step: int, metrics: dict[str, Any]) -> None:
        """Persist and broadcast one timestamped metric batch."""
        if not self.saving_enabled("metrics"):
            return
        selected_metrics = _select_metrics(
            json_safe(metrics),
            self.settings.get("metrics", {}),
        )
        if not selected_metrics:
            return
        event = self._publish(EventKind.METRICS, selected_metrics, step)
        self._append_compatibility_stream("metrics.jsonl", event)

    def log_telemetry(self, telemetry: dict[str, Any]) -> None:
        """Persist and broadcast one structured hardware sample."""
        if not self.saving_enabled("telemetry"):
            return
        event = self._publish(EventKind.TELEMETRY, json_safe(telemetry))
        self._append_compatibility_stream("telemetry.jsonl", event)

    def log_progress(
        self,
        completed: int,
        total: int,
        description: str = "Running",
    ) -> None:
        """Broadcast progress without persisting high-frequency updates."""
        self._publish(
            EventKind.PROGRESS,
            {
                "completed": completed,
                "total": total,
                "description": description,
            },
            persist=False,
        )

    def log_artifact(
        self,
        source: str | Path,
        kind: str = "artifact",
        name: str | None = None,
    ) -> Path:
        """Copy a file or directory into the run and register it."""
        source_path = Path(source)
        target_name = Path(name).name if name else source_path.name
        destination = self.run_dir / "artifacts" / _safe_segment(kind) / target_name
        if not self.saving_enabled("artifacts"):
            return destination
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        _copy_artifact(source_path, destination)
        self._publish(
            EventKind.ARTIFACT,
            {
                "path": str(destination.relative_to(self.run_dir)),
                "kind": kind,
            },
        )
        return destination

    def save_result(
        self,
        filename: str,
        result: dict[str, Any],
        category: str = "analysis",
    ) -> Path:
        """Atomically save a JSON result below a run category directory.

        Returns the intended path even when result saving is disabled, allowing
        callers to keep a stable control flow.
        """
        output_path = _safe_run_destination(self.run_dir, category, filename)
        if not self.saving_enabled("results"):
            return output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json_write(output_path, result)
        self._publish(
            EventKind.RESULT,
            {
                "path": str(output_path.relative_to(self.run_dir)),
                "category": category,
            },
        )
        return output_path

    def saving_enabled(self, category: str) -> bool:
        """Return whether the global and category-specific save switches are on."""
        saving = self.settings.get("saving", {})
        return bool(saving.get("enabled", True) and saving.get(category, True))

    def complete(self, metadata: dict[str, Any] | None = None) -> None:
        """Finalize the run successfully and merge optional result metadata."""
        self._finish("completed", metadata)

    def fail(self, error: BaseException | str) -> None:
        """Finalize the run as failed with a serialized error message."""
        message = str(error)
        if not message and isinstance(error, BaseException):
            message = type(error).__name__
        self._finish("failed", {"error": message})

    def _finish(self, status: str, metadata: dict[str, Any] | None) -> None:
        """Finalize storage, publish status, and release runtime resources."""
        with self._finish_lock:
            if self.manifest["status"] != "running":
                return
            self.manifest["status"] = status
            self.manifest["ended_at"] = utc_now()
            if metadata:
                self.manifest.setdefault("result", {}).update(json_safe(metadata))
            _atomic_json_write(self.run_dir / "manifest.json", self.manifest)
            self.catalog.upsert_run(self.manifest, self.run_dir)
            self._sync_event_exports()
            kind = (
                EventKind.RUN_COMPLETED
                if status == "completed"
                else EventKind.RUN_FAILED
            )
            self._publish(kind, json_safe(metadata or {}))
        self._close_runtime()

    def _publish(
        self,
        kind: EventKind,
        payload: dict[str, Any],
        step: int | None = None,
        persist: bool = True,
    ) -> RunEvent:
        event = RunEvent(kind, self.run_id, utc_now(), payload, step)
        if persist:
            self.catalog.append_event(event)
        self._event_bus.publish(event)
        return event

    def _append_compatibility_stream(
        self,
        filename: str,
        event: RunEvent,
    ) -> None:
        """Append a parent-owned JSONL record for legacy readers."""
        if not self._owns_streams:
            return
        path = self.run_dir / "metrics" / filename
        record = _compatibility_record(event)
        with self._stream_lock, path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _sync_event_exports(self) -> None:
        """Materialize complete JSONL views from the canonical event catalog."""
        if not self._owns_streams:
            return
        mappings = {
            EventKind.METRICS: "metrics.jsonl",
            EventKind.TELEMETRY: "telemetry.jsonl",
        }
        for kind, filename in mappings.items():
            events = self.catalog.list_events(self.run_id, kind)
            if events:
                _atomic_jsonl_write(
                    self.run_dir / "metrics" / filename,
                    [_compatibility_record(event) for event in events],
                )

    def _close_runtime(self) -> None:
        closer, self._runtime_closer = self._runtime_closer, None
        if closer is not None:
            closer()


def _event_from_row(row: sqlite3.Row) -> RunEvent:
    return RunEvent(
        kind=EventKind(row["kind"]),
        run_id=row["run_id"],
        timestamp=row["timestamp"],
        step=row["step"],
        payload=json.loads(row["payload_json"]),
    )


def _new_run_id(run_type: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{run_type}-{timestamp}-{uuid.uuid4().hex[:8]}"


def _create_run_directories(
    run_id: str,
    root_dir: str | Path | None,
    settings: dict[str, Any],
) -> tuple[Path, Path]:
    """Resolve the artifact root and create configured run subdirectories."""
    artifacts_settings = settings.get("artifacts", {})
    artifacts_root = Path(
        root_dir
        or os.environ.get(ARTIFACTS_ENVIRONMENT_VARIABLE)
        or artifacts_settings.get("root_dir", DEFAULT_ARTIFACTS_DIR)
    )
    date_path = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    run_dir = artifacts_root / "runs" / date_path / run_id
    configured = artifacts_settings.get("directories", [])
    directories = {"logs", "metrics", "artifacts", *configured}
    for directory in directories:
        (run_dir / _safe_segment(directory)).mkdir(parents=True, exist_ok=True)
    return artifacts_root, run_dir


def _write_initial_run_files(
    run_dir: Path,
    manifest: dict[str, Any],
    config: dict[str, Any] | None,
) -> None:
    _atomic_json_write(run_dir / "manifest.json", manifest)
    if config is not None:
        _atomic_json_write(run_dir / "config.json", config)


def _compatibility_record(event: RunEvent) -> dict[str, Any]:
    record = {"timestamp": event.timestamp}
    if event.step is not None:
        record["step"] = event.step
    record.update(json_safe(event.payload))
    return record


def _atomic_jsonl_write(path: Path, records: list[dict[str, Any]]) -> None:
    """Replace a JSONL stream atomically from complete records."""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary_path.replace(path)


def _copy_artifact(source: Path, destination: Path) -> None:
    """Copy one file or directory unless it already occupies the target."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == destination.resolve():
        return
    if source.is_dir():
        shutil.copytree(source, destination, dirs_exist_ok=True)
    else:
        shutil.copy2(source, destination)


def _safe_segment(value: str) -> str:
    """Validate a single user-configurable path segment."""
    segment = str(value).strip()
    if not segment or Path(segment).name != segment or segment in {".", ".."}:
        raise ValueError(f"Invalid path segment: {value}")
    return segment


def _safe_run_destination(run_dir: Path, category: str, filename: str) -> Path:
    """Resolve a result path while preventing escape from the run directory."""
    category_path = Path(category)
    filename_path = Path(filename)
    relative_path = category_path / filename_path
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("Result path must stay inside the run directory")
    return run_dir / relative_path


def _select_metrics(
    metrics: dict[str, Any],
    filters: dict[str, Any],
) -> dict[str, Any]:
    """Filter stored metric keys with shell-style include and exclude rules."""
    include = filters.get("include", ["*"])
    exclude = filters.get("exclude", [])
    return {
        name: value
        for name, value in metrics.items()
        if any(fnmatch.fnmatch(name, pattern) for pattern in include)
        and not any(fnmatch.fnmatch(name, pattern) for pattern in exclude)
    }


def _build_manifest(
    identity: dict[str, str],
    config: dict[str, Any] | None,
    metadata: dict[str, Any] | None,
    logger_settings: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the versioned, JSON-safe provenance record for a new run."""
    command = " ".join(shlex.quote(argument) for argument in sys.argv)
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": identity["run_id"],
        "run_type": identity["run_type"],
        "name": identity["name"],
        "status": "running",
        "started_at": identity["started_at"],
        "ended_at": None,
        "command": command,
        "python": sys.version,
        "git": _git_metadata(),
        "config": json_safe(config or {}),
        "metadata": json_safe(metadata or {}),
        "logger_settings": json_safe(logger_settings),
    }
