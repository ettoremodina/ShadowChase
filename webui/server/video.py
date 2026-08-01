"""Background video export.

The Tkinter dialog blocked on a modal progress bar. Here the export runs on a
worker thread and the browser polls for progress, so the board stays usable
while a replay renders.
"""
from __future__ import annotations

import threading
import uuid
from pathlib import Path
from typing import Dict, Optional

EXPORT_DIR = Path("exports")


class ExportJob:
    def __init__(self, job_id: str, game_id: str, output: Path):
        self.job_id = job_id
        self.game_id = game_id
        self.output = output
        self.status = "running"  # running | done | failed
        self.current = 0
        self.total = 0
        self.message = "Preparing frames"
        self.error: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "jobId": self.job_id,
            "gameId": self.game_id,
            "status": self.status,
            "current": self.current,
            "total": self.total,
            "message": self.message,
            "output": str(self.output).replace("\\", "/"),
            "error": self.error,
        }


class ExportManager:
    """Tracks every export started this session."""

    def __init__(self):
        self._jobs: Dict[str, ExportJob] = {}
        self._lock = threading.Lock()

    def get(self, job_id: str) -> Optional[ExportJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def start(self, game, game_id: str, *, filename: str, frame_duration: float,
              end_delay: float) -> ExportJob:
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)

        # Filenames only: an export never escapes the exports folder.
        safe_name = Path(filename or f"{game_id}.mp4").name
        if not safe_name.lower().endswith(".mp4"):
            safe_name += ".mp4"
        output = EXPORT_DIR / safe_name

        job = ExportJob(uuid.uuid4().hex[:12], game_id, output)
        with self._lock:
            self._jobs[job.job_id] = job

        thread = threading.Thread(
            target=self._run,
            args=(job, game, game_id, frame_duration, end_delay),
            daemon=True,
        )
        thread.start()
        return job

    def _run(self, job: ExportJob, game, game_id: str, frame_duration: float,
             end_delay: float) -> None:
        try:
            from ShadowChase.ui.video_exporter import GameVideoExporter

            def progress(current, total, message):
                job.current = int(current)
                job.total = int(total)
                job.message = str(message)

            exporter = GameVideoExporter(
                game,
                game_id,
                str(job.output),
                frame_duration,
                end_delay_seconds=end_delay,
            )
            result = exporter.export_video(progress)
            job.output = Path(result) if result else job.output
            job.status = "done"
            job.message = "Export finished"
        except Exception as error:  # noqa: BLE001 - reported to the browser
            job.status = "failed"
            job.error = str(error)
            job.message = "Export failed"


export_manager = ExportManager()
