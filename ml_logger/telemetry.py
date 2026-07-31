"""Background collection of numeric process and host telemetry."""

from __future__ import annotations

import os
import subprocess
import threading
from typing import Any

import psutil

from .storage import RunContext

MEGABYTE = 1024 * 1024


class TelemetrySampler:
    """Sample hardware outside the rendering loop and publish cached events."""

    def __init__(self, context: RunContext, settings: dict[str, Any]):
        self.context = context
        self.settings = settings
        self.interval = float(settings.get("sample_interval_sec", 5))
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._process = psutil.Process(os.getpid())

    def start(self) -> None:
        """Capture an initial sample and start the background sampler."""
        if self._thread is not None:
            return
        self.context.log_telemetry(self.collect())
        self._thread = threading.Thread(
            target=self._sample_until_stopped,
            name=f"ml-logger-telemetry-{self.context.run_id}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop sampling and join the worker without waiting for its interval."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    def collect(self) -> dict[str, Any]:
        """Return one structured sample using the configured providers."""
        sample: dict[str, Any] = {}
        if self.settings.get("include_system", True):
            sample["system"] = _system_metrics()
        if self.settings.get("include_process", True):
            sample["process"] = _process_metrics(self._process)
        provider = self.settings.get("gpu_provider", "auto")
        if provider != "off":
            sample["gpu"] = _gpu_metrics(provider)
        return sample

    def _sample_until_stopped(self) -> None:
        while not self._stop_event.wait(self.interval):
            self.context.log_telemetry(self.collect())


def _system_metrics() -> dict[str, float]:
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage(os.getcwd())
    return {
        "cpu_percent": float(psutil.cpu_percent()),
        "memory_percent": float(memory.percent),
        "memory_used_mb": round(memory.used / MEGABYTE, 3),
        "memory_total_mb": round(memory.total / MEGABYTE, 3),
        "disk_percent": float(disk.percent),
    }


def _process_metrics(process: psutil.Process) -> dict[str, float | int]:
    memory = process.memory_info()
    return {
        "pid": process.pid,
        "cpu_percent": float(process.cpu_percent()),
        "rss_mb": round(memory.rss / MEGABYTE, 3),
        "threads": process.num_threads(),
    }


def _gpu_metrics(provider: str) -> dict[str, Any]:
    """Collect one numeric NVIDIA sample or an explicit unavailable state."""
    if provider not in {"auto", "nvidia-smi"}:
        return {"provider": provider, "available": False, "devices": []}
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,"
                "memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
        devices = [_parse_gpu_line(line) for line in result.stdout.splitlines()]
        return {
            "provider": "nvidia-smi",
            "available": bool(devices),
            "devices": devices,
        }
    except (OSError, subprocess.SubprocessError, ValueError):
        return {"provider": "nvidia-smi", "available": False, "devices": []}


def _parse_gpu_line(line: str) -> dict[str, Any]:
    index, name, utilization, used, total, temperature = [
        value.strip() for value in line.split(",", maxsplit=5)
    ]
    return {
        "index": int(index),
        "name": name,
        "utilization_percent": float(utilization),
        "memory_used_mb": float(used),
        "memory_total_mb": float(total),
        "temperature_c": float(temperature),
    }
