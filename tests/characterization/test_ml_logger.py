"""Characterize the bundled ml_logger lifecycle and persisted schema."""

import json
import os
import sys

import pytest

from ml_logger import get_logger, run


logger = get_logger(__name__)


def _write_quiet_logger_config(path):
    """Write a deterministic logger configuration for isolated tests."""
    path.write_text(
        """version: 2
logging:
  enabled: true
  level: INFO
  console: false
  file: true
dashboard:
  mode: off
telemetry:
  enabled: false
report:
  enabled: false
""",
        encoding="utf-8",
    )
    return path


@pytest.mark.integration
def test_run_persists_manifest_metrics_and_summary(tmp_path):
    """Freeze successful run finalization and the metric JSONL contract."""
    config_path = _write_quiet_logger_config(tmp_path / "logger.yaml")
    artifact_root = tmp_path / "artifacts"

    with run(
        "characterization",
        name="baseline",
        config={"seed": 20260801},
        root_dir=artifact_root,
        logger_config_path=config_path,
    ) as context:
        context.log_params({"agent": "random"})
        context.log_metrics(0, {"game/turns": 3, "game/mrx_win": 0})
        context.log_summary({"games": 1})
        run_dir = context.run_dir

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    metrics = [
        json.loads(line)
        for line in (run_dir / "metrics" / "metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert manifest["schema_version"] == 2
    assert manifest["status"] == "completed"
    assert manifest["params"] == {"agent": "random"}
    assert manifest["result"] == {"games": 1}
    assert metrics[0]["step"] == 0
    assert metrics[0]["game/turns"] == 3


@pytest.mark.integration
@pytest.mark.xfail(
    condition=sys.platform == "win32",
    reason="Default ml_logger runtime can retain a Windows lock on catalog.sqlite",
    strict=False,
)
def test_default_run_releases_catalog_file(tmp_path):
    """Specify that completed runs should release their SQLite catalog file."""
    artifact_root = tmp_path / "default-artifacts"
    with run("catalog-release", root_dir=artifact_root):
        pass

    catalog_path = artifact_root / "catalog.sqlite"
    moved_path = artifact_root / "catalog-moved.sqlite"
    os.replace(catalog_path, moved_path)
    assert moved_path.exists()

