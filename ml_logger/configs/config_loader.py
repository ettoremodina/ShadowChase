"""Load and validate the central ml_logger runtime configuration."""

import os
from copy import deepcopy
from pathlib import Path

import yaml

from ..terminal import parse_utc_offset

DEFAULT_CONFIG_PATH = Path(__file__).parent / "default_config.yaml"
PROJECT_CONFIG_NAME = "logger_config.yaml"
CONFIG_ENVIRONMENT_VARIABLE = "ML_LOGGER_CONFIG"
VALID_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
VALID_RECORDING_LEVELS = {"summary", "actions", "full"}
VALID_DASHBOARD_MODES = {"auto", "live", "compact", "off"}
VALID_REPORT_VISUALIZATIONS = {"auto", "line", "table"}


def load_config(config_path: str | Path | None = None, run_type=None) -> dict:
    """Load defaults, user configuration, and an optional run-type override."""
    default_config = _read_yaml(DEFAULT_CONFIG_PATH)
    selected_path = _select_config_path(config_path)
    config = deepcopy(default_config)
    if selected_path != DEFAULT_CONFIG_PATH:
        _deep_merge(config, _read_yaml(selected_path))
    overrides = config.get("run_type_overrides", {}).get(run_type, {})
    _deep_merge(config, overrides)
    config["config_path"] = str(selected_path)
    _validate_config(config)
    return config


def _select_config_path(config_path):
    """Resolve explicit, environment, project, then packaged configuration."""
    if config_path:
        return Path(config_path)
    environment_path = os.environ.get(CONFIG_ENVIRONMENT_VARIABLE)
    if environment_path:
        return Path(environment_path)
    project_path = Path.cwd() / PROJECT_CONFIG_NAME
    return project_path if project_path.exists() else DEFAULT_CONFIG_PATH


def _read_yaml(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def _deep_merge(destination, source):
    """Recursively copy ``source`` values over a destination mapping."""
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(destination.get(key), dict):
            _deep_merge(destination[key], value)
        else:
            destination[key] = deepcopy(value)
    return destination


def _validate_config(config):
    """Normalize and validate settings that affect runtime behavior."""
    logging_level = str(config["logging"]["level"]).upper()
    if logging_level not in VALID_LEVELS:
        raise ValueError(f"Unsupported logging level: {logging_level}")
    config["logging"]["level"] = logging_level
    parse_utc_offset(config["terminal"]["timezone"])
    _validate_dashboard(config.get("dashboard", {}))
    _validate_report(config.get("report", {}))
    _validate_recording(config)


def _validate_dashboard(settings):
    """Normalize dashboard mode and reject invalid refresh settings."""
    raw_mode = settings.get("mode", "auto")
    if isinstance(raw_mode, bool):
        mode = "live" if raw_mode else "off"
    else:
        mode = str(raw_mode).lower()
    if mode not in VALID_DASHBOARD_MODES:
        raise ValueError(f"Unsupported dashboard mode: {mode}")
    settings["mode"] = mode
    if settings.get("refresh_rate", 1) <= 0:
        raise ValueError("dashboard.refresh_rate must be positive")


def _validate_report(settings):
    """Normalize the configured HTML report visualization."""
    visualization = str(settings.get("visualization", "auto")).lower()
    if visualization not in VALID_REPORT_VISUALIZATIONS:
        raise ValueError(f"Unsupported report visualization: {visualization}")
    settings["visualization"] = visualization


def _validate_recording(config):
    """Validate optional Regicide adapter settings, including legacy keys."""
    recording = (
        config.get("integrations", {})
        .get("regicide", {})
        .get("recording")
    )
    games = config.get("games")
    settings = recording or games
    if not settings:
        return
    recording_level = settings.get(
        "level",
        settings.get("recording_level", "actions"),
    )
    if recording_level not in VALID_RECORDING_LEVELS:
        raise ValueError(f"Unsupported recording level: {recording_level}")
