"""Shared selection helpers for event views."""

from __future__ import annotations

import fnmatch
from typing import Any


def flatten_values(values: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested dictionaries and lists using slash-delimited names."""
    flattened: dict[str, Any] = {}
    for key, value in values.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(flatten_values(value, name))
        elif isinstance(value, list):
            flattened.update(_flatten_list(value, name))
        else:
            flattened[name] = value
    return flattened


def matches_metric(name: str, filters: dict[str, Any]) -> bool:
    """Return whether a metric name passes include and exclude patterns."""
    include = filters.get("include", ["*"])
    exclude = filters.get("exclude", [])
    included = any(fnmatch.fnmatch(name, pattern) for pattern in include)
    excluded = any(fnmatch.fnmatch(name, pattern) for pattern in exclude)
    return included and not excluded


def _flatten_list(values: list[Any], prefix: str) -> dict[str, Any]:
    """Flatten indexed structured values below an existing prefix."""
    flattened: dict[str, Any] = {}
    for index, value in enumerate(values):
        name = f"{prefix}/{index}"
        if isinstance(value, dict):
            flattened.update(flatten_values(value, name))
        else:
            flattened[name] = value
    return flattened
