from __future__ import annotations

from pathlib import Path

from .base import LanguageAdapter
from .jinja_dbt import JinjaDbtAdapter
from .python import PythonAdapter


ADAPTERS: dict[str, type[LanguageAdapter]] = {
    "jinja-dbt": JinjaDbtAdapter,
    "python": PythonAdapter,
}


def detect_language(path: str | Path, source: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix == ".py":
        return "python"
    if suffix in {".jinja", ".jinja2", ".j2"}:
        return "jinja-dbt"
    if suffix == ".sql" and ("{%" in source or "{{" in source or "{#" in source):
        return "jinja-dbt"
    raise ValueError(
        f"Cannot detect a supported language for {path}. "
        "Pass --language python or --language jinja-dbt."
    )


def make_adapter(language: str) -> LanguageAdapter:
    try:
        adapter_type = ADAPTERS[language]
    except KeyError as exc:
        supported = ", ".join(sorted(ADAPTERS))
        raise ValueError(f"Unsupported language {language!r}. Supported: {supported}") from exc
    return adapter_type()


__all__ = ["detect_language", "make_adapter"]

