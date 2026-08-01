"""Web interface for Shadow Chase.

A local FastAPI server renders the board as SVG in the browser. The game engine
in ``ShadowChase.core`` is used unchanged: this package only translates between
engine calls and JSON.

Run it with ``python -m webui``.
"""

__all__ = ["server"]
