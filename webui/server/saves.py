"""Finding and opening saved games.

The Tkinter build used a native file dialog rooted at ``saved_games/``. The
browser has no such dialog, so the server does the walking. The tree holds
thousands of pickles, so listing reads directory entries only — a game is
unpickled when the player actually opens it.
"""
from __future__ import annotations

import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from ShadowChase.core.game import ShadowChaseGame
from ShadowChase.services.game_loader import GameLoader

SAVED_GAMES_ROOT = Path("saved_games")

# Games are also filed under by_date/by_graph_type/by_outcome shortcuts; walking
# those would list the same game several times.
SKIP_DIRECTORIES = {"by_date", "by_graph_type", "by_outcome", "exports", "statistics"}


@dataclass
class SaveEntry:
    game_id: str
    path: str
    folder: str
    size: int
    modified: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "gameId": self.game_id,
            "path": self.path,
            "folder": self.folder,
            "size": self.size,
            "modified": self.modified,
            "modifiedLabel": time.strftime(
                "%Y-%m-%d %H:%M", time.localtime(self.modified)
            ),
        }


def list_saves(limit: int = 400, query: str = "") -> List[Dict[str, object]]:
    """Most recently written games first, newest-modified wins."""
    root = SAVED_GAMES_ROOT
    if not root.exists():
        return []

    needle = query.strip().lower()
    entries: List[SaveEntry] = []

    for current_root, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in SKIP_DIRECTORIES]
        folder = os.path.relpath(current_root, root).replace("\\", "/")
        for filename in filenames:
            if not filename.endswith(".pkl"):
                continue
            game_id = filename[:-4]
            # Games are filed by matchup, so the folder is as searchable as the
            # id: "heuristic_vs_random" is what someone actually looks for.
            if needle and needle not in filename.lower() and needle not in folder.lower():
                continue
            full_path = os.path.join(current_root, filename)
            try:
                stat = os.stat(full_path)
            except OSError:
                continue
            entries.append(
                SaveEntry(
                    game_id=game_id,
                    path=full_path.replace("\\", "/"),
                    folder="" if folder == "." else folder,
                    size=stat.st_size,
                    modified=stat.st_mtime,
                )
            )

    entries.sort(key=lambda entry: entry.modified, reverse=True)
    return [entry.to_dict() for entry in entries[:limit]]


def resolve_save_path(path: str) -> Path:
    """Reject anything outside ``saved_games/`` before touching the disk."""
    candidate = Path(path)
    root = SAVED_GAMES_ROOT.resolve()
    resolved = (candidate if candidate.is_absolute() else Path.cwd() / candidate).resolve()
    if root not in resolved.parents and resolved != root:
        raise ValueError("Saved games can only be opened from the saved_games folder.")
    if not resolved.exists():
        raise ValueError(f"No file at {path}")
    return resolved


def load_game_file(path: str, loader: Optional[GameLoader] = None):
    """Load a saved game, accepting both bare games and GameRecord wrappers."""
    resolved = resolve_save_path(path)
    loader = loader or GameLoader()

    with open(resolved, "rb") as handle:
        payload = pickle.load(handle)

    if isinstance(payload, ShadowChaseGame):
        return payload, resolved.stem

    if hasattr(payload, "game_history") and hasattr(payload, "game_config"):
        game = loader._reconstruct_game_from_record(payload)
        if game is None:
            raise ValueError("That record could not be rebuilt into a game.")
        game_id = getattr(payload, "game_id", resolved.stem)
        return game, game_id

    raise ValueError(f"Unrecognized save format: {type(payload).__name__}")
