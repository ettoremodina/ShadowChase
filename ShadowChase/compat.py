"""Resolve module names recorded by earlier releases of this project.

A pickle stores the import path of every class it references and resolves that
path at load time. This package has been renamed twice, and one subpackage was
renamed with it:

    cops_and_robbers.storage.game_loader
    ScotlandYard.storage.game_loader
    ScotlandYard.services.game_loader
    ShadowChase.services.game_loader   (current)

Each rename silently made the games saved before it unreadable. Installing an
import alias for the historical names makes them resolve to the modules that
replaced them, which is what keeps ``saved_games/`` loadable across renames.

The same mechanism covers future moves: add the old name here rather than
leaving a corpus behind.
"""

from __future__ import annotations

import importlib
import sys
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
from types import ModuleType
from typing import Sequence


CURRENT_PACKAGE = "ShadowChase"
LEGACY_PACKAGES = ("ScotlandYard", "cops_and_robbers")
LEGACY_SUBPACKAGES = {"storage": "services"}


def current_module_name(legacy_name: str) -> str:
    """Translate one historical module path into its current equivalent."""
    parts = legacy_name.split(".")
    parts[0] = CURRENT_PACKAGE
    if len(parts) > 1:
        parts[1] = LEGACY_SUBPACKAGES.get(parts[1], parts[1])
    return ".".join(parts)


class _AliasLoader(Loader):
    """Bind a historical module name to the module that replaced it.

    The alias is a thin proxy rather than the replacement module itself. The
    import machinery stamps ``__name__`` and ``__spec__`` onto whatever
    ``create_module`` returns, so handing it the real module would rename that
    module to its own historical name. Attribute lookups fall through to the
    replacement, which is what keeps class identity intact: a pickle and live
    code must resolve to the same class object, or ``isinstance`` and enum
    member comparisons start failing.
    """

    def __init__(self, current_name: str) -> None:
        self._current_name = current_name

    def create_module(self, spec: ModuleSpec) -> ModuleType:
        target = importlib.import_module(self._current_name)
        proxy = ModuleType(spec.name)
        proxy.__dict__["__aliased_module__"] = target
        proxy.__dict__["__getattr__"] = lambda name: getattr(target, name)
        if hasattr(target, "__path__"):
            proxy.__path__ = []  # a package, so submodules keep resolving here
        return proxy

    def exec_module(self, module: ModuleType) -> None:
        """Do nothing: the replacement module is already executed."""


class _LegacyModuleFinder(MetaPathFinder):
    """Redirect imports of renamed packages to their current modules."""

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None = None,
        target: ModuleType | None = None,
    ) -> ModuleSpec | None:
        if fullname.split(".")[0] not in LEGACY_PACKAGES:
            return None
        return ModuleSpec(fullname, _AliasLoader(current_module_name(fullname)))


def install_legacy_aliases() -> None:
    """Register the historical import names once per interpreter.

    The finder goes in front of the standard path finder. Behind it, a legacy
    name whose parent already resolves to this package would be found on disk
    and executed a second time under the historical name, producing duplicate
    classes instead of aliases.
    """
    if any(isinstance(finder, _LegacyModuleFinder) for finder in sys.meta_path):
        return
    sys.meta_path.insert(0, _LegacyModuleFinder())
