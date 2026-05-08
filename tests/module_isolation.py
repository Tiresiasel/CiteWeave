"""Test helpers for temporary import stubs.

Several focused tests load modules by file path while replacing heavyweight
runtime dependencies with tiny fakes.  Those replacements must never leak into
other tests through ``sys.modules``.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import ModuleType
from typing import Any


_MISSING = object()


class ModuleSandbox:
    """Temporarily replace modules and restore ``sys.modules`` exactly."""

    def __init__(self):
        self._originals: dict[str, ModuleType | object] = {}

    def _remember(self, name: str) -> None:
        if name not in self._originals:
            self._originals[name] = sys.modules.get(name, _MISSING)

    def stub(self, name: str, **attrs: Any) -> ModuleType:
        self._remember(name)
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        sys.modules[name] = module
        return module

    def set(self, name: str, module: ModuleType) -> ModuleType:
        self._remember(name)
        sys.modules[name] = module
        return module

    def load(self, path: Path, module_name: str) -> ModuleType:
        self._remember(module_name)
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    def restore(self) -> None:
        for name in reversed(list(self._originals)):
            original = self._originals[name]
            if original is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original  # type: ignore[assignment]
        self._originals.clear()

    def __enter__(self) -> "ModuleSandbox":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.restore()
