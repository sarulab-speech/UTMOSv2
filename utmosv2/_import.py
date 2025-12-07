from __future__ import annotations

import importlib
import types
from typing import Any


class _LazyImport(types.ModuleType):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._name = name
        self._module: types.ModuleType | None = None

    def __getattr__(self, name: str) -> Any:
        if self._module is None:
            self._module = importlib.import_module(self._name)
            self.__dict__.update(self._module.__dict__)
        return getattr(self._module, name)
