from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class DatasetSchema:
    file_path: Path
    dataset: str
    mos: int | None = None
