from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class DatasetItem:
    file_path: Path
    dataset_name: str
    mos: int | None = None


