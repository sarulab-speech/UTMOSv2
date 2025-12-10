from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class DatasetItem:
    file_path: Path
    dataset_name: str
    mos: int | None = None


@dataclass(frozen=True)
class InMemoryData:
    # TODO(kAIto47802): Replace np.ndarray with torch.Tensor to eliminate
    # unnecessary CPU–GPU data transfers. This will require refactoring the
    # preprocessing pipeline, which currently relies on NumPy arrays.
    data: np.ndarray
    dataset_name: str

    def __post_init__(self) -> None:
        assert (
            self.data.ndim <= 2
        ), "InMemoryDataset only supports 1D or 2D data arrays."
        if self.data.ndim == 1:
            object.__setattr__(self, "data", self.data[np.newaxis, :])

    def __len__(self) -> int:
        return self.data.shape[0]
