from __future__ import annotations

import abc
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

from utmosv2._settings._config import Config

if TYPE_CHECKING:
    import pandas as pd

    from utmosv2.dataset._schema import DatasetSchema


class BaseDataset(torch.utils.data.Dataset, abc.ABC):
    def __init__(
        self,
        cfg: Config,
        data: "pd.DataFrame" | list[DatasetSchema],
        phase: str,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        self.cfg = cfg
        self.data = data
        self.phase = phase
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    @abc.abstractmethod
    def __getitem__(self, idx: int) -> tuple[torch.Tensor,...]:
        pass
