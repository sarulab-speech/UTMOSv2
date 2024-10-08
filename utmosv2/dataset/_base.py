from __future__ import annotations

import abc
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import pandas as pd

    from utmosv2.dataset._schema import DatasetSchema


class BaseDataset(torch.utils.data.Dataset, abc.ABC):
    def __init__(
        self,
        cfg,
        data: "pd.DataFrame" | list[DatasetSchema],
        phase: str,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        self.cfg = cfg
        self.data = data
        self.phase = phase
        self.transform = transform

    def __len__(self):
        return len(self.data)

    @abc.abstractmethod
    def __getitem__(self, idx):
        pass
