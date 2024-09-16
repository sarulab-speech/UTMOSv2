import abc
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import pandas as pd


class BaseDataset(torch.utils.data.Dataset, abc.ABC):
    def __init__(
        self,
        cfg,
        data: "pd.DataFrame",
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
