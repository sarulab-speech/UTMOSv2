from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

from utmosv2.dataset import MultiSpecDataset, SSLExtDataset
from utmosv2.dataset._base import BaseDataset

if TYPE_CHECKING:
    import pandas as pd

    from utmosv2.dataset._schema import DatasetSchema


class SSLLMultiSpecExtDataset(BaseDataset):
    def __init__(
        self,
        cfg,
        data: "pd.DataFrame" | list[DatasetSchema],
        phase: str,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        super().__init__(cfg, data, phase, transform)
        self.ssl = SSLExtDataset(cfg, data, phase)
        self.multi_spec = MultiSpecDataset(cfg, data, phase, transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x1, d, target = self.ssl[idx]
        x2, _ = self.multi_spec[idx]

        return x1, x2, d, target
