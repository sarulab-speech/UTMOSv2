from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from utmosv2.dataset._base import BaseDataset
from utmosv2.dataset._utils import (
    extend_audio,
    get_dataset_map,
    load_audio,
    select_random_start,
)

if TYPE_CHECKING:
    import pandas as pd

    from utmosv2.dataset._schema import DatasetSchema


class SSLDataset(BaseDataset):
    def __getitem__(self, idx):
        row = self.data[idx] if isinstance(self.data, list) else self.data.iloc[idx]
        file = row.file_path
        y = load_audio(self.cfg, file)
        length = int(self.cfg.dataset.ssl.duration * self.cfg.sr)
        y = extend_audio(self.cfg, y, length, type="tile")
        y = select_random_start(y, length)

        target = row.mos or 0.0
        target = torch.tensor(target, dtype=torch.float32)

        return y, target


class SSLExtDataset(SSLDataset):
    def __init__(self, cfg, data: "pd.DataFrame" | list[DatasetSchema], phase: str):
        super().__init__(cfg, data, phase)
        self.dataset_map = get_dataset_map(cfg)

    def __getitem__(self, idx):
        y, target = super().__getitem__(idx)
        row = self.data[idx] if isinstance(self.data, list) else self.data.iloc[idx]

        d = np.zeros(len(self.dataset_map))
        d[self.dataset_map[row.dataset]] = 1
        d = torch.tensor(d, dtype=torch.float32)

        return y, d, target
