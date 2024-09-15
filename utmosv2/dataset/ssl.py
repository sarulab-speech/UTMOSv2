from typing import TYPE_CHECKING

import numpy as np
import torch

from utmosv2.dataset._utils import (
    extend_audio,
    get_dataset_map,
    load_audio,
    select_random_start,
)

if TYPE_CHECKING:
    import pandas as pd


class SSLDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, data: pd.DataFrame, phase: str):
        self.cfg = cfg
        self.data = data
        self.phase = phase

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        file = row["file_path"]
        y = load_audio(self.cfg, file)
        length = int(self.cfg.dataset.ssl.duration * self.cfg.sr)
        y = extend_audio(self.cfg, y, length, type="tile")
        y = select_random_start(y, length)

        target = row["mos"]
        target = torch.tensor(target, dtype=torch.float32)

        return y, target


class SSLExtDataset(SSLDataset):
    def __init__(self, cfg, data: pd.DataFrame, phase: str):
        super().__init__(cfg, data, phase)
        self.dataset_map = get_dataset_map(cfg)

    def __getitem__(self, idx):
        y, target = super().__getitem__(idx)

        d = np.zeros(len(self.dataset_map))
        d[self.dataset_map[self.data.iloc[idx]["dataset"]]] = 1
        d = torch.tensor(d, dtype=torch.float32)

        return y, d, target
