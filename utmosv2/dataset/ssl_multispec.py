from typing import TYPE_CHECKING

import torch

from utmosv2.dataset import MultiSpecDataset, SSLExtDataset

if TYPE_CHECKING:
    import pandas as pd


class SSLLMultiSpecExtDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, data: pd.DataFrame, phase: str, transform=None):
        self.data = data
        self.ssl = SSLExtDataset(cfg, data, phase)
        self.multi_spec = MultiSpecDataset(cfg, data, phase, transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x1, d, target = self.ssl[idx]
        x2, _ = self.multi_spec[idx]

        return x1, x2, d, target
