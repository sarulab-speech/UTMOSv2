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
    """
    Dataset class for SSL (Self-Supervised Learning) feature extractor.
    This class handles audio loading, extending, and random selection of a segment from the audio.

    Args:
        cfg (SimpleNamespace | ModuleType):
            The configuration object containing dataset and model settings.
        data (pd.DataFrame | list[DatasetSchema]):
            The dataset containing file paths and MOS labels.
        phase (str):
            The phase of the dataset, either "train" or any other phase (e.g., "valid").
    """

    def __getitem__(self, idx):
        """
        Get the processed audio, and target MOS for a given index.

        Args:
            idx (int): Index of the sample.
        Returns:
            tuple: A tuple containing the processed audio (torch.Tensor), and target MOS (torch.Tensor).
        """
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
    """
    Dataset class for SSL (Self-Supervised Learning) feature extractor with data-domein embedding.

    Args:
        cfg (SimpleNamespace | ModuleType):
            The configuration object containing dataset and model settings.
        data (pd.DataFrame | list[DatasetSchema]):
            The dataset containing file paths and MOS labels.
        phase (str):
            The phase of the dataset, either "train" or any other phase (e.g., "valid").
    """

    def __init__(self, cfg, data: "pd.DataFrame" | list[DatasetSchema], phase: str):
        super().__init__(cfg, data, phase)
        self.dataset_map = get_dataset_map(cfg)

    def __getitem__(self, idx):
        """
        Get the processed audio, data-domain embedding, and target MOS for a given index.

        Args:
            idx (int): Index of the sample.
        Returns:
            tuple: A tuple containing the processed audio (torch.Tensor), data-domain embedding (torch.Tensor),
            and target MOS (torch.Tensor).
        """
        y, target = super().__getitem__(idx)
        row = self.data[idx] if isinstance(self.data, list) else self.data.iloc[idx]

        d = np.zeros(len(self.dataset_map))
        d[self.dataset_map[row.dataset]] = 1
        d = torch.tensor(d, dtype=torch.float32)

        return y, d, target
