from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

from utmosv2._settings._config import Config
from utmosv2.dataset import MultiSpecDataset, SSLExtDataset
from utmosv2.dataset._base import BaseDataset

if TYPE_CHECKING:
    import pandas as pd

    from utmosv2.dataset._schema import DatasetSchema


class SSLLMultiSpecExtDataset(BaseDataset):
    """
    Dataset class that combines both SSL (Self-Supervised Learning) and Multi-Spectrogram datasets.
    This dataset uses both SSLExtDataset and MultiSpecDataset to provide different representations
    of the same audio sample.

    Args:
        cfg (SimpleNamespace | ModuleType):
            The configuration object containing dataset and model settings.
        data (pd.DataFrame | list[DatasetSchema]):
            The dataset containing file paths and MOS labels.
        phase (str):
            The phase of the dataset, either "train" or any other phase (e.g., "valid").
        transform (dict[str, Callable[[torch.Tensor], torch.Tensor]] | None):
            Transformation function to apply to spectrograms.
    """

    def __init__(
        self,
        cfg: Config,
        data: "pd.DataFrame" | list[DatasetSchema],
        phase: str,
        transform: dict[str, Callable[[torch.Tensor], torch.Tensor]] | None = None,
    ):
        super().__init__(cfg, data, phase, transform)
        self.ssl = SSLExtDataset(cfg, data, phase)
        self.multi_spec = MultiSpecDataset(cfg, data, phase, transform)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        """
        Get data for SSL feature extractor, mel-spectrogram feature extractor, data-domain embedding, and target MOS for a given index.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: data for SSL feature extractor (torch.Tensor), data for mel-spectrogram feature extractor (torch.Tensor),
            data-domain id (torch.Tensor), and target MOS (torch.Tensor).
        """
        x1, d, target = self.ssl[idx]
        x2, _ = self.multi_spec[idx]

        return x1, x2, d, target
