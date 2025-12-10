from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from utmosv2.dataset._base import BaseDataset, DataDomainMixin
from utmosv2.dataset._utils import (
    extend_audio,
    select_random_start,
)

if TYPE_CHECKING:
    import pandas as pd

    from utmosv2._settings._config import Config
    from utmosv2.dataset._schema import DatasetItem, InMemoryData


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
        transform (dict[str, Callable[[torch.Tensor], torch.Tensor]] | None):
            Transformation function to apply to spectrograms.
    """

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        """
        Get the processed audio, and target MOS for a given index.

        Args:
            idx (int): Index of the sample.
        Returns:
            tuple: A tuple containing the processed audio (torch.Tensor), and target MOS (torch.Tensor).
        """
        y, target = self._get_audio(idx)
        length = int(self.cfg.dataset.ssl.duration * self.cfg.sr)
        y = extend_audio(y, length, method="tile")
        y = select_random_start(y, length)

        return torch.from_numpy(y), target


class SSLExtDataset(SSLDataset, DataDomainMixin):
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

    def __init__(
        self,
        cfg: Config,
        data: pd.DataFrame | list[DatasetItem] | InMemoryData,
        phase: str,
    ) -> None:
        SSLDataset.__init__(self, cfg, data, phase)
        DataDomainMixin.__init__(self, cfg)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        """
        Get the processed audio, data-domain embedding, and target MOS for a given index.

        Args:
            idx (int): Index of the sample.
        Returns:
            tuple: A tuple containing the processed audio (torch.Tensor), data-domain embedding (torch.Tensor),
            and target MOS (torch.Tensor).
        """
        y, target = super().__getitem__(idx)
        dt = self._get_data_domain_embedding(idx)

        return y, dt, target
