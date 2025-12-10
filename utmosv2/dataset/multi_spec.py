from __future__ import annotations

from typing import TYPE_CHECKING

import librosa
import numpy as np
import torch

from utmosv2.dataset._base import BaseDataset, DataDomainMixin
from utmosv2.dataset._utils import (
    extend_audio,
    select_random_start,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd

    from utmosv2._settings._config import Config
    from utmosv2.dataset._schema import DatasetItem, InMemoryData


class MultiSpecDataset(BaseDataset):
    """
    Dataset class for mel-spectrogram feature extractor. This class is responsible for
    loading audio data, generating multiple spectrograms for each sample, and
    applying the necessary transformations.

    Args:
        cfg (SimpleNamespace): The configuration object containing dataset and model settings.
        data (list[DatasetSchema] | pd.DataFrame): The dataset containing file paths and labels.
        phase (str): The phase of the dataset, either "train" or any other phase (e.g., "valid").
        transform (str, dict[Callable[[torch.Tensor], torch.Tensor]] | None): Transformation function to apply to spectrograms.
    """

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        """
        Get the spectrogram and target MOS for a given index.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: The spectrogram (torch.Tensor) and target MOS (torch.Tensor) for the sample.
        """
        y, target = self._get_audio(idx)
        specs = []
        length = int(self.cfg.dataset.spec_frames.frame_sec * self.cfg.sr)
        y = extend_audio(y, length, method=self.cfg.dataset.spec_frames.extend)
        for _ in range(self.cfg.dataset.spec_frames.num_frames):
            y1 = select_random_start(y, length)
            for spec_cfg in self.cfg.dataset.specs:
                spec = _make_spctrogram(self.cfg, spec_cfg, y1)
                if self.cfg.dataset.spec_frames.mixup_inner:
                    y2 = select_random_start(y, length)
                    spec2 = _make_spctrogram(self.cfg, spec_cfg, y2)
                    lmd = np.random.beta(
                        self.cfg.dataset.spec_frames.mixup_alpha,
                        self.cfg.dataset.spec_frames.mixup_alpha,
                    )
                    spec = lmd * spec + (1 - lmd) * spec2
                spec = np.stack([spec, spec, spec], axis=0)
                # spec = np.transpose(spec, (1, 2, 0))
                spec_tensor = torch.tensor(spec, dtype=torch.float32)
                phase = "train" if self.phase == "train" else "valid"
                assert self.transform is not None, "Transform must be provided."
                spec_tensor = self.transform[phase](spec_tensor)
                specs.append(spec_tensor)
        spec_tensor = torch.stack(specs).float()

        return spec_tensor, target


class MultiSpecExtDataset(MultiSpecDataset, DataDomainMixin):
    """
    Dataset class for mel-spectrogram feature extractor with data-domain embedding.

    Args:
        cfg (SimpleNamespace | ModuleType):
            The configuration object containing dataset and model settings.
        data (pd.DataFrame | list[DatasetSchema]):
            The dataset containing file paths and labels.
        phase (str):
            The phase of the dataset, either "train" or any other phase (e.g., "valid").
        transform (dict[str, Callable[[torch.Tensor], torch.Tensor]] | None):
            Transformation function to apply to spectrograms.
    """

    def __init__(
        self,
        cfg: Config,
        data: pd.DataFrame | list[DatasetItem] | InMemoryData,
        phase: str,
        transform: dict[str, Callable[[torch.Tensor], torch.Tensor]] | None = None,
    ) -> None:
        MultiSpecDataset.__init__(self, cfg, data, phase, transform)
        DataDomainMixin.__init__(self, cfg)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        """
        Get the spectrogram, data-domain embedding, and target MOS for a given index.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: A tuple containing the generated spectrogram (torch.Tensor), data-domain embedding (torch.Tensor),
            and target MOS (torch.Tensor).
        """
        spec, target = super().__getitem__(idx)
        dt = self._get_data_domain_embedding(idx)

        return spec, dt, target


def _make_spctrogram(cfg: Config, spec_cfg: Config, y: np.ndarray) -> np.ndarray:
    return {
        "melspec": _make_melspec,
        "stft": _make_stft,
    }[spec_cfg.mode](cfg, spec_cfg, y)


def _make_melspec(cfg: Config, spec_cfg: Config, y: np.ndarray) -> np.ndarray:
    spec = librosa.feature.melspectrogram(
        y=y,
        sr=cfg.sr,
        n_fft=spec_cfg.n_fft,
        hop_length=spec_cfg.hop_length,
        n_mels=spec_cfg.n_mels,
        win_length=spec_cfg.win_length,
    )
    spec = librosa.power_to_db(spec, ref=np.max)
    if spec_cfg.norm is not None:
        spec = (spec + spec_cfg.norm) / spec_cfg.norm
    return spec


def _make_stft(cfg: Config, spec_cfg: Config, y: np.ndarray) -> np.ndarray:
    spec = librosa.stft(y=y, n_fft=spec_cfg.n_fft, hop_length=spec_cfg.hop_length)
    spec = np.abs(spec)
    spec = librosa.amplitude_to_db(spec)
    return spec
