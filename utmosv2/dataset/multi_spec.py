from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import librosa
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


class MultiSpecDataset(BaseDataset):
    def __getitem__(self, idx):
        row = self.data[idx] if isinstance(self.data, list) else self.data.iloc[idx]
        file = row.file_path
        y = load_audio(self.cfg, file)
        specs = []
        length = int(self.cfg.dataset.spec_frames.frame_sec * self.cfg.sr)
        y = extend_audio(self.cfg, y, length, type=self.cfg.dataset.spec_frames.extend)
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
                spec = torch.tensor(spec, dtype=torch.float32)
                phase = "train" if self.phase == "train" else "valid"
                spec = self.transform[phase](spec)
                specs.append(spec)
        spec = torch.stack(specs).float()

        target = row.mos or 0.0
        target = torch.tensor(target, dtype=torch.float32)

        return spec, target


class MultiSpecExtDataset(MultiSpecDataset):
    def __init__(
        self,
        cfg,
        data: "pd.DataFrame" | list[DatasetSchema],
        phase: str,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        super().__init__(cfg, data, phase, transform)
        self.dataset_map = get_dataset_map(cfg)

    def __getitem__(self, idx):
        spec, target = super().__getitem__(idx)
        row = self.data[idx] if isinstance(self.data, list) else self.data.iloc[idx]

        d = np.zeros(len(self.dataset_map))
        d[self.dataset_map[row.dataset]] = 1
        d = torch.tensor(d, dtype=torch.float32)

        return spec, d, target


def _make_spctrogram(cfg, spec_cfg, y: np.ndarray) -> np.ndarray:
    if spec_cfg.mode == "melspec":
        return _make_melspec(cfg, spec_cfg, y)
    elif spec_cfg.mode == "stft":
        return _make_stft(cfg, spec_cfg, y)
    else:
        raise NotImplementedError


def _make_melspec(cfg, spec_cfg, y: np.ndarray) -> np.ndarray:
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


def _make_stft(cfg, spec_cfg, y: np.ndarray) -> np.ndarray:
    spec = librosa.stft(y=y, n_fft=spec_cfg.n_fft, hop_length=spec_cfg.hop_length)
    spec = np.abs(spec)
    spec = librosa.amplitude_to_db(spec)
    return spec
