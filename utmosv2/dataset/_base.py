from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd

    from utmosv2._settings._config import Config
    from utmosv2.dataset._schema import DatasetSchema


class _BaseDataset(torch.utils.data.Dataset, abc.ABC):
    def __init__(
        self,
        cfg: Config,
        data: pd.DataFrame | list[DatasetItem] | InMemoryData,
        phase: str,
        transform: dict[str, Callable[[torch.Tensor], torch.Tensor]] | None = None,
    ) -> None:
        self.cfg = cfg
        self.data = data
        self.phase = phase
        self.transform = transform

    def _get_audio_and_mos(self, idx: int) -> tuple[np.ndarray, torch.Tensor]:
        if isinstance(self.data, InMemoryData):
            y = self.data.data[idx]
            mos = None
        else:
            row = self.data[idx] if isinstance(self.data, list) else self.data.iloc[idx]
            file = row.file_path
            assert file is not None
            y = load_audio(self.cfg, file)
            mos = row.mos

        if getattr(self.cfg.dataset, "remove_silent_section", None):
            y = remove_silent_section(y)

        return y, torch.tensor(mos or 0.0, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.data)

    @abc.abstractmethod
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        pass
