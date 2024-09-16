import abc
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm

from utmosv2.dataset._schema import DatasetSchema
from utmosv2.utils import get_dataset

if TYPE_CHECKING:
    import torch.nn as nn


class UTMOSv2ModelMixin(abc.ABC):
    @property
    @abc.abstractmethod
    def _cfg(self) -> SimpleNamespace:
        pass

    @abc.abstractmethod
    def eval(self) -> "nn.Module":
        pass

    @abc.abstractmethod
    def __call__(self, *args, **kwargs) -> torch.Tensor:
        pass

    def predict(
        self,
        *,
        input_path: Path | str | None = None,
        input_dir: Path | str | None = None,
        val_list: list[str] | None = None,
        val_list_path: Path | str | None = None,
        predict_dataset: str = "sarulab",
        device: str | torch.device = "cuda:0",
        num_workers: int = 4,
        batch_size: int = 16,
        num_repetitions: int = 1,
        verbose: bool = True,
    ) -> float | list[dict[str, str | float]]:
        if not ((input_path is None) ^ (input_dir is None)):
            raise ValueError(
                "Either `input_path` or `input_dir` must be provided, but not both."
            )
        data = self._prepare_data(
            input_path,
            input_dir,
            val_list,
            val_list_path,
            predict_dataset,
        )
        dataset = get_dataset(self._cfg, data, self._cfg.phase)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        pred = self._predict_impl(dataloader, num_repetitions, device, verbose)

        if input_path is not None:
            return float(pred[0])
        else:
            return [
                {"file_path": d.file_path.as_posix(), "predicted_mos": float(p)}
                for d, p in zip(data, pred)
            ]

    def _prepare_data(
        self,
        input_path: Path | str | None,
        input_dir: Path | str | None,
        val_list: list[str] | None,
        val_list_path: Path | str | None,
        predict_dataset: str,
    ) -> list[DatasetSchema]:
        assert (input_path is None) ^ (input_dir is None)
        if isinstance(input_path, str):
            input_path = Path(input_path)
        if isinstance(input_dir, str):
            input_dir = Path(input_dir)
        if input_path is not None and not input_path.exists():
            raise FileNotFoundError(f"File not found: {input_path}")
        if input_dir is not None and not input_dir.exists():
            raise FileNotFoundError(f"Directory not found: {input_dir}")
        if val_list_path is not None:
            if val_list:
                warnings.warn(
                    "Both `val_list` and `val_list_path` are provided. "
                    "The union of the two will be used."
                )
            if val_list is None:
                val_list = []
            if isinstance(val_list_path, str):
                val_list_path = Path(val_list_path)
            if not val_list_path.exists():
                raise FileNotFoundError(f"File not found: {val_list_path}")
            with open(val_list_path, "r") as f:
                val_list.extend(f.read().splitlines())
        res: list[DatasetSchema]
        if input_path is not None:
            res = [
                DatasetSchema(
                    file_path=input_path,
                    dataset=predict_dataset,
                )
            ]
        if input_dir is not None:
            res = [
                DatasetSchema(
                    file_path=p,
                    dataset=predict_dataset,
                )
                for p in sorted(input_dir.glob("*.wav"))
            ]
            if not res:
                raise ValueError(f"No wav files found in {input_dir}")
        if val_list is not None:
            val_list = [d.replace(".wav", "") for d in val_list]
            res = [
                d
                for d in res
                if d.file_path.as_posix().split("/")[-1].replace(".wav", "")
                in set(val_list)
            ]
        if not res:
            raise ValueError(
                f"None of the data were found in the validation list: {val_list_path}"
            )
        return res

    def _predict_impl(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_repetitions: int,
        device: str | torch.device,
        verbose: bool,
    ) -> np.ndarray:
        self.eval().to(device)
        res = 0.0
        for i in range(num_repetitions):
            pred = []
            pbar = tqdm(
                dataloader,
                disable=not verbose,
                total=len(dataloader),
                desc=(
                    f"Predicting [{i + 1}/{num_repetitions}]: "
                    if num_repetitions > 1
                    else "Predicting: "
                ),
            )
            with torch.no_grad():
                for t in pbar:
                    x = t[:-1]
                    x = [t.to(device, non_blocking=True) for t in x]
                    with autocast():
                        output = self.__call__(*x).squeeze(1)
                    pred.append(output.cpu().numpy())
            res += np.concatenate(pred) / num_repetitions
        assert isinstance(res, np.ndarray)
        return res
