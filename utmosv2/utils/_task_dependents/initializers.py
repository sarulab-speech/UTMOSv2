from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import scipy.stats
import torch
import torch.nn as nn

from utmosv2._import import _LazyImport
from utmosv2.dataset import (
    MultiSpecDataset,
    MultiSpecExtDataset,
    SSLDataset,
    SSLExtDataset,
    SSLLMultiSpecExtDataset,
)
from utmosv2.model import (
    MultiSpecExtModel,
    MultiSpecModelV2,
    SSLExtModel,
    SSLMultiSpecExtModelV1,
    SSLMultiSpecExtModelV2,
)
from utmosv2.preprocess import add_sys_mean, preprocess, preprocess_test
from utmosv2.utils._constants import _UTMOSV2_CHACHE
from utmosv2.utils._download import download_pretrained_weights_from_hf

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd

    from utmosv2._settings._config import Config
    from utmosv2.dataset._schema import DatasetSchema

else:
    pd = _LazyImport("pandas")


def get_data(cfg: Config) -> pd.DataFrame:
    train_mos_list = pd.read_csv(cfg.input_dir / "sets/train_mos_list.txt", header=None)
    val_mos_list = pd.read_csv(cfg.input_dir / "sets/val_mos_list.txt", header=None)
    test_mos_list = pd.read_csv(cfg.input_dir / "sets/test_mos_list.txt", header=None)
    data = pd.concat([train_mos_list, val_mos_list, test_mos_list], axis=0)
    data.columns = ["utt_id", "mos"]
    data["file_path"] = data["utt_id"].apply(lambda x: cfg.input_dir / f"wav/{x}")
    return data


def get_dataset(
    cfg: Config, data: pd.DataFrame | list[DatasetSchema], phase: str
) -> torch.utils.data.Dataset:
    if cfg.print_config:
        print(f"Using dataset: {cfg.dataset.name}")
    res: torch.utils.data.Dataset
    if cfg.dataset.name == "multi_spec":
        res = MultiSpecDataset(cfg, data, phase, cfg.transform)
    elif cfg.dataset.name == "ssl":
        res = SSLDataset(cfg, data, phase)
    elif cfg.dataset.name == "sslext":
        res = SSLExtDataset(cfg, data, phase)
    elif cfg.dataset.name == "ssl_multispec_ext":
        res = SSLLMultiSpecExtDataset(cfg, data, phase, cfg.transform)
    elif cfg.dataset.name == "multi_spec_ext":
        res = MultiSpecExtDataset(cfg, data, phase, cfg.transform)
    else:
        raise NotImplementedError
    return res


def get_model(cfg: Config, device: torch.device) -> nn.Module:
    if cfg.print_config:
        print(f"Using model: {cfg.model.name}")
    model: nn.Module
    if cfg.model.name == "multi_specv2":
        model = MultiSpecModelV2(cfg)
    elif cfg.model.name == "sslext":
        model = SSLExtModel(cfg)
    elif cfg.model.name == "multi_spec_ext":
        model = MultiSpecExtModel(cfg)
    elif cfg.model.name == "ssl_multispec_ext":
        model = SSLMultiSpecExtModelV1(cfg)
    elif cfg.model.name == "ssl_multispec_ext_v2":
        model = SSLMultiSpecExtModelV2(cfg)
    else:
        raise NotImplementedError
    model = model.to(device)
    if cfg.weight is not None:
        if cfg.weight.endswith(".pth"):
            weight_path = cfg.weight
        else:
            weight_path = (
                Path("models")
                / cfg.weight
                / f"fold{cfg.now_fold}_s{cfg.split.seed}_best_model.pth"
            )
            if not weight_path.exists():
                weight_path_cache = _UTMOSV2_CHACHE / weight_path
                if not weight_path_cache.exists():
                    download_pretrained_weights_from_hf(cfg.weight, cfg.now_fold)
                weight_path = weight_path_cache
        model.load_state_dict(torch.load(weight_path))
        print(f"Loaded weight from {weight_path}")
    return model


def get_metrics() -> dict[str, Callable[[np.ndarray, np.ndarray], float]]:
    return {
        "mse": lambda x, y: np.mean((x - y) ** 2),
        "lcc": lambda x, y: np.corrcoef(x, y)[0][1],
        "srcc": lambda x, y: scipy.stats.spearmanr(x, y)[0],
        "ktau": lambda x, y: scipy.stats.kendalltau(x, y)[0],
    }


def _get_testdata(cfg: Config, data: pd.DataFrame) -> pd.DataFrame:
    with open(cfg.inference.val_list_path, "r") as f:
        val_lists = [s.replace("\n", "") + ".wav" for s in f.readlines()]
    test_data = data[data["utt_id"].isin(set(val_lists))]
    return test_data


def get_inference_data(cfg: Config) -> pd.DataFrame:
    if cfg.reproduce:
        data = get_data(cfg)
        data = preprocess_test(cfg, data)
        data = _get_testdata(cfg, data)
    else:
        if cfg.input_dir:
            files = sorted(glob.glob(str(cfg.input_dir / "*.wav")))
            data = pd.DataFrame({"file_path": files})
        else:
            data = pd.DataFrame({"file_path": [cfg.input_path.as_posix()]})
        data["utt_id"] = data["file_path"].apply(
            lambda x: x.split("/")[-1].replace(".wav", "")
        )
        data["file_path"] = data["file_path"].apply(lambda x: Path(x))
        data["sys_id"] = data["utt_id"].apply(lambda x: x.split("-")[0])
        if cfg.inference.val_list_path:
            with open(cfg.inference.val_list_path, "r") as f:
                val_lists = [s.replace(".wav", "") for s in f.read().splitlines()]
                print(val_lists)
            data = data[data["utt_id"].isin(set(val_lists))]
        data["dataset"] = cfg.predict_dataset
        data["mos"] = 0
    return data


def get_train_data(cfg: Config) -> pd.DataFrame:
    if cfg.reproduce:
        data = get_data(cfg)
        data = preprocess(cfg, data)
    else:
        with open(cfg.data_config, "r") as f:
            datasets = json.load(f)
        data = []
        for dt in datasets["data"]:
            files = sorted(glob.glob(str(Path(dt["dir"]) / "*.wav")))
            d = pd.DataFrame({"file_path": files})
            d["dataset"] = dt["name"]
            d["utt_id"] = d["file_path"].apply(
                lambda x: x.split("/")[-1].replace(".wav", "")
            )
            d["file_path"] = d["file_path"].apply(lambda x: Path(x))
            mos_list = pd.read_csv(dt["mos_list"], header=None)
            mos_list.columns = ["utt_id", "mos"]
            mos_list["utt_id"] = mos_list["utt_id"].apply(
                lambda x: x.replace(".wav", "")
            )
            d = d.merge(mos_list, on="utt_id", how="inner")
            d["sys_id"] = d["utt_id"].apply(lambda x: x.split("-")[0])
            add_sys_mean(d)
            data.append(d)
        data = pd.concat(data, axis=0)

    return data


def _get_test_save_name(cfg: Config) -> str:
    return f"{cfg.config_name}_[fold{cfg.inference.fold}_tta{cfg.inference.num_tta}_s{cfg.split.seed}]"
