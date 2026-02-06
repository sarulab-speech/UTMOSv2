from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import librosa
import numpy as np
from tqdm import tqdm

from utmosv2._import import _LazyImport

if TYPE_CHECKING:
    import pandas as pd

    from utmosv2._settings._config import Config
else:
    pd = _LazyImport("pandas")


def remove_silent_section(audio: np.ndarray, min_length: int = 4800) -> np.ndarray:
    mask = audio < 0.1
    mask = np.pad(mask, (1, 0)) ^ np.pad(mask, (0, 1))
    indices = np.where(mask)[0]
    length = indices[1::2] - indices[::2]
    indices_mask = np.repeat(length > min_length, 2)
    indices = indices[indices_mask]
    mask2 = np.zeros(audio.shape[0] + 1, dtype=int)
    mask2[indices] = np.where(np.arange(indices.shape[0]) % 2, -1, 1)
    mask2 = np.cumsum(mask2).astype(bool)[:-1]
    return audio[~mask2]


def _clip_audio(cfg: Config, data: pd.DataFrame, data_name: str = "bvcc") -> None:
    (cfg.preprocess.save_path / data_name).mkdir(parents=True, exist_ok=True)
    for file in tqdm(data["file_path"].values, desc="Clipping audio files"):
        y, _ = librosa.load(file, sr=None)
        y, _ = librosa.effects.trim(y, top_db=cfg.preprocess.top_db)
        np.save(
            cfg.preprocess.save_path
            / data_name
            / file.as_posix().split("/")[-1].replace(".wav", ".npy"),
            y,
        )


def _select_audio(
    cfg: Config, data: pd.DataFrame, data_name: str = "bvcc"
) -> pd.DataFrame:
    if cfg.preprocess.min_seconds is None:
        return data
    select_file_name = f"min_seconds={cfg.preprocess.min_seconds}.txt"
    if select_file_name in os.listdir(cfg.preprocess.save_path / data_name):
        with open(
            cfg.preprocess.save_path / data_name / select_file_name,
            "r",
        ) as f:
            select = f.read().split("\n")
    else:
        select = []
        for file in tqdm(data["file_path"].values, desc="Selecting audio files"):
            y = np.load(file)
            if y.shape[0] >= cfg.preprocess.min_seconds * cfg.sr:
                select.append(file)
        with open(
            cfg.preprocess.save_path / data_name / select_file_name,
            "w",
        ) as f:
            f.write("\n".join(select))
    _change_file_path(cfg, data)
    data = data[data["file_path"].isin(set(select))]
    return data


def _clip_and_select_audio(
    cfg: Config, data: pd.DataFrame, data_name: str = "bvcc"
) -> pd.DataFrame:
    if not (cfg.preprocess.save_path / data_name).exists():
        _clip_audio(cfg, data)
    _change_file_path(cfg, data)
    data = _select_audio(cfg, data)
    print(f"{len(data)} files are selected.")
    return data


def _change_file_path(cfg: Config, data: pd.DataFrame, data_name: str = "bvcc") -> None:
    data.loc[:, "file_path"] = data.loc[:, "file_path"].apply(
        lambda x: (
            cfg.preprocess.save_path
            / data_name
            / x.as_posix().split("/")[-1].replace(".wav", ".npy")
        )
    )


def _add_metadata(cfg: Config, data: pd.DataFrame) -> None:
    metadata = []
    for t in ["TRAINSET", "DEVSET", "TESTSET"]:
        meta = pd.read_csv(cfg.input_dir / f"sets/{t}")
        meta.columns = ["sys_id", "utt_id", "rating", "ignore", "listener_info"]
        meta = meta.groupby("utt_id", as_index=False).first()[["utt_id", "sys_id"]]
        metadata.append(meta)
    metadata = pd.concat(metadata, axis=0)
    dt = pd.merge(data, metadata, on="utt_id", how="left")
    data["sys_id"] = dt["sys_id"]


def add_sys_mean(data: pd.DataFrame) -> None:
    sys_mean = (
        data.groupby("sys_id", as_index=False)["mos"].mean().reset_index(drop=True)
    )
    sys_mean.columns = ["sys_id", "sys_mos"]
    dt = pd.merge(data, sys_mean, on="sys_id", how="left")
    data["sys_mos"] = dt["sys_mos"]


def preprocess(cfg: Config, data: pd.DataFrame) -> pd.DataFrame:
    data = _clip_and_select_audio(cfg, data)
    _add_metadata(cfg, data)
    add_sys_mean(data)
    data["dataset"] = "bvcc"
    if cfg.external_data:
        exdata = _get_external_data(cfg, data)
        add_sys_mean(exdata)
        for col in data.columns:
            if col not in exdata.columns:
                exdata[col] = None
        data = pd.concat([data, exdata], axis=0)
        print("Using dataset:", data["dataset"].unique())
    if not cfg.use_bvcc:
        data = data[data["dataset"] != "bvcc"]
    return data


def preprocess_test(cfg: Config, data: pd.DataFrame) -> pd.DataFrame:
    _change_file_path(cfg, data)
    _add_metadata(cfg, data)
    add_sys_mean(data)
    data["dataset"] = cfg.predict_dataset
    return data


def _get_external_data(cfg: Config, data: pd.DataFrame) -> pd.DataFrame:
    exdata = []
    if cfg.external_data == "all" or "sarulab" in cfg.external_data:
        ysdata = pd.read_csv(
            "data2/sarulab/VMC2024_MOS.csv", header=None, names=["utt_id", "mos"]
        )
        ysdata["mos"] = ysdata["mos"].astype(float)
        ysdata["sys_id"] = ysdata["utt_id"].apply(
            lambda x: "sarulab-" + x.split("-")[0]
        )
        ysdata["file_path"] = ysdata["utt_id"].apply(
            lambda x: cfg.preprocess.save_path / "bvcc" / x.replace(".wav", ".npy")
        )
        ysdata["dataset"] = "sarulab"
        exdata.append(ysdata)

    for name in ["blizzard2008", "blizzard2009", "blizzard2011"]:
        if cfg.external_data != "all" and name not in cfg.external_data:
            continue
        bzdata = pd.read_csv(
            f"data2/{name}/{name}_mos.csv",
            header=None,
            names=["utt_id", "mos"],
        )
        bzdata["mos"] = bzdata["mos"].astype(float)
        bzdata["sys_id"] = bzdata["utt_id"].apply(
            lambda x: f"{name}-" + x.split("_")[0]
        )
        bzdata["file_path"] = bzdata["utt_id"].apply(
            lambda x: Path(f"data2/{name}/{name}_wavs") / x
        )
        bzdata["dataset"] = name
        exdata.append(bzdata)

    for a in ["EH1", "EH2", "ES1", "ES3"]:
        if cfg.external_data != "all" and f"blizzard2010-{a}" not in cfg.external_data:
            continue
        d = pd.read_csv(
            f"data2/blizzard2010/blizzard2010_mos_{a}.csv",
            header=None,
            names=["utt_id", "mos"],
        )
        d["mos"] = d["mos"].astype(float)
        d["sys_id"] = d["utt_id"].apply(
            lambda x: f"blizzard2010-{a}-" + x.split("_")[0]
        )
        d["file_path"] = d["utt_id"].apply(
            lambda x: Path(f"data2/blizzard2010/blizzard2010_wavs_{a}") / x
        )
        d["dataset"] = f"blizzard2010-{a}"
        exdata.append(d)

    if cfg.external_data == "all" or "somos" in cfg.external_data:
        train_mos_list = pd.read_csv(
            "data2/somos/training_files/split1/clean/train_mos_list.txt",
        )
        val_mos_list = pd.read_csv(
            "data2/somos/training_files/split1/clean/valid_mos_list.txt",
        )
        test_mos_list = pd.read_csv(
            "data2/somos/training_files/split1/clean/test_mos_list.txt",
        )
        somosdata = pd.concat([train_mos_list, val_mos_list, test_mos_list], axis=0)
        somosdata.columns = ["utt_id", "mos"]
        somosdata["mos"] = somosdata["mos"].astype(float)
        somosdata["sys_id"] = somosdata["utt_id"].apply(
            lambda x: "somos-" + x.replace(".wav", "").split("_")[-1]
        )
        somosdata["file_path"] = somosdata["utt_id"].apply(
            lambda x: Path("data2/somos/audios") / x
        )
        somosdata["dataset"] = "somos"
        exdata.append(somosdata)

    exdata = pd.concat(exdata, axis=0)

    return exdata
