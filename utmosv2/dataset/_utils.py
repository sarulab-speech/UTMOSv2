import json
from pathlib import Path

import librosa
import numpy as np

from utmosv2._settings._config import Config


def load_audio(cfg: Config, file: Path) -> np.ndarray:
    if file.suffix in [".wav", ".flac"]:
        y, sr = librosa.load(file, sr=None)
        y = librosa.resample(y, orig_sr=sr, target_sr=cfg.sr)
    else:
        y = np.load(file)
    return y


def extend_audio(cfg: Config, y: np.ndarray, length: int, type: str) -> np.ndarray:
    if y.shape[0] > length:
        return y
    elif type == "tile":
        n = length // y.shape[0] + 1
        y = np.tile(y, n)
        return y
    else:
        raise NotImplementedError


def select_random_start(y: np.ndarray, length: int) -> np.ndarray:
    start = np.random.randint(0, y.shape[0] - length)
    return y[start : start + length]


def get_dataset_map(cfg: Config) -> dict[str, int]:
    if cfg.data_config:
        with open(cfg.data_config, "r") as f:
            datasets = json.load(f)
        return {d["name"]: i for i, d in enumerate(datasets["data"])}
    else:
        return {
            "bvcc": 0,
            "sarulab": 1,
            "blizzard2008": 2,
            "blizzard2009": 3,
            "blizzard2010-EH1": 4,
            "blizzard2010-EH2": 5,
            "blizzard2010-ES1": 6,
            "blizzard2010-ES3": 7,
            "blizzard2011": 8,
            "somos": 9,
        }


def get_dataset_num(cfg: Config) -> int:
    return len(get_dataset_map(cfg))
