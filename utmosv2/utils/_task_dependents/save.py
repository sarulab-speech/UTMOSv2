from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np

from utmosv2._import import _LazyImport
from utmosv2._settings._config import Config
from utmosv2.utils._task_dependents.initializers import _get_test_save_name

if TYPE_CHECKING:
    import pandas as pd
else:
    pd = _LazyImport("pandas")


def save_test_preds(
    cfg: Config,
    data: "pd.DataFrame",
    test_preds: np.ndarray,
    test_metrics: dict[str, float],
):
    test_df = pd.DataFrame({cfg.id_name: data[cfg.id_name], "test_preds": test_preds})
    cfg.inference.save_path.mkdir(parents=True, exist_ok=True)
    save_path = (
        cfg.inference.save_path
        / f"{_get_test_save_name(cfg)}_({cfg.predict_dataset})_test_preds{'_final' if cfg.final else ''}.csv"
    )
    test_df.to_csv(save_path, index=False)
    save_path = (
        cfg.inference.save_path
        / f"{_get_test_save_name(cfg)}_({cfg.predict_dataset})_val_score{'_final' if cfg.final else ''}.json"
    )
    with open(save_path, "w") as f:
        json.dump(test_metrics, f)
    print(f"Test predictions are saved to {save_path}")


def make_submission_file(cfg: Config, data: "pd.DataFrame", test_preds: np.ndarray):
    submit = pd.DataFrame({cfg.id_name: data[cfg.id_name], "prediction": test_preds})
    (
        cfg.inference.submit_save_path
        / f"{_get_test_save_name(cfg)}_({cfg.predict_dataset})"
    ).mkdir(parents=True, exist_ok=True)
    sub_file = (
        cfg.inference.submit_save_path
        / f"{_get_test_save_name(cfg)}_({cfg.predict_dataset})"
        / "answer.txt"
    )
    submit.to_csv(
        sub_file,
        index=False,
        header=False,
    )
    print(f"Submission file is saved to {sub_file}")


def save_preds(cfg: Config, data: "pd.DataFrame", test_preds: np.ndarray):
    pred = pd.DataFrame({cfg.id_name: data[cfg.id_name], "mos": test_preds})
    if cfg.out_path is None:
        print("Predictions:")
        print(pred)
    else:
        pred.to_csv(cfg.out_path, index=False)
        print(f"Predictions are saved to {cfg.out_path}")
