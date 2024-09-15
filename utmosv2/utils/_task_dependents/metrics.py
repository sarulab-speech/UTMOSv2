from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.stats


def calc_metrics(data: pd.DataFrame, preds: np.ndarray) -> dict[str, float]:
    data = data.copy()
    data["preds"] = preds
    data_sys = data.groupby("sys_id", as_index=False)[["mos", "preds"]].mean()
    res = {}
    for name, d in {"utt": data, "sys": data_sys}.items():
        res[f"{name}_mse"] = np.mean((d["mos"].values - d["preds"].values) ** 2)
        res[f"{name}_lcc"] = np.corrcoef(d["mos"].values, d["preds"].values)[0][1]
        res[f"{name}_srcc"] = scipy.stats.spearmanr(d["mos"].values, d["preds"].values)[
            0
        ]
        res[f"{name}_ktau"] = scipy.stats.kendalltau(
            d["mos"].values, d["preds"].values
        )[0]
    return res
