from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    import numpy as np


def save_oof_preds(cfg, data: pd.DataFrame, oof_preds: "np.ndarray", fold: int):
    oof_df = pd.DataFrame({cfg.id_name: data[cfg.id_name], "oof_preds": oof_preds})
    oof_df.to_csv(
        cfg.save_path / f"fold{fold}_s{cfg.split.seed}_oof_preds.csv", index=False
    )
