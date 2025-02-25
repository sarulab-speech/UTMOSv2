from typing import TYPE_CHECKING

from utmosv2._import import _LazyImport
from utmosv2._settings._config import Config

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
else:
    pd = _LazyImport("pandas")


def save_oof_preds(
    cfg: Config, data: "pd.DataFrame", oof_preds: "np.ndarray", fold: int
) -> None:
    """
    Save out-of-fold (OOF) predictions to a CSV file.

    Args:
        cfg (SimpleNamespace):
            Configuration object containing settings for saving OOF predictions.
            Includes `id_name` for the ID column and `save_path` for the save directory.
        data (pd.DataFrame):
            The original dataset containing the ID column.
        oof_preds (np.ndarray):
            The array of OOF predictions.
        fold (int):
            The current fold number used in cross-validation.

    Returns:
        None: The function saves the OOF predictions to a CSV file in the specified save path.
    """
    oof_df = pd.DataFrame({cfg.id_name: data[cfg.id_name], "oof_preds": oof_preds})
    oof_df.to_csv(
        cfg.save_path / f"fold{fold}_s{cfg.split.seed}_oof_preds.csv", index=False
    )
