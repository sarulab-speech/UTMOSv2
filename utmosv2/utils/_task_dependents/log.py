from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def show_inference_data(data: pd.DataFrame) -> None:
    print(
        data[[c for c in data.columns if c != "mos"]]
        .rename(columns={"dataset": "predict_dataset"})
        .head()
    )
