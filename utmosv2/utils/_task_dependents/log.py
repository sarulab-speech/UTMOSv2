import pandas as pd


def show_inference_data(data: pd.DataFrame):
    print(
        data[[c for c in data.columns if c != "mos"]]
        .rename(columns={"dataset": "predict_dataset"})
        .head()
    )
