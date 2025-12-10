from __future__ import annotations

import tempfile

import numpy as np
import pytest
import soundfile as sf
import torch

from utmosv2._core.create import create_model


@pytest.mark.parametrize(
    "data",
    [
        np.random.randn(32000 * 2),
        torch.randn(32000 * 2),
        np.random.randn(2, 32000 * 2),
        torch.randn(2, 32000 * 2),
    ],
)
def test_predict_data(data: np.ndarray | torch.Tensor) -> None:
    model = create_model(pretrained=False)

    pred = model.predict(data=data, device="cpu")
    assert type(pred) is type(data)
    assert pred.shape == (data.shape[0],) if data.ndim == 2 else (1,)


def test_predict_input_path() -> None:
    model = create_model(pretrained=False)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, np.random.randn(32000 * 2), samplerate=32000)
        pred = model.predict(input_path=tmp.name, device="cpu")
        assert isinstance(pred, float)


def test_predict_input_dir() -> None:
    model = create_model(pretrained=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        file_paths = []
        for i in range(2):
            file_path = f"{tmpdir}/test_{i}.wav"
            sf.write(file_path, np.random.randn(32000 * 2), samplerate=32000)
            file_paths.append(file_path)

        preds = model.predict(input_dir=tmpdir, device="cpu")
        assert isinstance(preds, list)
        assert len(preds) == 2
        for item in preds:
            assert "file_path" in item
            assert "predicted_mos" in item
            assert item["file_path"] in file_paths
            assert isinstance(item["predicted_mos"], float)
