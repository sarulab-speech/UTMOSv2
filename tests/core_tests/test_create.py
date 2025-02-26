from __future__ import annotations

import pytest
from utmosv2._core.create import create_model

import sys


@pytest.mark.parametrize(
    "pretrained",
    [
        pytest.param(
            True,
            marks=pytest.mark.skipif(
                sys.version_info[:2] != (3, 11),
                reason="To avoid downloading the model weights multiple times",
            ),
        ),
        False,
    ],
)
def test_create_model(pretrained: bool) -> None:
    model = create_model(pretrained=pretrained)
    assert hasattr(model, "forward")
    assert hasattr(model, "predict")
