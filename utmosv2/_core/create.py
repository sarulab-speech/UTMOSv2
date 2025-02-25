from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import torch

from utmosv2._core.model import UTMOSv2Model
from utmosv2._settings import configure_execution
from utmosv2.utils._constants import _UTMOSV2_CHACHE
from utmosv2.utils._download import download_pretrained_weights_from_hf


def create_model(
    pretrained: bool = True,
    config: str = "fusion_stage3",
    fold: int = 0,
    checkpoint_path: Path | str | None = None,
    seed: int = 42,
) -> UTMOSv2Model:
    """
    Create a UTMOSv2 model with the specified configuration and optional pretrained weights.

    Args:
        pretrained (bool):
            If True, loads pretrained weights. Defaults to True.
        config (str):
            The configuration name to load for the model. Defaults to "fusion_stage3".
        fold (int):
            The fold number for the pretrained weights (used for model selection). Defaults to 0.
        checkpoint_path (Path | str | None):
            Path to a specific model checkpoint. If None, the checkpoint downloaded from GitHub is used. Defaults to None.
        seed (int):
            The seed used for model training to select the correct checkpoint. Defaults to 42.

    Returns:
        UTMOSv2Model: The initialized UTMOSv2 model.

    Raises:
        FileNotFoundError: If the specified checkpoint file is not found.

    Notes:
        - The configuration is dynamically loaded from `utmosv2.config`.
        - If `pretrained` is True and `checkpoint_path` is not provided, the function attempts to download pretrained weights from GitHub.
    """
    _cfg = importlib.import_module(f"utmosv2.config.{config}")
    # Avoid issues with pickling `types.ModuleType`,
    # making it easier to use with multiprocessing, DDP, etc.
    cfg = SimpleNamespace(
        **{k: v for k, v in _cfg.__dict__.items() if not k.startswith("__")}
    )
    configure_execution(cfg)

    model = UTMOSv2Model(cfg)

    if pretrained:
        if checkpoint_path is None:
            checkpoint_path = (
                _UTMOSV2_CHACHE
                / "models"
                / config
                / f"fold{fold}_s{seed}_best_model.pth"
            )
            if not checkpoint_path.exists():
                download_pretrained_weights_from_hf(config, fold)
        if isinstance(checkpoint_path, str):
            checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        model.load_state_dict(torch.load(checkpoint_path))
        print(f"Loaded checkpoint from {checkpoint_path}")

    return model
