from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm

from utmosv2.utils import calc_metrics, print_metrics

if TYPE_CHECKING:
    import pandas as pd


def run_inference(
    cfg,
    model: torch.nn.Module,
    test_dataloader: torch.utils.data.DataLoader,
    cycle: int,
    test_data: "pd.DataFrame",
    device: torch.device,
) -> tuple[np.ndarray, dict[str, float] | None]:
    """
    Run inference on the test dataset using the provided model.

    Args:
        cfg (SimpleNamespace | ModuleType):
            Configuration object containing inference settings.
            It includes settings for test-time augmentation (TTA) and reproducibility.
        model (torch.nn.Module):
            The trained model to be used for inference.
        test_dataloader (torch.utils.data.DataLoader):
            Dataloader for the test dataset.
        cycle (int):
            Current cycle of test-time augmentation (TTA) if used.
        test_data (pd.DataFrame):
            DataFrame containing test data, used for metric calculation if reproducibility is enabled.
        device (torch.device):
            Device to run inference on (e.g., 'cuda' or 'cpu').

    Returns:
        tuple[np.ndarray, dict[str, float] | None]:
            - test_preds: Array containing the model's predictions for the test dataset.
            - test_metrics: Dictionary containing the calculated metrics if reproducibility is enabled; otherwise, None.
    """
    model.eval()
    test_preds_ls = []
    pbar = tqdm(
        test_dataloader,
        total=len(test_dataloader),
        desc=f"  [Inference] ({cycle + 1}/{cfg.inference.num_tta})",
    )

    with torch.no_grad():
        for t in pbar:
            x = t[:-1]
            x = [t.to(device, non_blocking=True) for t in x]
            with autocast():
                output = model(*x).squeeze(1)
            test_preds_ls.append(output.cpu().numpy())
    test_preds = np.concatenate(test_preds_ls)
    if cfg.reproduce:
        test_metrics = calc_metrics(test_data, test_preds)
        print_metrics(test_metrics)
    else:
        test_metrics = None

    return test_preds, test_metrics
