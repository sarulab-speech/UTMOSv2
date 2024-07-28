from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm

from utmosv2.utils import calc_metrics, print_metrics


def run_inference(
    cfg,
    model: torch.nn.Module,
    test_dataloader: torch.utils.data.DataLoader,
    cycle: int,
    test_data: pd.DataFrame,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, float] | None]:
    model.eval()
    test_preds = []
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
            test_preds.append(output.cpu().numpy())
    test_preds = np.concatenate(test_preds)
    if cfg.reproduce:
        test_metrics = calc_metrics(test_data, test_preds)
        print_metrics(test_metrics)
    else:
        test_metrics = None

    return test_preds, test_metrics
