from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from utmosv2.utils import calc_metrics, print_metrics
from utmosv2.utils._pure import _LazyImport

if TYPE_CHECKING:
    import pandas as pd
    import wandb
else:
    wandb = _LazyImport("wandb")


def train_1epoch(
    cfg,
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    train_loss: defaultdict[str, float] = defaultdict(float)
    scaler = GradScaler()
    print(f"  (lr: {scheduler.get_last_lr()[0]:.6f})")
    pbar = tqdm(train_dataloader, total=len(train_dataloader))
    for i, t in enumerate(pbar):
        x, y = t[:-1], t[-1]
        x = [t.to(device, non_blocking=True) for t in x]
        y = y.to(device, non_blocking=True)

        if cfg.run.mixup:
            lmd = np.random.beta(cfg.run.mixup_alpha, cfg.run.mixup_alpha)
            perm = torch.randperm(x[0].shape[0]).to(device)
            x2 = [t[perm, :] for t in x]
            y2 = y[perm]

        optimizer.zero_grad()
        with autocast():
            if cfg.run.mixup:
                output = model(
                    *[lmd * t + (1 - lmd) * t2 for t, t2 in zip(x, x2)]
                ).squeeze(1)
                if isinstance(cfg.loss, list):
                    loss = [
                        (w1, lmd * l1 + (1 - lmd) * l2)
                        for (w1, l1), (_, l2) in zip(
                            criterion(output, y), criterion(output, y2)
                        )
                    ]
                else:
                    loss = lmd * criterion(output, y) + (1 - lmd) * criterion(
                        output, y2
                    )
            else:
                output = model(*x).squeeze(1)
                loss = criterion(output, y)
            if isinstance(loss, list):
                loss_total = sum(w * ls for w, ls in loss)
            else:
                loss_total = loss

        scaler.scale(loss_total).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        train_loss["loss"] += loss_total.detach().float().cpu().item()
        if isinstance(loss, list):
            for (cl, _), (_, ls) in zip(cfg.loss, loss):
                train_loss[cl.name] += ls.detach().float().cpu().item()

        pbar.set_description(
            f'  loss: {train_loss["loss"] / (i + 1):.4f}'
            + (
                f' ({", ".join([f"{cl.name}: {train_loss[cl.name] / (i + 1):.4f}" for cl, _ in cfg.loss])})'
                if isinstance(loss, list)
                else ""
            )
        )

    return {name: v / len(train_dataloader) for name, v in train_loss.items()}


def validate_1epoch(
    cfg,
    model: torch.nn.Module,
    valid_dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    metrics: dict[str, Callable[[np.ndarray, np.ndarray], float]],
    device: torch.device,
) -> tuple[dict[str, float], dict[str, float], np.ndarray]:
    model.eval()
    valid_loss: defaultdict[str, float] = defaultdict(float)
    valid_metrics = {name: 0.0 for name in metrics}
    valid_preds = []
    pbar = tqdm(valid_dataloader, total=len(valid_dataloader))

    with torch.no_grad():
        for i, t in enumerate(pbar):
            x, y = t[:-1], t[-1]
            x = [t.to(device, non_blocking=True) for t in x]
            y_cpu = y
            y = y.to(device, non_blocking=True)
            with autocast():
                output = model(*x).squeeze(1)
                loss = criterion(output, y)
            if isinstance(loss, list):
                loss_total = sum(w * ls for w, ls in loss)
            else:
                loss_total = loss
            valid_loss["loss"] += loss_total.detach().float().cpu().item()
            if isinstance(loss, list):
                for (cl, _), (_, ls) in zip(cfg.loss, loss):
                    valid_loss[cl.name] += ls.detach().float().cpu().item()
            output = output.cpu().numpy()
            for name, metric in metrics.items():
                valid_metrics[name] += metric(output, y_cpu.numpy())
            pbar.set_description(
                f'  val_loss: {valid_loss["loss"] / (i + 1):.4f} '
                + (
                    f'({", ".join([f"{cl.name}: {valid_loss[cl.name] / (i + 1):.4f}" for cl, _ in cfg.loss])}) '
                    if isinstance(loss, list)
                    else ""
                )
                + " - ".join(
                    [
                        f"val_{name}: {v / (i + 1):.4f}"
                        for name, v in valid_metrics.items()
                    ]
                )
            )
            valid_preds.append(output)

    valid_loss_dic = {name: v / len(valid_dataloader) for name, v in valid_loss.items()}
    valid_metrics = {
        name: v / len(valid_dataloader) for name, v in valid_metrics.items()
    }
    valid_preds_arr = np.concatenate(valid_preds)

    return valid_loss_dic, valid_metrics, valid_preds_arr


def run_train(
    cfg,
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    valid_dataloader: torch.utils.data.DataLoader,
    valid_data: "pd.DataFrame",
    oof_preds: np.ndarray,
    now_fold: int,
    criterion: torch.nn.Module,
    metrics: dict[str, Callable[[np.ndarray, np.ndarray], float]],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
) -> None:
    best_metric = 0.0
    os.makedirs(cfg.save_path, exist_ok=True)

    for epoch in range(cfg.run.num_epochs):
        print(f"[Epoch {epoch + 1}/{cfg.run.num_epochs}]")
        train_loss = train_1epoch(
            cfg, model, train_dataloader, criterion, optimizer, scheduler, device
        )
        valid_loss, _, valid_preds = validate_1epoch(
            cfg, model, valid_dataloader, criterion, metrics, device
        )

        print(f"Validation dataset: {cfg.validation_dataset}")
        if cfg.validation_dataset == "each":
            dataset = valid_data["dataset"].unique()
            val_metrics_ls = [
                calc_metrics(
                    valid_data[valid_data["dataset"] == ds],
                    valid_preds[valid_data["dataset"] == ds],
                )
                for ds in dataset
            ]
            val_metrics = {
                name: sum([m[name] for m in val_metrics_ls]) / len(val_metrics_ls)
                for name in val_metrics_ls[0].keys()
            }
        elif cfg.validation_dataset == "all":
            print("Validation dataset: ALL")
            val_metrics = calc_metrics(valid_data, valid_preds)
        else:
            val_metrics = calc_metrics(
                valid_data[valid_data["dataset"] == cfg.validation_dataset],
                valid_preds[valid_data["dataset"] == cfg.validation_dataset],
            )
        print_metrics(val_metrics)

        if val_metrics[cfg.main_metric] > best_metric:
            new_metric = val_metrics[cfg.main_metric]
            print(f"(Found best metric: {best_metric:.4f} -> {new_metric:.4f})")
            best_metric = new_metric
            save_path = (
                cfg.save_path / f"fold{now_fold}_s{cfg.split.seed}_best_model.pth"
            )
            torch.save(model.state_dict(), save_path)
            print(f"Save best model: {save_path}")
            oof_preds[valid_data.index] = valid_preds

        save_path = cfg.save_path / f"fold{now_fold}_s{cfg.split.seed}_last_model.pth"
        torch.save(model.state_dict(), save_path)
        print()

        val_metrics["train_loss"] = train_loss["loss"]
        val_metrics["val_loss"] = valid_loss["loss"]
        for cl, _ in cfg.loss:
            val_metrics[f"train_loss_{cl.name}"] = train_loss[cl.name]
            val_metrics[f"val_loss_{cl.name}"] = valid_loss[cl.name]
        if cfg.wandb:
            wandb.log(val_metrics)
