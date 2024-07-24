from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
)

from utmosv2.loss import CombinedLoss, PairwizeDiffLoss


def split_data(
    cfg, data: pd.DataFrame
) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
    if cfg.print_config:
        print(f"Using split: {cfg.split.type}")
    if cfg.split.type == "simple":
        kf = KFold(n_splits=cfg.num_folds, shuffle=True, random_state=cfg.split.seed)
        for train_idx, valid_idx in kf.split(data):
            yield train_idx, valid_idx
    elif cfg.split.type == "stratified":
        kf = StratifiedKFold(
            n_splits=cfg.num_folds, shuffle=True, random_state=cfg.split.seed
        )
        for train_idx, valid_idx in kf.split(data, data[cfg.split.target].astype(int)):
            yield train_idx, valid_idx
    elif cfg.split.type == "group":
        kf = GroupKFold(n_splits=cfg.num_folds)
        for train_idx, valid_idx in kf.split(data, groups=data[cfg.split.group]):
            yield train_idx, valid_idx
    elif cfg.split.type == "stratified_group":
        kf = StratifiedGroupKFold(
            n_splits=cfg.num_folds, shuffle=True, random_state=cfg.split.seed
        )
        for train_idx, valid_idx in kf.split(
            data, data[cfg.split.target].astype(int), groups=data[cfg.split.group]
        ):
            yield train_idx, valid_idx
    elif cfg.split.type == "sgkf_kind":
        kind = data[cfg.split.kind].unique()
        kf = [
            StratifiedGroupKFold(
                n_splits=cfg.num_folds, shuffle=True, random_state=cfg.split.seed
            )
            for _ in range(len(kind))
        ]
        kf = [
            kf_i.split(
                data[data[cfg.split.kind] == ds],
                data[data[cfg.split.kind] == ds][cfg.split.target].astype(int),
                groups=data[data[cfg.split.kind] == ds][cfg.split.group],
            )
            for kf_i, ds in zip(kf, kind)
        ]
        for ds_idx in zip(*kf):
            train_idx = np.concatenate([d[0] for d in ds_idx])
            valid_idx = np.concatenate([d[1] for d in ds_idx])
            yield train_idx, valid_idx
    else:
        raise NotImplementedError


def get_dataloader(
    cfg, dataset: torch.utils.data.Dataset, phase: str
) -> torch.utils.data.DataLoader:
    if phase == "train":
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )
    elif phase == "valid":
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )
    elif phase == "test":
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.inference.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )
    else:
        raise ValueError(f"Phase must be one of [train, valid, test], but got {phase}")


def _get_unit_loss(loss_cfg) -> nn.Module:
    if loss_cfg.name == "pairwize_diff":
        return PairwizeDiffLoss(loss_cfg.margin, loss_cfg.norm)
    elif loss_cfg.name == "mse":
        return nn.MSELoss()
    else:
        raise NotImplementedError


def _get_combined_loss(cfg) -> nn.Module:
    if cfg.print_config:
        print(
            "Using losses: "
            + ", ".join([f"{loss_cfg.name} ({w})" for loss_cfg, w in cfg.loss])
        )
    weighted_losses = [(_get_unit_loss(loss_cfg), w) for loss_cfg, w in cfg.loss]
    return CombinedLoss(weighted_losses)


def get_loss(cfg) -> nn.Module:
    if isinstance(cfg.loss, list):
        return _get_combined_loss(cfg)
    else:
        return _get_unit_loss(cfg.loss)


def get_optimizer(cfg, model: nn.Module) -> optim.Optimizer:
    if cfg.print_config:
        print(f"Using optimizer: {cfg.optimizer.name}")
    if cfg.optimizer.name == "adam":
        return optim.Adam(model.parameters(), lr=cfg.optimizer.lr)
    elif cfg.optimizer.name == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay,
        )
    elif cfg.optimizer.name == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay,
        )
    else:
        raise NotImplementedError


def get_scheduler(
    cfg, optimizer: optim.Optimizer, n_iterations: int
) -> optim.lr_scheduler._LRScheduler:
    if cfg.print_config:
        print(f"Using scheduler: {cfg.scheduler}")
    if cfg.scheduler is None:
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1)
    if cfg.scheduler.name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.scheduler.T_max or n_iterations,
            eta_min=cfg.scheduler.eta_min,
        )
    else:
        raise NotImplementedError


def print_metrics(metrics: dict[str, float]):
    print(", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))


def save_oof_preds(cfg, data: pd.DataFrame, oof_preds: np.ndarray, fold: int):
    oof_df = pd.DataFrame({cfg.id_name: data[cfg.id_name], "oof_preds": oof_preds})
    oof_df.to_csv(
        cfg.save_path / f"fold{fold}_s{cfg.split.seed}_oof_preds.csv", index=False
    )


def configure_args(cfg, args):
    cfg.fold = args.fold
    cfg.split.seed = args.seed
    cfg.config_name = args.config
    cfg.input_dir = args.input_dir and Path(args.input_dir)
    cfg.num_workers = args.num_workers
    cfg.weight = args.weight
    cfg.save_path = Path("models") / cfg.config_name
    cfg.wandb = args.wandb
    cfg.reproduce = args.reproduce
    cfg.data_config = args.data_config
    cfg.phase = "train"


def configure_inference_args(cfg, args):
    cfg.inference.fold = args.fold
    cfg.split.seed = args.seed
    cfg.config_name = args.config
    cfg.input_dir = args.input_dir and Path(args.input_dir)
    cfg.input_path = args.input_path and Path(args.input_path)
    cfg.num_workers = args.num_workers
    cfg.weight = args.weight
    if not cfg.weight:
        cfg.weight = cfg.config_name
    cfg.inference.val_list_path = args.val_list_path and Path(args.val_list_path)
    cfg.save_path = Path("models") / cfg.config_name
    cfg.predict_dataset = args.predict_dataset
    cfg.final = args.final
    cfg.inference.num_tta = args.num_repetitions
    cfg.reproduce = args.reproduce
    cfg.out_path = args.out_path and Path(args.out_path)
    cfg.data_config = None
    cfg.phase = "inference"
