from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from utmosv2.loss import CombinedLoss, PairwizeDiffLoss


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
) -> optim.lr_scheduler.LRScheduler:
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
