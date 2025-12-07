from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.optim as optim

from utmosv2.loss import CombinedLoss, PairwizeDiffLoss

if TYPE_CHECKING:
    from utmosv2._settings._config import Config


def get_dataloader(
    cfg: Config, dataset: torch.utils.data.Dataset, phase: str
) -> torch.utils.data.DataLoader:
    """
    Return a DataLoader for the specified dataset and phase.

    Args:
        cfg (SimpleNamespace | ModuleType):
            Configuration object containing settings for batch size, number of workers, and pin memory.
        dataset (torch.utils.data.Dataset):
            The dataset to load data from.
        phase (str):
            The phase of the training process. Must be one of ["train", "valid", "test"].

    Returns:
        torch.utils.data.DataLoader: A DataLoader for the given dataset and phase.

    Raises:
        ValueError: If the phase is not one of ["train", "valid", "test"].
    """
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


def _get_unit_loss(loss_cfg: Config) -> nn.Module:
    if loss_cfg.name == "pairwize_diff":
        return PairwizeDiffLoss(loss_cfg.margin, loss_cfg.norm)
    elif loss_cfg.name == "mse":
        return nn.MSELoss()
    else:
        raise NotImplementedError


def _get_combined_loss(cfg: Config) -> nn.Module:
    if cfg.print_config:
        print(
            "Using losses: "
            + ", ".join([f"{loss_cfg.name} ({w})" for loss_cfg, w in cfg.loss])
        )
    weighted_losses = [(_get_unit_loss(loss_cfg), w) for loss_cfg, w in cfg.loss]
    return CombinedLoss(weighted_losses)


def get_loss(cfg: Config) -> nn.Module:
    """
    Return the appropriate loss function based on the configuration.

    Args:
        cfg (SimpleNamespace | ModuleType):
            Configuration object containing the loss settings.
            If `cfg.loss` is a list, a combined loss is returned.
            Otherwise, a single loss function is returned.

    Returns:
        nn.Module: The configured loss function, either a single loss or a combined loss module.
    """
    if isinstance(cfg.loss, list):
        return _get_combined_loss(cfg)
    else:
        return _get_unit_loss(cfg.loss)


def get_optimizer(cfg: Config, model: nn.Module) -> optim.Optimizer:
    """
    Return the optimizer based on the configuration settings.

    Args:
        cfg (SimpleNamespace | ModuleType):
            Configuration object containing optimizer settings.
            The optimizer name and learning rate are specified in `cfg.optimizer`.
        model (nn.Module):
            The model whose parameters will be optimized.

    Returns:
        optim.Optimizer: The configured optimizer (Adam, AdamW, or SGD).

    Raises:
        NotImplementedError: If the specified optimizer is not implemented.
    """
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
    cfg: Config, optimizer: optim.Optimizer, n_iterations: int
) -> optim.lr_scheduler.LRScheduler:
    """
    Return the learning rate scheduler based on the configuration settings.

    Args:
        cfg (SimpleNamespace | ModuleType):
            Configuration object containing scheduler settings.
            The scheduler name, T_max, and eta_min are specified in `cfg.scheduler`.
        optimizer (optim.Optimizer):
            The optimizer for which the learning rate will be scheduled.
        n_iterations (int):
            The number of iterations for the scheduler (used in CosineAnnealingLR).

    Returns:
        optim.lr_scheduler.LRScheduler: The configured learning rate scheduler.

    Raises:
        NotImplementedError: If the specified scheduler is not implemented.
    """
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
