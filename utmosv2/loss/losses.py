from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PairwizeDiffLoss(nn.Module):
    def __init__(self, margin: float = 0.2, norm: str = "l1"):
        super().__init__()
        self.margin = margin
        self.norm = norm

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        s = input.unsqueeze(1) - input.unsqueeze(0)
        t = target.unsqueeze(1) - target.unsqueeze(0)
        if self.norm not in ["l1", "l2_squared"]:
            raise ValueError(
                f'Unknown norm: {self.norm}. Must be one of ["l1", "l2_squared"]'
            )
        norm_fn = {
            "l1": torch.abs,
            "l2_squared": lambda x: x**2,
        }[self.norm]
        loss = F.relu(norm_fn(s - t) - self.margin)
        return loss.mean().div(2)


class CombinedLoss(nn.Module):
    def __init__(self, weighted_losses: list[tuple[nn.Module, float]]):
        super().__init__()
        self.weighted_losses = weighted_losses

    def forward(
        self, input: torch.Tensor, target: torch.Tensor
    ) -> list[tuple[float, torch.Tensor]]:
        return [(w, loss(input, target)) for loss, w in self.weighted_losses]
