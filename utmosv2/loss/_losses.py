from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PairwizeDiffLoss(nn.Module):
    """
    Pairwise difference loss function for comparing input and target tensors.
    The loss is based on the difference between pairs of inputs and pairs of targets,
    with a specified margin and norm ("l1" or "l2_squared").
    """

    def __init__(self, margin: float = 0.2, norm: str = "l1"):
        """
        Initialize the PairwizeDiffLoss with the specified margin and norm.

        Args:
            margin (float):
                The margin value used for the loss function. Defaults to 0.2.
            norm (str):
                The norm to use for the difference calculation. Must be "l1" or "l2_squared". Defaults to "l1".
        """
        super().__init__()
        self.margin = margin
        self.norm = norm

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the pairwise difference loss between input and target tensors.

        Args:
            input (torch.Tensor): The input tensor.
            target (torch.Tensor): The target tensor.

        Returns:
            torch.Tensor: The computed loss.
        """
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
        loss = F.relu(norm_fn(s - t) - self.margin)  # type: ignore
        return loss.mean().div(2)


class CombinedLoss(nn.Module):
    """
    A combined loss function that allows for multiple loss functions to be weighted and combined.

    Args:
        weighted_losses (list[tuple[nn.Module, float]]):
            A list of loss functions and their associated weights.
    """

    def __init__(self, weighted_losses: list[tuple[nn.Module, float]]):
        super().__init__()
        self.weighted_losses = weighted_losses

    def forward(
        self, input: torch.Tensor, target: torch.Tensor
    ) -> list[tuple[float, torch.Tensor]]:
        """
        Compute the weighted loss for each loss function in the list.

        Args:
            input (torch.Tensor): The input tensor.
            target (torch.Tensor): The target tensor.

        Returns:
            list[tuple[float, torch.Tensor]]:
                A list of tuples where each contains a weight and the corresponding computed loss.
        """
        return [(w, loss(input, target)) for loss, w in self.weighted_losses]
