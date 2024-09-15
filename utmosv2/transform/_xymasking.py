from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch


class XYMasking:
    def __init__(
        self,
        num_masks_x: int | tuple[int, int],
        num_masks_y: int | tuple[int, int],
        mask_x_length: int | tuple[int, int],
        mask_y_length: int | tuple[int, int],
        fill_value: int,
        p: float = 1.0,
    ):
        self.num_masks_x = num_masks_x
        self.num_masks_y = num_masks_y
        self.mask_x_length = mask_x_length
        self.mask_y_length = mask_y_length
        self.fill_value = fill_value
        self.p = p

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if np.random.rand() < self.p:
            return img
        _, width, height = img.shape
        num_masks_x = (
            np.random.randint(*self.num_masks_x)
            if isinstance(self.num_masks_x, tuple)
            else self.num_masks_x
        )
        for _ in range(num_masks_x):
            mask_x_length = (
                np.random.randint(*self.mask_x_length)
                if isinstance(self.mask_x_length, tuple)
                else self.mask_x_length
            )
            x = np.random.randint(0, width - mask_x_length)
            img[:, :, x : x + mask_x_length] = self.fill_value

        num_masks_y = (
            np.random.randint(*self.num_masks_y)
            if isinstance(self.num_masks_y, tuple)
            else self.num_masks_y
        )
        for _ in range(num_masks_y):
            mask_y_length = (
                np.random.randint(*self.mask_y_length)
                if isinstance(self.mask_y_length, tuple)
                else self.mask_y_length
            )
            y = np.random.randint(0, height - mask_y_length)
            img[:, y : y + mask_y_length, :] = self.fill_value

        return img
