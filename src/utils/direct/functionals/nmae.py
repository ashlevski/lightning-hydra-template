# coding=utf-8
# Copyright (c) DIRECT Contributors

import torch
import torch.nn as nn

__all__ = ["NMAELoss"]


class NMAELoss(nn.Module):
    """Computes the Normalized Mean Absolute Error (NMAE), i.e.:

    .. math::
        \frac{||u - v||_1}{||u||_1},

    where :math:`u` and :math:`v` denote the target and the input.
    """

    def __init__(self, reduction="mean"):
        """Inits :class:`NMAE`

        Parameters
        ----------
        reduction: str
             Specifies the reduction to apply to the output. Can be "none", "mean" or "sum".
             Note that "mean" or "sum" will yield the same output. Default: "mean".
        """
        super().__init__()
        self.mae_loss = nn.L1Loss(reduction=reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """Forward method of :class:`NMAE`.

        Parameters
        ----------
        input: torch.Tensor
            Tensor of shape (*), where * means any number of dimensions.
        target: torch.Tensor
            Tensor of same shape as the input.
        """
        return self.mae_loss(input, target) / self.mae_loss(
            torch.zeros_like(target, dtype=target.dtype, device=target.device), target
        )
