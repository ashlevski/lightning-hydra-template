# coding=utf-8
# Copyright (c) DIRECT Contributors

# Code was borrowed and reformatted from https://github.com/kornia/kornia/blob/master/kornia/filters/sobel.py
# part of "Kornia: an Open Source Differentiable Computer Vision Library for PyTorch" with an Apache License.

from enum import Enum
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SobelGradL1Loss", "SobelGradL2Loss"]


def get_sobel_kernel2d() -> torch.Tensor:
    r"""Returns the Sobel kernel matrices :math:`G_{x}` and :math:`G_{y}`:

    ..math::

        G_{x} = \begin{matrix}
                    -1 & 0 & 1 \\
                    -2 & 0 & 2 \\
                    -1 & 0 & 1
                \end{matrix}, \quad
        G_{y} = \begin{matrix}
                    -1 & -2 & -1 \\
                     0 & 0 & 0 \\
                     1 & 2 & 1
                \end{matrix}.
    """
    kernel_x: torch.Tensor = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
    kernel_y: torch.Tensor = kernel_x.transpose(0, 1)
    return torch.stack([kernel_x, kernel_y])


def normalize_kernel(input: torch.Tensor) -> torch.Tensor:
    r"""Normalize both derivative kernel.

    Parameters
    ----------
    input: torch.Tensor

    Returns
    -------
    torch.Tensor
        Normalized kernel.
    """
    norm: torch.Tensor = input.abs().sum(dim=-1).sum(dim=-1)
    return input / (norm.unsqueeze(-1).unsqueeze(-1))


def spatial_gradient(input: torch.Tensor, normalized: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Computes the first order image derivatives in :math:`x` and :math:`y` directions using a Sobel operator.

    Parameters
    ----------
    input: torch.Tensor
        Input image tensor with shape :math:`(B, C, H, W)`.
    normalized: bool
        Whether the output is normalized. Default: True.

    Returns
    -------
    grad_x, grad_y: (torch.Tensor, torch.Tensor)
        The derivatives in :math:`x` and :math:`y:` directions of the input each of same shape as input.
    """
    if not len(input.shape) == 4:
        raise ValueError(f"Invalid input shape, we expect BxCxHxW. Got: {input.shape}")
    # allocate kernel
    kernel: torch.Tensor = get_sobel_kernel2d()
    if normalized:
        kernel = normalize_kernel(kernel)

    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel: torch.Tensor = kernel.to(input).detach()
    tmp_kernel = tmp_kernel.unsqueeze(1).unsqueeze(1)

    # convolve input tensor with sobel kernel
    kernel_flip: torch.Tensor = tmp_kernel.flip(-3)

    # Pad with "replicate for spatial dims, but with zeros for channel
    spatial_pad = [kernel.size(1) // 2, kernel.size(1) // 2, kernel.size(2) // 2, kernel.size(2) // 2]
    padded_inp: torch.Tensor = F.pad(input.reshape(b * c, 1, h, w), spatial_pad, "replicate")[:, :, None]

    grad = F.conv3d(padded_inp, kernel_flip, padding=0).view(b, c, 2, h, w)
    grad_x, grad_y = grad[:, :, 0], grad[:, :, 1]

    return (grad_x, grad_y)


class SobelGradLossType(str, Enum):
    l1 = "l1"
    l2 = "l2"


class SobelGradLoss(nn.Module):
    r"""Computes the sum of the l1-loss between the gradient of input and target:

    It returns

    .. math ::

        ||u_x - v_x ||_k^k + ||u_y - v_y||_k^k

    where :math:`u` and :math:`v` denote the input and target images and :math:`k` is 1 if `type_loss`="l1" or 2 if
    `type_loss`="l2". The gradients w.r.t. to :math:`x` and :math:`y` directions are computed using the Sobel operators.
    """

    def __init__(self, type_loss: SobelGradLossType, reduction: str = "mean", normalized_grad: bool = True):
        """Inits :class:`SobelGradLoss`.

        Parameters
        ----------
        type_loss: SobelGradLossType
            Type of loss to be used. Can be "l1" or "l2".
        reduction: str
            Loss reduction. Can be 'mean' or "sum". Default: "mean".
        normalized_grad: bool
            Whether the computed gradients are normalized. Default: True.
        """
        super().__init__()

        self.reduction = reduction
        if type_loss == "l1":
            self.loss = nn.L1Loss(reduction=reduction)
        else:
            self.loss = nn.MSELoss(reduction=reduction)
        self.normalized_grad = normalized_grad

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Performs forward pass of :class:`SobelGradLoss`.

        Parameters
        ----------
        input: torch.Tensor
            Input tensor.
        target: torch.Tensor
            Target tensor.

        Returns
        -------
        loss: torch.Tensor
            Sum of the l1-loss between the gradient of input and target.
        """
        input_grad_x, input_grad_y = spatial_gradient(input, self.normalized_grad)
        target_grad_x, target_grad_y = spatial_gradient(target, self.normalized_grad)
        return self.loss(input_grad_x, target_grad_x) + self.loss(input_grad_y, target_grad_y)


class SobelGradL1Loss(SobelGradLoss):
    r"""Computes the sum of the l1-loss between the gradient of input and target:

    It returns

    .. math ::

        ||u_x - v_x ||_1 + ||u_y - v_y||_1

    where :math:`u` and :math:`v` denote the input and target images. The gradients w.r.t. to :math:`x` and :math:`y`
    directions are computed using the Sobel operators.
    """

    def __init__(self, reduction: str = "mean", normalized_grad: bool = True):
        """Inits :class:`SobelGradL1Loss`.

        Parameters
        ----------
        reduction: str
            Loss reduction. Can be 'mean' or "sum". Default: "mean".
        normalized_grad: bool
            Whether the computed gradients are normalized. Default: True.
        """
        super().__init__(SobelGradLossType.l1, reduction, normalized_grad)


class SobelGradL2Loss(SobelGradLoss):
    r"""Computes the sum of the l1-loss between the gradient of input and target:

    It returns

    .. math ::

        ||u_x - v_x ||_2^2 + ||u_y - v_y||_2^2

    where :math:`u` and :math:`v` denote the input and target images. The gradients w.r.t. to :math:`x` and :math:`y`
    directions are computed using the Sobel operators.
    """

    def __init__(self, reduction: str = "mean", normalized_grad: bool = True):
        """Inits :class:`SobelGradL2Loss`.

        Parameters
        ----------
        reduction: str
            Loss reduction. Can be 'mean' or "sum". Default: "mean".
        normalized_grad: bool
            Whether the computed gradients are normalized. Default: True.
        """
        super().__init__(SobelGradLossType.l2, reduction, normalized_grad)
