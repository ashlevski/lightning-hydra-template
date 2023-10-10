# coding=utf-8
# Copyright (c) DIRECT Contributors
import torch
import torch.nn as nn

__all__ = ("batch_psnr", "PSNRLoss")


def batch_psnr(input_data, target_data, reduction="mean"):
    """This function is a torch implementation of skimage.metrics.compare_psnr.

    Parameters
    ----------
    input_data: torch.Tensor
    target_data: torch.Tensor
    reduction: str

    Returns
    -------
    torch.Tensor
    """
    batch_size = target_data.size(0)
    input_view = input_data.view(batch_size, -1)
    target_view = target_data.view(batch_size, -1)
    maximum_value = torch.max(input_view, 1)[0]

    mean_square_error = torch.mean((input_view - target_view) ** 2, 1)
    psnrs = 20.0 * torch.log10(maximum_value) - 10.0 * torch.log10(mean_square_error)

    if reduction == "mean":
        return psnrs.mean()
    if reduction == "sum":
        return psnrs.sum()
    if reduction == "none":
        return psnrs
    raise ValueError(f"Reduction is either `mean`, `sum` or `none`. Got {reduction}.")


class PSNRLoss(nn.Module):
    __constants__ = ["reduction"]

    def __init__(self, reduction="mean"):
        self.reduction = reduction

    def forward(self, input_data, target_data):
        return batch_psnr(input_data, target_data, reduction=self.reduction)
