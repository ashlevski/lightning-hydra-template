# coding=utf-8
# Copyright (c) DIRECT Contributors

from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch import nn

import src.utils.direct.data.transforms as T
from src.utils.transforms import NormalizeSampleTransform, normalizeSampleTransform


class UNETEngine(nn.Module):
    """End-to-End Variational Network Engine."""

    def __init__(
        self,
        model: nn.Module,
        forward_operator: Optional[Callable] = None,
        backward_operator: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__()
        """Inits :class:`EndToEndVarNetEngine."""
        self.model = model
        self.forward_operator = forward_operator
        self.backward_operator = backward_operator
        self._coil_dim = 1
        self._complex_dim = -1
        self._spatial_dim = (2,3)

    def forward(self, data: Dict[str, Any], compute_target = True) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            if compute_target:
                target_img = T.root_sum_of_squares(
                    self.backward_operator(data["kspace"], dim=(2, 3)),
                    dim=1,
                )  # shape (batch, height,  width)
            else:
                target_img = 0
            data['kspace'] = (data["kspace"] * data["acs_mask"])
            input_img = T.root_sum_of_squares(
                self.backward_operator(data["kspace"], dim=(2, 3)),
                dim=1,
            )  # shape (batch, height,  width)

        output_image = self.model(
            input_img.unsqueeze(1)
        ).squeeze(1)

        return output_image, 0, target_img

