# coding=utf-8
# Copyright (c) DIRECT Contributors

from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch import nn

import src.utils.direct.data.transforms as T




class EndToEndVarNetEngine(nn.Module):
    """End-to-End Variational Network Engine."""

    def __init__(
        self,
        model: nn.Module,
        forward_operator: Optional[Callable] = None,
        backward_operator: Optional[Callable] = None,
        sensitivity: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__()
        """Inits :class:`EndToEndVarNetEngine."""
        self.model = model
        self.forward_operator = forward_operator
        self.backward_operator = backward_operator
        self.sensitivity = sensitivity

    def forward(self, data: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            data = self.sensitivity(data)
            # target_image = T.root_sum_of_squares(
            #     self.backward_operator(data["kspace"], dim=(2, 3)),
            #     dim=1,
            # )  # shape (batch, height,  width)
        data['kspace'] = (data["kspace"]*data["acs_mask"])
        output_kspace = self.model(
            masked_kspace=data["kspace"],
            sampling_mask=data["acs_mask"],
            sensitivity_map=data["sensitivity_map"],
        )
        output_image = T.root_sum_of_squares(
            self.backward_operator(output_kspace, dim=(2,3)),
            dim=1,
        )  # shape (batch, height,  width)

        return output_image, output_kspace
