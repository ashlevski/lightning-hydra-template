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
        sensitivity_model: nn.Module = None,
        **kwargs,
    ):
        super().__init__()
        """Inits :class:`EndToEndVarNetEngine."""
        self.model = model
        self.forward_operator = forward_operator
        self.backward_operator = backward_operator
        self.sensitivity = sensitivity
        self.sensitivity_model = sensitivity_model
        self._coil_dim = 1
        self._complex_dim = -1
        self._spatial_dim = (2,3)

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
            sensitivity_map=self.compute_sensitivity_map(data["sensitivity_map"]),
        )
        output_image = T.root_sum_of_squares(
            self.backward_operator(output_kspace, dim=self._spatial_dim),
            dim=self._coil_dim,
        )  # shape (batch, height,  width)

        return output_image, output_kspace

    def compute_sensitivity_map(self, sensitivity_map: torch.Tensor) -> torch.Tensor:
        r"""Computes sensitivity maps :math:`\{S^k\}_{k=1}^{n_c}` if `sensitivity_model` is available.

        :math:`\{S^k\}_{k=1}^{n_c}` are normalized such that

        .. math::
            \sum_{k=1}^{n_c}S^k {S^k}^* = I.

        Parameters
        ----------
        sensitivity_map: torch.Tensor
            Sensitivity maps of shape (batch, coil, height,  width, complex=2).

        Returns
        -------
        sensitivity_map: torch.Tensor
            Normalized and refined sensitivity maps of shape (batch, coil, height,  width, complex=2).
        """
        # ToDo: add transform per coil
        # Some things can be done with the sensitivity map here, e.g. apply a u-net
        if True:#"sensitivity_model" in self.models:
            # Move channels to first axis
            sensitivity_map = sensitivity_map.permute(
                (0, 1, 4, 2, 3)
            )  # shape (batch, coil, complex=2, height,  width)

            sensitivity_map = self.compute_model_per_coil("sensitivity_model", sensitivity_map).permute(
                (0, 1, 3, 4, 2)
            )  # has channel last: shape (batch, coil, height,  width, complex=2)

        # The sensitivity map needs to be normalized such that
        # So \sum_{i \in \text{coils}} S_i S_i^* = 1

        sensitivity_map_norm = torch.sqrt(
            ((sensitivity_map**2).sum(self._complex_dim)).sum(self._coil_dim)
        )  # shape (batch, height, width)
        sensitivity_map_norm = sensitivity_map_norm.unsqueeze(self._coil_dim).unsqueeze(self._complex_dim)

        return T.safe_divide(sensitivity_map, sensitivity_map_norm)

    def compute_model_per_coil(self, model_name: str, data: torch.Tensor) -> torch.Tensor:
        """Performs forward pass of model `model_name` in `self.models` per coil.

        Parameters
        ----------
        model_name: str
            Model to run.
        data: torch.Tensor
            Multi-coil data of shape (batch, coil, complex=2, height, width).

        Returns
        -------
        output: torch.Tensor
            Computed output per coil.
        """
        output = []
        for idx in range(data.size(self._coil_dim)):
            subselected_data = data.select(self._coil_dim, idx)
            output.append(self.sensitivity_model(subselected_data))

        return torch.stack(output, dim=self._coil_dim)
