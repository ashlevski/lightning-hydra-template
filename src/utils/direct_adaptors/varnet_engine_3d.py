# coding=utf-8
# Copyright (c) DIRECT Contributors

from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch import nn

import src.utils.direct.data.transforms as T
from src.models.components.attention_adjacent import Attention
from src.utils.direct.nn.unet import UnetModel2d
from src.utils.transforms import NormalizeSampleTransform, normalizeSampleTransform



class EndToEndVarNetEngine(nn.Module):
    """End-to-End Variational Network Engine."""

    def __init__(
        self,
        model: nn.Module,
        forward_operator: Optional[Callable] = None,
        backward_operator: Optional[Callable] = None,
        sensitivity: Optional[Callable] = None,
        sensitivity_model: nn.Module = None,
        dim_reduction = None,
        **kwargs,
    ):
        super().__init__()
        """Inits :class:`EndToEndVarNetEngine."""
        self.model = model
        self.forward_operator = forward_operator
        self.backward_operator = backward_operator
        self.sensitivity = sensitivity
        self.sensitivity_model = sensitivity_model
        self.attention = Attention()
        self._coil_dim = 1
        self._complex_dim = -1
        self._spatial_dim = (2,3)
        self.u_net = UnetModel2d(in_channels=3,out_channels=1,num_filters=8,num_pool_layers=4,dropout_probability=0.0)

    def forward(self, data: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            target_img = T.root_sum_of_squares(
                self.backward_operator(data["kspace"][:, :, 1].squeeze(), dim=(2, 3)),
                dim=1,
            )  # shape (batch, height,  width)

        data['kspace'] = (data["kspace"]*data["acs_mask"].unsqueeze(2))

        # with torch.no_grad():
        #     before_img,_ = self.do(data.copy(),0)
        #     after_img,_ = self.do(data.copy(),2)
        before_img = T.root_sum_of_squares(
            self.backward_operator(data["kspace"][:, :, 0].squeeze(), dim=(2, 3)),
            dim=1,
        )  # shape (batch, height,  width)
        after_img = T.root_sum_of_squares(
            self.backward_operator(data["kspace"][:, :, 2].squeeze(), dim=(2, 3)),
            dim=1,
        )  # shape (batch, height,  width)
        output_image, output_kspace = self.do(data,1)
        res = self.u_net(torch.stack((before_img,output_image,after_img),dim=1)).squeeze()
        # output_imag = self.attention(output_image,torch.stack((before_img,after_img),dim=1))
        return output_image+0.01*res, output_kspace, target_img

    def do(self,data,i):
        with torch.no_grad():
            data['kspace'] = data['kspace'][:, :, i]
            data = self.sensitivity(data)

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
