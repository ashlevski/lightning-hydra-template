# coding=utf-8
# Copyright (c) DIRECT Contributors

from typing import Callable

import torch
import torch.nn as nn

import direct.data.transforms as T
from direct.nn.multidomainnet.multidomain import MultiDomainUnet2d


class StandardizationLayer(nn.Module):
    r"""Multi-channel data standardization method. Inspired by AIRS model submission to the Fast MRI 2020 challenge. Given individual coil images :math:`\{x_i\}_{i=1}^{N_c}` and sensitivity coil maps :math:`\{S_i\}_{i=1}^{N_c}` it returns

    .. math::
        [(x_{\text{sense}}, {x_{\text{res}}}_1), ..., (x_{\text{sense}}, {x_{\text{res}}}_{N_c})]

    where :math:`{x_{\text{res}}}_i = xi - S_i \times x_{\text{sense}}` and :math:`x_{\text{sense}} = \sum_{i=1}^{N_c} {S_i}^{*} \times x_i`.

    """

    def __init__(self, coil_dim: int = 1, channel_dim: int = -1):
        """Inits :class:`StandardizationLayer`.

        Parameters
        ----------
        coil_dim: int
            Coil dimension. Default: 1.
        channel_dim: int
            Channel dimension. Default: -1.
        """
        super().__init__()
        self.coil_dim = coil_dim
        self.channel_dim = channel_dim

    def forward(self, coil_images: torch.Tensor, sensitivity_map: torch.Tensor) -> torch.Tensor:
        """Performs forward pass of :class:`StandardizationLayer`.

        Parameters
        ----------
        coil_images: torch.Tensor
            Coil images tensor.
        sensitivity_map: torch.Tensor
            Sensitivity maps.

        Returns
        -------
        torch.Tensor
        """
        combined_image = T.reduce_operator(coil_images, sensitivity_map, self.coil_dim)
        residual_image = combined_image.unsqueeze(self.coil_dim) - T.complex_multiplication(
            sensitivity_map, combined_image.unsqueeze(self.coil_dim)
        )
        concat = torch.cat(
            [
                torch.cat([combined_image, residual_image.select(self.coil_dim, idx)], self.channel_dim).unsqueeze(
                    self.coil_dim
                )
                for idx in range(coil_images.size(self.coil_dim))
            ],
            self.coil_dim,
        )
        return concat


class MultiDomainNet(nn.Module):
    """Feature-level multi-domain module.

    Inspired by AIRS Medical submission to the Fast MRI 2020 challenge.
    """

    def __init__(
        self,
        forward_operator: Callable,
        backward_operator: Callable,
        standardization: bool = True,
        num_filters: int = 16,
        num_pool_layers: int = 4,
        dropout_probability: float = 0.0,
        **kwargs,
    ):
        """Inits :class:`MultiDomainNet`.

        Parameters
        ----------
        forward_operator: Callable
            Forward Operator.
        backward_operator: Callable
            Backward Operator.
        standardization: bool
            If True standardization is used. Default: True.
        num_filters: int
            Number of filters for the :class:`MultiDomainUnet` module. Default: 16.
        num_pool_layers: int
            Number of pooling layers for the :class:`MultiDomainUnet` module. Default: 4.
        dropout_probability: float
            Dropout probability for the :class:`MultiDomainUnet` module. Default: 0.0.
        """
        super().__init__()
        self.forward_operator = forward_operator
        self.backward_operator = backward_operator
        self._coil_dim = 1
        self._complex_dim = -1
        self._spatial_dims = (2, 3)

        if standardization:
            self.standardization = StandardizationLayer(self._coil_dim, self._complex_dim)

        self.unet = MultiDomainUnet2d(
            forward_operator,
            backward_operator,
            in_channels=4 if standardization else 2,  # if standardization, in_channels is 4 due to standardized input
            out_channels=2,
            num_filters=num_filters,
            num_pool_layers=num_pool_layers,
            dropout_probability=dropout_probability,
        )

    def _compute_model_per_coil(self, model: nn.Module, data: torch.Tensor) -> torch.Tensor:
        """Computes model per coil.

        Parameters
        ----------
        model: nn.Module
            Model to compute.
        data: torch.Tensor
            Data to pass in the model.

        Returns
        -------
        output: torch.Tensor
        """
        output = []
        for idx in range(data.size(self._coil_dim)):
            subselected_data = data.select(self._coil_dim, idx)
            output.append(model(subselected_data))
        output = torch.stack(output, dim=self._coil_dim)
        return output

    def forward(self, masked_kspace: torch.Tensor, sensitivity_map: torch.Tensor) -> torch.Tensor:
        """Performs forward pass of :class:`MultiDomainNet`.

        Parameters
        ----------
        masked_kspace: torch.Tensor
            Masked k-space of shape (N, coil, height, width, complex=2).
        sensitivity_map: torch.Tensor
            Sensitivity map of shape (N, coil, height, width, complex=2).

        Returns
        -------
        output_image: torch.Tensor
            Multi-coil output image of shape (N, coil, height, width, complex=2).
        """
        input_image = self.backward_operator(masked_kspace, dim=self._spatial_dims)
        if hasattr(self, "standardization"):
            input_image = self.standardization(input_image, sensitivity_map)
        output_image = self._compute_model_per_coil(self.unet, input_image.permute(0, 1, 4, 2, 3)).permute(
            0, 1, 3, 4, 2
        )
        return output_image
