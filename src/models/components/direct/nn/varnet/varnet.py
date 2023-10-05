# coding=utf-8
# Copyright (c) DIRECT Contributors

from typing import Callable

import torch
import torch.nn as nn

from direct.data.transforms import expand_operator, reduce_operator
from direct.nn.unet import UnetModel2d


class EndToEndVarNet(nn.Module):
    """End-to-End Variational Network based on [1]_.

    References
    ----------

    .. [1] Sriram, Anuroop, et al. “End-to-End Variational Networks for Accelerated MRI Reconstruction.”
        ArXiv:2004.06688 [Cs, Eess], Apr. 2020. arXiv.org, http://arxiv.org/abs/2004.06688.
    """

    def __init__(
        self,
        forward_operator: Callable,
        backward_operator: Callable,
        num_layers: int,
        regularizer_num_filters: int = 18,
        regularizer_num_pull_layers: int = 4,
        regularizer_dropout: float = 0.0,
        in_channels: int = 2,
        **kwargs,
    ):
        """Inits :class:`EndToEndVarNet`.

        Parameters
        ----------
        forward_operator: Callable
            Forward Operator.
        backward_operator: Callable
            Backward Operator.
        num_layers: int
            Number of cascades.
        regularizer_num_filters: int
            Regularizer model number of filters.
        regularizer_num_pull_layers: int
            Regularizer model number of pulling layers.
        regularizer_dropout: float
            Regularizer model dropout probability.
        """
        super().__init__()
        extra_keys = kwargs.keys()
        for extra_key in extra_keys:
            if extra_key not in [
                "model_name",
            ]:
                raise ValueError(f"{type(self).__name__} got key `{extra_key}` which is not supported.")

        self.layers_list = nn.ModuleList()

        for _ in range(num_layers):
            self.layers_list.append(
                EndToEndVarNetBlock(
                    forward_operator=forward_operator,
                    backward_operator=backward_operator,
                    regularizer_model=UnetModel2d(
                        in_channels=in_channels,
                        out_channels=in_channels,
                        num_filters=regularizer_num_filters,
                        num_pool_layers=regularizer_num_pull_layers,
                        dropout_probability=regularizer_dropout,
                    ),
                )
            )

    def forward(
        self, masked_kspace: torch.Tensor, sampling_mask: torch.Tensor, sensitivity_map: torch.Tensor
    ) -> torch.Tensor:
        """Performs the forward pass of :class:`EndToEndVarNet`.

        Parameters
        ----------
        masked_kspace: torch.Tensor
            Masked k-space of shape (N, coil, height, width, complex=2).
        sampling_mask: torch.Tensor
            Sampling mask of shape (N, 1, height, width, 1).
        sensitivity_map: torch.Tensor
            Sensitivity map of shape (N, coil, height, width, complex=2).

        Returns
        -------
        kspace_prediction: torch.Tensor
            K-space prediction of shape (N, coil, height, width, complex=2).
        """

        kspace_prediction = masked_kspace.clone()
        for layer in self.layers_list:
            kspace_prediction = layer(kspace_prediction, masked_kspace, sampling_mask, sensitivity_map)
        return kspace_prediction


class EndToEndVarNetBlock(nn.Module):
    """End-to-End Variational Network block."""

    def __init__(
        self,
        forward_operator: Callable,
        backward_operator: Callable,
        regularizer_model: nn.Module,
    ):
        """Inits :class:`EndToEndVarNetBlock`.

        Parameters
        ----------
        forward_operator: Callable
            Forward Operator.
        backward_operator: Callable
            Backward Operator.
        regularizer_model: nn.Module
            Regularizer model.
        """
        super().__init__()
        self.regularizer_model = regularizer_model
        self.forward_operator = forward_operator
        self.backward_operator = backward_operator
        self.learning_rate = nn.Parameter(torch.tensor([1.0]))
        self._coil_dim = 1
        self._complex_dim = -1
        self._spatial_dims = (2, 3)

    def forward(
        self,
        current_kspace: torch.Tensor,
        masked_kspace: torch.Tensor,
        sampling_mask: torch.Tensor,
        sensitivity_map: torch.Tensor,
    ) -> torch.Tensor:
        """Performs the forward pass of :class:`EndToEndVarNetBlock`.

        Parameters
        ----------
        current_kspace: torch.Tensor
            Current k-space prediction of shape (N, coil, height, width, complex=2).
        masked_kspace: torch.Tensor
            Masked k-space of shape (N, coil, height, width, complex=2).
        sampling_mask: torch.Tensor
            Sampling mask of shape (N, 1, height, width, 1).
        sensitivity_map: torch.Tensor
            Sensitivity map of shape (N, coil, height, width, complex=2).

        Returns
        -------
        torch.Tensor
            Next k-space prediction of shape (N, coil, height, width, complex=2).
        """
        kspace_error = torch.where(
            sampling_mask == 0,
            torch.tensor([0.0], dtype=masked_kspace.dtype).to(masked_kspace.device),
            current_kspace - masked_kspace,
        )
        regularization_term = torch.cat(
            [
                reduce_operator(
                    self.backward_operator(kspace, dim=self._spatial_dims), sensitivity_map, dim=self._coil_dim
                )
                for kspace in torch.split(current_kspace, 2, self._complex_dim)
            ],
            dim=self._complex_dim,
        ).permute(0, 3, 1, 2)
        regularization_term = self.regularizer_model(regularization_term).permute(0, 2, 3, 1)
        regularization_term = torch.cat(
            [
                self.forward_operator(
                    expand_operator(image, sensitivity_map, dim=self._coil_dim), dim=self._spatial_dims
                )
                for image in torch.split(regularization_term, 2, self._complex_dim)
            ],
            dim=self._complex_dim,
        )
        return current_kspace - self.learning_rate * kspace_error + regularization_term
