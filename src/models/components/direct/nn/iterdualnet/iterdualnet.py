# coding=utf-8
# Copyright (c) DIRECT Contributors

from typing import Callable

import torch
from torch import nn

import direct.data.transforms as T
from direct.constants import COMPLEX_SIZE
from direct.nn.unet.unet_2d import NormUnetModel2d, UnetModel2d


class IterDualNet(nn.Module):
    r"""Iterative Dual Network solves iteratively the following problem

    .. math ::

        \min_{x} ||A(x) - y||_2^2 + \lambda_I ||x - D_I(x)||_2^2 + \lambda_F ||x - \mathcal{Q}(D_F(f))||_2^2, \quad
        \left\{ \begin{array} Q = \mathcal{F}^{-1}, f = \mathcal{F}(x) & \text{if compute_per_coil is False} \\
        Q = \mathcal{F}^{-1} \circ \mathcal{E}, f = \mathcal{R} \circ \mathcal{F}(x) & \text{otherwise} \end{array}

    by unrolling a gradient descent scheme where :math:`\mathcal{E}` and :math:`\mathcal{R}` are the expand and
    reduce operators which use the sensitivity maps. :math:`D_I` and :math:`D_F` are trainable U-Nets operating
    in the image and k-space domain.
    """

    def __init__(
        self,
        forward_operator: Callable,
        backward_operator: Callable,
        num_iter: int = 10,
        image_normunet: bool = False,
        kspace_normunet: bool = False,
        image_no_parameter_sharing: bool = True,
        kspace_no_parameter_sharing: bool = True,
        compute_per_coil: bool = True,
        **kwargs,
    ):
        """Inits :class:`IterDualNet`.

        Parameters
        ----------
        forward_operator : Callable
            Forward Operator.
        backward_operator : Callable
            Backward Operator.
        num_iter : int
            Number of iterations. Default: 10.
        image_normunet : bool
            If True will use NormUNet for the image model. Default: False.
        kspace_normunet : bool
            If True will use NormUNet for the kspace model. Default: False.
        image_no_parameter_sharing : bool
            If False, a single image model will be shared across all iterations. Default: True.
        kspace_no_parameter_sharing : bool
            If False, a single kspace model will be shared across all iterations. Default: True.
        compute_per_coil : bool
            If True :math:`f` will be transformed into a multi-coil kspace.
        kwargs : dict
            Kwargs for unet models.
        """
        super().__init__()

        self.forward_operator = forward_operator
        self.backward_operator = backward_operator
        self.num_iter = num_iter

        self.image_no_parameter_sharing = image_no_parameter_sharing
        self.kspace_no_parameter_sharing = kspace_no_parameter_sharing
        image_unet_architecture = NormUnetModel2d if image_normunet else UnetModel2d
        kspace_unet_architecture = NormUnetModel2d if kspace_normunet else UnetModel2d

        self.image_block_list = nn.ModuleList()
        self.kspace_block_list = nn.ModuleList()

        for _ in range(self.num_iter if self.image_no_parameter_sharing else 1):
            self.image_block_list.append(
                image_unet_architecture(
                    in_channels=COMPLEX_SIZE,
                    out_channels=COMPLEX_SIZE,
                    num_filters=kwargs.get("image_unet_num_filters", 8),
                    num_pool_layers=kwargs.get("image_unet_num_pool_layers", 4),
                    dropout_probability=kwargs.get("image_unet_dropout", 0.0),
                )
            )
        for _ in range(self.num_iter if self.kspace_no_parameter_sharing else 1):
            self.kspace_block_list.append(
                kspace_unet_architecture(
                    in_channels=COMPLEX_SIZE,
                    out_channels=COMPLEX_SIZE,
                    num_filters=kwargs.get("kspace_unet_num_filters", 8),
                    num_pool_layers=kwargs.get("kspace_unet_num_pool_layers", 4),
                    dropout_probability=kwargs.get("kspace_unet_dropout", 0.0),
                )
            )
        self.compute_per_coil = compute_per_coil

        self.lr = nn.Parameter(torch.ones(num_iter))
        self.reg_param_I = nn.Parameter(torch.ones(num_iter))
        self.reg_param_F = nn.Parameter(torch.ones(num_iter))

        self._coil_dim = 1
        self._complex_dim = -1
        self._spatial_dims = (2, 3)

    def _image_model(self, image: torch.Tensor, step: int) -> torch.Tensor:
        image = image.permute(0, 3, 1, 2)
        block_idx = step if self.image_no_parameter_sharing else 0
        return self.image_block_list[block_idx](image).permute(0, 2, 3, 1).contiguous()

    def _kspace_model(self, kspace: torch.Tensor, step: int) -> torch.Tensor:
        block_idx = step if self.kspace_no_parameter_sharing else 0
        if self.compute_per_coil:
            kspace = (
                self._compute_model_per_coil(self.kspace_block_list[block_idx], kspace.permute(0, 1, 4, 2, 3))
                .permute(0, 1, 3, 4, 2)
                .contiguous()
            )
        else:
            kspace = self.kspace_block_list[block_idx](kspace.permute(0, 3, 1, 2)).permute(0, 2, 3, 1).contiguous()
        return kspace

    def _compute_model_per_coil(self, model: nn.Module, data: torch.Tensor) -> torch.Tensor:
        output = []
        for idx in range(data.size(self._coil_dim)):
            subselected_data = data.select(self._coil_dim, idx)
            output.append(model(subselected_data))
        return torch.stack(output, dim=self._coil_dim)

    def _forward_operator(
        self, image: torch.Tensor, sampling_mask: torch.Tensor, sensitivity_map: torch.Tensor
    ) -> torch.Tensor:
        return T.apply_mask(
            self.forward_operator(T.expand_operator(image, sensitivity_map, self._coil_dim), dim=self._spatial_dims),
            sampling_mask,
            return_mask=False,
        )

    def _backward_operator(
        self, kspace: torch.Tensor, sampling_mask: torch.Tensor, sensitivity_map: torch.Tensor
    ) -> torch.Tensor:
        return T.reduce_operator(
            self.backward_operator(T.apply_mask(kspace, sampling_mask, return_mask=False), self._spatial_dims),
            sensitivity_map,
            self._coil_dim,
        )

    def forward(
        self,
        masked_kspace: torch.Tensor,
        sampling_mask: torch.Tensor,
        sensitivity_map: torch.Tensor,
    ) -> torch.Tensor:
        """Computes forward pass of :class:`IterDualNet`.

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
        out_image: torch.Tensor
            Output image of shape (N, height, width, complex=2).
        """
        x = T.reduce_operator(
            self.backward_operator(masked_kspace, self._spatial_dims), sensitivity_map, self._coil_dim
        )

        for step in range(self.num_iter):
            f = (
                self.forward_operator(T.expand_operator(x, sensitivity_map, self._coil_dim), dim=self._spatial_dims)
                if self.compute_per_coil
                else self.forward_operator(x, dim=[d - 1 for d in self._spatial_dims])
            )
            kspace_model_out = self._kspace_model(f, step)
            kspace_model_out = (
                T.reduce_operator(
                    self.backward_operator(kspace_model_out, self._spatial_dims),
                    sensitivity_map,
                    self._coil_dim,
                )
                if self.compute_per_coil
                else self.backward_operator(kspace_model_out, dim=[d - 1 for d in self._spatial_dims])
            )

            img_model_out = self._image_model(x, step)

            dc_out = self._backward_operator(
                self._forward_operator(x, sampling_mask, sensitivity_map) - masked_kspace,
                sampling_mask,
                sensitivity_map,
            )
            x = (1 - self.lr[step] * (self.reg_param_I[step] + self.reg_param_F[step])) * x + self.lr[step] * (
                self.reg_param_I[step] * img_model_out + self.reg_param_F[step] * kspace_model_out - dc_out
            )
        return x
