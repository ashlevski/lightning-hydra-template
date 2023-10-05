# coding=utf-8
# Copyright (c) DIRECT Contributors

from typing import Callable, Optional

import torch
from torch import nn

from direct.constants import COMPLEX_SIZE
from direct.data.transforms import expand_operator, reduce_operator
from direct.nn.get_nn_model_config import ModelName, _get_model_config
from direct.nn.types import InitType


class MRIVarSplitNet(nn.Module):
    r"""MRI reconstruction network that solves the variable split optimisation problem.

    It solves the following:

    .. math ::
         z^{i-1} = \arg \min_{z} \mu * ||x^{i-1} - z||_2^2 + \mathcal{R}(z)

         x^{i} = \arg \min_{x} ||y - A(x)||_2^2 + \mu * ||x - z^{i-1}||_2^2

    by unrolling twice using the gradient descent algorithm and replacing :math:`R` with a neural network.
    More specifically, for :math:`z_0, x_0 = \text{SENSE}(\tilde{y})`:

    .. math ::
        z^{i} = \alpha_{i-1} \times f_{\theta_{i-1}}\Big(\mu(z^{i-1} - x^{i-1}), z^{i-1}\Big), \quad i=1,\cdots,T_{reg}

    where :math:`x^{i}` is the output of

    .. math ::
        (x^{i})^{j} = (x^{i})^{j-1} - \beta_{j-1} \Big[ A^{*}\big( A( (x^{i})^{j-1} ) - \tilde{y} \big) +
        \mu ((x^{i})^{j-1} - z^{i}) \Big], \quad j=1,\cdots,T_{dc},

    i.e. :math:`x^{i}=(x^{i}^{T_{reg}})`.
    """

    def __init__(
        self,
        forward_operator: Callable,
        backward_operator: Callable,
        num_steps_reg: int,
        num_steps_dc: int,
        image_init: str = InitType.sense,
        no_parameter_sharing: bool = True,
        image_model_architecture: ModelName = ModelName.unet,
        kspace_no_parameter_sharing: Optional[bool] = True,
        kspace_model_architecture: Optional[ModelName] = None,
        **kwargs,
    ):
        """Inits :class:`MRIVarSplitNet`."""
        super().__init__()
        self.num_steps_reg = num_steps_reg
        self.num_steps_dc = num_steps_dc

        self.image_nets = nn.ModuleList()
        if kspace_model_architecture:
            self.kspace_nets = nn.ModuleList()
        else:
            self.kspace_nets = None

        self.no_parameter_sharing = no_parameter_sharing

        if image_model_architecture not in ["unet", "normunet", "resnet", "didn", "conv"]:
            raise ValueError(f"Invalid value {image_model_architecture} for `image_model_architecture`.")
        if kspace_model_architecture not in ["unet", "normunet", "resnet", "didn", "conv", None]:
            raise ValueError(f"Invalid value {kspace_model_architecture} for `kspace_model_architecture`.")

        image_model, image_model_kwargs = _get_model_config(
            image_model_architecture,
            in_channels=4,
            **{k.replace("image_", ""): v for (k, v) in kwargs.items() if "image_" in k},
        )
        for _ in range(self.num_steps_reg if self.no_parameter_sharing else 1):
            self.image_nets.append(image_model(**image_model_kwargs))

        if kspace_model_architecture:
            self.kspace_no_parameter_sharing = kspace_no_parameter_sharing
            kspace_model, kspace_model_kwargs = _get_model_config(
                kspace_model_architecture,
                in_channels=COMPLEX_SIZE * 2 + 1,
                **{k.replace("kspace_", ""): v for (k, v) in kwargs.items() if "kspace_" in k},
            )
            for _ in range(self.num_steps_reg if self.kspace_no_parameter_sharing else 1):
                self.kspace_nets.append(kspace_model(**kspace_model_kwargs))
            self.learning_rate_k = nn.Parameter(torch.ones(num_steps_reg, requires_grad=True))
            nn.init.trunc_normal_(self.learning_rate_k, 0.0, 1.0, 0.0)

        self.learning_rate_reg = nn.Parameter(torch.ones(num_steps_reg, requires_grad=True))
        nn.init.trunc_normal_(self.learning_rate_reg, 0.0, 1.0, 0.0)
        self.learning_rate_dc = nn.Parameter(torch.ones(num_steps_dc, requires_grad=True))
        nn.init.trunc_normal_(self.learning_rate_dc, 0.0, 1.0, 0.0)
        self.mu = nn.Parameter(torch.ones(1, requires_grad=True))
        nn.init.trunc_normal_(self.mu, 0, 0.1, 0.0)

        self.forward_operator = forward_operator
        self.backward_operator = backward_operator

        if image_init not in ["sense", "zero_filled"]:
            raise ValueError(f"Unknown image_initialization. Expected 'sense' or 'zero_filled'. " f"Got {image_init}.")

        self.image_init = image_init

        self._coil_dim = 1
        self._complex_dim = -1
        self._spatial_dims = (2, 3)

    def forward(
        self,
        masked_kspace: torch.Tensor,
        sensitivity_map: torch.Tensor,
        sampling_mask: torch.Tensor,
        scaling_factor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes forward pass of :class:`MRIVarSplitNet`.

        Parameters
        ----------
        masked_kspace: torch.Tensor
            Masked k-space of shape (N, coil, height, width, complex=2).
        sensitivity_map: torch.Tensor
            Sensitivity map of shape (N, coil, height, width, complex=2). Default: None.
        sampling_mask: torch.Tensor
        scaling_factor: torch.Tensor

        Returns
        -------
        image: torch.Tensor
            Output image of shape (N, height, width, complex=2).
        """
        if self.image_init == "sense":
            image = reduce_operator(
                coil_data=self.backward_operator(masked_kspace, dim=self._spatial_dims),
                sensitivity_map=sensitivity_map,
                dim=self._coil_dim,
            )
        else:
            image = self.backward_operator(masked_kspace, dim=self._spatial_dims).sum(self._coil_dim)

        if scaling_factor is None:
            scaling_factor = torch.tensor([1.0], dtype=masked_kspace.dtype).to(masked_kspace.device)
        scaling_factor = scaling_factor.reshape(-1, *(torch.ones(len(masked_kspace.shape) - 1).int()))

        z = image.clone()

        for iz in range(self.num_steps_reg):
            z = self.learning_rate_reg[iz] * self.image_nets[iz if self.no_parameter_sharing else 0](
                torch.cat([z, self.mu * (z - image)], dim=self._complex_dim).permute(0, 3, 1, 2)
            ).permute(0, 2, 3, 1)

            if self.kspace_nets is not None:
                kspace_z = torch.cat(
                    [
                        self.forward_operator(
                            expand_operator(z, sensitivity_map, self._coil_dim),
                            dim=self._spatial_dims,
                        ),
                        masked_kspace.clone(),
                        torch.repeat_interleave(sampling_mask, masked_kspace.size(self._coil_dim), self._coil_dim),
                    ],
                    self._complex_dim,
                )
                kspace_z = self.compute_model_per_coil(
                    self.kspace_nets[iz if self.kspace_no_parameter_sharing else 0],
                    kspace_z.permute(0, 1, 4, 2, 3),
                ).permute(0, 1, 3, 4, 2)

                z = z + self.learning_rate_k[iz] * reduce_operator(
                    coil_data=self.backward_operator(kspace_z.contiguous(), dim=self._spatial_dims),
                    sensitivity_map=sensitivity_map,
                    dim=self._coil_dim,
                )

            for ix in range(self.num_steps_dc):
                mul = scaling_factor * expand_operator(image, sensitivity_map, self._coil_dim)
                mr_forward = torch.where(
                    sampling_mask == 0,
                    torch.tensor([0.0], dtype=masked_kspace.dtype).to(masked_kspace.device),
                    self.forward_operator(mul, dim=self._spatial_dims),
                )
                error = mr_forward - scaling_factor * torch.where(
                    sampling_mask == 0,
                    torch.tensor([0.0], dtype=masked_kspace.dtype).to(masked_kspace.device),
                    masked_kspace,
                )
                mr_backward = self.backward_operator(error, dim=self._spatial_dims)
                dc = reduce_operator(mr_backward, sensitivity_map, self._coil_dim)

                image = image - self.learning_rate_dc[ix] * (dc + self.mu * (image - z))

        return image

    def compute_model_per_coil(self, model: nn.Module, data: torch.Tensor) -> torch.Tensor:
        """Performs forward pass of model per coil.

        Parameters
        ----------
        model: nn.Module
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
            output.append(model(subselected_data))

        return torch.stack(output, dim=self._coil_dim)
