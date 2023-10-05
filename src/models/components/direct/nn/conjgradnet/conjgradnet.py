# coding=utf-8
# Copyright (c) DIRECT Contributors

from typing import Callable, Optional, Tuple

import torch
from torch import nn

from direct.data.transforms import reduce_operator
from direct.nn.conjgradnet.conjgrad import CGUpdateType, ConjGrad
from direct.nn.get_nn_model_config import ModelName, _get_model_config
from direct.types import DirectEnum


class ConjGradNetInitType(DirectEnum):
    sense = "sense"
    zero_filled = "zero_filled"
    zeros = "zeros"


class ConjGradNet(nn.Module):
    r"""Conjugate Gradient Network for MRI Reconstruction.

    Solves iteratively the following:

    .. math ::
        z^i =  \arg \min_{z} \mu ||x^i - z||_2^2 + \mathcal{R}(z)
        x^i+1 = \arg \min_{x} ||A(x) - y||_2^2 + \mu ||x - z^i||_2^2

    where A is the forward operator of Accelerated MRI Reconstruction. The former equation is solved by a denoiser
    :math:`D_{i_\theta}` who takes as input :math:`x^i` and the former is solved by the conjugate gradient
    algorithm [1]_.

    References
    ----------
    .. [1] Jonathan Richard Shewchuk (1994) An introduction to the conjugate gradient method without the agonizing pain.
     Available at: https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf.
    """

    def __init__(
        self,
        forward_operator: Callable,
        backward_operator: Callable,
        num_steps: int,
        denoiser_architecture: ModelName = ModelName.resnet,
        image_init: ConjGradNetInitType = ConjGradNetInitType.sense,
        no_parameter_sharing: bool = True,
        cg_iters: int = 15,
        cg_tol: float = 1e-7,
        cg_param_update_type: CGUpdateType = CGUpdateType.FR,
        **kwargs,
    ):
        """Inits :class:`ConjGradNet`.

        Parameters
        ----------
        forward_operator : Callable
            Forward Operator.
        backward_operator : Callable
            Backward Operator.
        num_steps : int
            Number of unrolled optimization steps.
        denoiser_architecture : ModelName
            Type of architecture to use as a denoiser. Can be "resnet", "unet", "normunet", "didn" or "conv".
            Default: "resnet".
        image_init : ConjGradNetInitType
            Initialization type for `z`. Can be "sense", "zero_filled" or "zeros". Default: "zeros".
        no_parameter_sharing: bool
            If False, a single denoiser is used for all num_steps. Default: True.
        cg_iters : int
            Number of maximum conjugate gradient iterations. Defualt: 15.
        cg_tol : float
            Convergence tolerance for conjugate gradient. Default: 1e-7.
        cg_param_update_type : CGUpdateType
            How to compute :math:`b_k` in conjugate gradient. Can be "FR", "PRP", "DY" and "BAN". Default "FR".
        kwargs : dictionary
            Key word arguments should include denoiser architecture parameters. For example if `denoiser_architecture`
            is "unet" or "norm_unet" then `unet_num_filters`, `unet_num_pool_layers` and `unet_dropout_probability`
            should be passed.
        """
        super().__init__()
        self.num_steps = num_steps
        self.nets = nn.ModuleList()

        self.no_parameter_sharing = no_parameter_sharing
        for _ in range(self.num_steps if self.no_parameter_sharing else 1):
            model, model_kwargs = _get_model_config(denoiser_architecture, in_channels=2, out_channels=2, **kwargs)
            self.nets.append(model(**model_kwargs))
        self.learning_rate = nn.Parameter(torch.ones(num_steps, requires_grad=True))
        nn.init.normal_(self.learning_rate, 0, 1.0)
        self.conj_grad = ConjGrad(forward_operator, backward_operator, cg_iters, cg_tol, cg_param_update_type)
        self.mu = nn.Parameter(torch.ones(1), requires_grad=True)

        if image_init not in ["sense", "zero_filled", "zeros"]:
            raise ValueError(
                f"Unknown `image_initialization`. Expected 'sense', 'zero_filled' or 'zeros'. Got {image_init}."
            )
        self.image_init = image_init

        self.forward_operator = forward_operator
        self.backward_operator = backward_operator
        self._coil_dim = 1
        self._spatial_dims = (2, 3)

    def forward(
        self,
        masked_kspace: torch.Tensor,
        sensitivity_map: torch.Tensor,
        sampling_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Computes forward pass of :class:`ConjGradNet`.

        Parameters
        ----------
        masked_kspace: torch.Tensor
            Masked k-space of shape (N, coil, height, width, complex=2).
        sensitivity_map: torch.Tensor
            Sensitivity map of shape (N, coil, height, width, complex=2). Default: None.
        sampling_mask: torch.Tensor

        Returns
        -------
        image: torch.Tensor
            Output image of shape (N, height, width, complex=2).
        """
        z = self.init_z(
            self.image_init,
            self.backward_operator,
            masked_kspace,
            self._coil_dim,
            self._spatial_dims,
            sensitivity_map if self.image_init == "sense" else None,
        )
        x = self.conj_grad(masked_kspace, sensitivity_map, sampling_mask, z, self.mu)
        for i in range(self.num_steps):
            z = self.learning_rate[i] * self.nets[i if self.no_parameter_sharing else 0](
                x.permute(0, 3, 1, 2)
            ).permute(0, 2, 3, 1)
            x = self.conj_grad(masked_kspace, sensitivity_map, sampling_mask, z, self.mu)

        return x

    @staticmethod
    def init_z(
        image_init: ConjGradNetInitType,
        backward_operator: Callable,
        kspace: torch.Tensor,
        coil_dim: int,
        spatial_dims: Tuple[int, int],
        sensitivity_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if image_init == "zeros":
            image = torch.zeros([kspace.shape[0]] + list(kspace.shape[coil_dim + 1 :]), device=kspace.device)
        elif image_init == "sense":
            image = reduce_operator(
                coil_data=backward_operator(kspace.clone(), dim=spatial_dims),
                sensitivity_map=sensitivity_map,
                dim=coil_dim,
            )
        else:
            image = backward_operator(kspace, dim=spatial_dims).sum(coil_dim)

        return image
