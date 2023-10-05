# coding=utf-8
# Copyright (c) DIRECT Contributors

from typing import Callable

import torch
import torch.nn as nn

import direct.data.transforms as T
from direct.nn.conv.conv import Conv2d
from direct.nn.didn.didn import DIDN
from direct.nn.mwcnn.mwcnn import MWCNN
from direct.nn.unet.unet_2d import NormUnetModel2d, UnetModel2d


class DualNet(nn.Module):
    """Dual Network for Learned Primal Dual Network."""

    def __init__(self, num_dual: int, **kwargs):
        """Inits :class:`DualNet`.

        Parameters
        ----------
        num_dual: int
            Number of dual for LPD algorithm.
        kwargs: dict
        """
        super().__init__()

        if kwargs.get("dual_architectue") is None:
            n_hidden = kwargs.get("n_hidden")
            if n_hidden is None:
                raise ValueError("Missing argument n_hidden.")
            self.dual_block = nn.Sequential(
                *[
                    nn.Conv2d(2 * (num_dual + 2), n_hidden, kernel_size=3, padding=1),
                    nn.PReLU(),
                    nn.Conv2d(n_hidden, n_hidden, kernel_size=3, padding=1),
                    nn.PReLU(),
                    nn.Conv2d(n_hidden, 2 * num_dual, kernel_size=3, padding=1),
                ]
            )
        else:
            self.dual_block = kwargs.get("dual_architectue")  # type: ignore

    @staticmethod
    def compute_model_per_coil(model: nn.Module, data: torch.Tensor) -> torch.Tensor:
        """Computes model per coil.

        Parameters
        ----------
        model: nn.Module
            Model to compute.
        data: torch.Tensor
            Multi-coil input.

        Returns
        -------
        output: torch.Tensor
            Multi-coil output.
        """
        output = []
        for idx in range(data.size(1)):
            subselected_data = data.select(1, idx)
            output.append(model(subselected_data))

        return torch.stack(output, dim=1)

    def forward(self, h: torch.Tensor, forward_f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([h, forward_f, g], dim=-1).permute(0, 1, 4, 2, 3)
        return self.compute_model_per_coil(self.dual_block, inp).permute(0, 1, 3, 4, 2)


class PrimalNet(nn.Module):
    """Primal Network for Learned Primal Dual Network."""

    def __init__(self, num_primal: int, **kwargs):
        """Inits :class:`PrimalNet`.

        Parameters
        ----------
        num_primal: int
            Number of primal for LPD algorithm.
        """
        super().__init__()

        if kwargs.get("primal_architectue") is None:
            n_hidden = kwargs.get("n_hidden")
            if n_hidden is None:
                raise ValueError("Missing argument n_hidden.")
            self.primal_block = nn.Sequential(
                *[
                    nn.Conv2d(2 * (num_primal + 1), n_hidden, kernel_size=3, padding=1),
                    nn.PReLU(),
                    nn.Conv2d(n_hidden, n_hidden, kernel_size=3, padding=1),
                    nn.PReLU(),
                    nn.Conv2d(n_hidden, 2 * num_primal, kernel_size=3, padding=1),
                ]
            )
        else:
            self.primal_block = kwargs.get("primal_architectue")  # type: ignore

    def forward(self, f: torch.Tensor, backward_h: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([f, backward_h], dim=-1).permute(0, 3, 1, 2)
        return self.primal_block(inp).permute(0, 2, 3, 1)


class LPDNet(nn.Module):
    """Learned Primal Dual network implementation inspired by [1]_.

    References
    ----------

    .. [1] Adler, Jonas, and Ozan Öktem. “Learned Primal-Dual Reconstruction.” IEEE Transactions on Medical Imaging, vol. 37, no. 6, June 2018, pp. 1322–32. arXiv.org, https://doi.org/10.1109/TMI.2018.2799231.
    """

    def __init__(
        self,
        forward_operator: Callable,
        backward_operator: Callable,
        num_iter: int,
        num_primal: int,
        num_dual: int,
        primal_model_architecture: str = "MWCNN",
        dual_model_architecture: str = "DIDN",
        **kwargs,
    ):
        """Inits :class:`LPDNet`.

        Parameters
        ----------
        forward_operator: Callable
            Forward Operator.
        backward_operator: Callable
            Backward Operator.
        num_iter: int
            Number of unrolled iterations.
        num_primal: int
            Number of primal networks.
        num_dual: int
            Number of dual networks.
        primal_model_architecture: str
            Primal model architecture. Currently only implemented for MWCNN and (NORM)UNET. Default: 'MWCNN'.
        dual_model_architecture: str
            Dual model architecture. Currently only implemented for CONV and DIDN and (NORM)UNET. Default: 'DIDN'.
        kwargs: dict
            Keyword arguments for model architectures.
        """
        super().__init__()

        self.forward_operator = forward_operator
        self.backward_operator = backward_operator
        self.num_iter = num_iter
        self.num_primal = num_primal
        self.num_dual = num_dual

        primal_model: nn.Module
        if primal_model_architecture == "MWCNN":
            primal_model = nn.Sequential(
                *[
                    MWCNN(
                        input_channels=2 * (num_primal + 1),
                        first_conv_hidden_channels=kwargs.get("primal_mwcnn_hidden_channels", 32),
                        num_scales=kwargs.get("primal_mwcnn_num_scales", 4),
                        bias=kwargs.get("primal_mwcnn_bias", False),
                        batchnorm=kwargs.get("primal_mwcnn_batchnorm", False),
                    ),
                    nn.Conv2d(2 * (num_primal + 1), 2 * num_primal, kernel_size=1),
                ]
            )
        elif primal_model_architecture in ["UNET", "NORMUNET"]:
            unet = UnetModel2d if primal_model_architecture == "UNET" else NormUnetModel2d
            primal_model = unet(
                in_channels=2 * (num_primal + 1),
                out_channels=2 * num_primal,
                num_filters=kwargs.get("primal_unet_num_filters", 8),
                num_pool_layers=kwargs.get("primal_unet_num_pool_layers", 4),
                dropout_probability=kwargs.get("primal_unet_dropout_probability", 0.0),
            )
        else:
            raise NotImplementedError(
                f"XPDNet is currently implemented only with primal_model_architecture == 'MWCNN', 'UNET' or 'NORMUNET."
                f"Got {primal_model_architecture}."
            )
        dual_model: nn.Module
        if dual_model_architecture == "CONV":
            dual_model = Conv2d(
                in_channels=2 * (num_dual + 2),
                out_channels=2 * num_dual,
                hidden_channels=kwargs.get("dual_conv_hidden_channels", 16),
                n_convs=kwargs.get("dual_conv_n_convs", 4),
                batchnorm=kwargs.get("dual_conv_batchnorm", False),
            )
        elif dual_model_architecture == "DIDN":
            dual_model = DIDN(
                in_channels=2 * (num_dual + 2),
                out_channels=2 * num_dual,
                hidden_channels=kwargs.get("dual_didn_hidden_channels", 16),
                num_dubs=kwargs.get("dual_didn_num_dubs", 6),
                num_convs_recon=kwargs.get("dual_didn_num_convs_recon", 9),
            )
        elif dual_model_architecture in ["UNET", "NORMUNET"]:
            unet = UnetModel2d if dual_model_architecture == "UNET" else NormUnetModel2d
            dual_model = unet(
                in_channels=2 * (num_dual + 2),
                out_channels=2 * num_dual,
                num_filters=kwargs.get("dual_unet_num_filters", 8),
                num_pool_layers=kwargs.get("dual_unet_num_pool_layers", 4),
                dropout_probability=kwargs.get("dual_unet_dropout_probability", 0.0),
            )
        else:
            raise NotImplementedError(
                f"XPDNet is currently implemented for dual_model_architecture == 'CONV', 'DIDN',"
                f" 'UNET' or 'NORMUNET'. Got dual_model_architecture == {dual_model_architecture}."
            )

        self._coil_dim = 1
        self._complex_dim = -1
        self._spatial_dims = (2, 3)

        self.primal_net = nn.ModuleList(
            [PrimalNet(num_primal, primal_architectue=primal_model) for _ in range(num_iter)]
        )
        self.dual_net = nn.ModuleList([DualNet(num_dual, dual_architectue=dual_model) for _ in range(num_iter)])

    def _forward_operator(
        self, image: torch.Tensor, sampling_mask: torch.Tensor, sensitivity_map: torch.Tensor
    ) -> torch.Tensor:
        forward = torch.where(
            sampling_mask == 0,
            torch.tensor([0.0], dtype=image.dtype).to(image.device),
            self.forward_operator(T.expand_operator(image, sensitivity_map, self._coil_dim), dim=self._spatial_dims),
        )
        return forward

    def _backward_operator(
        self, kspace: torch.Tensor, sampling_mask: torch.Tensor, sensitivity_map: torch.Tensor
    ) -> torch.Tensor:
        backward = T.reduce_operator(
            self.backward_operator(
                torch.where(
                    sampling_mask == 0,
                    torch.tensor([0.0], dtype=kspace.dtype).to(kspace.device),
                    kspace,
                ).contiguous(),
                self._spatial_dims,
            ),
            sensitivity_map,
            self._coil_dim,
        )
        return backward

    def forward(
        self,
        masked_kspace: torch.Tensor,
        sensitivity_map: torch.Tensor,
        sampling_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Computes forward pass of :class:`LPDNet`.

        Parameters
        ----------
        masked_kspace: torch.Tensor
            Masked k-space of shape (N, coil, height, width, complex=2).
        sensitivity_map: torch.Tensor
            Sensitivity map of shape (N, coil, height, width, complex=2).
        sampling_mask: torch.Tensor
            Sampling mask of shape (N, 1, height, width, 1).

        Returns
        -------
        output: torch.Tensor
            Output image of shape (N, height, width, complex=2).
        """
        input_image = self._backward_operator(masked_kspace, sampling_mask, sensitivity_map)
        dual_buffer = torch.cat([masked_kspace] * self.num_dual, self._complex_dim).to(masked_kspace.device)
        primal_buffer = torch.cat([input_image] * self.num_primal, self._complex_dim).to(masked_kspace.device)

        for curr_iter in range(self.num_iter):
            # Dual
            f_2 = primal_buffer[..., 2:4].clone()
            dual_buffer = self.dual_net[curr_iter](
                dual_buffer, self._forward_operator(f_2, sampling_mask, sensitivity_map), masked_kspace
            )

            # Primal
            h_1 = dual_buffer[..., 0:2].clone()
            primal_buffer = self.primal_net[curr_iter](
                primal_buffer, self._backward_operator(h_1, sampling_mask, sensitivity_map)
            )

        output = primal_buffer[..., 0:2]
        return output
