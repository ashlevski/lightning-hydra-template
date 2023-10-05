# coding=utf-8
# Copyright (c) DIRECT Contributors

import warnings
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from direct.data import transforms as T
from direct.nn.recurrent.recurrent import Conv2dGRU, NormConv2dGRU
from direct.utils.asserts import assert_positive_integer


class MRILogLikelihood(nn.Module):
    r"""Defines the MRI loglikelihood assuming one noise vector for the complex images for all coils:

    .. math::
         \frac{1}{\sigma^2} \sum_{i}^{N_c} {S}_i^{\text{H}} \mathcal{F}^{-1} P^{*} (P \mathcal{F} S_i x_{\tau} - y_{\tau})

    for each time step :math:`\tau`.
    """

    def __init__(
        self,
        forward_operator: Callable,
        backward_operator: Callable,
    ):
        """Inits :class:`MRILogLikelihood`.

        Parameters
        ----------
        forward_operator: Callable
            Forward Operator.
        backward_operator: Callable
            Backward Operator.
        """
        super().__init__()

        self.forward_operator = forward_operator
        self.backward_operator = backward_operator

        self._coil_dim = 1
        self._spatial_dims = (2, 3)

    def forward(
        self,
        input_image,
        masked_kspace,
        sensitivity_map,
        sampling_mask,
        loglikelihood_scaling=None,
    ) -> torch.Tensor:
        """Performs forward pass of :class:`MRILogLikelihood`.

        Parameters
        ----------
        input_image: torch.Tensor
            Initial or previous iteration of image with complex first
            of shape (N, complex, height, width).
        masked_kspace: torch.Tensor
            Masked k-space of shape (N, coil, height, width, complex).
        sensitivity_map: torch.Tensor
            Sensitivity Map of shape (N, coil, height, width, complex).
        sampling_mask: torch.Tensor
        loglikelihood_scaling: torch.Tensor
            Multiplier for loglikelihood, for instance for the k-space noise, of shape (1,).

        Returns
        -------
        out: torch.Tensor
            The MRI Loglikelihood.
        """

        input_image = input_image.permute(0, 2, 3, 1)  # shape (N, height, width, complex)

        if loglikelihood_scaling is not None:
            loglikelihood_scaling = loglikelihood_scaling
        else:
            loglikelihood_scaling = torch.tensor([1.0], dtype=masked_kspace.dtype).to(masked_kspace.device)
        loglikelihood_scaling = loglikelihood_scaling.reshape(
            -1, *(torch.ones(len(sensitivity_map.shape) - 1).int())
        )  # shape (1, 1, 1, 1, 1)

        # We multiply by the loglikelihood_scaling here to prevent fp16 information loss,
        # as this value is typically <<1, and the operators are linear.

        mul = loglikelihood_scaling * T.complex_multiplication(
            sensitivity_map, input_image.unsqueeze(1)  # (N, 1, height, width, complex)
        )  # shape (N, coil, height, width, complex)

        mr_forward = torch.where(
            sampling_mask == 0,
            torch.tensor([0.0], dtype=masked_kspace.dtype).to(masked_kspace.device),
            self.forward_operator(mul, dim=self._spatial_dims),
        )  # shape (N, coil, height, width, complex)

        error = mr_forward - loglikelihood_scaling * torch.where(
            sampling_mask == 0,
            torch.tensor([0.0], dtype=masked_kspace.dtype).to(masked_kspace.device),
            masked_kspace,
        )  # shape (N, coil, height, width, complex)

        mr_backward = self.backward_operator(error, dim=self._spatial_dims)  # shape (N, coil, height, width, complex)

        if sensitivity_map is not None:
            out = T.complex_multiplication(T.conjugate(sensitivity_map), mr_backward).sum(self._coil_dim)
        else:
            out = mr_backward.sum(self._coil_dim)
        # out has shape (N, complex=2, height, width)

        out = out.permute(0, 3, 1, 2)  # complex first: shape (N, height, width, complex=2)

        return out


class RIMInit(nn.Module):
    """Learned initializer for RIM, based on multi-scale context aggregation with dilated convolutions, that replaces
    zero initializer for the RIM hidden vector. Inspired by [1]_.

    References
    ----------

    .. [1] Yu, Fisher, and Vladlen Koltun. “Multi-Scale Context Aggregation by Dilated Convolutions.” ArXiv:1511.07122 [Cs], Apr. 2016. arXiv.org, http://arxiv.org/abs/1511.07122.
    """

    def __init__(
        self,
        x_ch: int,
        out_ch: int,
        channels: Tuple[int, ...],
        dilations: Tuple[int, ...],
        depth: int = 2,
        multiscale_depth: int = 1,
    ):
        """Inits :class:`RIMInit`.

        Parameters
        ----------
        x_ch: int
            Input channels.
        out_ch: int
            Number of hidden channels in the RIM.
        channels: tuple
            Channels in the convolutional layers of initializer. Typical it could be e.g. (32, 32, 64, 64).
        dilations: tuple
            Dilations of the convolutional layers of the initializer. Typically it could be e.g. (1, 1, 2, 4).
        depth: int
            RIM depth
        multiscale_depth: 1
            Number of feature layers to aggregate for the output, if 1, multi-scale context aggregation is disabled.
        """
        super().__init__()

        self.conv_blocks = nn.ModuleList()
        self.out_blocks = nn.ModuleList()
        self.depth = depth
        self.multiscale_depth = multiscale_depth
        tch = x_ch
        for curr_channels, curr_dilations in zip(channels, dilations):
            block = [
                nn.ReplicationPad2d(curr_dilations),
                nn.Conv2d(tch, curr_channels, 3, padding=0, dilation=curr_dilations),
            ]
            tch = curr_channels
            self.conv_blocks.append(nn.Sequential(*block))
        tch = np.sum(channels[-multiscale_depth:])
        for _ in range(depth):
            block = [nn.Conv2d(tch, out_ch, 1, padding=0)]
            self.out_blocks.append(nn.Sequential(*block))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = []
        for block in self.conv_blocks:
            x = F.relu(block(x), inplace=True)
            if self.multiscale_depth > 1:
                features.append(x)
        if self.multiscale_depth > 1:
            x = torch.cat(features[-self.multiscale_depth :], dim=1)
        output_list = []
        for block in self.out_blocks:
            y = F.relu(block(x), inplace=True)
            output_list.append(y)
        out = torch.stack(output_list, dim=-1)
        return out


class RIM(nn.Module):
    """Recurrent Inference Machine Module as in [1]_.

    References
    ----------

    .. [1] Putzky, Patrick, and Max Welling. “Recurrent Inference Machines for Solving Inverse Problems.” ArXiv:1706.04008 [Cs], June 2017. arXiv.org, http://arxiv.org/abs/1706.04008.
    """

    def __init__(
        self,
        forward_operator: Callable,
        backward_operator: Callable,
        hidden_channels: int,
        x_channels: int = 2,
        length: int = 8,
        depth: int = 1,
        no_parameter_sharing: bool = True,
        instance_norm: bool = False,
        dense_connect: bool = False,
        skip_connections: bool = True,
        replication_padding: bool = True,
        image_initialization: str = "zero_filled",
        learned_initializer: bool = False,
        initializer_channels: Optional[Tuple[int, ...]] = (32, 32, 64, 64),
        initializer_dilations: Optional[Tuple[int, ...]] = (1, 1, 2, 4),
        initializer_multiscale: int = 1,
        normalized: bool = False,
        **kwargs,
    ):
        """Inits :class:`RIM`.

        Parameters
        ----------
        forward_operator: Callable
            Forward Operator.
        backward_operator: Callable
            Backward Operator.
        hidden_channels: int
            Number of hidden channels in recurrent unit of RIM.
        x_channels: int
            Number of input channels. Default: 2 (complex data).
        length: int
            Number of time-steps. Default: 8.
        depth: int
            Number of layers of recurrent unit of RIM. Default: 1.
        no_parameter_sharing: bool
            If False, a single recurrent unit will be used for each time-step. Default: True.
        instance_norm: bool
            If True, instance normalization is applied in the recurrent unit of RIM. Default: False.
        dense_connect: bool
            Use dense connection in the recurrent unit of RIM. Default: False.
        skip_connections: bool
            If True, the previous prediction is added to the next. Default: True.
        replication_padding: bool
            Replication padding for the recurrent unit of RIM. Defaul: True.
        image_initialization: str
            Input image initialization for RIM. Can be "sense", "input_kspace", "input_image" or "zero_filled". Default: "zero_filled".
        learned_initializer: bool
            If True, an initializer is trained to learn image initialization. Default: False.
        initializer_channels: Optional[Tuple[int, ...]]
            Number of channels for learned_initializer. If "learned_initializer=False" this is ignored. Default: (32, 32, 64, 64).
        initializer_dilations: Optional[Tuple[int, ...]]
            Number of dilations for learned_initializer. Must have the same length as "initialize_channels".
            If "learned_initializer=False" this is ignored. Default: (1, 1, 2, 4)
        initializer_multiscale: int
            Number of initializer multiscale. If "learned_initializer=False" this is ignored. Default: 1.
        normalized: bool
            If True, :class:`NormConv2dGRU` will be used instead of :class:`Conv2dGRU`. Default: False.
        """
        super().__init__()

        extra_keys = kwargs.keys()
        for extra_key in extra_keys:
            if extra_key not in [
                "steps",
                "sensitivity_map_model",
                "model_name",
                "z_reduction_frequency",
                "kspace_context",
                "scale_loglikelihood",
                "whiten_input",  # should be passed!
            ]:
                raise ValueError(f"{type(self).__name__} got key `{extra_key}` which is not supported.")

        assert_positive_integer(x_channels, hidden_channels, length, depth)
        # assert_bool(no_parameter_sharing, instance_norm, dense_connect, skip_connections, replication_padding)

        self.initializer: Optional[nn.Module] = None
        if learned_initializer and initializer_channels is not None and initializer_dilations is not None:
            # List is because of a omegaconf bug.
            self.initializer = RIMInit(
                x_channels,
                hidden_channels,
                channels=initializer_channels,
                dilations=initializer_dilations,
                depth=depth,
                multiscale_depth=initializer_multiscale,
            )

        self.image_initialization = image_initialization

        self.forward_operator = forward_operator
        self.backward_operator = backward_operator

        self.grad_likelihood = MRILogLikelihood(forward_operator, backward_operator)

        self.skip_connections = skip_connections

        self.x_channels = x_channels
        self.hidden_channels = hidden_channels

        self.cell_list = nn.ModuleList()
        self.no_parameter_sharing = no_parameter_sharing

        conv_unit_params = {
            "in_channels": x_channels * 2,  # double channels as input is concatenated image and gradient
            "out_channels": x_channels,
            "hidden_channels": hidden_channels,
            "num_layers": depth,
            "instance_norm": instance_norm,
            "dense_connect": dense_connect,
            "replication_padding": replication_padding,
        }
        for _ in range(length if no_parameter_sharing else 1):
            self.cell_list.append(
                NormConv2dGRU(**conv_unit_params) if normalized else Conv2dGRU(**conv_unit_params)  # type: ignore
            )

        self.length = length
        self.depth = depth

        self._coil_dim = 1
        self._spatial_dims = (2, 3)

    def compute_sense_init(self, kspace: torch.Tensor, sensitivity_map: torch.Tensor) -> torch.Tensor:
        # kspace is of shape: (N, coil, height, width, complex)
        # sensitivity_map is of shape (N, coil, height, width, complex)

        input_image = T.complex_multiplication(
            T.conjugate(sensitivity_map),
            self.backward_operator(kspace, dim=self._spatial_dims),
        )  # shape (N, coil, height, width, complex=2)

        input_image = input_image.sum(self._coil_dim)

        # shape (N, height, width, complex=2)
        return input_image

    def forward(
        self,
        input_image: torch.Tensor,
        masked_kspace: torch.Tensor,
        sampling_mask: torch.Tensor,
        sensitivity_map: Optional[torch.Tensor] = None,
        previous_state: Optional[torch.Tensor] = None,
        loglikelihood_scaling: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Performs forward pass of :class:`RIM`.

        Parameters
        ----------
        input_image: torch.Tensor
            Initial or intermediate guess of input. Has shape (N, height, width, complex=2).
        masked_kspace: torch.Tensor
            Masked k-space of shape (N, coil, height, width, complex=2).
        sensitivity_map: torch.Tensor
            Sensitivity map of shape (N, coil, height, width, complex=2).
        sampling_mask: torch.Tensor
            Sampling mask of shape (N, 1, height, width, 1).
        previous_state: torch.Tensor
        loglikelihood_scaling: torch.Tensor
            Float tensor of shape (1,).

        Returns
        -------
        torch.Tensor
        """
        if input_image is None:
            if self.image_initialization == "sense":
                input_image = self.compute_sense_init(
                    kspace=masked_kspace,
                    sensitivity_map=sensitivity_map,
                )
            elif self.image_initialization == "input_kspace":
                if "initial_kspace" not in kwargs:
                    raise ValueError(
                        f"`'initial_kspace` is required as input if initialization is {self.image_initialization}."
                    )
                input_image = self.compute_sense_init(
                    kspace=kwargs["initial_kspace"],
                    sensitivity_map=sensitivity_map,
                )
            elif self.image_initialization == "input_image":
                if "initial_image" not in kwargs:
                    raise ValueError(
                        f"`'initial_image` is required as input if initialization is {self.image_initialization}."
                    )
                input_image = kwargs["initial_image"]

            elif self.image_initialization == "zero_filled":
                input_image = self.backward_operator(masked_kspace, dim=self._spatial_dims).sum(self._coil_dim)
            else:
                raise ValueError(
                    f"Unknown image_initialization. Expected `sense`, `input_kspace`, `input_image` or `zero_filled`. "
                    f"Got {self.image_initialization}."
                )
        # Provide an initialization for the first hidden state.
        if (self.initializer is not None) and (previous_state is None):
            previous_state = self.initializer(
                input_image.permute(0, 3, 1, 2)
            )  # shape (N, hidden_channels, height, width, depth)
        # TODO: This has to be made contiguous

        input_image = input_image.permute(0, 3, 1, 2).contiguous()  # shape (N, complex=2, height, width)

        batch_size = input_image.size(0)
        spatial_shape = [input_image.size(self._spatial_dims[0]), input_image.size(self._spatial_dims[1])]
        # Initialize zero state for RIM
        state_size = [batch_size, self.hidden_channels] + list(spatial_shape) + [self.depth]
        if previous_state is None:
            # shape (N, hidden_channels, height, width, depth)
            previous_state = torch.zeros(*state_size, dtype=input_image.dtype).to(input_image.device)

        cell_outputs = []
        intermediate_image = input_image  # shape (N, complex=2, height, width)

        for cell_idx in range(self.length):
            cell = self.cell_list[cell_idx] if self.no_parameter_sharing else self.cell_list[0]

            grad_loglikelihood = self.grad_likelihood(
                intermediate_image,
                masked_kspace,
                sensitivity_map,
                sampling_mask,
                loglikelihood_scaling,
            )  # shape (N, complex=2, height, width)

            if grad_loglikelihood.abs().max() > 150.0:
                warnings.warn(
                    f"Very large values for the gradient loglikelihood ({grad_loglikelihood.abs().max()}). "
                    f"Might cause difficulties."
                )

            cell_input = torch.cat(
                [intermediate_image, grad_loglikelihood],
                dim=1,
            )  # shape (N, complex=4, height, width)

            cell_output, previous_state = cell(cell_input, previous_state)
            # shapes (N, complex=2, height, width), (N, hidden_channels, height, width, depth)

            if self.skip_connections:
                # shape (N, complex=2, height, width)
                intermediate_image = intermediate_image + cell_output
            else:
                # shape (N, complex=2, height, width)
                intermediate_image = cell_output

            if not self.training:
                # If not training, memory can be significantly reduced by clearing the previous cell.
                cell_output.set_()
                grad_loglikelihood.set_()
                del cell_output, grad_loglikelihood

            # Only save intermediate reconstructions at training step
            if self.training or cell_idx == (self.length - 1):
                cell_outputs.append(intermediate_image)  # type: ignore

        return cell_outputs, previous_state
