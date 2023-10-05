# coding=utf-8
# Copyright (c) DIRECT Contributors

from typing import Any, Callable, List, Optional, Tuple, Union

import torch
from torch import nn

from direct.data.transforms import expand_operator, reduce_operator
from direct.nn.rim.rim import MRILogLikelihood


class ConvRNNStack(nn.Module):
    """
    A stack of convolutional RNNs.

    Takes as input a sequence of recurrent and convolutional layers.
    """

    def __init__(self, convs, recurrent):
        """
        Parameters
        ----------
        convs: List[torch.nn.Module]
            List of convolutional layers.
        recurrent: torch.nn.Module
            Recurrent layer.
        """
        super().__init__()
        self.convs = convs
        self.recurrent = recurrent

    def forward(self, _input, hidden):
        """
        Parameters
        ----------
        _input: torch.Tensor
            Input tensor. (batch_size, seq_len, input_size)
        hidden: torch.Tensor
            Hidden state. (num_layers * num_directions, batch_size, hidden_size)

        Returns
        -------
        output: torch.Tensor
            Output tensor. (batch_size, seq_len, hidden_size)
        """
        return self.recurrent(self.convs(_input), hidden)


class ConvNonlinear(nn.Module):
    """A convolutional layer with nonlinearity."""

    def __init__(self, input_size, features, kernel_size, dilation, bias):
        """
        Initializes the convolutional layer.

        Parameters
        ----------
        input_size: int
            Size of the input.
        features: int
            Number of features.
        kernel_size: int
            Size of the kernel.
        dilation: int
            Dilation of the kernel.
        bias: bool
            Whether to use bias.
        """
        super().__init__()

        self.padding = torch.nn.ReplicationPad2d(
            torch.div(dilation * (kernel_size - 1), 2, rounding_mode="trunc").item()
        )

        self.conv_layer = nn.Conv2d(
            in_channels=input_size,
            out_channels=features,
            kernel_size=kernel_size,
            padding=0,
            dilation=dilation,
            bias=bias,
        )
        self.nonlinear = torch.nn.ReLU()

        self.reset_parameters()

    def reset_parameters(self):
        """Resets the parameters of the convolutional layer."""
        torch.nn.init.kaiming_normal_(self.conv_layer.weight, nonlinearity="relu")

        if self.conv_layer.bias is not None:
            nn.init.zeros_(self.conv_layer.bias)

    def forward(self, _input):
        """
        Forward pass of the convolutional layer.

        Parameters
        ----------
        _input: torch.Tensor
            Input tensor. (batch_size, seq_len, input_size)

        Returns
        -------
        output: torch.Tensor
            Output tensor. (batch_size, seq_len, features)
        """
        return self.nonlinear(self.conv_layer(self.padding(_input)))


class IndRNNCell(nn.Module):
    """
    Base class for Independently RNN cells as presented in [1]_.

    References
    ----------

    .. [1] Li, S. et al. (2018) ‘Independently Recurrent Neural Network (IndRNN): Building A Longer and Deeper RNN’,
        Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition, (1),
        pp. 5457–5466. doi: 10.1109/CVPR.2018.00572.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int = 1,
        dilation: int = 1,
        bias: bool = True,
    ):
        """

        Parameters
        ----------
        in_channels : int
            Number of input channels
        hidden_channels : int
            Number of hidden channels
        kernel_size : int
            Kernel size. Default: 1.
        dilation : int
            Dilation size. Default: 1.
        bias : bool
            Whether to use bias. Default: True.
        """

        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.bias = bias

        self.ih = nn.Conv2d(
            in_channels,
            hidden_channels,
            kernel_size,
            padding=torch.div(dilation * (kernel_size - 1), 2, rounding_mode="trunc").item(),
            dilation=dilation,
            bias=bias,
        )
        self.hh = nn.Parameter(
            nn.init.normal_(
                torch.empty(1, hidden_channels, 1, 1), std=1.0 / (hidden_channels * (1 + kernel_size**2))
            )
        )

        self.reset_parameters()

    def reset_parameters(self):
        """Reset the parameters."""
        self.ih.weight.data = self.orthotogonalize_weights(self.ih.weight.data)

        nn.init.normal_(self.ih.weight, std=1.0 / (self.hidden_channels * (1 + self.kernel_size**2)))

        if self.bias is True:
            nn.init.zeros_(self.ih.bias)

    @staticmethod
    def orthotogonalize_weights(weights, chunks=1):
        """
        Orthogonalize weights.

        Parameters
        ----------
        weights: torch.Tensor
            The weights to orthogonalize.
        chunks: int
            Number of chunks. Default: 1.

        Returns
        -------
        weights: torch.Tensor
            The orthogonalized weights.
        """
        return torch.cat([nn.init.orthogonal_(w) for w in weights.chunk(chunks, 0)], 0)

    def forward(self, _input, hx):
        """
        Forward pass of the cell.

        Parameters
        ----------
        _input: torch.Tensor
            Input tensor. (batch_size, seq_len, input_size), tensor containing input features.
        hx: torch.Tensor
            Hidden state. (batch_size, hidden_channels, 1, 1), tensor containing hidden state features.

        Returns
        -------
        output: torch.Tensor
            Output tensor. (batch_size, seq_len, hidden_channels), tensor containing the next hidden state.
        """
        return nn.ReLU()(self.ih(_input) + self.hh * hx)


class CIRIM(nn.Module):
    """
    Cascades of Independently Recurrent Inference Machines implementation as presented in [1]_.

    References
    ----------

    .. [1] Karkalousos, D. et al. (2021) ‘Assessment of Data Consistency through Cascades of Independently Recurrent
        Inference Machines for fast and robust accelerated MRI reconstruction’.
        Available at: https://arxiv.org/abs/2111.15498v1
    """

    def __init__(
        self,
        forward_operator: Callable,
        backward_operator: Callable,
        depth: int = 2,
        in_channels: int = 2,
        time_steps: int = 8,
        recurrent_hidden_channels: int = 64,
        num_cascades: int = 8,
        no_parameter_sharing: bool = True,
        **kwargs,
    ):
        """

        Parameters
        ----------
        forward_operator : Callable
            Forward Operator.
        backward_operator : Callable
            Backward Operator.
        depth: int
            Number of layers.
        time_steps : int
            Number of iterations :math:`T`.
        in_channels : int
            Input channel number. Default is 2 for complex data.
        recurrent_hidden_channels : int
            Hidden channels number for the recurrent unit of the CIRIM Blocks. Default: 64.
        recurrent_num_layers : int
            Number of layers for the recurrent unit of the CIRIM Block (:math:`n_l`). Default: 4.
        no_parameter_sharing : bool
            If False, the same CIRIM Block is used for all time_steps. Default: True.
        """
        super().__init__()

        extra_keys = kwargs.keys()
        for extra_key in extra_keys:
            if extra_key not in [
                "model_name",
            ]:
                raise ValueError(f"{type(self).__name__} got key `{extra_key}` which is not supported.")

        self.forward_operator = forward_operator
        self.backward_operator = backward_operator
        self.no_parameter_sharing = no_parameter_sharing

        # Create cascades of Recurrent Inference Machines blocks
        self.block_list = nn.ModuleList(
            [
                RIMBlock(
                    forward_operator=self.forward_operator,
                    backward_operator=self.backward_operator,
                    depth=depth,
                    in_channels=in_channels,
                    hidden_channels=recurrent_hidden_channels,
                    time_steps=time_steps,
                    no_parameter_sharing=self.no_parameter_sharing,
                )
                for _ in range(num_cascades)
            ]
        )

        self._coil_dim = 1
        self._spatial_dims = (2, 3)

        self.dc_weight = nn.Parameter(torch.ones(1))

    def forward(
        self,
        masked_kspace: torch.Tensor,
        sampling_mask: torch.Tensor,
        sensitivity_map: torch.Tensor,
    ) -> List[List[Union[torch.Tensor, Any]]]:
        """
        Parameters
        ----------
        masked_kspace : torch.Tensor
            Masked k-space of shape (N, coil, height, width, complex=2).
        sampling_mask : torch.Tensor
            Sampling mask of shape (N, 1, height, width, 1).
        sensitivity_map : torch.Tensor
            Coil sensitivities of shape (N, coil, height, width, complex=2).

        Returns
        -------
        imspace_prediction: torch.Tensor
            imspace prediction.
        """
        previous_state: Optional[torch.Tensor] = None
        current_prediction = masked_kspace.clone()

        cascades_etas = []
        for i, cascade in enumerate(self.block_list):
            # Forward pass through the cascades
            current_prediction, previous_state = cascade(
                current_prediction,
                masked_kspace,
                sampling_mask,
                sensitivity_map,
                previous_state,
                parameter_sharing=False if i == 0 else self.no_parameter_sharing,
                coil_dim=self._coil_dim,
                spatial_dims=self._spatial_dims,
            )

            if self.no_parameter_sharing:
                _current_prediction = [torch.abs(torch.view_as_complex(x)) for x in current_prediction]
            else:
                _current_prediction = [
                    torch.abs(
                        torch.view_as_complex(
                            reduce_operator(
                                self.backward_operator(x, dim=self._spatial_dims), sensitivity_map, self._coil_dim
                            )
                        )
                    )
                    for x in current_prediction
                ]

            # Compute the prediction for the current cascade
            cascades_etas.append(_current_prediction)

        yield cascades_etas


class RIMBlock(nn.Module):
    """
    Recurrent Inference Machines block as presented in [1]_.

    References
    ----------

    .. [1] Karkalousos, D. et al. (2021) ‘Assessment of Data Consistency through Cascades of Independently
        Recurrent Inference Machines for fast and robust accelerated MRI reconstruction’.
        Available at: https://arxiv.org/abs/2111.15498v1
    """

    def __init__(
        self,
        forward_operator: Callable,
        backward_operator: Callable,
        depth: int = 2,
        in_channels: int = 2,
        hidden_channels: int = 64,
        time_steps: int = 4,
        no_parameter_sharing: bool = False,
    ):
        """
        Parameters
        ----------
        forward_operator: Callable
            Forward Fourier Transform.
        backward_operator: Callable
            Backward Fourier Transform.
        depth: int
            Number of layers in the RIM block.
        in_channels: int,
            Input channel number. Default is 2 for complex data.
        hidden_channels: int,
            Hidden channels. Default: 64.
        time_steps: int,
            Number of layers of :math:`n_l` recurrent unit. Default: 4.
        data_consistency: bool,
            If False, the DC component is removed from the input.
        """
        super().__init__()

        self.input_size = in_channels * 2
        self.hidden_channels = hidden_channels
        self.depth = depth
        self.time_steps = time_steps

        # Create Recurrent Inference Machines
        self.layers = nn.ModuleList()
        for i in range(depth):
            conv_layer = None
            if i != depth:
                conv_layer = ConvNonlinear(
                    self.input_size,
                    hidden_channels,
                    kernel_size=5 if i == 0 else 3,
                    dilation=2 if i == 1 else 1,
                    bias=True,
                )
                self.input_size = hidden_channels
            if i != depth:
                rnn_layer = IndRNNCell(
                    self.input_size,
                    hidden_channels,
                    kernel_size=1,
                    dilation=1,
                    bias=True,
                )
                self.input_size = hidden_channels
                self.layers.append(ConvRNNStack(conv_layer, rnn_layer))

        self.final_layer = torch.nn.Sequential(
            ConvNonlinear(
                self.input_size,
                2,
                kernel_size=3,
                dilation=1,
                bias=False,
            )
        )

        self.no_parameter_sharing = no_parameter_sharing

        if not self.no_parameter_sharing:
            self.dc_weight = nn.Parameter(torch.ones(1))

        self.forward_operator = forward_operator
        self.backward_operator = backward_operator

        # Initialize the log-likelihood gradient
        self.grad_likelihood = MRILogLikelihood(self.forward_operator, self.backward_operator)

    def forward(
        self,
        current_prediction: torch.Tensor,
        masked_kspace: torch.Tensor,
        sampling_mask: torch.Tensor,
        sensitivity_map: torch.Tensor,
        hidden_state: Union[None, torch.Tensor],
        parameter_sharing: bool = False,
        coil_dim: int = 1,
        spatial_dims: Tuple[int, int] = (2, 3),
    ) -> Union[Tuple[List, None], Tuple[List, Union[List, torch.Tensor]]]:
        """
        Parameters
        ----------
        current_prediction : torch.Tensor
            Current k-space.
        masked_kspace : torch.Tensor
            Masked k-space of shape (N, coil, height, width, complex=2).
        sampling_mask : torch.Tensor
            Sampling mask of shape (N, 1, height, width, 1).
        sensitivity_map : torch.Tensor
            Coil sensitivities of shape (N, coil, height, width, complex=2).
        hidden_state: torch.Tensor or None
            IndRNN hidden state of shape (N, hidden_channels, height, width, num_layers) if not None. Optional.
        parameter_sharing: bool
            If True, the weights of the convolutional layers are shared between the forward and backward pass.
        coil_dim: int
            Coil dimension. Default: 1.
        spatial_dims: tuple of ints
            Spatial dimensions. Default: (2, 3).

        Returns
        -------
        new_kspace: torch.Tensor
            New k-space prediction of shape (N, coil, height, width, complex=2).
        hidden_state: torch.Tensor or None
            Next hidden state of shape (N, hidden_channels, height, width, num_layers) if parameter_sharing else None.
        """
        # Initialize the hidden states
        if hidden_state is None:
            hidden_state = [
                masked_kspace.new_zeros((masked_kspace.size(0), self.hidden_channels, *masked_kspace.size()[2:-1]))
                for _ in range(self.depth)
            ]

        # Initialize the k-space prediction to the last time-step of the current prediction
        if isinstance(current_prediction, list):
            current_prediction = current_prediction[-1].detach()

        # Compute the current estimation
        intermediate_image = (
            reduce_operator(self.backward_operator(current_prediction, dim=spatial_dims), sensitivity_map, coil_dim)
            if not parameter_sharing
            else current_prediction
        )
        intermediate_image = intermediate_image.permute(0, 3, 1, 2)

        # Iterate over the time-steps
        intermediate_images = []
        for _ in range(self.time_steps):
            # Compute the log-likelihood gradient
            llg = self.grad_likelihood(intermediate_image, masked_kspace, sensitivity_map, sampling_mask)
            llg_eta = torch.cat([llg, intermediate_image], dim=coil_dim).contiguous()

            for hs, convrnn in enumerate(self.layers):
                # Compute the hidden state
                hidden_state[hs] = convrnn(llg_eta, hidden_state[hs])
                # Compute the next intermediate image
                llg_eta = hidden_state[hs]
            # Compute the estimation of the last time-step
            llg_eta = self.final_layer(llg_eta)
            # Accumulate the intermediate images with the log-likelihood gradient
            llg_eta = (intermediate_image + llg_eta).permute(0, 2, 3, 1)
            intermediate_images.append(llg_eta)

        if self.no_parameter_sharing:
            # Return the estimation on image space
            return intermediate_images, None

        # Compute the soft data consistency term
        soft_dc = torch.where(
            sampling_mask == 0,
            torch.tensor([0.0], dtype=masked_kspace.dtype).to(masked_kspace.device),
            current_prediction - masked_kspace,
        )

        # Return the estimation on kspace space and the hidden state
        current_kspace = [
            masked_kspace
            - soft_dc
            - self.forward_operator(expand_operator(x, sensitivity_map, dim=coil_dim), dim=spatial_dims)
            for x in intermediate_images
        ]

        return current_kspace, hidden_state
