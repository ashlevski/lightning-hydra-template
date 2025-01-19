# coding=utf-8
# Copyright (c) DIRECT Contributors

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2dGRU(nn.Module):
    """2D Convolutional GRU Network."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: Optional[int] = None,
        num_layers: int = 2,
        gru_kernel_size=1,
        orthogonal_initialization: bool = True,
        instance_norm: bool = False,
        dense_connect: int = 0,
        replication_padding: bool = True,
    ):
        """Inits :class:`Conv2dGRU`.

        Parameters
        ----------
        in_channels: int
            Number of input channels.
        hidden_channels: int
            Number of hidden channels.
        out_channels: Optional[int]
            Number of output channels. If None, same as in_channels. Default: None.
        num_layers: int
            Number of layers. Default: 2.
        gru_kernel_size: int
            Size of the GRU kernel. Default: 1.
        orthogonal_initialization: bool
            Orthogonal initialization is used if set to True. Default: True.
        instance_norm: bool
            Instance norm is used if set to True. Default: False.
        dense_connect: int
            Number of dense connections.
        replication_padding: bool
            If set to true replication padding is applied.
        """
        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.dense_connect = dense_connect

        self.reset_gates = nn.ModuleList([])
        self.update_gates = nn.ModuleList([])
        self.out_gates = nn.ModuleList([])
        self.conv_blocks = nn.ModuleList([])

        # Create convolutional blocks
        for idx in range(num_layers + 1):
            in_ch = in_channels if idx == 0 else (1 + min(idx, dense_connect)) * hidden_channels
            out_ch = hidden_channels if idx < num_layers else out_channels
            padding = 0 if replication_padding else (2 if idx == 0 else 1)
            block: List[nn.Module] = []
            if replication_padding:
                if idx == 1:
                    block.append(nn.ReplicationPad2d(2))
                else:
                    block.append(nn.ReplicationPad2d(2 if idx == 0 else 1))
            block.append(
                nn.Conv2d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=5 if idx == 0 else 3,
                    dilation=(2 if idx == 1 else 1),
                    padding=padding,
                )
            )
            self.conv_blocks.append(nn.Sequential(*block))

        # Create GRU blocks
        for idx in range(num_layers):
            for gru_part in [self.reset_gates, self.update_gates, self.out_gates]:
                gru_block: List[nn.Module] = []
                if instance_norm:
                    gru_block.append(nn.InstanceNorm2d(2 * hidden_channels))
                gru_block.append(
                    nn.Conv2d(
                        in_channels=2 * hidden_channels,
                        out_channels=hidden_channels,
                        kernel_size=gru_kernel_size,
                        padding=gru_kernel_size // 2,
                    )
                )
                gru_part.append(nn.Sequential(*gru_block))

        if orthogonal_initialization:
            for reset_gate, update_gate, out_gate in zip(self.reset_gates, self.update_gates, self.out_gates):
                nn.init.orthogonal_(reset_gate[-1].weight)
                nn.init.orthogonal_(update_gate[-1].weight)
                nn.init.orthogonal_(out_gate[-1].weight)
                nn.init.constant_(reset_gate[-1].bias, -1.0)
                nn.init.constant_(update_gate[-1].bias, 0.0)
                nn.init.constant_(out_gate[-1].bias, 0.0)

    def forward(
        self,
        cell_input: torch.Tensor,
        previous_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes Conv2dGRU forward pass given tensors `cell_input` and `previous_state`.

        Parameters
        ----------
        cell_input: torch.Tensor
            Input tensor.
        previous_state: torch.Tensor
            Tensor of previous hidden state.

        Returns
        -------
        out, new_states: (torch.Tensor, torch.Tensor)
            Output and new states.
        """
        new_states: List[torch.Tensor] = []
        conv_skip: List[torch.Tensor] = []

        if previous_state is None:
            batch_size, spatial_size = cell_input.size(0), (cell_input.size(2), cell_input.size(3))
            state_size = [batch_size, self.hidden_channels] + list(spatial_size) + [self.num_layers]
            previous_state = torch.zeros(*state_size, dtype=cell_input.dtype).to(cell_input.device)

        for idx in range(self.num_layers):
            if len(conv_skip) > 0:
                cell_input = F.relu(
                    self.conv_blocks[idx](torch.cat([*conv_skip[-self.dense_connect :], cell_input], dim=1)),
                    inplace=True,
                )
            else:
                cell_input = F.relu(self.conv_blocks[idx](cell_input), inplace=True)
            if self.dense_connect > 0:
                conv_skip.append(cell_input)

            stacked_inputs = torch.cat([cell_input, previous_state[:, :, :, :, idx]], dim=1)

            update = torch.sigmoid(self.update_gates[idx](stacked_inputs))
            reset = torch.sigmoid(self.reset_gates[idx](stacked_inputs))
            delta = torch.tanh(
                self.out_gates[idx](torch.cat([cell_input, previous_state[:, :, :, :, idx] * reset], dim=1))
            )
            cell_input = previous_state[:, :, :, :, idx] * (1 - update) + delta * update
            new_states.append(cell_input)
            cell_input = F.relu(cell_input, inplace=False)
        if len(conv_skip) > 0:
            out = self.conv_blocks[self.num_layers](torch.cat([*conv_skip[-self.dense_connect :], cell_input], dim=1))
        else:
            out = self.conv_blocks[self.num_layers](cell_input)

        return out, torch.stack(new_states, dim=-1)


class NormConv2dGRU(nn.Module):
    """Normalized 2D Convolutional GRU Network.

    Normalization methods adapted from NormUnet of [1]_.

    References
    ----------

    .. [1] https://github.com/facebookresearch/fastMRI/blob/
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: Optional[int] = None,
        num_layers: int = 2,
        gru_kernel_size=1,
        orthogonal_initialization: bool = True,
        instance_norm: bool = False,
        dense_connect: int = 0,
        replication_padding: bool = True,
        norm_groups: int = 2,
    ):
        """Inits :class:`NormConv2dGRU`.

        Parameters
        ----------
        in_channels: int
            Number of input channels.
        hidden_channels: int
            Number of hidden channels.
        out_channels: Optional[int]
            Number of output channels. If None, same as in_channels. Default: None.
        num_layers: int
            Number of layers. Default: 2.
        gru_kernel_size: int
            Size of the GRU kernel. Default: 1.
        orthogonal_initialization: bool
            Orthogonal initialization is used if set to True. Default: True.
        instance_norm: bool
            Instance norm is used if set to True. Default: False.
        dense_connect: int
            Number of dense connections.
        replication_padding: bool
            If set to true replication padding is applied.
        norm_groups: int,
            Number of normalization groups.
        """
        super().__init__()
        self.convgru = Conv2dGRU(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            gru_kernel_size=gru_kernel_size,
            orthogonal_initialization=orthogonal_initialization,
            instance_norm=instance_norm,
            dense_connect=dense_connect,
            replication_padding=replication_padding,
        )
        self.norm_groups = norm_groups

    @staticmethod
    def norm(input_data: torch.Tensor, num_groups: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs group normalization."""
        b, c, h, w = input_data.shape
        input_data = input_data.reshape(b, num_groups, -1)

        mean = input_data.mean(-1, keepdim=True)
        std = input_data.std(-1, keepdim=True)

        output = (input_data - mean) / std
        output = output.reshape(b, c, h, w)

        return output, mean, std

    @staticmethod
    def unnorm(input_data: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, num_groups: int) -> torch.Tensor:
        b, c, h, w = input_data.shape
        input_data = input_data.reshape(b, num_groups, -1)
        return (input_data * std + mean).reshape(b, c, h, w)

    def forward(
        self,
        cell_input: torch.Tensor,
        previous_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes :class:`NormConv2dGRU` forward pass given tensors `cell_input` and `previous_state`.

        It performs group normalization on the input before the forward pass to the Conv2dGRU.
        Output of Conv2dGRU is then un-normalized.

        Parameters
        ----------
        cell_input: torch.Tensor
            Input tensor.
        previous_state: torch.Tensor
            Tensor of previous hidden state.

        Returns
        -------
        out, new_states: (torch.Tensor, torch.Tensor)
            Output and new states.

        """
        # Normalize
        cell_input, mean, std = self.norm(cell_input, self.norm_groups)
        # Pass normalized input
        cell_input, previous_state = self.convgru(cell_input, previous_state)
        # Unnormalize output
        cell_input = self.unnorm(cell_input, mean, std, self.norm_groups)

        return cell_input, previous_state
