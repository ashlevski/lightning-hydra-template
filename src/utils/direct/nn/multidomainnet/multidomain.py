# coding=utf-8
# Copyright (c) DIRECT Contributors

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiDomainConv2d(nn.Module):
    def __init__(
        self,
        forward_operator: Callable,
        backward_operator: Callable,
        in_channels: int,
        out_channels: int,
        **kwargs,
    ):
        """Inits :class:`MultiDomainConv2d`.

        Parameters
        ----------
        forward_operator: Callable
            Forward Operator.
        backward_operator: Callable
            Backward Operator.
        in_channels: int
            Number of input channels.
        out_channels: int
            Number of output channels.
        """
        super().__init__()

        self.image_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels // 2, **kwargs)
        self.kspace_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels // 2, **kwargs)
        self.forward_operator = forward_operator
        self.backward_operator = backward_operator
        self._channels_dim = 1
        self._spatial_dims = (1, 2)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Performs forward pass of of :class:`MultiDomainConv2d`.

        Parameters
        ----------
        image: torch.Tensor

        Returns
        -------
        torch.Tensor
        """
        kspace = torch.cat(
            tensors=[
                self.forward_operator(
                    im,
                    dim=self._spatial_dims,
                )
                for im in torch.split(image.permute(0, 2, 3, 1).contiguous(), 2, -1)
            ],
            dim=-1,
        ).permute(0, 3, 1, 2)
        kspace = self.kspace_conv(kspace)

        backward = torch.cat(
            tensors=[
                self.backward_operator(
                    ks,
                    dim=self._spatial_dims,
                )
                for ks in torch.split(kspace.permute(0, 2, 3, 1).contiguous(), 2, -1)
            ],
            dim=-1,
        ).permute(0, 3, 1, 2)

        image = self.image_conv(image)
        image = torch.cat([image, backward], dim=self._channels_dim)
        return image


class MultiDomainConvTranspose2d(nn.Module):
    def __init__(
        self,
        forward_operator: Callable,
        backward_operator: Callable,
        in_channels: int,
        out_channels: int,
        **kwargs,
    ):
        """Inits :class:`MultiDomainConvTranspose2d`.

        Parameters
        ----------
        forward_operator: Callable
            Forward Operator.
        backward_operator: Callable
            Backward Operator.
        in_channels: int
            Number of input channels.
        out_channels: int
            Number of output channels.
        """
        super().__init__()

        self.image_conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels // 2, **kwargs)
        self.kspace_conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels // 2, **kwargs)
        self.forward_operator = forward_operator
        self.backward_operator = backward_operator
        self._channels_dim = 1
        self._spatial_dims = (1, 2)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Performs forward pass of of :class:`MultiDomainConvTranspose2d`.

        Parameters
        ----------
        image: torch.Tensor

        Returns
        -------
        torch.Tensor
        """
        kspace = torch.cat(
            tensors=[
                self.forward_operator(
                    im,
                    dim=self._spatial_dims,
                )
                for im in torch.split(image.permute(0, 2, 3, 1).contiguous(), 2, -1)
            ],
            dim=-1,
        ).permute(0, 3, 1, 2)
        kspace = self.kspace_conv(kspace)

        backward = torch.cat(
            tensors=[
                self.backward_operator(
                    ks,
                    dim=self._spatial_dims,
                )
                for ks in torch.split(kspace.permute(0, 2, 3, 1).contiguous(), 2, -1)
            ],
            dim=-1,
        ).permute(0, 3, 1, 2)

        image = self.image_conv(image)
        return torch.cat([image, backward], dim=self._channels_dim)


class MultiDomainConvBlock(nn.Module):
    """A multi-domain convolutional block that consists of two multi-domain convolution layers each followed by instance
    normalization, LeakyReLU activation and dropout."""

    def __init__(
        self,
        forward_operator: Callable,
        backward_operator: Callable,
        in_channels: int,
        out_channels: int,
        dropout_probability: float,
    ):
        """Inits :class:`MultiDomainConvBlock`.

        Parameters
        ----------
        forward_operator: Callable
            Forward Operator.
        backward_operator: Callable
            Backward Operator.
        in_channels: int
            Number of input channels.
        out_channels: int
            Number of output channels.
        dropout_probability: float
            Dropout probability.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout_probability = dropout_probability

        self.layers = nn.Sequential(
            MultiDomainConv2d(
                forward_operator, backward_operator, in_channels, out_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(dropout_probability),
            MultiDomainConv2d(
                forward_operator, backward_operator, out_channels, out_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(dropout_probability),
        )

    def forward(self, _input: torch.Tensor) -> torch.Tensor:
        """Performs forward pass of of :class:`MultiDomainConvBlock`.

        Parameters
        ----------
        _input: torch.Tensor

        Returns
        -------
        torch.Tensor
        """
        return self.layers(_input)

    def __repr__(self):
        """Representation of :class:`MultiDomainConvBlock`."""
        return (
            f"MultiDomainConvBlock(in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"dropout_probability={self.dropout_probability})"
        )


class TransposeMultiDomainConvBlock(nn.Module):
    """A Transpose Convolutional Block that consists of one convolution transpose layers followed by instance
    normalization and LeakyReLU activation."""

    def __init__(self, forward_operator, backward_operator, in_channels: int, out_channels: int):
        """
        Parameters
        ----------
        in_channels: int
            Number of input channels.
        out_channels: int
            Number of output channels.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layers = nn.Sequential(
            MultiDomainConvTranspose2d(
                forward_operator, backward_operator, in_channels, out_channels, kernel_size=2, stride=2, bias=False
            ),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, input_data: torch.Tensor):
        """

        Parameters
        ----------
        input_data: torch.Tensor

        Returns
        -------
        torch.Tensor
        """
        return self.layers(input_data)

    def __repr__(self):
        return f"MultiDomainConvBlock(in_channels={self.in_channels}, out_channels={self.out_channels})"


class MultiDomainUnet2d(nn.Module):
    """Unet modification to be used with Multi-domain network as in AIRS Medical submission to the Fast MRI 2020
    challenge."""

    def __init__(
        self,
        forward_operator: Callable,
        backward_operator: Callable,
        in_channels: int,
        out_channels: int,
        num_filters: int,
        num_pool_layers: int,
        dropout_probability: float,
    ):
        """

        Parameters
        ----------
        forward_operator: Callable
            Forward Operator.
        backward_operator: Callable
            Backward Operator.
        in_channels: int
            Number of input channels to the u-net.
        out_channels: int
            Number of output channels to the u-net.
        num_filters: int
            Number of output channels of the first convolutional layer.
        num_pool_layers: int
            Number of down-sampling and up-sampling layers (depth).
        dropout_probability: float
            Dropout probability.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_filters = num_filters
        self.num_pool_layers = num_pool_layers
        self.dropout_probability = dropout_probability

        self.down_sample_layers = nn.ModuleList(
            [MultiDomainConvBlock(forward_operator, backward_operator, in_channels, num_filters, dropout_probability)]
        )
        ch = num_filters
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers += [
                MultiDomainConvBlock(forward_operator, backward_operator, ch, ch * 2, dropout_probability)
            ]
            ch *= 2
        self.conv = MultiDomainConvBlock(forward_operator, backward_operator, ch, ch * 2, dropout_probability)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv += [TransposeMultiDomainConvBlock(forward_operator, backward_operator, ch * 2, ch)]
            self.up_conv += [
                MultiDomainConvBlock(forward_operator, backward_operator, ch * 2, ch, dropout_probability)
            ]
            ch //= 2

        self.up_transpose_conv += [TransposeMultiDomainConvBlock(forward_operator, backward_operator, ch * 2, ch)]
        self.up_conv += [
            nn.Sequential(
                MultiDomainConvBlock(forward_operator, backward_operator, ch * 2, ch, dropout_probability),
                nn.Conv2d(ch, self.out_channels, kernel_size=1, stride=1),
            )
        ]

    def forward(self, input_data: torch.Tensor):
        """

        Parameters
        ----------
        input_data: torch.Tensor

        Returns
        -------
        torch.Tensor
        """
        stack = []
        output = input_data

        # Apply down-sampling layers
        for _, layer in enumerate(self.down_sample_layers):
            output = layer(output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        output = self.conv(output)

        # Apply up-sampling layers
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            output = transpose_conv(output)

            # Reflect pad on the right/bottom if needed to handle odd input dimensions.
            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1  # Padding right
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1  # Padding bottom
            if sum(padding) != 0:
                output = F.pad(output, padding, "reflect")

            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output)

        return output
