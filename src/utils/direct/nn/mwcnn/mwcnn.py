# coding=utf-8
# Copyright (c) DIRECT Contributors

from collections import OrderedDict
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DWT(nn.Module):
    """2D Discrete Wavelet Transform as implemented in [1]_.

    References
    ----------

    .. [1] Liu, Pengju, et al. “Multi-Level Wavelet-CNN for Image Restoration.” ArXiv:1805.07071 [Cs], May 2018. arXiv.org, http://arxiv.org/abs/1805.07071.
    """

    def __init__(self):
        """Inits :class:`DWT`."""
        super().__init__()
        self.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes DWT(`x`) given tensor `x`.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor.

        Returns
        -------
        out: torch.Tensor
            DWT of `x`.
        """
        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]
        x_LL = x1 + x2 + x3 + x4
        x_HL = -x1 - x2 + x3 + x4
        x_LH = -x1 + x2 - x3 + x4
        x_HH = x1 - x2 - x3 + x4

        return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)


class IWT(nn.Module):
    """2D Inverse Wavelet Transform as implemented in [1]_.

    References
    ----------

    .. [1] Liu, Pengju, et al. “Multi-Level Wavelet-CNN for Image Restoration.” ArXiv:1805.07071 [Cs], May 2018. arXiv.org, http://arxiv.org/abs/1805.07071.
    """

    def __init__(self):
        """Inits :class:`IWT`."""
        super().__init__()
        self.requires_grad = False
        self._r = 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes IWT(`x`) given tensor `x`.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor.

        Returns
        -------
        h: torch.Tensor
            IWT of `x`.
        """
        batch, in_channel, in_height, in_width = x.size()
        out_channel, out_height, out_width = int(in_channel / (self._r**2)), self._r * in_height, self._r * in_width

        x1 = x[:, 0:out_channel, :, :] / 2
        x2 = x[:, out_channel : out_channel * 2, :, :] / 2
        x3 = x[:, out_channel * 2 : out_channel * 3, :, :] / 2
        x4 = x[:, out_channel * 3 : out_channel * 4, :, :] / 2

        h = torch.zeros([batch, out_channel, out_height, out_width], dtype=x.dtype).to(x.device)

        h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
        h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
        h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
        h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

        return h


class ConvBlock(nn.Module):
    """Convolution Block for :class:`MWCNN` as implemented in [1]_.

    References
    ----------

    .. [1] Liu, Pengju, et al. “Multi-Level Wavelet-CNN for Image Restoration.” ArXiv:1805.07071 [Cs], May 2018. arXiv.org, http://arxiv.org/abs/1805.07071.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        bias: bool = True,
        batchnorm: bool = False,
        activation: nn.Module = nn.ReLU(True),
        scale: Optional[float] = 1.0,
    ):
        """Inits :class:`ConvBlock`.

        Parameters
        ----------
        in_channels: int
            Number of input channels.
        out_channels: int
            Number of output channels.
        kernel_size: int
            Conv kernel size.
        bias: bool
            Use convolution bias. Default: True.
        batchnorm: bool
            Use batch normalization. Default: False.
        activation: nn.Module
            Activation function. Default: nn.ReLU(True).
        scale: float, optional
            Scale. Default: 1.0.
        """
        super().__init__()

        net: List[nn.Module] = []
        net.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                bias=bias,
                padding=kernel_size // 2,
            )
        )
        if batchnorm:
            net.append(nn.BatchNorm2d(num_features=out_channels, eps=1e-4, momentum=0.95))
        net.append(activation)

        self.net = nn.Sequential(*net)
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs forward pass of :class:`ConvBlock`.

        Parameters
        ----------
        x: torch.Tensor
            Input with shape (N, C, H, W).

        Returns
        -------
        output: torch.Tensor
            Output with shape (N, C', H', W').
        """
        output = self.net(x) * self.scale
        return output


class DilatedConvBlock(nn.Module):
    """Double dilated Convolution Block fpr :class:`MWCNN` as implemented in [1]_.

    References
    ----------

    .. [1] Liu, Pengju, et al. “Multi-Level Wavelet-CNN for Image Restoration.” ArXiv:1805.07071 [Cs], May 2018. arXiv.org, http://arxiv.org/abs/1805.07071.
    """

    def __init__(
        self,
        in_channels: int,
        dilations: Tuple[int, int],
        kernel_size: int,
        out_channels: Optional[int] = None,
        bias: bool = True,
        batchnorm: bool = False,
        activation: nn.Module = nn.ReLU(True),
        scale: Optional[float] = 1.0,
    ):
        """Inits :class:`DilatedConvBlock`.

        Parameters
        ----------
        in_channels: int
            Number of input channels.
        dilations: (int, int)
            Number of dilations.
        kernel_size: int
            Conv kernel size.
        out_channels: int
            Number of output channels.
        bias: bool
            Use convolution bias. Default: True.
        batchnorm: bool
            Use batch normalization. Default: False.
        activation: nn.Module
            Activation function. Default: nn.ReLU(True).
        scale: float, optional
            Scale. Default: 1.0.
        """
        super().__init__()
        net: List[nn.Module] = []
        net.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                bias=bias,
                dilation=dilations[0],
                padding=kernel_size // 2 + dilations[0] - 1,
            )
        )
        if batchnorm:
            net.append(nn.BatchNorm2d(num_features=in_channels, eps=1e-4, momentum=0.95))
        net.append(activation)
        if out_channels is None:
            out_channels = in_channels
        net.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                bias=bias,
                dilation=dilations[1],
                padding=kernel_size // 2 + dilations[1] - 1,
            )
        )
        if batchnorm:
            net.append(nn.BatchNorm2d(num_features=in_channels, eps=1e-4, momentum=0.95))
        net.append(activation)

        self.net = nn.Sequential(*net)
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs forward pass of :class:`DilatedConvBlock`.

        Parameters
        ----------
        x: torch.Tensor
            Input with shape (N, C, H, W).

        Returns
        -------
        output: torch.Tensor
            Output with shape (N, C', H', W').
        """
        output = self.net(x) * self.scale
        return output


class MWCNN(nn.Module):
    """Multi-level Wavelet CNN (MWCNN) implementation as implemented in [1]_.

    References
    ----------

    .. [1] Liu, Pengju, et al. “Multi-Level Wavelet-CNN for Image Restoration.” ArXiv:1805.07071 [Cs], May 2018. arXiv.org, http://arxiv.org/abs/1805.07071.
    """

    def __init__(
        self,
        input_channels: int,
        first_conv_hidden_channels: int,
        num_scales: int = 4,
        bias: bool = True,
        batchnorm: bool = False,
        activation: nn.Module = nn.ReLU(True),
    ):
        """Inits :class:`MWCNN`.

        Parameters
        ----------
        input_channels: int
            Input channels dimension.
        first_conv_hidden_channels: int
            First convolution output channels dimension.
        num_scales: int
            Number of scales. Default: 4.
        bias: bool
            Convolution bias. If True, adds a learnable bias to the output. Default: True.
        batchnorm: bool
            If True, a batchnorm layer is added after each convolution. Default: False.
        activation: nn.Module
            Activation function applied after each convolution. Default: nn.ReLU().
        """
        super().__init__()
        self._kernel_size = 3
        self.DWT = DWT()
        self.IWT = IWT()

        self.down = nn.ModuleList()
        for idx in range(0, num_scales):
            in_channels = input_channels if idx == 0 else first_conv_hidden_channels * 2 ** (idx + 1)
            out_channels = first_conv_hidden_channels * 2**idx
            dilations = (2, 1) if idx != num_scales - 1 else (2, 3)
            self.down.append(
                nn.Sequential(
                    OrderedDict(
                        [
                            (
                                f"convblock{idx}",
                                ConvBlock(
                                    in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=self._kernel_size,
                                    bias=bias,
                                    batchnorm=batchnorm,
                                    activation=activation,
                                ),
                            ),
                            (
                                f"dilconvblock{idx}",
                                DilatedConvBlock(
                                    in_channels=out_channels,
                                    dilations=dilations,
                                    kernel_size=self._kernel_size,
                                    bias=bias,
                                    batchnorm=batchnorm,
                                    activation=activation,
                                ),
                            ),
                        ]
                    )
                )
            )
        self.up = nn.ModuleList()
        for idx in range(num_scales)[::-1]:
            in_channels = first_conv_hidden_channels * 2**idx
            out_channels = input_channels if idx == 0 else first_conv_hidden_channels * 2 ** (idx + 1)
            dilations = (2, 1) if idx != num_scales - 1 else (3, 2)
            self.up.append(
                nn.Sequential(
                    OrderedDict(
                        [
                            (
                                f"invdilconvblock{num_scales - 2 - idx}",
                                DilatedConvBlock(
                                    in_channels=in_channels,
                                    dilations=dilations,
                                    kernel_size=self._kernel_size,
                                    bias=bias,
                                    batchnorm=batchnorm,
                                    activation=activation,
                                ),
                            ),
                            (
                                f"invconvblock{num_scales - 2 - idx}",
                                ConvBlock(
                                    in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=self._kernel_size,
                                    bias=bias,
                                    batchnorm=batchnorm,
                                    activation=activation,
                                ),
                            ),
                        ]
                    )
                )
            )
        self.num_scales = num_scales

    @staticmethod
    def pad(x: torch.Tensor) -> torch.Tensor:
        padding = [0, 0, 0, 0]

        if x.shape[-2] % 2 != 0:
            padding[3] = 1  # Padding right - width
        if x.shape[-1] % 2 != 0:
            padding[1] = 1  # Padding bottom - height
        if sum(padding) != 0:
            x = F.pad(x, padding, "reflect")
        return x

    @staticmethod
    def crop_to_shape(x: torch.Tensor, shape: tuple) -> torch.Tensor:
        h, w = x.shape[-2:]

        if h > shape[0]:
            x = x[:, :, : shape[0], :]
        if w > shape[1]:
            x = x[:, :, :, : shape[1]]
        return x

    def forward(self, input_tensor: torch.Tensor, res: bool = False) -> torch.Tensor:
        """Computes forward pass of :class:`MWCNN`.

        Parameters
        ----------
        input_tensor: torch.Tensor
            Input tensor.
        res: bool
            If True, residual connection is applied to the output. Default: False.

        Returns
        -------
        x: torch.Tensor
            Output tensor.
        """
        res_values = []
        x = self.pad(input_tensor.clone())
        for idx in range(self.num_scales):
            if idx == 0:
                x = self.pad(self.down[idx](x))
                res_values.append(x)
            elif idx == self.num_scales - 1:
                x = self.down[idx](self.DWT(x))
            else:
                x = self.pad(self.down[idx](self.DWT(x)))
                res_values.append(x)

        for idx in range(self.num_scales):
            if idx != self.num_scales - 1:
                x = (
                    self.crop_to_shape(self.IWT(self.up[idx](x)), res_values[self.num_scales - 2 - idx].shape[-2:])
                    + res_values[self.num_scales - 2 - idx]
                )
            else:
                x = self.crop_to_shape(self.up[idx](x), input_tensor.shape[-2:])
                if res:
                    x += input_tensor
        return x
