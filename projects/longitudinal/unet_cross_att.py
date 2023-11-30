# coding=utf-8
# Copyright (c) DIRECT Contributors

# Code borrowed / edited from: https://github.com/facebookresearch/fastMRI/blob/
import math
from typing import Callable, List, Optional, Tuple

import torch
from torch import nn, Tensor
from torch.nn import functional as F, MultiheadAttention

from src.utils.direct.data import transforms as T
class MHA_2d(nn.Module):
    def __init__(self,embed_dim=4, num_heads=4, batch_first=True,kernel=8):
        super(MHA_2d, self).__init__()
        self.embed_dim=embed_dim
        self.kernel = kernel
        self.slice_conv1 = nn.Conv2d(embed_dim, embed_dim*8, kernel_size=kernel, stride=kernel, padding=0)
        self.slice_conv2 = nn.Conv2d(1, embed_dim*8, kernel_size=kernel, stride=kernel, padding=0)
        self.att = MultiheadAttention(embed_dim=embed_dim*8, num_heads=num_heads, batch_first=batch_first)
        self.proj = nn.ConvTranspose2d(embed_dim*8, embed_dim, kernel_size=kernel, stride=kernel, padding=0)
        self.pos = PositionalEncoding(d_model=embed_dim*8)
    def forward(self,x,y):
        x, pad = pad_to_nearest_multiple(x,self.kernel)
        y , _= pad_to_nearest_multiple(y,self.kernel)
        x = self.slice_conv1(x)
        b,c, h,w = x.shape
        x = x.view(x.shape[0], self.embed_dim*8, -1).transpose(1, 2)
        y = self.slice_conv2(y).view(y.shape[0], self.embed_dim*8, -1).transpose(1, 2)
        x = self.pos(x)
        y = self.pos(y)
        x, _ = self.att(x,y,y)
        x = self.proj(x.transpose(1, 2).view(-1, self.embed_dim*8,  h, w))
        return x[:,:,:-pad[0],:-pad[1]]


def pad_to_nearest_multiple(tensor, Z):
    """
    Pads the input tensor to the nearest multiple of Z.

    :param tensor: Input tensor of shape (b, c, h, w)
    :param Z: The number to which height and width should be padded
    :return: Padded tensor
    """
    _, _, h, w = tensor.shape

    # Calculate the padding needed to reach the nearest multiple of Z
    h_padding = (Z - (h % Z)) % Z
    w_padding = (Z - (w % Z)) % Z

    # Apply symmetric padding (dividing the padding equally on both sides)
    # The format for pad is (left, right, top, bottom)
    padded_tensor = torch.nn.functional.pad(tensor, (w_padding // 2, w_padding - w_padding // 2, h_padding // 2, h_padding - h_padding // 2), mode='constant', value=0)

    return padded_tensor,(h_padding,w_padding)



class ConvBlock(nn.Module):
    """U-Net convolutional block.

    It consists of two convolution layers each followed by instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, in_channels: int, out_channels: int, dropout_probability: float):
        """Inits ConvBlock.

        Parameters
        ----------
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
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(dropout_probability),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(dropout_probability),
        )

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass of :class:`ConvBlock`.

        Parameters
        ----------
        input_data: torch.Tensor

        Returns
        -------
        torch.Tensor
        """
        return self.layers(input_data)

    def __repr__(self):
        """Representation of :class:`ConvBlock`."""
        return (
            f"ConvBlock(in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"dropout_probability={self.dropout_probability})"
        )


class TransposeConvBlock(nn.Module):
    """U-Net Transpose Convolutional Block.

    It consists of one convolution transpose layers followed by instance normalization and LeakyReLU activation.
    """

    def __init__(self, in_channels: int, out_channels: int):
        """Inits :class:`TransposeConvBlock`.

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
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Performs forward pass of :class:`TransposeConvBlock`.

        Parameters
        ----------
        input_data: torch.Tensor

        Returns
        -------
        torch.Tensor
        """
        return self.layers(input_data)

    def __repr__(self):
        """Representation of "class:`TransposeConvBlock`."""
        return f"ConvBlock(in_channels={self.in_channels}, out_channels={self.out_channels})"

class UnetModel2d_att(nn.Module):
    """PyTorch implementation of a U-Net model based on [1]_.

    References
    ----------

    .. [1] Ronneberger, Olaf, et al. “U-Net: Convolutional Networks for Biomedical Image Segmentation.” Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015, edited by Nassir Navab et al., Springer International Publishing, 2015, pp. 234–41. Springer Link, https://doi.org/10.1007/978-3-319-24574-4_28.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_filters: int,
        num_pool_layers: int,
        dropout_probability: float,
    ):
        """Inits :class:`UnetModel2d`.

        Parameters
        ----------
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

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_channels, num_filters, dropout_probability)])
        ch = num_filters
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvBlock(ch, ch * 2, dropout_probability)]
            ch *= 2
        self.conv = ConvBlock(ch, ch * 2, dropout_probability)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv += [TransposeConvBlock(ch * 2, ch)]
            self.up_conv += [ConvBlock(ch * 2, ch, dropout_probability)]
            ch //= 2

        self.up_transpose_conv += [TransposeConvBlock(ch * 2, ch)]
        self.up_conv += [
            nn.Sequential(
                ConvBlock(ch * 2, ch, dropout_probability),
                nn.Conv2d(ch, self.out_channels, kernel_size=1, stride=1),
            )
        ]

        self.att_layers = nn.ModuleList([])

        ch = num_filters
        for _ in range(num_pool_layers):
            self.att_layers += [MHA_2d(embed_dim=ch, num_heads=8, batch_first=True,kernel=16)]
            ch*=2



    def forward(self, input_data: torch.Tensor, att_data: torch.Tensor) -> torch.Tensor:
        """Performs forward pass of :class:`UnetModel2d`.

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
        for layer,att in zip(self.down_sample_layers,self.att_layers):
            output = layer(output)
            att_output = att(output,att_data)
            att_output = output+att_output
            stack.append(att_output)
            output = F.avg_pool2d(att_output, kernel_size=2, stride=2, padding=0)

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

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
