from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, ndim, in_channels, out_channels, middle_channels=None):
        Conv = (None, nn.Conv1d, nn.Conv2d, nn.Conv3d)[ndim]
        BatchNorm = (None, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)[ndim]
        super().__init__()
        if not middle_channels:
            middle_channels = out_channels
        self.conv_block = nn.Sequential(
            Conv(in_channels, middle_channels, kernel_size=3, padding=1),
            BatchNorm(middle_channels),
            nn.ReLU(inplace=True),
            Conv(middle_channels, out_channels, kernel_size=3, padding=1),
            BatchNorm(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv_block(x)


class DownScaling(nn.Module):
    def __init__(self, ndim, in_channels, out_channels):
        Pool = (None, nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d)[ndim]
        super().__init__()
        self.pool = nn.Sequential(Pool(2), ConvBlock(ndim, in_channels, out_channels))

    def forward(self, x):
        return self.pool(x)


class UpScaling(nn.Module):
    def __init__(self, ndim, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = ConvBlock(ndim, in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff = [el2 - el1 for el1, el2 in zip(x1.shape[2:], x2.shape[2:])][
            ::-1
        ]  # flip order for correct padding
        pad = list(chain.from_iterable((d // 2, d - d // 2) for d in diff))
        x1 = F.pad(x1, pad)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, ndim, in_channels, out_channels):
        super(OutConv, self).__init__()
        Conv = (None, nn.Conv1d, nn.Conv2d, nn.Conv3d)[ndim]
        self.conv = Conv(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, ndim, initial_features=16):
        super().__init__()
        Conv = (None, nn.Conv1d, nn.Conv2d, nn.Conv3d)[ndim]
        n_channels = 2
        self.ndim = ndim
        self.inconv = ConvBlock(ndim, n_channels, initial_features)
        self.downs1 = DownScaling(ndim, initial_features, initial_features * 2)
        self.downs2 = DownScaling(ndim, initial_features * 2, initial_features * 4)
        self.downs3 = DownScaling(ndim, initial_features * 4, initial_features * 8)
        self.downs4 = DownScaling(ndim, initial_features * 8, initial_features * 8)
        self.ups1 = UpScaling(ndim, initial_features * 16, initial_features * 4)
        self.ups2 = UpScaling(ndim, initial_features * 8, initial_features * 2)
        self.ups3 = UpScaling(ndim, initial_features * 4, initial_features)
        self.ups4 = UpScaling(ndim, initial_features * 2, initial_features)
        self.outconv = Conv(initial_features, ndim, kernel_size=1)

    def forward(self, fixed, moving=None):
        x = fixed
        if moving is not None:
            x = torch.cat((fixed, moving), 1)
        x1 = self.inconv(x)
        x2 = self.downs1(x1)
        x3 = self.downs2(x2)
        x4 = self.downs3(x3)
        x5 = self.downs4(x4)
        x = self.ups1(x5, x4)
        x = self.ups2(x, x3)
        x = self.ups3(x, x2)
        x = self.ups4(x, x1)
        return self.outconv(x)
