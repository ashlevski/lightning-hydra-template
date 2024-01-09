import torch
from torch import nn

from projects.longitudinal.unet import UNetModel

unet = UNetModel(in_channels=1,out_channels=1,channels=32,n_res_blocks=1,attention_levels=[2],channel_multipliers=[1,2,4,8],n_heads=4,d_cond=1)

x = unet(torch.rand((1,1,218,170)),torch.rand((1,218*170,1)))

slice_conv1 = nn.Conv2d(1, 16, kernel_size=16, stride=16,padding=0)
slice_conv1(torch.rand((1,1,218,170)))
pass