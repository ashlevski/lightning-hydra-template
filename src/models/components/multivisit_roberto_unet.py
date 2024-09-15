import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from monai.networks.nets import AutoEncoder
from sympy import reduced_totient
from torch.nn import MultiheadAttention

import src.utils.direct.data.transforms as T
from projects.longitudinal.stable_diff import pad_to_nearest_multiple, UNet2DConditionModel
from projects.longitudinal.unet import UNetModel
from projects.longitudinal.unet_cross_att import UnetModel2d_att
from src.utils.direct.nn.unet import UnetModel2d
from torchmetrics.image  import StructuralSimilarityIndexMeasure

from src.utils.unet_yousef import UNetBlock, UNetBlock_3d
from projects.longitudinal.DiT_custom import DiT_S_8, DiT_S_2, DiT_S_4, DiT_B_4, DiT_L_8, DiT_L_4
from projects.longitudinal.Patch_embeeding import PatchEmbed

from monai.networks import nets
class MV(nn.Module):
    def __init__(self, dim=4):
        super(MV, self).__init__()
        self.unet = UNetBlock(2,1)


    def forward(self, x_slice, x_volume):
        x_slice, pad = pad_to_nearest_multiple((x_slice), 16)
        x_volume, pad = pad_to_nearest_multiple((x_volume), 16)
        z = self.unet(torch.cat((x_slice, x_volume), dim=1))
        z = z[..., pad[0]//2:-pad[0]//2, pad[1]//2:-pad[1]//2]
        return z.squeeze(1),x_volume


class MultiVisitNet(nn.Module):
    def __init__(self, freeze=True):
        super(MultiVisitNet, self).__init__()
        self.multi_visit_net = MV()

    def forward(self, x):
        # Forward pass through the multi-visit network
        # It's assumed here that the multi-visit network takes the output of the single-visit network as input
        output_image_mv, x_volume  = self.multi_visit_net(x["data"],x['baseline'])
        output_image = x["data"].squeeze() + output_image_mv
        return output_image, 0 , x["target"].squeeze(), output_image_mv, x['baseline'].squeeze()
