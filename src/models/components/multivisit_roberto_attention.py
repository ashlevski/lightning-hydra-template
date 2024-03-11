import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from einops import rearrange
from sympy import reduced_totient
from torch.nn import MultiheadAttention

import src.utils.direct.data.transforms as T
from projects.longitudinal.Patch_embeeding import PatchEmbed
from projects.longitudinal.stable_diff import pad_to_nearest_multiple, UNet2DConditionModel
from projects.longitudinal.unet import UNetModel
from projects.longitudinal.unet_cross_att import UnetModel2d_att
from projects.longitudinal.unetr_dual import UNETR
from src.utils.direct.nn.unet import UnetModel2d
from torchmetrics.image  import StructuralSimilarityIndexMeasure
from torch.nn import Conv3d
from src.utils.unet_yousef import UNetBlock, UNetBlock_att
from projects.longitudinal.DiT_custom import DiT_S_8, DiT_S_4, DiT_B_4, DiT_L_8, DiT_L_4

class MV(nn.Module):
    def __init__(self, dim=4):
        super(MV, self).__init__()
        self.model = DiT_S_4(learn_sigma=False,input_size=224,in_channels=1)
        self.unet = UNetBlock(2,1)
        self.y_embedder = PatchEmbed(224, 4, 1, 384, bias=True)
        self.pos_embed = nn.Parameter(torch.zeros(1, 3136, 384), requires_grad=False)
    def cal_ssim(self,input,target):
        return self.ssim(input, target.unsqueeze(1))
    def forward(self, x_slice, x_volume):

        # print(x_volume.shape)
        x_slice, pad = pad_to_nearest_multiple((x_slice), 224)
        x_volume, pad = pad_to_nearest_multiple((x_volume), 224)
        # print(x_slice.shape)
        # print(x_volume.shape)
        x_volume = self.y_embedder(x_volume.unsqueeze(1)) + self.pos_embed
        z = self.model(x_slice,x_volume)
        z = self.unet(torch.cat((x_slice, z), dim=1))[..., pad[0]//2:-pad[0]//2, pad[1]//2:-pad[1]//2]
        return z.squeeze(1),x_volume


class MultiVisitNet(nn.Module):
    def __init__(self,  multi_visit_net,freeze=True):
        super(MultiVisitNet, self).__init__()
        self.multi_visit_net = MV()

    def forward(self, x):

        output_image_mv,x_volume  = self.multi_visit_net(x["data"],x['baseline'])
        output_image = x["data"].squeeze() + output_image_mv

        return output_image, 0 , x["target"].squeeze(), output_image_mv, x_volume.squeeze()#x["baseline"].squeeze()
