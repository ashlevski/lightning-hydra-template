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
        self.dit = DiT_S_8(learn_sigma=False, input_size=224, in_channels=1,drop=0.0)
        # self.unet = UNetBlock(2,1)
        self.y_embedder = PatchEmbed(224, 8, 1, 384, bias=True)
        self.pos_embed = nn.Parameter(torch.zeros(1, 12544//16, 384), requires_grad=False)
        # self.unet = nets.RegUNet(spatial_dims=2, in_channels=2, num_channel_initial=32, depth=5, out_channels=1)
        # self.unet = nets.SwinUNETR((224,192), in_channels=2, out_channels=1,spatial_dims=2)
    def forward(self, x_slice, x_volume):
        x_slice, pad = pad_to_nearest_multiple((x_slice), 224)
        x_volume, pad = pad_to_nearest_multiple((x_volume), 224)
        x_volume = self.y_embedder(x_volume) + self.pos_embed
        z = self.dit(x_slice, x_volume)
        # z = self.unet(torch.cat((x_slice, x_volume), dim=1))
        z = z[..., pad[0]//2:-pad[0]//2, pad[1]//2:-pad[1]//2]
        return z.squeeze(1),x_volume

# Example usage
# Assuming x_slice is a 2D slice (e.g., [batch_size, channels, height, width])
# and x_volume is a 3D scan (e.g., [batch_size, channels, depth, height, width])
# mri_net = MRINet()
# x_slice = torch.randn(1, 1, 256, 256)  # Example 2D slice
# x_volume = torch.randn(1, 1, 64, 256, 256)  # Example 3D scan
# reconstructed_volume = mri_net(x_slice, x_volume)

class MultiVisitNet(nn.Module):
    def __init__(self,  multi_visit_net,freeze=True):
        super(MultiVisitNet, self).__init__()
        # Initialize the single-visit network and load weights
        # self.single_visit_net = single_visit_net
        # if weights_path is not None:
        #     self.single_visit_net.load_state_dict(torch.load(weights_path))
        #     if freeze:
        #         for param in single_visit_net.parameters():
        #             param.requires_grad = False
        # Initialize the multi-visit network
        # self.unet = UnetModel2d(in_channels=156,out_channels=1,num_filters=8,num_pool_layers=2,dropout_probability=0)

        self.multi_visit_net = MV()

    def forward(self, x):
        # Forward pass through the single-visit network
        # with torch.no_grad():
        #     output_image, output_kspace, target_img = self.single_visit_net(x)
        # Forward pass through the multi-visit network
        # It's assumed here that the multi-visit network takes the output of the single-visit network as input
        output_image_mv,x_volume  = self.multi_visit_net(x["data"],x['baseline'])
        output_image = x["data"].squeeze() + output_image_mv
        # plt.imshow(output_image.cpu().detach()[0, :, :])
        # plt.title(x['metadata']["File name"])
        # plt.show()+ 0*multi_visit_output #+ self.unet(x['img_pre']).squeeze()
        # del output_kspace
        return output_image, 0 , x["target"].squeeze(), output_image_mv, x['baseline'].squeeze()

# Example usage:
# # Define the single-visit and multi-visit networks (they should be instances of nn.Module with the same input/output dimensions)
# single_visit_net = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 10))
# multi_visit_net = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 1))
#
# # Path to the weights file
# weights_path = None
#
# # Create an instance of the custom module
# my_module = MultiVisitNet(single_visit_net, weights_path, multi_visit_net)
#
# # Now you can use my_module for a forward pass with some input tensor `x`
# x = torch.randn(12, 10) # Example input
# output = my_module(x)
