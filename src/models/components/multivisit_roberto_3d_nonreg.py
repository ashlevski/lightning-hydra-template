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
from projects.longitudinal.torchir.networks.globalnet import AIRNet
from projects.longitudinal.torchir.transformers import AffineTransformer,BsplineTransformer
from src.utils.unet_yousef import UNetBlock, UNetBlock_3d
from projects.longitudinal.torchir.networks.dirnet import DIRNet


class MV(nn.Module):
    def __init__(self, dim=4):
        super(MV, self).__init__()
        grid_spacing = (32, 32, 32)
        # self.param = DIRNet(kernels=5, grid_spacing=grid_spacing,ndim=3)
        # self.trans = BsplineTransformer(ndim=3, upsampling_factors = grid_spacing)

        self.param0 = AIRNet(kernels=3,ndim=3,num_conv_layers=6)
        self.trans0 = AffineTransformer(ndim=3)
        # self.unet = UNetBlock_3d(2,1)
 
    def forward(self, x_slice, x_volume):
        params = self.param0(x_slice.unsqueeze(1), x_volume.unsqueeze(1))
        x_volume = self.trans0(params, x_slice.unsqueeze(1), x_volume.unsqueeze(1))
        # params = self.param(x_slice.unsqueeze(1), x_volume)
        # x_volume = self.trans(params, x_slice.unsqueeze(1), x_volume)
        # x_slice, pad = pad_to_nearest_multiple((x_slice), 16)
        # x_volume, pad = pad_to_nearest_multiple((x_volume), 16)
        # print(x_slice.shape)
        # print(x_volume.shape)
        z= x_volume#[..., pad[0]//2:-pad[0]//2, pad[1]//2:-pad[1]//2]
        # z = self.unet(torch.cat((x_slice.unsqueeze(1), x_volume), dim=1))[..., pad[0]//2:-pad[0]//2, pad[1]//2:-pad[1]//2]
        return z.squeeze(1),x_volume.squeeze(1)#[..., pad[0]//2:-pad[0]//2, pad[1]//2:-pad[1]//2]
    # def __init__(self, dim=4, n=3):
    #     super(MV, self).__init__()
    #     grid_spacing = (32, 32, 32)
        
    #     self.param_list = nn.ModuleList([AIRNet(kernels=3, ndim=3, num_conv_layers=4) for _ in range(n)])
    #     self.trans_list = nn.ModuleList([AffineTransformer(ndim=3) for _ in range(n)])
        
    #     self.unet = UNetBlock_3d(2, 1)

    # def forward(self, x_slice, x_volume):
    #     x_slice = x_slice.unsqueeze(1)
    #     x_volume = x_volume.unsqueeze(1)
    #     for param, trans in zip(self.param_list, self.trans_list):
    #         params = param(x_slice, x_volume)
    #         x_volume = trans(params, x_slice, x_volume)
        
    #     z = x_volume
        
    #     return z.squeeze(1), x_volume.squeeze(1)

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
        return output_image, 0 , x["target"].squeeze(1), output_image_mv, x_volume.squeeze(1)

