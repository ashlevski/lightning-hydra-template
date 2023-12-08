import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from monai.networks.nets import AutoEncoder
from torch.nn import MultiheadAttention

import src.utils.direct.data.transforms as T
from projects.longitudinal.stable_diff import pad_to_nearest_multiple, UNet2DConditionModel
from projects.longitudinal.unet_cross_att import UnetModel2d_att
from src.utils.direct.nn.unet import UnetModel2d


class MV(nn.Module):
    def __init__(self, dim=4):
        super(MV, self).__init__()
        # self.attention = nn.ModuleList()
        # self.layers2 = nn.Sequential(
        #     # First Convolution Block
        #     nn.Conv2d(1, 4, kernel_size=3, stride=2, padding=1),  # Output: (batch, 8, 128, 109, 85)
        #     nn.BatchNorm2d(4),
        #     nn.LeakyReLU(),
        #
        #     # Second Convolution Block
        #     nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1),  # Output: (batch, 16, 64, 55, 43)
        #     nn.BatchNorm2d(8),
        #     nn.LeakyReLU(),
        #
        #     nn.Conv2d(8, dim, kernel_size=3, stride=2, padding=1),  # Output: (batch, 16, 64, 55, 43)
        #     nn.BatchNorm2d(8),
        #     nn.LeakyReLU(),
        # )
        #
        # self.layers3 = nn.Sequential(
        #     # First Convolution Block
        #     nn.Conv3d(1, 4, kernel_size=3, stride=2, padding=1),  # Output: (batch, 8, 128, 109, 85)
        #     nn.BatchNorm3d(4),
        #     nn.LeakyReLU(),
        #
        #     # Second Convolution Block
        #     nn.Conv3d(4, 8, kernel_size=3, stride=2, padding=1),  # Output: (batch, 16, 64, 55, 43)
        #     nn.BatchNorm3d(8),
        #     nn.LeakyReLU(),
        #
        #     # Second Convolution Block
        #     nn.Conv3d(8, dim, kernel_size=3, stride=2, padding=1),  # Output: (batch, 16, 64, 55, 43)
        #     nn.BatchNorm3d(dim),
        #     nn.LeakyReLU(),
        # )

        # for i in range(1):
        #     self.attention.append(MultiheadAttention(embed_dim=dim, num_heads=4, batch_first=True))

        kernel = 16
        # Feature extraction layers for current slice
        # self.slice_conv1 = nn.Conv2d(1, dim, kernel_size=1, stride=1,padding=0)
        # self.slice_conv2 = nn.Conv2d(dim, dim*2, kernel_size=1, padding=0)

        # Feature extraction layers for past scan
        # self.volume_conv1 = nn.Conv3d(1, dim, kernel_size=kernel, stride=kernel,padding=(0,3,3))
        # self.volume_conv2 = nn.Conv3d(1, dim, kernel_size=kernel, stride=kernel,padding=(0,3,3))

        # Cross-attention layer
        # self.cross_attention = CrossAttentionLayer(dim*2)
        # self.softmax = nn.Softmax(dim=-3)
        # Reconstruction layers

        # self.recon_conv1 = nn.Conv3d(dim*2, dim, kernel_size=3, padding=1)
        # self.recon_conv2 = nn.Conv3d(dim*2, 1, kernel_size=1, padding=0)
        # self.recon_conv3 = nn.Conv2d(256, 1, kernel_size=1, padding=0)
        # self.deconv1 = nn.ConvTranspose3d(dim,1, kernel_size=kernel, stride=kernel,padding=(0,3,3))
        # self.deconv2 = nn.ConvTranspose2d(dim, 4, kernel_size=kernel, stride=kernel, padding=(3,3))
        # self.deconv3 = nn.Conv2d(int(dim/2), 1, kernel_size=15, stride=1,padding=1)
        # self.deconv4 = nn.Conv3d(1, 1, kernel_size=(16, 1, 1), stride=(16, 1, 1))
        # self.unet = UnetModel2d_att(in_channels=1,out_channels=1,num_filters=8,num_pool_layers=4,dropout_probability=0)
        self.unet = UnetModel2d(in_channels=2, out_channels=1, num_filters=8, num_pool_layers=4,dropout_probability=0)
        # self.dim = dim
        # self.norm1 = torch.nn.LayerNorm(dim)
        # self.norm2 = torch.nn.LayerNorm(dim)
        # self.norm3 = torch.nn.LayerNorm(dim)
        # self.deconv2 = nn.ConvTranspose2d(dim, 1, kernel_size=1, stride=1, padding=0)
        # self.lin1 = nn.Linear(dim,4*dim)
        # self.lin2 = nn.Linear(4*dim, dim)
        # self.gelu = nn.GELU()
        # self.proj = nn.Linear(dim, 16*16)
        # self.transformers = nn.Transformer(batch_first=True,num_encoder_layers=1,num_decoder_layers=1)
        # self.act = nn.LeakyReLU()
        # self.vae_2d = AutoEncoder(spatial_dims=2, in_channels=1, out_channels=1, channels=(4, 4, 4), strides=(2, 2, 2))
        # self.vae_3d = AutoEncoder(spatial_dims=3, in_channels=1, out_channels=256, channels=(32, 256, 512),
        #                      strides=(2, 2, 2))
        # self.unet = UNet2DConditionModel(in_channels=4, out_channels=4, cross_attention_dim=512,layers_per_block=1)
    def forward(self, x_slice, x_volume):
        # x_slice_ = F.interpolate(x_slice.unsqueeze(1), scale_factor=0.25, mode='bilinear', align_corners=False)
        # x_volume_ = F.interpolate(x_volume.unsqueeze(1), scale_factor=(1, 0.25, 0.25)
        #                                    , mode='trilinear', align_corners=False)
        x_volume = x_volume.permute(0,2,3,1)
        max_id = torch.softmax(torch.einsum('bdhw, bshw -> bs', torch.exp(x_slice.unsqueeze(1)), torch.exp(x_volume)),
                               dim=1).argmax(dim=1)
        # result_tensors = []
        # # Index the original tensor for each batch
        # for batch_idx, index in enumerate(max_id):
        #     result_tensors.append(x_volume[batch_idx, index, :, :].unsqueeze(0))
        x_volume = x_volume[torch.arange(x_volume.size(0)), max_id]
        # Stack the individual tensors along a new batch dimension
        # x_volume = torch.cat(result_tensors, dim=0)
        x_slice = ((x_slice / torch.amax(x_slice, dim=(-1, -2), keepdim=True)))
        x_volume = (x_volume / torch.amax(x_volume, dim=(-1, -2), keepdim=True))
        z = self.unet(torch.cat((x_slice.unsqueeze(1), x_volume.unsqueeze(1)), dim=1))
        # x_slice, pad = pad_to_nearest_multiple(x_slice.unsqueeze(1), 256)
        # latent_2d = self.vae_2d.encode(x_slice)
        # latent_3d = self.vae_3d.encode(x_volume.unsqueeze(1))
        # latent_3d = latent_3d.view(latent_3d.shape[0], latent_3d.shape[1], 1, -1)
        # latent_2d = self.unet(latent_2d, latent_3d)
        # z = self.vae_2d.decode(latent_2d[0])
        # with torch.no_grad():
        # key = self.slice_conv1(x_slice.unsqueeze(1)).view(x_slice.shape[0],self.dim,-1).transpose(1,2)
        # patch_3d = self.volume_conv1(x_volume.unsqueeze(1)).view(x_slice.shape[0],self.dim,-1).transpose(1,2)

        # z = self.unet(x_slice.unsqueeze(1),x_volume.view(x_slice.shape[0],x_slice.shape[1],-1).unsqueeze(1))
        # z = self.unet(x_slice.unsqueeze(1))
        # value = self.volume_conv2(x_volume.unsqueeze(1)).view(x_slice.shape[0],self.dim,-1).transpose(1,2)

        # key, query = self.norm1(key),self.norm2(query)
        # attn_output = self.transformers(key,query)
        # attn_output = 0
        # for attend in self.attention:
        #     attn_output += key
        #     attn_output, _ = attend(attn_output, query, query)
            # attn_output = self.act(attn_output)
        # attn_output = self.lin2(self.gelu(self.lin1(self.norm3(attn_output))))
        # attn_output = attn_output
        # z = self.proj(attn_output).reshape(-1, 1, 14*16, 11*16)[:,:,:-6,:-6]
        # z = attn_output.transpose(1, 2).reshape(-1, self.dim,  218, 170)
        # z = self.act(z)
        # z = self.deconv2(z)
        # z = self.act(z)
        # z = self.unet(z)
        # z = self.softmax(z)

        # z = z.repeat_interleave(16, dim=-1).repeat_interleave(16, dim=-2).repeat_interleave(16, dim=-3)
        # z = x_volume.unsqueeze(1)*(z[:,:,:,:-6,:-6])
        # z = self.deconv4(self.deconv3(z))
        # z= self.lin(attn_output).view(1, 1,218,170)
        # attn_output_weights = attn_output_weights
        #
        #
        # # Feature extraction for current slice
        # x_slice = (self.slice_conv1(x_slice.unsqueeze(1)))
        # x_slice = (self.slice_conv2(x_slice))
        #
        # # Feature extraction for past scan
        # x_volume = (self.volume_conv1(x_volume.unsqueeze(1)))
        # x_volume = (self.volume_conv2(x_volume))
        #
        # # Apply cross-attention
        # x_volume = self.cross_attention(x_slice, x_volume)
        # # x_volume = x_volume.mean((1,2))
        # # Reconstruction
        # # x_volume = F.relu(self.recon_conv1(x_volume))
        # x_volume = self.recon_conv2(x_volume)
        # x_volume = self.recon_conv3(x_volume.squeeze(1))
        return z.squeeze(1)

# Example usage
# Assuming x_slice is a 2D slice (e.g., [batch_size, channels, height, width])
# and x_volume is a 3D scan (e.g., [batch_size, channels, depth, height, width])
# mri_net = MRINet()
# x_slice = torch.randn(1, 1, 256, 256)  # Example 2D slice
# x_volume = torch.randn(1, 1, 64, 256, 256)  # Example 3D scan
# reconstructed_volume = mri_net(x_slice, x_volume)

class MultiVisitNet(nn.Module):
    def __init__(self, single_visit_net, weights_path, multi_visit_net,freeze=True):
        super(MultiVisitNet, self).__init__()
        # Initialize the single-visit network and load weights
        self.single_visit_net = single_visit_net
        if weights_path is not None:
            self.single_visit_net.load_state_dict(torch.load(weights_path))
            if freeze:
                for param in single_visit_net.parameters():
                    param.requires_grad = False
        # Initialize the multi-visit network
        # self.unet = UnetModel2d(in_channels=156,out_channels=1,num_filters=8,num_pool_layers=2,dropout_probability=0)

        self.multi_visit_net = MV()

    def forward(self, x):
        # Forward pass through the single-visit network
        with torch.no_grad():
            output_image, output_kspace, target_img = self.single_visit_net(x)
        # Forward pass through the multi-visit network
        # It's assumed here that the multi-visit network takes the output of the single-visit network as input
        output_image = output_image+self.multi_visit_net(output_image,x['img_pre'])
        # plt.imshow(output_image.cpu().detach()[0, :, :])
        # plt.title(x['metadata']["File name"])
        # plt.show()+ 0*multi_visit_output #+ self.unet(x['img_pre']).squeeze()
        del output_kspace
        return output_image, 0 , target_img

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
