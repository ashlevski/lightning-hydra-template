import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.nn import MultiheadAttention

import src.utils.direct.data.transforms as T
from src.utils.direct.nn.unet import UnetModel2d


class CrossAttentionLayer(nn.Module):
    def __init__(self, channel_size, dk=4):
        super(CrossAttentionLayer, self).__init__()
        # Define layers for query, key, and value for attention
        self.query_conv = nn.Conv2d(channel_size, channel_size, kernel_size=1)
        self.key_conv = nn.Conv3d(channel_size, channel_size, kernel_size=1)
        self.value_conv = nn.Conv3d(channel_size, channel_size, kernel_size=1)
        self.softmax = nn.Softmax(dim=-3)
        self.dk = channel_size  # dimension of the key for scaling

    def forward(self, x_slice, x_volume):
        # Generate query, key, value
        query = self.query_conv(x_slice).unsqueeze(2)
        key = self.key_conv(x_volume)##.view(x_volume.shape[0], x_volume.shape[1], -1)
        value = self.value_conv(x_volume)#.view(x_volume.shape[0], x_volume.shape[1], -1)

        # Calculate attention
        attention = query.repeat(1,1,256,1,1)*key
        attention = self.softmax(attention/(self.dk ** 0.5))

        # Apply attention to value
        out = attention*value
        # out = out.view(x_volume.shape)

        return out


class MRINet(nn.Module):
    def __init__(self, dim=512):
        super(MRINet, self).__init__()
        # self.attention=MultiheadAttention(embed_dim=dim, num_heads=4,batch_first=True)
        kernel = 16
        # Feature extraction layers for current slice
        self.slice_conv1 = nn.Conv2d(1, dim, kernel_size=kernel, stride=kernel,padding=3)
        # self.slice_conv2 = nn.Conv2d(dim, dim*2, kernel_size=1, padding=0)

        # Feature extraction layers for past scan
        self.volume_conv1 = nn.Conv3d(1, dim, kernel_size=kernel, stride=kernel,padding=(0,3,3))
        # self.volume_conv2 = nn.Conv3d(1, dim, kernel_size=kernel, stride=kernel,padding=(0,3,3))

        # Cross-attention layer
        # self.cross_attention = CrossAttentionLayer(dim*2)

        # Reconstruction layers
        # self.recon_conv1 = nn.Conv3d(dim*2, dim, kernel_size=3, padding=1)
        # self.recon_conv2 = nn.Conv3d(dim*2, 1, kernel_size=1, padding=0)
        # self.recon_conv3 = nn.Conv2d(256, 1, kernel_size=1, padding=0)
        # self.deconv1 = nn.ConvTranspose2d(dim,dim, kernel_size=kernel, stride=kernel,padding=3)
        # self.unet = UnetModel2d(in_channels=dim,out_channels=1,num_filters=8,num_pool_layers=2,dropout_probability=0)
        self.dim = dim
        self.norm1 = torch.nn.LayerNorm(dim)
        self.norm2 = torch.nn.LayerNorm(dim)
        # self.norm3 = torch.nn.LayerNorm(dim)
        # self.deconv2 = nn.ConvTranspose2d(dim, 1, kernel_size=7, stride=1, padding=3)
        # self.lin1 = nn.Linear(dim,4*dim)
        # self.lin2 = nn.Linear(4*dim, dim)
        # self.gelu = nn.GELU()
        self.proj = nn.Linear(dim, 16*16)
        self.transformers = nn.Transformer(batch_first=True)
    def forward(self, x_slice, x_volume):
        # with torch.no_grad():
        key = self.slice_conv1(x_slice.unsqueeze(1)).view(x_slice.shape[0],self.dim,-1).transpose(1,2)
        query = self.volume_conv1(x_volume.unsqueeze(1)).view(x_slice.shape[0],self.dim,-1).transpose(1,2)

        # value = self.volume_conv2(x_volume.unsqueeze(1))
        key, query = self.norm1(key),self.norm2(query)
        attn_output = self.transformers(query,key)

        # attn_output, _ = self.attention(key, query, query)
        # attn_output = self.lin2(self.gelu(self.lin1(self.norm3(attn_output))))
        # attn_output = attn_output
        z = self.proj(attn_output).reshape(-1, 1, 14*16, 11*16)[:,:,:-6,:-6]
        # z = self.deconv1(attn_output.transpose(1, 2).reshape(-1, self.dim, 14, 11))
        # z = self.unet(z)

        # z = self.deconv2(z)
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
        self.unet = UnetModel2d(in_channels=156,out_channels=1,num_filters=8,num_pool_layers=2,dropout_probability=0)

        self.multi_visit_net =  MRINet()

    def forward(self, x):
        # Forward pass through the single-visit network
        output_image, output_kspace, target_img = self.single_visit_net(x)
        # Forward pass through the multi-visit network
        # It's assumed here that the multi-visit network takes the output of the single-visit network as input
        multi_visit_output = self.multi_visit_net(output_image,x['img_pre'])
        # plt.imshow(output_image.cpu().detach()[0, :, :])
        # plt.title(x['metadata']["File name"])
        # plt.show()+ 0*multi_visit_output #+ self.unet(x['img_pre']).squeeze()
        return output_image+multi_visit_output, output_kspace, target_img

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
