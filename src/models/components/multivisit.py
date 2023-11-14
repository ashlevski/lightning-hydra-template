import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

import src.utils.direct.data.transforms as T

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
    def __init__(self, dim=4):
        super(MRINet, self).__init__()
        # Feature extraction layers for current slice
        self.slice_conv1 = nn.Conv2d(1, dim, kernel_size=1, padding=0)
        self.slice_conv2 = nn.Conv2d(dim, dim*2, kernel_size=1, padding=0)

        # Feature extraction layers for past scan
        self.volume_conv1 = nn.Conv3d(1, dim, kernel_size=1, padding=0)
        self.volume_conv2 = nn.Conv3d(dim, dim*2, kernel_size=1, padding=0)

        # Cross-attention layer
        self.cross_attention = CrossAttentionLayer(dim*2)

        # Reconstruction layers
        # self.recon_conv1 = nn.Conv3d(dim*2, dim, kernel_size=3, padding=1)
        self.recon_conv2 = nn.Conv3d(dim*2, 1, kernel_size=1, padding=0)
        self.recon_conv3 = nn.Conv2d(256, 1, kernel_size=1, padding=0)

    def forward(self, x_slice, x_volume):
        # Feature extraction for current slice
        x_slice = (self.slice_conv1(x_slice.unsqueeze(1)))
        x_slice = (self.slice_conv2(x_slice))

        # Feature extraction for past scan
        x_volume = (self.volume_conv1(x_volume.unsqueeze(1)))
        x_volume = (self.volume_conv2(x_volume))

        # Apply cross-attention
        x_volume = self.cross_attention(x_slice, x_volume)
        # x_volume = x_volume.mean((1,2))
        # Reconstruction
        # x_volume = F.relu(self.recon_conv1(x_volume))
        x_volume = self.recon_conv2(x_volume)
        x_volume = self.recon_conv3(x_volume.squeeze(1))
        return x_volume

# Example usage
# Assuming x_slice is a 2D slice (e.g., [batch_size, channels, height, width])
# and x_volume is a 3D scan (e.g., [batch_size, channels, depth, height, width])
# mri_net = MRINet()
# x_slice = torch.randn(1, 1, 256, 256)  # Example 2D slice
# x_volume = torch.randn(1, 1, 64, 256, 256)  # Example 3D scan
# reconstructed_volume = mri_net(x_slice, x_volume)

class MultiVisitNet(nn.Module):
    def __init__(self, single_visit_net, weights_path, multi_visit_net):
        super(MultiVisitNet, self).__init__()
        # Initialize the single-visit network and load weights
        self.single_visit_net = single_visit_net
        if weights_path is not None:
            self.single_visit_net.load_state_dict(torch.load(weights_path))
            for param in single_visit_net.parameters():
                param.requires_grad = False
        # Initialize the multi-visit network
        self.multi_visit_net =  MRINet()

    def forward(self, x):
        # Forward pass through the single-visit network
        output_image, output_kspace, target_img = self.single_visit_net(x)

        # Forward pass through the multi-visit network
        # It's assumed here that the multi-visit network takes the output of the single-visit network as input
        multi_visit_output = self.multi_visit_net(output_image,x['img_pre'])
        # plt.imshow(output_image.cpu().detach()[0, :, :])
        # plt.title(x['metadata']["File name"])
        # plt.show()
        return output_image + multi_visit_output.squeeze(1), output_kspace, target_img

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
