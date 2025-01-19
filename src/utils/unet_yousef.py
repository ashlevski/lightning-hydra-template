import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetBlock, self).__init__()

        # Downsampling path
        self.conv1 = nn.Conv2d(in_channels, 48, kernel_size=3, padding="same")
        self.conv2 = nn.Conv2d(48, 48, kernel_size=3, padding="same")
        self.conv3 = nn.Conv2d(48, 48, kernel_size=3, padding="same")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(48, 64, kernel_size=3, padding="same")
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, padding="same")
        self.conv6 = nn.Conv2d(64, 64, kernel_size=3, padding="same")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv7 = nn.Conv2d(64, 128, kernel_size=3, padding="same")
        self.conv8 = nn.Conv2d(128, 128, kernel_size=3, padding="same")
        self.conv9 = nn.Conv2d(128, 128, kernel_size=3, padding="same")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv10 = nn.Conv2d(128, 256, kernel_size=3, padding="same")
        self.conv11 = nn.Conv2d(256, 256, kernel_size=3, padding="same")
        self.conv12 = nn.Conv2d(256, 256, kernel_size=3, padding="same")

        # Upsampling path
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv13 = nn.Conv2d(384, 128, kernel_size=3, padding="same")
        self.conv14 = nn.Conv2d(128, 128, kernel_size=3, padding="same")
        self.conv15 = nn.Conv2d(128, 128, kernel_size=3, padding="same")

        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv16 = nn.Conv2d(192, 64, kernel_size=3, padding="same")
        self.conv17 = nn.Conv2d(64, 64, kernel_size=3, padding="same")
        self.conv18 = nn.Conv2d(64, 64, kernel_size=3, padding="same")

        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv19 = nn.Conv2d(112, 48, kernel_size=3, padding="same")
        self.conv20 = nn.Conv2d(48, 48, kernel_size=3, padding="same")
        self.conv21 = nn.Conv2d(48, 48, kernel_size=3, padding="same")
        self.dropout = nn.Dropout(0.2)
        # Output layer
        self.conv22 = nn.Conv2d(48, out_channels, kernel_size=1)

    def forward(self, x):
        stack=[]
        # Downsample
        x = F.relu(self.conv1(x))
        # x = self.dropout(x)  # added dropout here
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        stack.append(x)
        x = self.pool1(x)



        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        stack.append(x)
        x = self.pool2(x)



        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        stack.append(x)
        x = self.pool3(x)


        x = F.relu(self.conv10(x))
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))


        # Upsample
        x = self.up1(x)
        x = torch.cat([x, stack.pop()], dim=1)

        x = F.relu(self.conv13(x))
        x = F.relu(self.conv14(x))
        x = F.relu(self.conv15(x))

        x = self.up2(x)
        # x = F.pad(x, (1, 0, 1, 0))
        x = torch.cat([x, stack.pop()], dim=1)

        x = F.relu(self.conv16(x))
        x = F.relu(self.conv17(x))
        x = F.relu(self.conv18(x))

        x = self.up3(x)
        x = torch.cat([x, stack.pop()], dim=1)

        x = F.relu(self.conv19(x))
        x = F.relu(self.conv20(x))
        x = F.relu(self.conv21(x))

        # Output layer
        x = self.conv22(x)

        return x

class UNetBlock_att(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetBlock_att, self).__init__()

        # Downsampling path
        self.conv1 = nn.Conv2d(in_channels, 48, kernel_size=3, padding="same")
        self.conv2 = nn.Conv2d(48, 48, kernel_size=3, padding="same")
        self.conv3 = nn.Conv2d(48, 48, kernel_size=3, padding="same")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(48, 64, kernel_size=3, padding="same")
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, padding="same")
        self.conv6 = nn.Conv2d(64, 64, kernel_size=3, padding="same")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)



        # Downsampling path 3D
        self.conv1_ = nn.Conv3d(in_channels, 48, kernel_size=3, padding="same")
        self.conv2_ = nn.Conv3d(48, 48, kernel_size=3, padding="same")
        self.conv3_ = nn.Conv3d(48, 48, kernel_size=3, padding="same")
        self.pool1_ = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv4_ = nn.Conv3d(48, 64, kernel_size=3, padding="same")
        self.conv5_ = nn.Conv3d(64, 64, kernel_size=3, padding="same")
        self.conv6_ = nn.Conv3d(64, 64, kernel_size=3, padding="same")
        self.pool2_ = nn.MaxPool3d(kernel_size=2, stride=2)

        # // RES ATTENTION
        self.conv7 = nn.Conv2d(64, 128, kernel_size=3, padding="same")
        self.conv8 = nn.Conv2d(128, 128, kernel_size=3, padding="same")
        self.conv9 = nn.Conv2d(128, 128, kernel_size=3, padding="same")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv10 = nn.Conv2d(128, 256, kernel_size=3, padding="same")
        self.conv11 = nn.Conv2d(256, 256, kernel_size=3, padding="same")
        self.conv12 = nn.Conv2d(256, 256, kernel_size=3, padding="same")


        self.conv7_ = nn.Conv3d(64, 128, kernel_size=3, padding="same")
        self.conv8_ = nn.Conv3d(128, 128, kernel_size=3, padding="same")
        self.conv9_ = nn.Conv3d(128, 128, kernel_size=3, padding="same")
        self.pool3_ = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv10_ = nn.Conv3d(128, 256, kernel_size=3, padding="same")
        self.conv11_ = nn.Conv3d(256, 256, kernel_size=3, padding="same")
        self.conv12_ = nn.Conv3d(256, 256, kernel_size=3, padding="same")


        self.multihead_attn1 = nn.MultiheadAttention(512, 8, batch_first=True)
        self.multihead_attn2 = nn.MultiheadAttention(512, 8, batch_first=True)
        self.multihead_attn3 = nn.MultiheadAttention(512, 8, batch_first=True)

        self.multihead_attn4 = nn.MultiheadAttention(1024, 8, batch_first=True)
        self.multihead_attn5 = nn.MultiheadAttention(1024, 8, batch_first=True)
        self.multihead_attn6 = nn.MultiheadAttention(1024, 8, batch_first=True)

        self.multihead_attn7 = nn.MultiheadAttention(1024, 8, batch_first=True)
        self.multihead_attn8 = nn.MultiheadAttention(1024, 8, batch_first=True)
        self.multihead_attn9 = nn.MultiheadAttention(1024, 8, batch_first=True)

        self.multihead_attn10 = nn.MultiheadAttention(512, 8, batch_first=True)
        self.multihead_attn11 = nn.MultiheadAttention(512, 8, batch_first=True)
        self.multihead_attn12 = nn.MultiheadAttention(512, 8, batch_first=True)

        # Upsampling path
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv13 = nn.Conv2d(384, 128, kernel_size=3, padding="same")
        self.conv14 = nn.Conv2d(128, 128, kernel_size=3, padding="same")
        self.conv15 = nn.Conv2d(128, 128, kernel_size=3, padding="same")

        self.up1_ = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv13_ = nn.Conv3d(384, 128, kernel_size=3, padding="same")
        self.conv14_ = nn.Conv3d(128, 128, kernel_size=3, padding="same")
        self.conv15_ = nn.Conv3d(128, 128, kernel_size=3, padding="same")

        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv16 = nn.Conv2d(192, 64, kernel_size=3, padding="same")
        self.conv17 = nn.Conv2d(64, 64, kernel_size=3, padding="same")
        self.conv18 = nn.Conv2d(64, 64, kernel_size=3, padding="same")

        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv19 = nn.Conv2d(112, 48, kernel_size=3, padding="same")
        self.conv20 = nn.Conv2d(48, 48, kernel_size=3, padding="same")
        self.conv21 = nn.Conv2d(48, 48, kernel_size=3, padding="same")

        # Output layer
        self.conv22 = nn.Conv2d(48, out_channels, kernel_size=1)

    def forward(self, x, x3):
        stack=[]
        stack3 = []
        # Downsample
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        stack.append(x)
        x = self.pool1(x)



        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        stack.append(x)
        x = self.pool2(x)


        # Downsample
        x3 = F.relu(self.conv1_(x3))
        x3 = F.relu(self.conv2_(x3))
        x3 = F.relu(self.conv3_(x3))
        # stack.append(x)
        x3 = self.pool1_(x3)



        x3 = F.relu(self.conv4_(x3))
        x3 = F.relu(self.conv5_(x3))
        x3 = F.relu(self.conv6_(x3))
        # stack.append(x)
        x3 = self.pool2_(x3)



        x = F.relu(self.conv7(x))
        x3 = F.relu(self.conv7_(x3))
        x = rearrange(x, "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=2, p2=2)
        x3_e = rearrange(x3, "b c (z) (h p1) (w p2) -> b  (z h w) (c p1 p2)", p1=2, p2=2)
        x = self.multihead_attn1(x,x3_e,x3_e)[0]
        x = rearrange(x, "b (h w) (c p1 p2) -> b c (h p1) (w p2)",
                  c=128, p1=2, p2=2, h=56 // 2, w=44 // 2)


        x = F.relu(self.conv8(x))
        x3 = F.relu(self.conv8_(x3))
        x = rearrange(x, "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=2, p2=2)
        x3_e = rearrange(x3, "b c (z) (h p1) (w p2) -> b  (z h w) (c p1 p2)", p1=2, p2=2)
        x = self.multihead_attn1(x,x3_e,x3_e)[0]
        x = rearrange(x, "b (h w) (c p1 p2) -> b c (h p1) (w p2)",
                  c=128, p1=2, p2=2, h=56 // 2, w=44 // 2)

        x = F.relu(self.conv9(x))
        x3 = F.relu(self.conv9_(x3))
        x = rearrange(x, "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=2, p2=2)
        x3_e = rearrange(x3, "b c (z) (h p1) (w p2) -> b  (z h w) (c p1 p2)", p1=2, p2=2)
        x = self.multihead_attn1(x,x3_e,x3_e)[0]
        x = rearrange(x, "b (h w) (c p1 p2) -> b c (h p1) (w p2)",
                  c=128, p1=2, p2=2, h=56 // 2, w=44 // 2)


        stack.append(x)
        stack3.append(x3)
        x = self.pool3(x)
        x3 = self.pool3_(x3)


        x = F.relu(self.conv10(x))
        x3 = F.relu(self.conv10_(x3))
        x = rearrange(x, "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=2, p2=2)
        x3_e= rearrange(x3, "b c (z) (h p1) (w p2) -> b  (z h w) (c p1 p2)", p1=2, p2=2)
        x = self.multihead_attn4(x,x3_e,x3_e)[0]
        x = rearrange(x, "b (h w) (c p1 p2) -> b c (h p1) (w p2)",
                  c=256, p1=2, p2=2, h= 28 // 2, w= 22 // 2)

        x = F.relu(self.conv11(x))
        x3 = F.relu(self.conv11_(x3))
        x = rearrange(x, "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=2, p2=2)
        x3_e= rearrange(x3, "b c (z) (h p1) (w p2) -> b  (z h w) (c p1 p2)", p1=2, p2=2)
        x = self.multihead_attn4(x,x3_e,x3_e)[0]
        x = rearrange(x, "b (h w) (c p1 p2) -> b c (h p1) (w p2)",
                  c=256, p1=2, p2=2, h= 28 // 2, w= 22 // 2)

        x = F.relu(self.conv12(x))
        x3 = F.relu(self.conv12_(x3))
        x = rearrange(x, "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=2, p2=2)
        x3_e= rearrange(x3, "b c (z) (h p1) (w p2) -> b  (z h w) (c p1 p2)", p1=2, p2=2)
        x = self.multihead_attn4(x,x3_e,x3_e)[0]
        x = rearrange(x, "b (h w) (c p1 p2) -> b c (h p1) (w p2)",
                  c=256, p1=2, p2=2, h= 28 // 2, w= 22 // 2)


        # Upsample
        x = self.up1(x)
        x3 = self.up1(x3)
        x = torch.cat([x, stack.pop()], dim=1)
        x3 = torch.cat([x3, stack3.pop()], dim=1)

        x = F.relu(self.conv13(x))
        x3 = F.relu(self.conv13_(x3))
        x = rearrange(x, "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=2, p2=2)
        x3_e = rearrange(x3, "b c (z) (h p1) (w p2) -> b  (z h w) (c p1 p2)", p1=2, p2=2)
        x = self.multihead_attn1(x, x3_e, x3_e)[0]
        x = rearrange(x, "b (h w) (c p1 p2) -> b c (h p1) (w p2)",
                      c=128, p1=2, p2=2, h=56 // 2, w=44 // 2)

        x = F.relu(self.conv14(x))
        x3 = F.relu(self.conv14_(x3))
        x = rearrange(x, "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=2, p2=2)
        x3_e = rearrange(x3, "b c (z) (h p1) (w p2) -> b  (z h w) (c p1 p2)", p1=2, p2=2)
        x = self.multihead_attn1(x, x3_e, x3_e)[0]
        x = rearrange(x, "b (h w) (c p1 p2) -> b c (h p1) (w p2)",
                      c=128, p1=2, p2=2, h=56 // 2, w=44 // 2)

        x = F.relu(self.conv15(x))
        x3 = F.relu(self.conv15_(x3))
        x = rearrange(x, "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=2, p2=2)
        x3_e = rearrange(x3, "b c (z) (h p1) (w p2) -> b  (z h w) (c p1 p2)", p1=2, p2=2)
        x = self.multihead_attn1(x, x3_e, x3_e)[0]
        x = rearrange(x, "b (h w) (c p1 p2) -> b c (h p1) (w p2)",
                      c=128, p1=2, p2=2, h=56 // 2, w=44 // 2)

        x = self.up2(x)
        # x = F.pad(x, (1, 0, 1, 0))
        x = torch.cat([x, stack.pop()], dim=1)

        x = F.relu(self.conv16(x))
        x = F.relu(self.conv17(x))
        x = F.relu(self.conv18(x))

        x = self.up3(x)
        x = torch.cat([x, stack.pop()], dim=1)

        x = F.relu(self.conv19(x))
        x = F.relu(self.conv20(x))
        x = F.relu(self.conv21(x))

        # Output layer
        x = self.conv22(x)

        return x


class UNetBlock_3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetBlock_3d, self).__init__()

        # Downsampling path
        self.conv1 = nn.Conv3d(in_channels, 48, kernel_size=3, padding="same")
        self.conv2 = nn.Conv3d(48, 48, kernel_size=3, padding="same")
        self.conv3 = nn.Conv3d(48, 48, kernel_size=3, padding="same")
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv3d(48, 64, kernel_size=3, padding="same")
        self.conv5 = nn.Conv3d(64, 64, kernel_size=3, padding="same")
        self.conv6 = nn.Conv3d(64, 64, kernel_size=3, padding="same")
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv7 = nn.Conv3d(64, 128, kernel_size=3, padding="same")
        self.conv8 = nn.Conv3d(128, 128, kernel_size=3, padding="same")
        self.conv9 = nn.Conv3d(128, 128, kernel_size=3, padding="same")
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv10 = nn.Conv3d(128, 256, kernel_size=3, padding="same")
        self.conv11 = nn.Conv3d(256, 256, kernel_size=3, padding="same")
        self.conv12 = nn.Conv3d(256, 256, kernel_size=3, padding="same")

        # Upsampling path
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv13 = nn.Conv3d(384, 128, kernel_size=3, padding="same")
        self.conv14 = nn.Conv3d(128, 128, kernel_size=3, padding="same")
        self.conv15 = nn.Conv3d(128, 128, kernel_size=3, padding="same")

        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv16 = nn.Conv3d(192, 64, kernel_size=3, padding="same")
        self.conv17 = nn.Conv3d(64, 64, kernel_size=3, padding="same")
        self.conv18 = nn.Conv3d(64, 64, kernel_size=3, padding="same")

        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv19 = nn.Conv3d(112, 48, kernel_size=3, padding="same")
        self.conv20 = nn.Conv3d(48, 48, kernel_size=3, padding="same")
        self.conv21 = nn.Conv3d(48, 48, kernel_size=3, padding="same")

        # Output layer
        self.conv22 = nn.Conv3d(48, out_channels, kernel_size=1)

    def forward(self, x):
        stack=[]
        # Downsample
        x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))
        stack.append(x)
        x = self.pool1(x)



        x = F.relu(self.conv4(x))
        # x = F.relu(self.conv5(x))
        # x = F.relu(self.conv6(x))
        stack.append(x)
        x = self.pool2(x)



        x = F.relu(self.conv7(x))
        # x = F.relu(self.conv8(x))
        # x = F.relu(self.conv9(x))
        stack.append(x)
        x = self.pool3(x)


        x = F.relu(self.conv10(x))
        # x = F.relu(self.conv11(x))
        # x = F.relu(self.conv12(x))


        # Upsample
        x = self.up1(x)
        x = torch.cat([x, stack.pop()], dim=1)

        x = F.relu(self.conv13(x))
        # x = F.relu(self.conv14(x))
        # x = F.relu(self.conv15(x))

        x = self.up2(x)
        # x = F.pad(x, (1, 0, 1, 0))
        x = torch.cat([x, stack.pop()], dim=1)

        x = F.relu(self.conv16(x))
        # x = F.relu(self.conv17(x))
        # x = F.relu(self.conv18(x))

        x = self.up3(x)
        x = torch.cat([x, stack.pop()], dim=1)

        x = F.relu(self.conv19(x))
        # x = F.relu(self.conv20(x))
        # x = F.relu(self.conv21(x))

        # Output layer
        x = self.conv22(x)

        return x

import torch

# def test_unet_block():
#     # Define the input tensor
#     input_tensor = torch.randn(8, 2, 218, 170)
#
#     # Instantiate the U-Net block
#     unet_block = UNetBlock(in_channels=2, out_channels=2)
#
#     # Forward pass
#     output_tensor = unet_block(input_tensor)
#
#     # Print input and output shapes
#     print(f"Input shape: {input_tensor.shape}")
#     print(f"Output shape: {output_tensor.shape}")

# Run the test function
# test_unet_block()
