import torch
import torch.nn as nn
import torch.nn.functional as F

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

        # Output layer
        self.conv22 = nn.Conv2d(48, out_channels, kernel_size=1)

    def forward(self, x):
        stack=[]
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
