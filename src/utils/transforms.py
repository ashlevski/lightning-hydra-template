import torch.nn as nn
import torch
import src.utils.direct.data.transforms as T
class NormalizeSampleTransform(nn.Module):
    def __init__(self, scale = 37060):
        super(NormalizeSampleTransform, self).__init__()
        self.scale = scale

    def forward(self, x):
        # Calculate the maximum along the spatial dimensions and the real/imaginary channel (last three dimensions)

        return (x / torch.abs(x).max()) * self.scale


def normalizeSampleTransform(x):
    # Calculate the maximum along the spatial dimensions and the real/imaginary channel (last three dimensions)
    max_values = torch.abs(x).max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0].max(dim=-3, keepdim=True)[0].max(dim=-4, keepdim=True)[0]
    # Expand the max_values tensor to match the input dimensions
    max_values = max_values.expand_as(x)
    # Normalize the input tensor by the max_values tensor
    normalized_x = x / (max_values + 1e-10)
    return normalized_x,max_values + 1e-10
def test_NormalizeSampleTransform():
    # Example usage:
    # Create a dummy tensor with a size of (batch * coil * x * y * 2)
    batch_size, coil, x, y = 5, 3, 64, 64
    dummy_data = torch.rand(batch_size, coil, x, y, 2)

    # Create the transform
    normalize_transform = NormalizeSampleTransform()

    # Apply the transform to the dummy data
    normalized_data = normalize_transform(dummy_data)

class rssTransform(nn.Module):
    def __init__(self, fourier_dim = (2,3), channel_dim= 1):
        super(rssTransform, self).__init__()

    def forward(self, x):
        #
        prev_img = T.root_sum_of_squares(
            self.backward_operator(x["kspace_pre"], dim=(2, 3)),
            dim=1,
        )  # shape (batch, height,  width)
        return prev_img