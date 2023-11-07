import torch
import torch.nn as nn
import torch.nn.functional as F


class EfficientCrossAttention(nn.Module):
    def __init__(self, feature_dim):
        super(EfficientCrossAttention, self).__init__()
        # Separate convolutions for query and keys
        self.query_conv = nn.Conv2d(1, feature_dim, kernel_size=3, padding=1)
        self.key_conv = nn.Conv2d(2, feature_dim, kernel_size=3, padding=1)  # 2 channels for adjacent slices
        self.value_conv = nn.Conv2d(2, feature_dim, kernel_size=3, padding=1)  # 2 channels for adjacent slices

    def forward(self, target, adjacent):
        # Extract query and key features
        query = F.relu(self.query_conv(target.unsqueeze(1)))
        key = self.key_conv(adjacent)
        value = self.value_conv(adjacent)

        # Compute attention scores
        # Here we use a simple dot product for attention; more sophisticated functions can be used
        attention_scores = torch.sum(query * key, dim=1, keepdim=True)

        # Apply softmax to get attention weights
        attention_weights = (F.softmax(attention_scores, dim=2)+F.softmax(attention_scores, dim=2))/2

        # Apply attention weights to the values
        attended_features = attention_weights * value

        # Sum features from the adjacent slices
        # attended_features = torch.sum(attended_features, dim=1, keepdim=True)
        # Combine attended features with target features
        combined_features = attended_features #+ query
        return combined_features

class Attention(nn.Module):
    def __init__(self, feature_dim=16):
        super(Attention, self).__init__()
        self.feature_dim = feature_dim
        # self.conv1 = nn.Conv2d(1, feature_dim, kernel_size=3, padding=1)
        self.efficient_cross_attention = EfficientCrossAttention(feature_dim)
        self.conv_final = nn.Conv2d(feature_dim, 1, kernel_size=3, padding=1)

    def forward(self, target_slice, adjacent_slices):
        # Extract features from the target slice
        # target_features = F.relu(self.conv1(target_slice.unsqueeze(1)))

        # Apply efficient cross-attention
        combined_features = self.efficient_cross_attention(target_slice, adjacent_slices)


        # Reconstruct the target slice
        reconstruction = self.conv_final(combined_features)
        return reconstruction.squeeze()

# Example usage:
# model = MRIReconstructionModel()
# # target_slice should be the slice you want to reconstruct
# # adjacent_slices should be the stack of adjacent slices
# target_slice = torch.randn(10, 1, 256, 256)  # Batch size of 10, 1 slice, 256x256 per slice
# adjacent_slices = torch.randn(10, 2, 256, 256)  # Batch size of 10, 2 adjacent slices, 256x256 per slice
# output = model(target_slice, adjacent_slices)
