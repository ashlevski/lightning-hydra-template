import torch
import torch.nn as nn

class BatchNormalizedMSE(nn.Module):
    def __init__(self):
        super(BatchNormalizedMSE, self).__init__()

    def forward(self, y_pred, y_true):
        """
        Calculate the Normalized Mean Square Error across a batch.
        :param y_pred: Predicted values (torch.Tensor) of shape [B, H, W]
        :param y_true: Actual values (torch.Tensor) of shape [B, H, W]
        :return: NMSE (torch.Tensor)
        """
        mse = torch.mean((y_true - y_pred) ** 2, dim=[1, 2])  # Mean over H and W dimensions
        norm_factor = torch.mean(y_true ** 2, dim=[1, 2])  # Normalization factor over H and W dimensions
        nmse = mse / norm_factor
        nmse = torch.mean(nmse)  # Mean over the batch
        return nmse
