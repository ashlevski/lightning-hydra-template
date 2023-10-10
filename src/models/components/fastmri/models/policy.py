"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import functools
import operator
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

import fastmri


class LOUPEPolicy(nn.Module):
    """
    LOUPE policy model.
    """

    def __init__(
        self,
        num_actions: int,
        budget: int,
        use_softplus: bool = True,
        slope: float = 10,
        sampler_detach_mask: bool = False,
        straight_through_slope: float = 10,
        fix_sign_leakage: bool = True,
        st_clamp: bool = False,
    ):
        super().__init__()
        # shape = [1, W]
        self.use_softplus = use_softplus
        self.slope = slope
        self.straight_through_slope = straight_through_slope
        self.fix_sign_leakage = fix_sign_leakage
        self.st_clamp = st_clamp

        if use_softplus:
            # Softplus will be applied
            self.sampler = nn.Parameter(
                torch.normal(
                    torch.ones((1, num_actions)), torch.ones((1, num_actions)) / 10
                )
            )
        else:
            # Sigmoid will be applied
            self.sampler = nn.Parameter(torch.zeros((1, num_actions)))

        self.binarizer = ThresholdSigmoidMask.apply
        self.budget = budget
        self.sampler_detach_mask = sampler_detach_mask

    def forward(self, mask: torch.Tensor, kspace: torch.Tensor):
        B, M, H, W, C = kspace.shape  # batch, coils, height, width, complex
        # Reshape to [B, W]
        sampler_out = self.sampler.expand(mask.shape[0], -1)
        if self.use_softplus:
            # Softplus to make positive
            prob_mask = F.softplus(sampler_out, beta=self.slope)
            prob_mask = prob_mask / torch.max(
                (1 - mask.reshape(prob_mask.shape[0], prob_mask.shape[1])) * prob_mask,
                dim=1,
            )[0].reshape(-1, 1)
        else:
            # Sigmoid to make positive
            prob_mask = torch.sigmoid(self.slope * sampler_out)
        # Mask out already sampled rows
        masked_prob_mask = prob_mask * (
            1 - mask.reshape(prob_mask.shape[0], prob_mask.shape[1])
        )
        # Take out zero (masked) probabilities, since we don't want to include
        # those in the normalisation
        nonzero_idcs = (mask.view(B, W) == 0).nonzero(as_tuple=True)
        probs_to_norm = masked_prob_mask[nonzero_idcs].reshape(B, -1)
        # Rescale probabilities to desired sparsity.
        normed_probs = self.rescale_probs(probs_to_norm)
        # Reassign to original array
        masked_prob_mask[nonzero_idcs] = normed_probs.flatten()
        # Binarize the mask
        flat_bin_mask = self.binarizer(
            masked_prob_mask, self.straight_through_slope, self.st_clamp
        )
        # BCHW --> BW --> B11W1
        acquisitions = flat_bin_mask.reshape(B, 1, 1, W, 1)
        final_prob_mask = masked_prob_mask.reshape(B, 1, 1, W, 1)
        # B11W1
        mask = mask + acquisitions
        # BMHWC
        masked_kspace = mask * kspace
        if self.sampler_detach_mask:
            mask = mask.detach()
        # Note that since masked_kspace = mask * kspace, this kspace_pred will leak sign information
        if self.fix_sign_leakage:
            fix_sign_leakage_mask = torch.where(
                torch.bitwise_and(kspace < 0.0, mask == 0.0), -1.0, 1.0
            )
            masked_kspace = masked_kspace * fix_sign_leakage_mask
        return mask, masked_kspace, final_prob_mask

    def rescale_probs(self, batch_x: torch.Tensor):
        """
        Rescale Probability Map
        given a prob map x, rescales it so that it obtains the desired sparsity,
        specified by self.budget and the image size.

        if mean(x) > sparsity, then rescaling is easy: x' = x * sparsity / mean(x)
        if mean(x) < sparsity, one can basically do the same thing by rescaling
                                (1-x) appropriately, then taking 1 minus the result.
        """

        batch_size, W = batch_x.shape
        sparsity = self.budget / W
        ret = []
        for i in range(batch_size):
            x = batch_x[i : i + 1]
            xbar = torch.mean(x)
            r = sparsity / (xbar)
            beta = (1 - sparsity) / (1 - xbar)

            # compute adjustment
            le = torch.le(r, 1).float()
            ret.append(le * x * r + (1 - le) * (1 - (1 - x) * beta))

        return torch.cat(ret, dim=0)


class StraightThroughPolicy(nn.Module):
    """
    Policy model for active acquisition.
    """

    def __init__(
        self,
        budget: int,
        crop_size: Tuple[int, int] = (128, 128),
        slope: float = 10,
        sampler_detach_mask: bool = False,
        use_softplus: bool = True,
        straight_through_slope: float = 10,
        fix_sign_leakage: bool = True,
        st_clamp: bool = False,
        fc_size: int = 256,
        drop_prob: float = 0.0,
        num_fc_layers: int = 3,
        activation: str = "leakyrelu",
    ):
        super().__init__()

        self.sampler = LineConvSampler(
            input_dim=(2, *crop_size),
            slope=slope,
            use_softplus=use_softplus,
            fc_size=fc_size,
            num_fc_layers=num_fc_layers,
            drop_prob=drop_prob,
            activation=activation,
        )

        self.binarizer = ThresholdSigmoidMask.apply
        self.slope = slope
        self.straight_through_slope = straight_through_slope
        self.budget = budget
        self.sampler_detach_mask = sampler_detach_mask
        self.use_softplus = use_softplus
        self.fix_sign_leakage = fix_sign_leakage
        self.st_clamp = st_clamp
        self.fc_size = fc_size
        self.drop_prob = drop_prob
        self.num_fc_layers = num_fc_layers
        self.activation = activation

    def forward(self, kspace_pred: torch.Tensor, mask: torch.Tensor):
        B, C, H, W = kspace_pred.shape
        flat_prob_mask = self.sampler(kspace_pred, mask)
        # Take out zero (masked) probabilities, since we don't want to include
        # those in the normalisation
        nonzero_idcs = (mask.view(B, W) == 0).nonzero(as_tuple=True)
        probs_to_norm = flat_prob_mask[nonzero_idcs].reshape(B, -1)
        # Rescale probabilities to desired sparsity.
        normed_probs = self.rescale_probs(probs_to_norm)
        # Reassign to original array
        flat_prob_mask[nonzero_idcs] = normed_probs.flatten()
        # Binarize the mask
        flat_bin_mask = self.binarizer(
            flat_prob_mask, self.straight_through_slope, self.st_clamp
        )
        return flat_bin_mask, flat_prob_mask

    def do_acquisition(
        self,
        kspace: torch.Tensor,
        kspace_pred: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor,
    ):
        B, M, H, W, C = kspace.shape  # batch, coils, height, width, complex
        # BMHWC --> BHWC --> BCHW
        current_recon = (
            self.sens_reduce(kspace_pred, sens_maps).squeeze(1).permute(0, 3, 1, 2)
        )

        # BCHW --> BW --> B11W1
        acquisitions, flat_prob_mask = self(current_recon, mask)
        acquisitions = acquisitions.reshape(B, 1, 1, W, 1)
        prob_mask = flat_prob_mask.reshape(B, 1, 1, W, 1)

        # B11W1
        mask = mask + acquisitions
        # BMHWC
        masked_kspace = mask * kspace
        if self.sampler_detach_mask:
            mask = mask.detach()
        # Note that since masked_kspace = mask * kspace, this kspace_pred will leak sign information.
        if self.fix_sign_leakage:
            fix_sign_leakage_mask = torch.where(
                torch.bitwise_and(kspace < 0.0, mask == 0.0), -1.0, 1.0
            )
            masked_kspace = masked_kspace * fix_sign_leakage_mask
        return mask, masked_kspace, prob_mask

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        x = fastmri.ifft2c(x)
        return fastmri.complex_mul(x, fastmri.complex_conj(sens_maps)).sum(
            dim=1, keepdim=True
        )

    def rescale_probs(self, batch_x: torch.Tensor):
        """
        Rescale Probability Map
        given a prob map x, rescales it so that it obtains the desired sparsity,
        specified by self.budget and the image size.

        if mean(x) > sparsity, then rescaling is easy: x' = x * sparsity / mean(x)
        if mean(x) < sparsity, one can basically do the same thing by rescaling
                                (1-x) appropriately, then taking 1 minus the result.
        """

        batch_size, W = batch_x.shape
        sparsity = self.budget / W
        ret = []
        for i in range(batch_size):
            x = batch_x[i : i + 1]
            xbar = torch.mean(x)
            r = sparsity / (xbar)
            beta = (1 - sparsity) / (1 - xbar)

            # compute adjustment
            le = torch.le(r, 1).float()
            ret.append(le * x * r + (1 - le) * (1 - (1 - x) * beta))

        return torch.cat(ret, dim=0)


class ThresholdSigmoidMask(Function):
    def __init__(self):
        """
        Straight through estimator.
        The forward step stochastically binarizes the probability mask.
        The backward step estimate the non differentiable > operator using sigmoid with large slope (10).
        """
        super(ThresholdSigmoidMask, self).__init__()

    @staticmethod
    def forward(ctx, inputs, slope, clamp):
        batch_size = len(inputs)
        probs = []
        results = []

        for i in range(batch_size):
            x = inputs[i : i + 1]
            count = 0
            while True:
                prob = x.new(x.size()).uniform_()
                result = (x > prob).float()
                if torch.isclose(torch.mean(result), torch.mean(x), atol=1e-3):
                    break
                count += 1
                if count > 1000:
                    print(torch.mean(prob), torch.mean(result), torch.mean(x))
                    raise RuntimeError(
                        "Rejection sampled exceeded number of tries. Probably this means all "
                        "sampling probabilities are 1 or 0 for some reason, leading to divide "
                        "by zero in rescale_probs()."
                    )
            probs.append(prob)
            results.append(result)
        results = torch.cat(results, dim=0)
        probs = torch.cat(probs, dim=0)

        slope = torch.tensor(slope, requires_grad=False)
        ctx.clamp = clamp
        ctx.save_for_backward(inputs, probs, slope)
        return results

    @staticmethod
    def backward(ctx, grad_output):
        input, prob, slope = ctx.saved_tensors
        if ctx.clamp:
            grad_output = F.hardtanh(grad_output)
        # derivative of sigmoid function
        current_grad = (
            slope
            * torch.exp(-slope * (input - prob))
            / torch.pow((torch.exp(-slope * (input - prob)) + 1), 2)
        )
        return current_grad * grad_output, None, None


class SingleConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, relu activation and dropout.
    """

    def __init__(
        self, in_chans: int, out_chans: int, drop_prob: float = 0, pool_size: int = 2
    ):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
            drop_prob (float): Dropout probability.
            pool_size (int): Size of 2D max-pooling operator.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob
        self.pool_size = pool_size

        layers = [
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_chans),
            nn.ReLU(),
            nn.Dropout2d(drop_prob),
        ]

        if pool_size > 1:
            layers.append(nn.MaxPool2d(pool_size))

        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        return self.layers(input)

    def __repr__(self):
        return (
            f"ConvBlock(in_chans={self.in_chans}, out_chans={self.out_chans}, "
            f"drop_prob={self.drop_prob}, max_pool_size={self.pool_size})"
        )


class LineConvSampler(nn.Module):
    def __init__(
        self,
        input_dim: tuple = (2, 128, 128),
        chans: int = 16,
        num_pool_layers: int = 4,
        fc_size: int = 256,
        drop_prob: float = 0,
        slope: float = 10,
        use_softplus: bool = True,
        num_fc_layers: int = 3,
        activation: str = "leakyrelu",
    ):
        """
        Args:
            in_chans (int): Number of channels in the input to the U-Net model.
            input_dim (tuple): Input size of reconstructed images (C, H, W).
            chans (int): Number of output channels of the first convolution layer.
            num_pool_layers (int): Number of down-sampling layers.
            fc_size (int): Number of hidden neurons for the fully connected layers.
            drop_prob (float): Dropout probability.
            num_fc_layers (int): Number of fully connected layers to use after convolutional part.
            use_softplus (bool): Whether to use softplus as final activation (otherwise sigmoid).
            activation (str): Activation function to use: leakyrelu or elu.
        """
        super().__init__()
        assert len(input_dim) == 3
        self.input_dim = input_dim
        self.in_chans = input_dim[0]
        self.num_actions = input_dim[-1]
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.fc_size = fc_size
        self.drop_prob = drop_prob
        self.pool_size = 2
        self.slope = slope
        self.use_softplus = use_softplus
        self.num_fc_layers = num_fc_layers
        self.activation = activation

        # Initial from in_chans to chans
        self.channel_layer = SingleConvBlock(
            self.in_chans, chans, drop_prob, pool_size=1
        )

        # Downsampling convolution
        # These are num_pool_layers layers where each layers 2x2 max pools, and doubles
        # the number of channels.
        self.down_sample_layers = nn.ModuleList(
            [
                SingleConvBlock(
                    chans * 2**i,
                    chans * 2 ** (i + 1),
                    drop_prob,
                    pool_size=self.pool_size,
                )
                for i in range(num_pool_layers)
            ]
        )

        self.feature_extractor = nn.Sequential(
            self.channel_layer, *self.down_sample_layers
        )

        # Dynamically determinte size of fc_layer
        self.flattened_size = functools.reduce(
            operator.mul, list(self.feature_extractor(torch.rand(1, *input_dim)).shape)
        )

        fc_out: List[nn.Module] = []
        for layer in range(self.num_fc_layers):
            in_features = fc_size
            out_features = fc_size
            if layer == 0:
                in_features = self.flattened_size
            if layer + 1 == self.num_fc_layers:
                out_features = self.num_actions
            fc_out.append(nn.Linear(in_features=in_features, out_features=out_features))

            if layer + 1 < self.num_fc_layers:
                act: nn.Module
                if activation == "leakyrelu":
                    act = nn.LeakyReLU()
                elif activation == "elu":
                    act = nn.ELU()
                else:
                    raise RuntimeError(
                        f"Invalid activation function {activation}. Should be leakyrelu or elu."
                    )
                fc_out.append(act)

        self.fc_out = nn.Sequential(*fc_out)

    def forward(self, image, mask):
        """
        Args:
            image (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
            mask (torch.Tensor): Input tensor of shape [resolution], containing 0s and 1s
        Returns:
            torch.Tensor: prob_mask [batch_size, num_actions] corresponding to all actions at the
            given observation. Gives probabilities of sampling a particular action.
        """

        # Image embedding
        image_emb = self.feature_extractor(image)
        # flatten all but batch dimension before FC layers
        image_emb = image_emb.flatten(start_dim=1)
        out = self.fc_out(image_emb)

        if self.use_softplus:
            # Softplus to make positive
            out = F.softplus(out, beta=self.slope)
            # Make sure max probability is 1, but ignore already sampled rows for this normalisation, since
            #  those get masked out later anyway.
            prob_mask = out / torch.max(
                (1 - mask.reshape(out.shape[0], out.shape[1])) * out, dim=1
            )[0].reshape(-1, 1)
        else:
            prob_mask = torch.sigmoid(self.slope * out)
        # Mask out already sampled rows
        prob_mask = prob_mask * (
            1 - mask.reshape(prob_mask.shape[0], prob_mask.shape[1])
        )
        assert len(prob_mask.shape) == 2
        return prob_mask
