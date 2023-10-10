"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from collections import defaultdict
from typing import Optional, Tuple

import torch
import torch.nn as nn

import fastmri
from fastmri.data import transforms

from .policy import LOUPEPolicy, StraightThroughPolicy
from .varnet import NormUnet


class AdaptiveSensitivityModel(nn.Module):
    """
    Model for learning sensitivity estimation from k-space data.

    This model applies an IFFT to multichannel k-space data and then a U-Net
    to the coil images to estimate coil sensitivities. It can be used with the
    end-to-end variational network.
    """

    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
        num_sense_lines: Optional[int] = None,
    ):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pools: Number of down-sampling and up-sampling layers.
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            drop_prob: Dropout probability.
            num_sense_lines: Number of low-frequency lines to use for sensitivity map
                computation, must be even or `None`. Default `None` will automatically
                compute the number from masks. Default behaviour may cause some slices to
                use more low-frequency lines than others, when used in conjunction with
                e.g. the EquispacedMaskFunc defaults.
        """
        super().__init__()

        self.num_sense_lines = num_sense_lines
        self.norm_unet = NormUnet(
            chans,
            num_pools,
            in_chans=in_chans,
            out_chans=out_chans,
            drop_prob=drop_prob,
        )

    def chans_to_batch_dim(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        b, c, h, w, comp = x.shape

        return x.view(b * c, 1, h, w, comp), b

    def batch_chans_to_chan_dim(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        bc, _, h, w, comp = x.shape
        c = bc // batch_size

        return x.view(batch_size, c, h, w, comp)

    def divide_root_sum_of_squares(self, x: torch.Tensor) -> torch.Tensor:
        return x / fastmri.rss_complex(x, dim=1).unsqueeze(-1).unsqueeze(1)

    def get_pad_and_num_low_freqs(
        self, mask: torch.Tensor, num_sense_lines: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # get low frequency line locations and mask them out
        squeezed_mask = mask[:, 0, 0, :, 0]
        cent = squeezed_mask.shape[1] // 2
        # running argmin returns the first non-zero
        left = torch.argmin(squeezed_mask[:, :cent].flip(1), dim=1)
        right = torch.argmin(squeezed_mask[:, cent:], dim=1)
        num_low_freqs = torch.max(
            2 * torch.min(left, right), torch.ones_like(left)
        )  # force a symmetric center unless 1

        if self.num_sense_lines is not None:  # Use pre-specified number instead
            if (num_low_freqs < num_sense_lines).all():
                raise RuntimeError(
                    "`num_sense_lines` cannot be greater than the actual number of "
                    "low-frequency lines in the mask: {}".format(num_low_freqs)
                )
            num_low_freqs = num_sense_lines * torch.ones(
                mask.shape[0], dtype=mask.dtype, device=mask.device
            )

        pad = (mask.shape[-2] - num_low_freqs + 1) // 2
        return pad, num_low_freqs

    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pad, num_low_freqs = self.get_pad_and_num_low_freqs(mask, self.num_sense_lines)
        x = transforms.batched_mask_center(masked_kspace, pad, pad + num_low_freqs)

        # convert to image space
        x = fastmri.ifft2c(x)
        x, b = self.chans_to_batch_dim(x)
        # NOTE: Channel dimensions have been converted to batch dimensions, so this
        #  acts like a UNet that treats every coil as a separate image!
        # estimate sensitivities
        x = self.norm_unet(x)
        x = self.batch_chans_to_chan_dim(x, b)
        x = self.divide_root_sum_of_squares(x)

        return x


class AdaptiveVarNet(nn.Module):
    """
    A full adaptive variational network model. This model uses a policy to do
    end-to-end adaptive acquisition and reconstruction.

    This model applies a combination of soft data consistency with a U-Net
    regularizer. To use non-U-Net regularizers, use VarNetBlock.
    """

    def __init__(
        self,
        budget: int = 22,
        num_cascades: int = 12,
        sens_chans: int = 8,
        sens_pools: int = 4,
        chans: int = 18,
        pools: int = 4,
        cascades_per_policy: int = 1,
        loupe_mask: bool = False,
        use_softplus: bool = True,
        crop_size: Tuple[int, int] = (128, 128),
        num_actions: Optional[int] = None,
        num_sense_lines: Optional[int] = None,
        hard_dc: bool = False,
        dc_mode: str = "simul",
        slope: float = 10,
        sparse_dc_gradients: bool = True,
        straight_through_slope: float = 10,
        st_clamp: bool = False,
        policy_fc_size: int = 256,
        policy_drop_prob: float = 0.0,
        policy_num_fc_layers: int = 3,
        policy_activation: str = "leakyrelu",
    ):
        """
        Args:
            budget: Total number of acquisitions to do.
            num_cascades: Number of cascades (i.e., layers) for variational
                network.
            sens_chans: Number of channels for sensitivity map U-Net.
            sens_pools Number of downsampling and upsampling layers for
                sensitivity map U-Net.
            chans: Number of channels for cascade U-Net.
            pools: Number of downsampling and upsampling layers for cascade
                U-Net.
            cascades_per_policy: How many cascades to use per policy step.
                Policies will be applied starting after first cascade, and then
                every cascades_per_policy cascades after. Note that
                num_cascades % cascades_per_policy should equal 1. There is an
                option to set cascades_per_policy equal to num_cascades as well,
                in which case the policy will be applied before the first
                cascade only.
            loupe_mask: Whether to use LOUPE-like mask instead of equispaced
                (still keeps center lines).
            use_softplus: Whether to use softplus or sigmoid in LOUPE.
            crop_size: tuple, crop size of MR images.
            num_actions: Number of possible actions to sample (=image width).
                Used only when loupe_mask is True.
            num_sense_lines: Number of low-frequency lines to use for
                sensitivity map computation, must be even or `None`. Default
                `None` will automatically compute the number from masks.
                Default behaviour may cause some slices to use more
                low-frequency lines than others, when used in conjunction with
                e.g. the EquispacedMaskFunc defaults.
            hard_dc: Whether to do hard DC layers instead of soft (learned).
            dc_mode: Whether to do DC before ('first'), after ('last') or
                simultaneously ('simul') with Refinement step. Default 'simul'.
            slope: Slope to use for sigmoid in LOUPE and Policy forward, or
                beta to use in softplus.
            sparse_dc_gradients: Whether to sparsify the gradients in DC by
                using torch.where() with the mask: this essentially removes
                gradients for the policy on unsampled rows.
            straight_through_slope: Slope to use in Straight Through estimator.
            st_clamp: Whether to clamp gradients between -1 and 1 in straight
                through estimator.
            policy_fc_size: int, size of fully connected layers in Policy
                architecture.
            policy_drop_prob: float, dropout probability of convolutional
                layers in Policy.
            policy_num_fc_layers: int, number of fully-connected layers to
                apply after the convolutional layers in the policy.
            policy_activation: str, "leakyrelu" or "elu". Activation function
                to use between fully-connected layers in the policy. Only used
                if policy_num_fc_layers > 1.
        """
        super().__init__()

        self.budget = budget
        self.cascades_per_policy = cascades_per_policy
        self.loupe_mask = loupe_mask
        self.use_softplus = use_softplus
        self.crop_size = crop_size
        self.num_actions = num_actions
        self.num_sense_lines = num_sense_lines
        self.hard_dc = hard_dc
        self.dc_mode = dc_mode

        self.slope = slope
        self.sparse_dc_gradients = sparse_dc_gradients
        self.straight_through_slope = straight_through_slope

        self.st_clamp = st_clamp

        self.policy_fc_size = policy_fc_size
        self.policy_drop_prob = policy_drop_prob
        self.policy_num_fc_layers = policy_num_fc_layers
        self.policy_activation = policy_activation

        self.sens_net = AdaptiveSensitivityModel(
            sens_chans, sens_pools, num_sense_lines=num_sense_lines
        )
        self.cascades = nn.ModuleList(
            [
                AdaptiveVarNetBlock(
                    NormUnet(chans, pools),
                    hard_dc=hard_dc,
                    dc_mode=dc_mode,
                    sparse_dc_gradients=sparse_dc_gradients,
                )
                for _ in range(num_cascades)
            ]
        )

        # LOUPE or adaptive policies
        if self.loupe_mask:
            assert isinstance(self.num_actions, int)
            self.loupe = LOUPEPolicy(
                self.num_actions,
                self.budget,
                use_softplus=self.use_softplus,
                slope=self.slope,
                straight_through_slope=self.straight_through_slope,
                st_clamp=self.st_clamp,
            )
        else:
            # Define policies. If budget is not cleanly divided by num_cascades - 1,
            # then put all remaining acquisitions in last cascade.
            remaining_budget = self.budget
            if cascades_per_policy > num_cascades:
                raise RuntimeError(
                    "Number of cascades {} cannot be smaller than number of cascades "
                    "per policy {}.".format(num_cascades, cascades_per_policy)
                )
            elif num_cascades != cascades_per_policy:
                base_budget = self.budget // ((num_cascades - 1) // cascades_per_policy)
                policies = []
                for i in range(1, num_cascades):  # First cascade has no policy
                    # 5/5 --> no i (special case, see if-else outside this for-loop)
                    # 5/4 --> i = 1
                    # 5/3 --> i = 2
                    # 5/2 --> i = 1, 3
                    # 5/1 --> i = 1, 2, 3, 4
                    if (
                        num_cascades - i
                    ) % cascades_per_policy == 0:  # Count from the back
                        if remaining_budget < 2 * base_budget:
                            policy = StraightThroughPolicy(
                                remaining_budget,
                                crop_size,
                                slope=self.slope,
                                use_softplus=self.use_softplus,
                                straight_through_slope=self.straight_through_slope,
                                st_clamp=self.st_clamp,
                                fc_size=self.policy_fc_size,
                                drop_prob=self.policy_drop_prob,
                                num_fc_layers=self.policy_num_fc_layers,
                                activation=self.policy_activation,
                            )
                            remaining_budget = 0
                        else:
                            policy = StraightThroughPolicy(
                                base_budget,
                                crop_size,
                                slope=self.slope,
                                use_softplus=self.use_softplus,
                                straight_through_slope=self.straight_through_slope,
                                st_clamp=self.st_clamp,
                                fc_size=self.policy_fc_size,
                                drop_prob=self.policy_drop_prob,
                                num_fc_layers=self.policy_num_fc_layers,
                                activation=self.policy_activation,
                            )
                            remaining_budget -= base_budget
                        policies.append(policy)
            else:  # Will do single policy immediately before first cascade
                policies = [
                    StraightThroughPolicy(
                        self.budget,
                        crop_size,
                        slope=self.slope,
                        use_softplus=self.use_softplus,
                        straight_through_slope=self.straight_through_slope,
                        st_clamp=self.st_clamp,
                        fc_size=self.policy_fc_size,
                        drop_prob=self.policy_drop_prob,
                        num_fc_layers=self.policy_num_fc_layers,
                        activation=self.policy_activation,
                    )
                ]

            self.policies = nn.ModuleList(policies)

    def forward(
        self,
        kspace: torch.Tensor,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
    ):

        extra_outputs = defaultdict(list)

        # Make it so that masked_kspace and mask are reduced to center only.
        mask, masked_kspace = self.extract_low_freq_mask(mask, masked_kspace)
        extra_outputs["masks"].append(mask)

        # Sensitivity
        sens_maps = self.sens_net(masked_kspace, mask)
        extra_outputs["sense"].append(sens_maps)

        # Store current reconstruction
        current_recon = fastmri.complex_abs(
            self.sens_reduce(masked_kspace, sens_maps)
        ).squeeze(1)
        extra_outputs["recons"].append(current_recon.detach().cpu())

        # Sample LOUPE mask
        if self.loupe_mask:
            mask, masked_kspace, prob_mask = self.loupe(mask, kspace)

            extra_outputs["masks"].append(mask)
            extra_outputs["prob_masks"].append(prob_mask)
            # Store current reconstruction
            current_recon = fastmri.complex_abs(
                self.sens_reduce(masked_kspace, sens_maps)
            ).squeeze(1)
            extra_outputs["recons"].append(current_recon.detach().cpu())

        if self.cascades_per_policy == len(self.cascades) and not self.loupe_mask:
            # Special setting: do policy once before any cascade only.
            if len(self.policies) != 1:
                raise ValueError(
                    "Must have only one policy when number of cascades "
                    f"{len(self.cascades)} equals the number of cascades_per_policy "
                    f"{self.cascades_per_policy}."
                )
            kspace_pred = masked_kspace.clone()
            mask, masked_kspace, prob_mask = self.policies[0].do_acquisition(
                kspace, kspace_pred, mask, sens_maps
            )
            extra_outputs["masks"].append(mask)
            extra_outputs["prob_masks"].append(prob_mask)

        kspace_pred = masked_kspace.clone()

        j = 0  # Keep track of policy number
        for i, cascade in enumerate(self.cascades):
            kspace_pred = cascade(
                kspace_pred, masked_kspace, mask, sens_maps, kspace=kspace
            )

            # Store current reconstruction
            current_recon = fastmri.complex_abs(
                self.sens_reduce(masked_kspace, sens_maps)
            ).squeeze(1)
            extra_outputs["recons"].append(current_recon.detach().cpu())

            if i == len(self.cascades) - 1 or self.loupe_mask:
                continue  # Don't do acquisition, just reconstruct

            # Count from the back
            if (
                len(self.cascades) - (i + 1)
            ) % self.cascades_per_policy == 0 and self.cascades_per_policy != len(
                self.cascades
            ):
                mask, masked_kspace, prob_mask = self.policies[j].do_acquisition(
                    kspace, kspace_pred, mask, sens_maps
                )
                j += 1

                extra_outputs["masks"].append(mask)
                extra_outputs["prob_masks"].append(prob_mask)

        # Could presumably do complex_abs(complex_rss()) instead and get same result?
        output = fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(kspace_pred)), dim=1)
        # Add final reconstruction image
        extra_outputs["recons"].append(output.detach().cpu())
        return output, extra_outputs

    def extract_low_freq_mask(self, mask: torch.Tensor, masked_kspace: torch.Tensor):
        """
        Extracts low frequency components that are used by sensitivity map
        computation. This serves as the starting point for active acquisition.
        """
        pad, num_low_freqs = self.sens_net.get_pad_and_num_low_freqs(
            mask, self.num_sense_lines
        )

        mask = transforms.batched_mask_center(mask, pad, pad + num_low_freqs)
        masked_kspace = transforms.batched_mask_center(
            masked_kspace, pad, pad + num_low_freqs
        )

        return mask, masked_kspace

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        x = fastmri.ifft2c(x)
        return fastmri.complex_mul(x, fastmri.complex_conj(sens_maps)).sum(
            dim=1, keepdim=True
        )


class AdaptiveVarNetBlock(nn.Module):
    """
    Model block for adaptive end-to-end variational network.

    This model applies a combination of soft data consistency with the input
    model as a regularizer. A series of these blocks can be stacked to form
    the full variational network.
    """

    def __init__(
        self,
        model: nn.Module,
        inter_sens: bool = True,
        hard_dc: bool = False,
        dc_mode: str = "simul",
        sparse_dc_gradients: bool = True,
    ):
        """
        Args:
            model: Module for "regularization" component of variational
                network.
            inter_sens: boolean, whether to do reduction and expansion using
                estimated sensitivity maps.
            hard_dc: boolean, whether to do hard DC layer instead of soft.
            dc_mode: str, whether to do DC before ('first'), after ('last') or
                simultaneously ('simul') with Refinement step. Default 'simul'.
            sparse_dc_gradients: Whether to sparsify the gradients in DC by
                using torch.where() with the mask: this essentially removes
                gradients for the policy on unsampled rows.
        """
        super().__init__()

        self.model = model
        self.inter_sens = inter_sens
        self.hard_dc = hard_dc
        self.dc_mode = dc_mode
        self.sparse_dc_gradients = sparse_dc_gradients

        if dc_mode not in ["first", "last", "simul"]:
            raise ValueError(
                "`dc_mode` must be one of 'first', 'last', or 'simul'. "
                "Not {}".format(dc_mode)
            )

        if hard_dc:
            self.dc_weight = 1
        else:
            self.dc_weight = nn.Parameter(torch.ones(1))  # type: ignore

    def sens_expand(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        return fastmri.fft2c(fastmri.complex_mul(x, sens_maps))

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        x = fastmri.ifft2c(x)
        return fastmri.complex_mul(x, fastmri.complex_conj(sens_maps)).sum(
            dim=1, keepdim=True
        )

    def forward(
        self,
        current_kspace: torch.Tensor,
        ref_kspace: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor,
        kspace: Optional[torch.Tensor],
    ) -> torch.Tensor:
        zero = torch.zeros(1, 1, 1, 1, 1).to(current_kspace)

        if self.dc_mode == "first":
            # DC before Refinement, this directly puts kspace rows from ref_kspace
            #  into current_kspace if dc_weight = 1.
            if self.sparse_dc_gradients:
                current_kspace = (
                    current_kspace
                    - torch.where(mask.byte(), current_kspace - ref_kspace, zero)
                    * self.dc_weight
                )
            else:
                # Values in current_kspace that should be replaced by actual sampled
                # information
                dc_kspace = current_kspace * mask
                # don't need to multiply ref_kspace by mask because ref_kspace is 0
                # where mask is 0
                current_kspace = (
                    current_kspace - (dc_kspace - ref_kspace) * self.dc_weight
                )

        model_term = self.sens_expand(
            self.model(self.sens_reduce(current_kspace, sens_maps)),
            sens_maps,
        )

        if self.dc_mode == "first":
            return current_kspace - model_term
        elif self.dc_mode == "simul":
            # Default implementation: simultaneous DC and Refinement
            if self.sparse_dc_gradients:
                soft_dc = (
                    torch.where(mask.byte(), current_kspace - ref_kspace, zero)
                    * self.dc_weight
                )
            else:
                # Values in current_kspace that should be replaced by actual sampled
                # information
                dc_kspace = current_kspace * mask
                soft_dc = (dc_kspace - ref_kspace) * self.dc_weight
            return current_kspace - soft_dc - model_term
        elif self.dc_mode == "last":
            combined_kspace = current_kspace - model_term

            if self.sparse_dc_gradients:
                combined_kspace = (
                    combined_kspace
                    - torch.where(mask.byte(), combined_kspace - ref_kspace, zero)
                    * self.dc_weight
                )
            else:
                # Values in combined_kspace that should be replaced by actual sampled
                # information
                dc_kspace = combined_kspace * mask
                # don't need to multiply ref_kspace by mask because ref_kspace is 0
                # where mask is 0
                combined_kspace = (
                    combined_kspace - (dc_kspace - ref_kspace) * self.dc_weight
                )

            return combined_kspace
        else:
            raise ValueError(
                "`dc_mode` must be one of 'first', 'last', or 'simul'. "
                "Not {}".format(self.dc_mode)
            )
