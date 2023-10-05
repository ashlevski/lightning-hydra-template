# coding=utf-8
# Copyright (c) DIRECT Contributors

"""This module contains mathematical optimization techniques specific to MRI."""

from typing import Any, Callable, Dict

import numpy as np
import torch
from torch import nn

from direct.algorithms.optimization import MaximumEigenvaluePowerMethod
from direct.data.transforms import crop_to_acs, view_as_complex, view_as_real
from direct.types import KspaceKey
from direct.utils import DirectModule


class EspiritCalibration(DirectModule):
    """Estimates sensitivity maps estimated with the ESPIRIT calibration method as described in [1]_.

    We adapted code for ESPIRIT method adapted from [2]_.

    References
    ----------

    .. [1] Uecker M, Lai P, Murphy MJ, Virtue P, Elad M, Pauly JM, Vasanawala SS, Lustig M. ESPIRiT--an eigenvalue
        approach to autocalibrating parallel MRI: where SENSE meets GRAPPA. Magn Reson Med. 2014 Mar;71(3):990-1001.
        doi: 10.1002/mrm.24751. PMID: 23649942; PMCID: PMC4142121.
    .. [2] https://github.com/mikgroup/sigpy/blob/1817ff849d34d7cbbbcb503a1b310e7d8f95c242/sigpy/mri/app.py#L388-L491

    """

    def __init__(
        self,
        backward_operator: Callable,
        threshold: float = 0.05,
        kernel_size: int = 6,
        crop: float = 0.95,
        max_iter: int = 100,
        kspace_key: KspaceKey = KspaceKey.masked_kspace,
    ):
        """Inits :class:`EstimateSensitivityMap`.

        Parameters
        ----------
        backward_operator: Callable
            The backward operator, e.g. some form of inverse FFT (centered or uncentered).
        threshold: float, optional
            Threshold for the calibration matrix. Default: 0.05.
        kernel_size: int, optional
            Kernel size for the calibration matrix. Default: 6.
        crop: float, optional
            Output eigenvalue cropping threshold. Default: 0.95.
        max_iter: int, optional
            Power method iterations. Default: 30.
        kspace_key: KspaceKey
            K-space key. Default KspaceKey.masked_kspace.
        """
        self.backward_operator = backward_operator
        self.threshold = threshold
        self.kernel_size = kernel_size
        self.crop = crop
        self.max_iter = max_iter
        self.kspace_key = kspace_key
        super().__init__()

    def calculate_sensitivity_map(self, acs_mask: torch.Tensor, kspace: torch.Tensor) -> torch.Tensor:
        """Calculates sensitivity map given as input the `acs_mask` and the `k-space`.

        Parameters
        ----------
        acs_mask : torch.Tensor
            Autocalibration mask.
        kspace : torch.Tensor
            K-space.

        Returns
        -------
        sensitivity_map : torch.Tensor
        """
        # pylint: disable=too-many-locals
        ndim = kspace.ndim - 2
        spatial_size = kspace.shape[1:-1]

        # Used in case the k-space is padded (e.g. for batches)
        non_padded_dim = kspace.clone().sum(dim=tuple(range(1, kspace.ndim))).bool()

        num_coils = non_padded_dim.sum()
        acs_kspace_cropped = view_as_complex(crop_to_acs(acs_mask.squeeze(), kspace[non_padded_dim]))

        # Get calibration matrix.
        calibration_matrix = (
            nn.functional.unfold(acs_kspace_cropped.unsqueeze(1), kernel_size=self.kernel_size, stride=1)
            .transpose(1, 2)
            .to(acs_kspace_cropped.device)
            .reshape(
                num_coils,
                *(np.array(acs_kspace_cropped.shape[1:3]) - self.kernel_size + 1),
                *([self.kernel_size] * ndim),
            )
        )
        calibration_matrix = calibration_matrix.reshape(num_coils, -1, self.kernel_size**ndim)
        calibration_matrix = calibration_matrix.permute(1, 0, 2)
        calibration_matrix = calibration_matrix.reshape(-1, num_coils * self.kernel_size**ndim)

        # Perform SVD on calibration matrix
        _, s, vh = torch.linalg.svd(calibration_matrix, full_matrices=True)
        vh = vh[s > (self.threshold * s.max()), :]

        # Get kernels
        num_kernels = vh.shape[0]
        kernels = vh.reshape([num_kernels, num_coils] + [self.kernel_size] * ndim)

        # Get covariance matrix in image domain
        covariance = torch.zeros(
            spatial_size[::-1] + (num_coils, num_coils),
            dtype=kernels.dtype,
            device=kernels.device,
        )
        for kernel in kernels:
            pad_h, pad_w = (
                spatial_size[0] - self.kernel_size,
                spatial_size[1] - self.kernel_size,
            )
            pad = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
            kernel_padded = torch.nn.functional.pad(kernel, pad)

            img_kernel = self.backward_operator(kernel_padded, dim=(1, 2), complex_input=False)
            aH = img_kernel.permute(*torch.arange(img_kernel.ndim - 1, -1, -1)).unsqueeze(-1)
            a = aH.transpose(-1, -2).conj()
            covariance += aH @ a

        covariance = covariance * (np.prod(spatial_size) / self.kernel_size**ndim)
        sensitivity_map = torch.ones(
            (*spatial_size[::-1], num_coils, 1),
            dtype=kernels.dtype,
            device=kernels.device,
        )

        def forward(x):
            return covariance @ x

        def normalize(x):
            return (x.abs() ** 2).sum(dim=-2, keepdims=True) ** 0.5

        power_method = MaximumEigenvaluePowerMethod(forward, max_iter=self.max_iter, norm_func=normalize)
        power_method.fit(x=sensitivity_map)

        temp_sensitivity_map = power_method.x.squeeze(-1)
        temp_sensitivity_map = temp_sensitivity_map.permute(
            *torch.arange(temp_sensitivity_map.ndim - 1, -1, -1)
        ).squeeze(-1)
        temp_sensitivity_map = temp_sensitivity_map * temp_sensitivity_map.conj() / temp_sensitivity_map.abs()

        max_eig = power_method.max_eig.squeeze()
        max_eig = max_eig.permute(*torch.arange(max_eig.ndim - 1, -1, -1))
        temp_sensitivity_map = temp_sensitivity_map * (max_eig > self.crop)

        sensitivity_map = torch.zeros_like(kspace, device=kspace.device, dtype=kspace.dtype)
        sensitivity_map[non_padded_dim] = view_as_real(temp_sensitivity_map)
        return sensitivity_map

    def forward(self, sample: Dict[str, Any]) -> torch.Tensor:
        """Forward method of :class:`EspiritCalibration`.

        Parameters
        ----------
        sample: Dict[str, Any]
             Contains key `kspace_key`.

        Returns
        -------
        sample: Dict[str, Any]
             Contains key 'sampling_mask'.
        """
        acs_mask = sample["acs_mask"]
        kspace = sample[self.kspace_key]
        sensitivity_map = torch.stack(
            [self.calculate_sensitivity_map(acs_mask[_], kspace[_]) for _ in range(kspace.shape[0])],
            dim=0,
        ).to(kspace.device)

        return sensitivity_map
