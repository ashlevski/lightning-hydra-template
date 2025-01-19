from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch import nn, Tensor, Size
import torch.nn.functional as F

from projects.longitudinal.torchir.utils import identity_grid, Resampler
from projects.longitudinal.torchir.utils import bspline_convolution_kernel
from projects.longitudinal.torchir.utils import (
    batch_transform_efficient,
    rotation_matrix_eff,
    shearing_matrix_eff,
    scaling_matrix_eff,
)


class Transformer(ABC, nn.Module):
    def __init__(self, ndim: int, coord_dim: int = 1):
        super().__init__()
        self.ndim = ndim
        self.coord_dim = coord_dim
        self._resampler = Resampler(coord_dim=coord_dim)

    @abstractmethod
    def apply_transform(
        self,
        parameters: Tensor,
        fixed_image: Tensor,
        moving_image: Tensor,
        coordinate_grid: Optional[Tensor] = None,
    ) -> Tensor:
        """apply the parameters to get the transformed coord_grid"""
        pass

    def forward(
        self,
        parameters: Tensor,
        fixed_image: Tensor,
        moving_image: Tensor,
        coordinate_grid: Optional[Tensor] = None,
        return_coordinate_grid: bool = False,
    ) -> Tuple[Tensor]:

        coordinate_grid = self.apply_transform(
            parameters, fixed_image, moving_image, coordinate_grid
        )

        ret = self._resampler(moving_image, coordinate_grid)
        if return_coordinate_grid:
            ret = (ret, coordinate_grid)
        return ret


class BsplineTransformer(Transformer):
    """
    third order bspline. upsampling via transposed convolutions.
    """

    def __init__(
        self,
        *args,
        upsampling_factors: Tuple[int],
        order: int = 3,
        trainable: bool = False,
        apply_diffeomorphic_limits: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.upsampling_factors = upsampling_factors
        ConvTranspose = (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)[
            self.ndim - 1
        ]

        bspline_kernel = self.make_bspline_kernel(self.upsampling_factors, order=order)
        kernel_size = bspline_kernel.shape
        crop_size = tuple(int(el * 3 / 8) if el != 5 else 2 for el in kernel_size)
        upsampler = ConvTranspose(
            1,
            1,
            kernel_size,
            stride=self.upsampling_factors,
            padding=crop_size,
            bias=False,
        )
        upsampler.weight = nn.Parameter(
            bspline_kernel[None, None], requires_grad=trainable
        )

        self._upsampler = upsampler
        self.apply_diffeomorphic_limits = apply_diffeomorphic_limits

    @staticmethod
    def make_bspline_kernel(upsampling_factors, order, dtype=torch.float32):
        # TODO: solve the issues with kernel shapes
        bspline_kernel = bspline_convolution_kernel(
            upsampling_factors, order=order, dtype=dtype
        )

        if (np.array(bspline_kernel.shape[::-1]) == 4).any() or (
            np.array(bspline_kernel.shape[::-1]) == 2
        ).any():  # hack to deal with 1 strides and kernel size of 4
            padding = list()
            for s in bspline_kernel.shape[::-1]:
                if s == 4 or s == 2:
                    padding.extend([1, 0])
                else:
                    padding.extend([0, 0])
            bspline_kernel = F.pad(bspline_kernel, padding, mode="constant")

        return bspline_kernel

    def create_dvf(self, bspline_parameters, output_shape):
        assert bspline_parameters.shape[1] == self.ndim
        shape = bspline_parameters.shape
        dvf = self._upsampler(
            bspline_parameters.view((shape[0] * self.ndim, 1) + shape[2:]),
            output_size=output_shape,
        )
        newshape = dvf.shape

        return dvf.view((shape[0], self.ndim) + newshape[2:])

    def apply_transform(
        self,
        bspline_coefficients: Tensor,
        fixed_image: Tensor,
        moving_image: Tensor,
        coordinate_grid: Tensor = None,
    ):
        device = bspline_coefficients.device
        dtype = bspline_coefficients.dtype
        if self.apply_diffeomorphic_limits:
            bspline_coefficients = torch.tanh(bspline_coefficients) * (
                0.4
                * torch.tensor(
                    self.upsampling_factors[::-1],
                    dtype=dtype,
                    device=device,
                ).view((1, self.ndim) + (1,) * self.ndim)
            )

        shape = fixed_image.shape[2:]
        dvf = self.create_dvf(bspline_coefficients, output_shape=shape)
        new_grid = dvf + identity_grid(
            dvf.shape[2:], stackdim=0, dtype=dtype, device=device
        )
        if coordinate_grid is not None:
            assert shape == coordinate_grid.shape[2:]
            new_grid = self._resampler(coordinate_grid, new_grid)
        return new_grid


class AffineTransformer(Transformer):
    def apply_transform(
        self,
        parameters: Tuple[Tensor],
        fixed_image: Tensor,
        moving_image: Tensor,
        coordinate_grid: Optional[Tensor] = None,
    ) -> Tensor:
        """
        :param parameters: translation, rotation, scale, shear
        :param fixed_image:
        :param moving_image:
        :param coordinate_grid:
        :return:

        Note: If ndim == 3, number of parameters are 3 for translation, rotation, and scale; and
        6 for shear. If ndim == 2, number of parameters are 2 for translation, scale, and shear;
        and 1 for rotation
        """
        translation, rotation, scale, shear = parameters

        if self.ndim == 2:
            rot_mat = rotation_matrix_eff(rotation, ndim=self.ndim)
        elif self.ndim == 3:
            rot_mat = (
                rotation_matrix_eff(rotation[:, 0].unsqueeze(1), axis=2, ndim=self.ndim)
                @ rotation_matrix_eff(rotation[:, 1].unsqueeze(1), axis=1, ndim=self.ndim)
                @ rotation_matrix_eff(rotation[:, 2].unsqueeze(1), axis=0, ndim=self.ndim)
            )

        Tmat = (
            scaling_matrix_eff(scale, ndim=self.ndim)
            @ shearing_matrix_eff(shear, ndim=self.ndim)
            @ rot_mat
        )

        f_origin = -(
            torch.tensor(
                fixed_image.shape[2:],
                dtype=fixed_image.dtype,
                device=fixed_image.device,
            )[None]
            / 2
        )
        m_origin = -(
            torch.tensor(
                moving_image.shape[2:],
                dtype=moving_image.dtype,
                device=moving_image.device,
            )[None]
            / 2
        )

        if coordinate_grid is None:
            coordinate_grid = identity_grid(
                fixed_image.shape[2:],
                stackdim=0,
                dtype=fixed_image.dtype,
                device=fixed_image.device,
            )[None].movedim(1, self.coord_dim)

        coordinate_grid = batch_transform_efficient(
            coordinate_grid, self.coord_dim, Tmat, translation, f_origin, m_origin
        )

        return coordinate_grid


class DirectDVFTransformer(Transformer):
    def apply_transform(
        self,
        dvf: Tensor,
        fixed_image: Tensor,
        moving_image: Tensor,
        coordinate_grid: Optional[Tensor] = None,
    ) -> Tensor:
        return dvf


class DiffeomorphicFlowTransformer(Transformer):
    """
    Implementation for diffeormorphic DVFs as used in `Voxelmorph <https://github.com/voxelmorph/voxelmorph>`_.
    Uses scaling and squaring of a velocity field to obtain a diffeomorphic DVF.
    """

    def __init__(self, *args, nsteps=8, **kwargs):
        super().__init__(*args, **kwargs)

        assert nsteps >= 0, f"nsteps should be >= 0"
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)

    def scaling_squaring(self, flow_field):
        flow_field = flow_field * self.scale
        for _ in range(self.nsteps):
            grid = identity_grid(
                flow_field.shape[2:],
                stackdim=0,
                device=flow_field.device,
                dtype=flow_field.dtype,
            )[None].movedim(1, self.coord_dim)
            flow_field = flow_field + self._resampler(flow_field, flow_field + grid)
        return flow_field

    def apply_transform(
        self,
        flow_field: Tensor,
        fixed_image: Tensor,
        moving_image: Tensor,
        coordinate_grid: Optional[Tensor] = None,
    ) -> Tensor:
        dvf = self.scaling_squaring(flow_field)
        device = dvf.device
        dtype = dvf.dtype
        new_grid = dvf + identity_grid(
            dvf.shape[2:], stackdim=0, dtype=dtype, device=device
        )
        if coordinate_grid:
            new_grid = self._resampler(coordinate_grid, new_grid)
        return new_grid
