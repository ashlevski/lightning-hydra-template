# coding=utf-8
# Copyright (c) DIRECT Contributors

from __future__ import annotations

import functools
import logging
import random
import warnings
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torchvision

from direct.algorithms.mri_algorithms import EspiritCalibration
from direct.data import transforms as T
from direct.exceptions import ItemNotFoundException
from direct.types import DirectEnum, IntegerListOrTupleString, KspaceKey
from direct.utils import DirectModule, DirectTransform
from direct.utils.asserts import assert_complex

logger = logging.getLogger(__name__)


class Compose(DirectTransform):
    """Compose several transformations together, for instance ClipAndScale and a flip.

    Code based on torchvision: https://github.com/pytorch/vision, but got forked from there as torchvision has some
    additional dependencies.
    """

    def __init__(self, transforms: Iterable[Callable]) -> None:
        """Inits :class:`Compose`.

        Parameters
        ----------
        transforms: Iterable[Callable]
            List of transforms.
        """
        super().__init__()
        self.transforms = transforms

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Calls :class:`Compose`.

        Parameters
        ----------
        sample: Dict[str, Any]
            Dict sample.

        Returns
        -------
        Dict[str, Any]
            Dict sample transformed by `transforms`.
        """
        for transform in self.transforms:
            sample = transform(sample)

        return sample

    def __repr__(self):
        """Representation of :class:`Compose`."""
        repr_string = self.__class__.__name__ + "("
        for transform in self.transforms:
            repr_string += "\n"
            repr_string += f"    {transform},"
        repr_string = repr_string[:-1] + "\n)"
        return repr_string


class RandomRotation(DirectTransform):
    r"""Random :math:`k`-space rotation.

    Performs a random rotation with probability :math:`p`. Rotation degrees must be multiples of 90.
    """

    def __init__(
        self,
        degrees: Sequence[int] = (-90, 90),
        p: float = 0.5,
        kspace_key: KspaceKey = KspaceKey.kspace,
    ):
        r"""Inits :class:`RandomRotation`.

        Parameters
        ----------
        degrees: sequence of ints
            Degrees of rotation. Must be a multiple of 90. If len(degrees) > 1, then a degree will be chosen at random.
            Default: (-90, 90).
        p: float
            Probability of the backprojected :math:`k`-space being rotated. Default: 0.5
        kspace_key: KspaceKey
            Default: KspaceKey.kspace.
        """
        super().__init__()

        assert all(degree % 90 == 0 for degree in degrees)

        self.degrees = degrees
        self.p = p
        self.kspace_key = kspace_key

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Calls :class:`RandomRotation`.

        Parameters
        ----------
        sample: Dict[str, Any]
            Dict sample.

        Returns
        -------
        Dict[str, Any]
            Sample with rotated :math:`k`-space.
        """
        if random.SystemRandom().random() <= self.p:
            kspace = T.view_as_complex(sample[self.kspace_key].clone())
            degree = random.SystemRandom().choice(self.degrees)
            k = degree // 90
            rotated_kspace = torch.rot90(kspace, k=k, dims=(1, 2))
            sample[self.kspace_key] = T.view_as_real(rotated_kspace)

            # If rotated by multiples of (n + 1) * 90 degrees, reconstruction size also needs to change
            reconstruction_size = sample.get("reconstruction_size", None)
            if reconstruction_size and (k % 2) == 1:
                sample["reconstruction_size"] = reconstruction_size[:2][::-1] + reconstruction_size[2:]

        return sample


class RandomFlipType(DirectEnum):
    horizontal = "horizontal"
    vertical = "vertical"
    random = "random"


class RandomFlip(DirectTransform):
    r"""Random k-space flip transform.

    Performs a random flip with probability :math:`p`. Flip can be horizontal, vertical, or a random choice of the two.
    """

    def __init__(
        self,
        flip: RandomFlipType = RandomFlipType.random,
        p: float = 0.5,
        kspace_key: KspaceKey = KspaceKey.kspace,
    ):
        r"""Inits :class:`RandomFlip`.

        Parameters
        ----------
        flip : RandomFlipType
            Horizontal, vertical, or random choice of the two. Default: RandomFlipType.random.
        p : float
            Probability of the backprojected :math:`k`-space being flipped. Default: 0.5
        kspace_key : KspaceKey
            Default: KspaceKey.kspace.
        """
        super().__init__()

        if flip == "horizontal":
            self.flipper = torchvision.transforms.RandomHorizontalFlip(p=p)
        elif flip == "vertical":
            self.flipper = torchvision.transforms.RandomVerticalFlip(p=p)
        else:
            self.flipper = random.SystemRandom().choice(
                [torchvision.transforms.RandomHorizontalFlip(p=p), torchvision.transforms.RandomVerticalFlip(p=p)]
            )
        self.kpace_key = kspace_key

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Calls :class:`RandomFlip`.

        Parameters
        ----------
        sample: Dict[str, Any]
            Dict sample.

        Returns
        -------
        Dict[str, Any]
            Sample with flipped :math:`k`-space.
        """
        kspace = T.view_as_complex(sample[self.kpace_key].clone())
        kspace = self.flipper(kspace)
        sample[self.kpace_key] = T.view_as_real(kspace)

        return sample


class CreateSamplingMask(DirectTransform):
    """Data Transformer for training MRI reconstruction models.

    Creates sampling mask.
    """

    def __init__(
        self,
        mask_func: Callable,
        shape: Optional[Tuple[int, ...]] = None,
        use_seed: bool = True,
        return_acs: bool = False,
    ) -> None:
        """Inits :class:`CreateSamplingMask`.

        Parameters
        ----------
        mask_func: Callable
            A function which creates a sampling mask of the appropriate shape.
        shape: tuple, optional
            Sampling mask shape. Default: None.
        use_seed: bool
            If true, a pseudo-random number based on the filename is computed so that every slice of the volume get
            the same mask every time. Default: True.
        return_acs: bool
            If True, it will generate an ACS mask. Default: False.
        """
        super().__init__()
        self.mask_func = mask_func
        self.shape = shape
        self.use_seed = use_seed
        self.return_acs = return_acs

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Calls :class:`CreateSamplingMask`.

        Parameters
        ----------
        sample: Dict[str, Any]
            Dict sample.

        Returns
        -------
        Dict[str, Any]
            Sample with `sampling_mask` key.
        """
        if not self.shape:
            shape = sample["kspace"].shape[1:]
        elif any(_ is None for _ in self.shape):  # Allow None as values.
            kspace_shape = list(sample["kspace"].shape[1:-1])
            shape = tuple(_ if _ else kspace_shape[idx] for idx, _ in enumerate(self.shape)) + (2,)
        else:
            shape = self.shape + (2,)

        seed = None if not self.use_seed else tuple(map(ord, str(sample["filename"])))

        sampling_mask = self.mask_func(shape=shape, seed=seed, return_acs=False)

        if "padding" in sample:
            sampling_mask = T.apply_padding(sampling_mask, sample["padding"])

        # Shape (1, height, width, 1)
        sample["sampling_mask"] = sampling_mask

        if self.return_acs:
            kspace_shape = sample["kspace"].shape[1:]
            sample["acs_mask"] = self.mask_func(shape=kspace_shape, seed=seed, return_acs=True)

        return sample


class ApplyMaskModule(DirectModule):
    """Data Transformer for training MRI reconstruction models.

    Masks the input k-space (with key `input_kspace_key`) using a sampling mask with key `sampling_mask_key` onto
    a new masked k-space with key `target_kspace_key`.
    """

    def __init__(
        self,
        sampling_mask_key: str = "sampling_mask",
        input_kspace_key: KspaceKey = KspaceKey.kspace,
        target_kspace_key: KspaceKey = KspaceKey.masked_kspace,
    ) -> None:
        """Inits :class:`ApplyMaskModule`.

        Parameters
        ----------
        sampling_mask_key: str
            Default: "sampling_mask".
        input_kspace_key: KspaceKey
            Default: KspaceKey.kspace.
        target_kspace_key: KspaceKey
            Default KspaceKey.masked_kspace.
        """
        super().__init__()
        self.logger = logging.getLogger(type(self).__name__)

        self.sampling_mask_key = sampling_mask_key
        self.input_kspace_key = input_kspace_key
        self.target_kspace_key = target_kspace_key

    def forward(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Forward pass of :class:`ApplyMaskModule`.

        Applies mask with key `sampling_mask_key` onto kspace `input_kspace_key`. Result is stored as a tensor with
        key `target_kspace_key`.

        Parameters
        ----------
        sample: Dict[str, Any]
            Dict sample containing keys `sampling_mask_key` and `input_kspace_key`.

        Returns
        -------
        Dict[str, Any]
            Sample with (new) key `target_kspace_key`.
        """
        if self.input_kspace_key not in sample:
            raise ValueError(f"Key {self.input_kspace_key} corresponding to `input_kspace_key` not found in sample.")
        input_kspace = sample[self.input_kspace_key]

        if self.sampling_mask_key not in sample:
            raise ValueError(f"Key {self.sampling_mask_key} corresponding to `sampling_mask_key` not found in sample.")
        sampling_mask = sample[self.sampling_mask_key]

        target_kspace, _ = T.apply_mask(input_kspace, sampling_mask)
        sample[self.target_kspace_key] = target_kspace
        return sample


class CropKspace(DirectTransform):
    """Data Transformer for training MRI reconstruction models.

    Crops the k-space by:
        * It first projects the k-space to the image-domain via the backward operator,
        * It crops the back-projected k-space to specified shape or key,
        * It transforms the cropped back-projected k-space to the k-space domain via the forward operator.
    """

    def __init__(
        self,
        crop: Union[str, Tuple[int, ...], List[int]],
        forward_operator: Callable = T.fft2,
        backward_operator: Callable = T.ifft2,
        image_space_center_crop: bool = False,
        random_crop_sampler_type: Optional[str] = "uniform",
        random_crop_sampler_use_seed: Optional[bool] = True,
        random_crop_sampler_gaussian_sigma: Optional[List[float]] = None,
    ) -> None:
        """Inits :class:`CropKspace`.

        Parameters
        ----------
        crop: tuple of ints or str
            Shape to crop the input to or a string pointing to a crop key (e.g. `reconstruction_size`).
        forward_operator: Callable
            The forward operator, e.g. some form of FFT (centered or uncentered).
            Default: :class:`direct.data.transforms.fft2`.
        backward_operator: Callable
            The backward operator, e.g. some form of inverse FFT (centered or uncentered).
            Default: :class:`direct.data.transforms.ifft2`.
        image_space_center_crop: bool
            If set, the crop in the data will be taken in the center
        random_crop_sampler_type: Optional[str]
            If "uniform" the random cropping will be done by uniformly sampling `crop`, as opposed to `gaussian` which
            will sample from a gaussian distribution. If `image_space_center_crop` is True, then this is ignored.
            Default: "uniform".
        random_crop_sampler_use_seed: bool
            If true, a pseudo-random number based on the filename is computed so that every slice of the volume
            is cropped the same way. Default: True.
        random_crop_sampler_gaussian_sigma: Optional[List[float]]
            Standard variance of the gaussian when `random_crop_sampler_type` is `gaussian`.
            If `image_space_center_crop` is True, then this is ignored. Default: None.
        """
        super().__init__()
        self.logger = logging.getLogger(type(self).__name__)

        self.image_space_center_crop = image_space_center_crop

        if not (isinstance(crop, (Iterable, str))):
            raise ValueError(
                f"Invalid input for `crop`. Received {crop}. Can be a list of tuple of integers or a string."
            )
        self.crop = crop

        if image_space_center_crop:
            self.crop_func = T.complex_center_crop
        else:
            self.crop_func = functools.partial(
                T.complex_random_crop,
                sampler=random_crop_sampler_type,
                sigma=random_crop_sampler_gaussian_sigma,
            )
            self.random_crop_sampler_use_seed = random_crop_sampler_use_seed

        self.forward_operator = forward_operator
        self.backward_operator = backward_operator

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Calls :class:`CropKspace`.

        Parameters
        ----------
        sample: Dict[str, Any]
            Dict sample containing key `kspace`.

        Returns
        -------
        Dict[str, Any]
            Cropped and masked sample.
        """

        kspace = sample["kspace"]  # shape (coil, height, width, complex=2)

        backprojected_kspace = self.backward_operator(kspace, dim=(1, 2))  # shape (coil, height, width, complex=2)

        if isinstance(self.crop, IntegerListOrTupleString):
            crop_shape = IntegerListOrTupleString(self.crop)
        elif isinstance(self.crop, str):
            assert self.crop in sample, f"Not found {self.crop} key in sample."
            crop_shape = sample[self.crop][:2]
        else:
            crop_shape = self.crop

        cropper_args = {
            "data_list": [backprojected_kspace],
            "crop_shape": crop_shape,
            "contiguous": False,
        }
        if not self.image_space_center_crop:
            cropper_args["seed"] = (
                None if not self.random_crop_sampler_use_seed else tuple(map(ord, str(sample["filename"])))
            )
        cropped_backprojected_kspace = self.crop_func(**cropper_args)

        # Compute new k-space for the cropped_backprojected_kspace
        # shape (coil, new_height, new_width, complex=2)
        sample["kspace"] = self.forward_operator(cropped_backprojected_kspace, dim=(1, 2))  # The cropped kspace

        return sample


class ComputeZeroPadding(DirectTransform):
    r"""Computes zero padding present in multi-coil kspace input.

    Zero-padding is computed from multi-coil kspace with no signal contribution, i.e. its magnitude
    is really close to zero:

    .. math ::

        \text{padding} = \sum_{i=1}^{n_c} |y_i| < \frac{1}{n_x \cdot n_y}
        \sum_{j=1}^{n_x \cdot n_y} \big\{\sum_{i=1}^{n_c} |y_i|\big\} * \epsilon.
    """

    def __init__(
        self,
        kspace_key: KspaceKey = KspaceKey.kspace,
        padding_key: str = "padding",
        eps: float = 0.0001,
    ) -> None:
        """Inits :class:`ComputeZeroPadding`.

        Parameters
        ----------
        kspace_key: KspaceKey
            K-space key. Default: KspaceKey.kspace.
        padding_key: str
            Target key. Default: "padding".
        eps: float
            Epsilon to multiply sum of signals. If really high, probably no padding will be produced. Default: 0.0001.
        """
        super().__init__()
        self.kspace_key = kspace_key
        self.padding_key = padding_key
        self.eps = eps

    def __call__(self, sample: Dict[str, Any], coil_dim: int = 0) -> Dict[str, Any]:
        """Updates sample with a key `padding_key` with value a binary tensor.

        Non-zero entries indicate samples in kspace with key `kspace_key` which have minor contribution, i.e. padding.

        Parameters
        ----------
        sample : Dict[str, Any]
            Dict sample containing key `kspace_key`.
        coil_dim : int
            Coil dimension. Default: 0.

        Returns
        -------
        sample : Dict[str, Any]
            Dict sample containing key `padding_key`.
        """

        kspace = T.modulus(sample[self.kspace_key]).sum(coil_dim)
        padding = (kspace < torch.mean(kspace) * self.eps).to(kspace.device).unsqueeze(coil_dim).unsqueeze(-1)

        sample[self.padding_key] = padding

        return sample


class ApplyZeroPadding(DirectTransform):
    """Applies zero padding present in multi-coil kspace input."""

    def __init__(self, kspace_key: KspaceKey = KspaceKey.kspace, padding_key: str = "padding") -> None:
        """Inits :class:`ApplyZeroPadding`.

        Parameters
        ----------
        kspace_key: kspaceKey
            K-space key. Default: kspaceKey.kspace.
        padding_key: str
            Target key. Default: "padding".
        """
        super().__init__()
        self.kspace_key = kspace_key
        self.padding_key = padding_key

    def __call__(self, sample: Dict[str, Any], coil_dim: int = 0) -> Dict[str, Any]:
        """Applies zero padding on `kspace_key` with value a binary tensor.

        Parameters
        ----------
        sample : Dict[str, Any]
            Dict sample containing key `kspace_key`.
        coil_dim : int
            Coil dimension. Default: 0.

        Returns
        -------
        sample : Dict[str, Any]
            Dict sample containing key `padding_key`.
        """

        sample[self.kspace_key] = T.apply_padding(sample[self.kspace_key], sample[self.padding_key])

        return sample


class ReconstructionType(str, Enum):
    """Reconstruction method for :class:`ComputeImage` transform."""

    rss = "rss"
    complex = "complex"
    complex_mod = "complex_mod"
    sense = "sense"
    sense_mod = "sense_mod"


class ComputeImageModule(DirectModule):
    """Compute Image transform."""

    def __init__(
        self,
        kspace_key: KspaceKey,
        target_key: str,
        backward_operator: Callable,
        type_reconstruction: ReconstructionType = ReconstructionType.rss,
    ) -> None:
        """Inits :class:`ComputeImageModule`.

        Parameters
        ----------
        kspace_key: KspaceKey
            K-space key.
        target_key: str
            Target key.
        backward_operator: callable
            The backward operator, e.g. some form of inverse FFT (centered or uncentered).
        type_reconstruction: ReconstructionType
            Type of reconstruction. Can be "complex", "complex_mod", "sense", "sense_mod" or "rss".
            Default: ReconstructionType.rss.
        """
        super().__init__()
        self.backward_operator = backward_operator
        self.kspace_key = kspace_key
        self.target_key = target_key
        self.type_reconstruction = type_reconstruction

    def forward(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Forward pass of :class:`ComputeImageModule`.

        Parameters
        ----------
        sample: Dict[str, Any]
            Contains key kspace_key with value a torch.Tensor of shape (coil,\*spatial_dims, complex=2).

        Returns
        -------
        sample: dict
            Contains key target_key with value a torch.Tensor of shape (\*spatial_dims) if `type_reconstruction` is
            "rss", "complex_mod" or "sense_mod", and of shape(\*spatial_dims, complex_dim=2) otherwise.
        """
        kspace_data = sample[self.kspace_key]

        # Get complex-valued data solution
        image = self.backward_operator(kspace_data, dim=self.spatial_dims)
        if self.type_reconstruction in [
            ReconstructionType.complex,
            ReconstructionType.complex_mod,
        ]:
            sample[self.target_key] = image.sum(self.coil_dim)
        elif self.type_reconstruction == ReconstructionType.rss:
            sample[self.target_key] = T.root_sum_of_squares(image, dim=self.coil_dim)
        else:
            if "sensitivity_map" not in sample:
                raise ItemNotFoundException(
                    "sensitivity map",
                    "Sensitivity map is required for SENSE reconstruction.",
                )
            sample[self.target_key] = T.complex_multiplication(T.conjugate(sample["sensitivity_map"]), image).sum(
                self.coil_dim
            )
        if self.type_reconstruction in [
            ReconstructionType.complex_mod,
            ReconstructionType.sense_mod,
        ]:
            sample[self.target_key] = T.modulus(sample[self.target_key], self.complex_dim)
        return sample


class EstimateBodyCoilImage(DirectTransform):
    """Estimates body coil image."""

    def __init__(self, mask_func: Callable, backward_operator: Callable, use_seed: bool = True) -> None:
        """Inits :class:`EstimateBodyCoilImage'.

        Parameters
        ----------
        mask_func: Callable
            A function which creates a sampling mask of the appropriate shape.
        backward_operator: callable
            The backward operator, e.g. some form of inverse FFT (centered or uncentered).
        use_seed: bool
            If true, a pseudo-random number based on the filename is computed so that every slice of the volume get
            the same mask every time. Default: True.
        """
        super().__init__()
        self.mask_func = mask_func
        self.use_seed = use_seed
        self.backward_operator = backward_operator

    def __call__(self, sample: Dict[str, Any], coil_dim: int = 0) -> Dict[str, Any]:
        """Calls :class:`EstimateBodyCoilImage`.

        Parameters
        ----------
        sample: Dict[str, Any]
            Contains key kspace_key with value a torch.Tensor of shape (coil, ..., complex=2).
        coil_dim: int
            Coil dimension. Default: 0.

        Returns
        ----------
        sample: Dict[str, Any]
            Contains key `"body_coil_image`.
        """
        kspace = sample["kspace"]
        # We need to create an ACS mask based on the shape of this kspace, as it can be cropped.

        seed = None if not self.use_seed else tuple(map(ord, str(sample["filename"])))
        kspace_shape = sample["kspace"].shape[1:]
        acs_mask = self.mask_func(shape=kspace_shape, seed=seed, return_acs=True)

        kspace = acs_mask * kspace + 0.0
        acs_image = self.backward_operator(kspace, dim=(1, 2))

        sample["body_coil_image"] = T.root_sum_of_squares(acs_image, dim=coil_dim)
        return sample


class SensitivityMapType(DirectEnum):
    espirit = "espirit"
    rss_estimate = "rss_estimate"
    unit = "unit"


class EstimateSensitivityMapModule(DirectModule):
    """Data Transformer for training MRI reconstruction models.

    Estimates sensitivity maps given masked k-space data using one of three methods:

    *   Unit: unit sensitivity map in case of single coil acquisition.
    *   RSS-estimate: sensitivity maps estimated by using the root-sum-of-squares of the autocalibration-signal.
    *   ESPIRIT: sensitivity maps estimated with the ESPIRIT method [1]_.

    References
    ----------

    .. [1] Uecker M, Lai P, Murphy MJ, Virtue P, Elad M, Pauly JM, Vasanawala SS, Lustig M. ESPIRiT--an eigenvalue
        approach to autocalibrating parallel MRI: where SENSE meets GRAPPA. Magn Reson Med. 2014 Mar;71(3):990-1001.
        doi: 10.1002/mrm.24751. PMID: 23649942; PMCID: PMC4142121.
    """

    def __init__(
        self,
        kspace_key: KspaceKey = KspaceKey.kspace,
        backward_operator: Callable = T.ifft2,
        type_of_map: Optional[SensitivityMapType] = SensitivityMapType.rss_estimate,
        gaussian_sigma: Optional[float] = None,
        espirit_threshold: Optional[float] = 0.05,
        espirit_kernel_size: Optional[int] = 6,
        espirit_crop: Optional[float] = 0.95,
        espirit_max_iters: Optional[int] = 30,
    ) -> None:
        """Inits :class:`EstimateSensitivityMapModule`.

        Parameters
        ----------
        kspace_key: kspaceKey
            K-space key. Default kspaceKey.kspace.
        backward_operator: callable
            The backward operator, e.g. some form of inverse FFT (centered or uncentered).
        type_of_map: SensitivityMapType, optional
            Type of map to estimate. Can be "unit", "rss_estimate" or "espirit". Default: "espirit".
        gaussian_sigma: float, optional
            If non-zero, acs_image well be calculated
        espirit_threshold: float, optional
            Threshold for the calibration matrix when `type_of_map`=="espirit". Default: 0.05.
        espirit_kernel_size: int, optional
            Kernel size for the calibration matrix when `type_of_map`=="espirit". Default: 6.
        espirit_crop: float, optional
            Output eigenvalue cropping threshold when `type_of_map`=="espirit". Default: 0.95.
        espirit_max_iters: int, optional
            Power method iterations when `type_of_map`=="espirit". Default: 30.
        """
        super().__init__()
        self.backward_operator = backward_operator
        self.kspace_key = kspace_key
        if type_of_map not in ["unit", "rss_estimate", "espirit"]:
            raise ValueError(
                f"Expected type of map to be either `unit`, `rss_estimate`, `espirit`. Got {type_of_map}."
            )
        self.type_of_map = type_of_map

        # RSS estimate attributes
        self.gaussian_sigma = gaussian_sigma
        # Espirit attributes
        if type_of_map == "espirit":
            self.espirit_calibrator = EspiritCalibration(
                backward_operator,
                espirit_threshold,
                espirit_kernel_size,
                espirit_crop,
                espirit_max_iters,
                kspace_key,
            )
        self.espirit_threshold = espirit_threshold
        self.espirit_kernel_size = espirit_kernel_size
        self.espirit_crop = espirit_crop
        self.espirit_max_iters = espirit_max_iters

    def estimate_acs_image(self, sample: Dict[str, Any], width_dim: int = -2) -> torch.Tensor:
        """Estimates the autocalibration (ACS) image by sampling the k-space using the ACS mask.

        Parameters
        ----------
        sample: Dict[str, Any]
            Sample dictionary,
        width_dim: int
            Dimension corresponding to width. Default: -2.

        Returns
        -------
        acs_image: torch.Tensor
            Estimate of the ACS image.
        """
        kspace_data = sample[self.kspace_key]  # Shape (coil, height, width, complex=2)

        if kspace_data.shape[self.coil_dim] == 1:
            warnings.warn(
                "Estimation of sensitivity map of Single-coil data. This warning will be displayed only once."
            )

        if "sensitivity_map" in sample:
            warnings.warn(
                "`sensitivity_map` is given, but will be overwritten. This warning will be displayed only once."
            )

        if self.gaussian_sigma == 0 or not self.gaussian_sigma:
            kspace_acs = kspace_data * sample["acs_mask"] + 0.0  # + 0.0 removes the sign of zeros.
        else:
            gaussian_mask = torch.linspace(-1, 1, kspace_data.size(width_dim), dtype=kspace_data.dtype)
            gaussian_mask = torch.exp(-((gaussian_mask / self.gaussian_sigma) ** 2))
            gaussian_mask_shape = torch.ones(len(kspace_data.shape)).int()
            gaussian_mask_shape[width_dim] = kspace_data.size(width_dim)
            gaussian_mask = gaussian_mask.reshape(tuple(gaussian_mask_shape))
            kspace_acs = kspace_data * sample["acs_mask"] * gaussian_mask + 0.0

        # Get complex-valued data solution
        # Shape (batch, coil, height, width, complex=2)
        acs_image = self.backward_operator(kspace_acs, dim=(2, 3))

        return acs_image

    def forward(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Calculates sensitivity maps for the input sample.

        Parameters
        ----------
        sample: Dict[str, Any]
            Must contain key matching kspace_key with value a (complex) torch.Tensor
            of shape (coil, height, width, complex=2).

        Returns
        -------
        sample: Dict[str, Any]
            Sample with key "sensitivity_map" with value the estimated sensitivity map.
        """
        if self.type_of_map == "unit":
            kspace = sample[self.kspace_key]
            sensitivity_map = torch.zeros(kspace.shape).float()
            # Assumes complex channel is last
            assert_complex(kspace, complex_last=True)
            sensitivity_map[..., 0] = 1.0
            # Shape (coil, height, width, complex=2)
            sensitivity_map = sensitivity_map.to(kspace.device)

        elif self.type_of_map == "rss_estimate":
            # Shape (batch, coil, height, width, complex=2)
            acs_image = self.estimate_acs_image(sample)
            # Shape (batch, height, width)
            acs_image_rss = T.root_sum_of_squares(acs_image, dim=self.coil_dim)
            # Shape (batch, 1, height, width, 1)
            acs_image_rss = acs_image_rss.unsqueeze(self.coil_dim).unsqueeze(self.complex_dim)
            # Shape (batch, coil, height, width, complex=2)
            sensitivity_map = T.safe_divide(acs_image, acs_image_rss)
        else:
            sensitivity_map = self.espirit_calibrator(sample)

        sensitivity_map_norm = torch.sqrt(
            (sensitivity_map**2).sum(self.complex_dim).sum(self.coil_dim)
        )  # shape (height, width)
        sensitivity_map_norm = sensitivity_map_norm.unsqueeze(self.coil_dim).unsqueeze(self.complex_dim)

        sample["sensitivity_map"] = T.safe_divide(sensitivity_map, sensitivity_map_norm)
        return sample


class DeleteKeysModule(DirectModule):
    """Remove keys from the sample if present."""

    def __init__(self, keys: List[str]):
        """Inits :class:`DeleteKeys`.

        Parameters
        ----------
        keys: List[str]
            Key(s) to delete.
        """
        super().__init__()
        self.keys = keys

    def forward(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Forward pass of :class:`DeleteKeys`.

        Parameters
        ----------
        sample: Dict[str, Any]
            Dictionary to look for keys and remove them.

        Returns
        -------
        Dict[str, Any]
            Dictionary with deleted specified keys.
        """
        for key in self.keys:
            if key in sample:
                del sample[key]

        return sample


class RenameKeysModule(DirectModule):
    """Rename keys from the sample if present."""

    def __init__(self, old_keys: List[str], new_keys: List[str]):
        """Inits :class:`RenameKeys`.

        Parameters
        ----------
        old_keys: List[str]
            Key(s) to rename.
        new_keys: List[str]
            Key(s) to replace old keys.
        """
        super().__init__()
        self.old_keys = old_keys
        self.new_keys = new_keys

    def forward(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Forward pass of :class:`RenameKeys`.

        Parameters
        ----------
        sample: Dict[str, Any]
            Dictionary to look for keys and rename them.

        Returns
        -------
        Dict[str, Any]
            Dictionary with renamed specified keys.
        """
        for old_key, new_key in zip(self.old_keys, self.new_keys):
            if old_key in sample:
                sample[new_key] = sample.pop(old_key)

        return sample


class PadCoilDimensionModule(DirectModule):
    """Pad the coils by zeros to a given number of coils.

    Useful if you want to collate volumes with different coil dimension.
    """

    def __init__(
        self,
        pad_coils: Optional[int] = None,
        key: str = "masked_kspace",
        coil_dim: int = 1,
    ):
        """Inits :class:`PadCoilDimensionModule`.

        Parameters
        ----------
        pad_coils: int, optional
            Number of coils to pad to. Default: None.
        key: str
            Key to pad in sample. Default: "masked_kspace".
        coil_dim: int
            Coil dimension along which the pad will be done. Default: 0.
        """
        super().__init__()
        self.num_coils = pad_coils
        self.key = key
        self.coil_dim = coil_dim

    def forward(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Forward pass of :class:`PadCoilDimensionModule`.

        Parameters
        ----------
        sample: Dict[str, Any]
            Dictionary with key `self.key`.

        Returns
        -------
        sample: Dict[str, Any]
            Dictionary with padded coils of sample[self.key] if self.num_coils is not None.
        """
        if not self.num_coils:
            return sample

        if self.key not in sample:
            return sample

        data = sample[self.key]

        curr_num_coils = data.shape[self.coil_dim]
        if curr_num_coils > self.num_coils:
            raise ValueError(
                f"Tried to pad to {self.num_coils} coils, but already have {curr_num_coils} for "
                f"{sample['filename']}."
            )
        if curr_num_coils == self.num_coils:
            return sample

        shape = data.shape
        num_coils = shape[self.coil_dim]
        padding_data_shape = list(shape).copy()
        padding_data_shape[self.coil_dim] = max(self.num_coils - num_coils, 0)
        zeros = torch.zeros(padding_data_shape, dtype=data.dtype, device=data.device)
        sample[self.key] = torch.cat([zeros, data], dim=self.coil_dim)

        return sample


class ComputeScalingFactorModule(DirectModule):
    """Calculates scaling factor.

    Scaling factor is for the input data based on either to the percentile or to the maximum of `normalize_key`.
    """

    def __init__(
        self,
        normalize_key: Union[None, str] = "masked_kspace",
        percentile: Union[None, float] = 0.99,
        scaling_factor_key: str = "scaling_factor",
    ):
        """Inits :class:`ComputeScalingFactorModule`.

        Parameters
        ----------
        normalize_key : str or None
            Key name to compute the data for. If the maximum has to be computed on the ACS, ensure the reconstruction
            on the ACS is available (typically `body_coil_image`). Default: "masked_kspace".
        percentile : float or None
            Rescale data with the given percentile. If None, the division is done by the maximum. Default: 0.99.
        scaling_factor_key : str
            Name of how the scaling factor will be stored. Default: 'scaling_factor'.
        """
        super().__init__()
        self.normalize_key = normalize_key
        self.percentile = percentile
        self.scaling_factor_key = scaling_factor_key

    def forward(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Forward pass of :class:`ComputeScalingFactorModule`.

        Parameters
        ----------
        sample: Dict[str, Any]
            Sample with key `normalize_key` to compute scaling_factor.

        Returns
        -------
        sample: Dict[str, Any]
            Sample with key `scaling_factor_key`.
        """
        if self.normalize_key == "scaling_factor":  # This is a real-valued given number
            scaling_factor = sample["scaling_factor"]
        elif not self.normalize_key:
            kspace = sample["masked_kspace"]
            scaling_factor = torch.tensor([1.0] * kspace.size(0), device=kspace.device, dtype=kspace.dtype)
        else:
            data = sample[self.normalize_key]
            scaling_factor: Union[List, torch.Tensor] = []
            # Compute the maximum and scale the input
            if self.percentile:
                for _ in range(data.size(0)):
                    # Used in case the k-space is padded (e.g. for batches)
                    non_padded_coil_data = data[_][data[_].sum(dim=tuple(range(1, data[_].ndim))).bool()]
                    tview = -1.0 * T.modulus(non_padded_coil_data).view(-1)
                    s, _ = torch.kthvalue(tview, int((1 - self.percentile) * tview.size()[0]) + 1)
                    scaling_factor += [-1.0 * s]
                scaling_factor = torch.tensor(scaling_factor, dtype=data.dtype, device=data.device)
            else:
                scaling_factor = T.modulus(data).amax(dim=list(range(data.ndim))[1:-1])
        sample[self.scaling_factor_key] = scaling_factor
        return sample


class NormalizeModule(DirectModule):
    """Normalize the input data."""

    def __init__(
        self,
        scaling_factor_key: str = "scaling_factor",
        keys_to_normalize: Optional[List[str]] = None,
    ):
        """Inits :class:`NormalizeModule`.

        Parameters
        ----------
        scaling_factor_key : str
            Name of scaling factor key expected in sample. Default: 'scaling_factor'.
        """
        super().__init__()
        self.scaling_factor_key = scaling_factor_key

        self.keys_to_normalize = (
            [
                "masked_kspace",
                "target",
                "kspace",
                "body_coil_image",  # sensitivity_map does not require normalization.
                "initial_image",
                "initial_kspace",
            ]
            if keys_to_normalize is None
            else keys_to_normalize
        )

    def forward(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Forward pass of :class:`NormalizeModule`.

        Parameters
        ----------
        sample: Dict[str, Any]
            Sample to normalize.

        Returns
        -------
        sample: Dict[str, Any]
            Sample with normalized values if their respective key is in `keys_to_normalize` and key
            `scaling_factor_key` exists in sample.
        """
        scaling_factor = sample.get(self.scaling_factor_key, None)
        # Normalize data
        if scaling_factor is not None:
            for key in sample.keys():
                if key not in self.keys_to_normalize:
                    continue
                sample[key] = T.safe_divide(
                    sample[key],
                    scaling_factor.reshape(-1, *[1 for _ in range(sample[key].ndim - 1)]),
                )

            sample["scaling_diff"] = 0.0
        return sample


class WhitenDataModule(DirectModule):
    """Whitens complex data Module."""

    def __init__(self, epsilon: float = 1e-10, key: str = "complex_image"):
        """Inits :class:`WhitenDataModule`.

        Parameters
        ----------
        epsilon: float
            Default: 1e-10.
        key: str
            Key to whiten. Default: "complex_image".
        """
        super().__init__()
        self.epsilon = epsilon
        self.key = key

    def complex_whiten(self, complex_image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Whiten complex image.

        Parameters
        ----------
        complex_image: torch.Tensor
            Complex image tensor to whiten.

        Returns
        -------
        mean, std, whitened_image: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        """
        # From: https://github.com/facebookresearch/fastMRI
        #       blob/da1528585061dfbe2e91ebbe99a5d4841a5c3f43/banding_removal/fastmri/data/transforms.py#L464  # noqa
        real = complex_image[..., 0]
        imag = complex_image[..., 1]

        # Center around mean.
        mean = complex_image.mean()
        centered_complex_image = complex_image - mean

        # Determine covariance between real and imaginary.
        n_elements = real.nelement()
        real_real = (real.mul(real).sum() - real.mean().mul(real.mean())) / n_elements
        real_imag = (real.mul(imag).sum() - real.mean().mul(imag.mean())) / n_elements
        imag_imag = (imag.mul(imag).sum() - imag.mean().mul(imag.mean())) / n_elements
        eig_input = torch.Tensor([[real_real, real_imag], [real_imag, imag_imag]])

        # Remove correlation by rotating around covariance eigenvectors.
        eig_values, eig_vecs = torch.linalg.eig(eig_input)

        # Scale by eigenvalues for unit variance.
        std = (eig_values.real + self.epsilon).sqrt()
        whitened_image = torch.matmul(centered_complex_image, eig_vecs.real) / std

        return mean, std, whitened_image

    def forward(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Forward pass of :class:`WhitenDataModule`.

        Parameters
        ----------
        sample: Dict[str, Any]
            Sample with key `key`.

        Returns
        -------
        sample: Dict[str, Any]
            Sample with value of `key` whitened.
        """
        _, _, whitened_image = self.complex_whiten(sample[self.key])
        sample[self.key] = whitened_image
        return sample


class ModuleWrapper:
    class SubWrapper:
        def __init__(self, transform, toggle_dims):
            self.toggle_dims = toggle_dims
            self._transform = transform

        def __call__(self, sample):
            if self.toggle_dims:
                for k, v in sample.items():
                    if isinstance(v, (torch.Tensor, np.ndarray)):
                        sample[k] = v[None]

            sample = self._transform.forward(sample)

            if self.toggle_dims:
                for k, v in sample.items():
                    if isinstance(v, (torch.Tensor, np.ndarray)):
                        sample[k] = v.squeeze(0)

            return sample

        def __repr__(self):
            return self._transform.__repr__()

    def __init__(self, module: Callable, toggle_dims: bool):
        self._module = module
        self.toggle_dims = toggle_dims

    def __call__(self, *args, **kwargs):
        return self.SubWrapper(self._module(*args, **kwargs), toggle_dims=self.toggle_dims)


ApplyMask = ModuleWrapper(ApplyMaskModule, toggle_dims=False)
ComputeImage = ModuleWrapper(ComputeImageModule, toggle_dims=True)
EstimateSensitivityMap = ModuleWrapper(EstimateSensitivityMapModule, toggle_dims=True)
DeleteKeys = ModuleWrapper(DeleteKeysModule, toggle_dims=False)
RenameKeys = ModuleWrapper(RenameKeysModule, toggle_dims=False)
PadCoilDimension = ModuleWrapper(PadCoilDimensionModule, toggle_dims=True)
ComputeScalingFactor = ModuleWrapper(ComputeScalingFactorModule, toggle_dims=True)
Normalize = ModuleWrapper(NormalizeModule, toggle_dims=False)
WhitenData = ModuleWrapper(WhitenDataModule, toggle_dims=False)


class ToTensor(DirectTransform):
    """Transforms all np.array-like values in sample to torch.tensors."""

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Calls :class:`ToTensor`.

        Parameters
        ----------
        sample: Dict[str, Any]
             Contains key 'kspace' with value a np.array of shape (coil, height, width) (2D)
             or (coil, slice, height, width) (3D)

        Returns
        -------
        sample: Dict[str, Any]
             Contains key 'kspace' with value a torch.Tensor of shape (coil, height, width) (2D)
             or (coil, slice, height, width) (3D)
        """

        ndim = sample["kspace"].ndim - 1

        if ndim not in [2, 3]:
            raise ValueError(f"Can only cast 2D and 3D data (+coil) to tensor. Got {ndim}.")

        # Shape:    2D: (coil, height, width, complex=2), 3D: (coil, slice, height, width, complex=2)
        sample["kspace"] = T.to_tensor(sample["kspace"]).float()
        # Sensitivity maps are not necessarily available in the dataset.
        if "initial_kspace" in sample:
            # Shape:    2D: (coil, height, width, complex=2), 3D: (coil, slice, height, width, complex=2)
            sample["initial_kspace"] = T.to_tensor(sample["initial_kspace"]).float()
        if "initial_image" in sample:
            # Shape:    2D: (height, width), 3D: (slice, height, width)
            sample["initial_image"] = T.to_tensor(sample["initial_image"]).float()

        if "sensitivity_map" in sample:
            # Shape:    2D: (coil, height, width, complex=2), 3D: (coil, slice, height, width, complex=2)
            sample["sensitivity_map"] = T.to_tensor(sample["sensitivity_map"]).float()
        if "target" in sample:
            # Shape:    2D: (coil, height, width), 3D: (coil, slice, height, width)
            sample["target"] = sample["target"]
        if "sampling_mask" in sample:
            sample["sampling_mask"] = torch.from_numpy(sample["sampling_mask"]).byte()
        if "acs_mask" in sample:
            sample["acs_mask"] = torch.from_numpy(sample["acs_mask"])
        if "scaling_factor" in sample:
            sample["scaling_factor"] = torch.tensor(sample["scaling_factor"]).float()
        if "loglikelihood_scaling" in sample:
            # Shape: (coil, )
            sample["loglikelihood_scaling"] = torch.from_numpy(np.asarray(sample["loglikelihood_scaling"])).float()

        return sample


def build_pre_mri_transforms(
    forward_operator: Callable,
    backward_operator: Callable,
    mask_func: Optional[Callable],
    crop: Optional[Union[Tuple[int, int], str]] = None,
    crop_type: Optional[str] = "uniform",
    image_center_crop: bool = True,
    random_rotation: bool = False,
    random_rotation_degrees: Optional[Sequence[int]] = (-90, 90),
    random_rotation_probability: Optional[float] = 0.5,
    random_flip: bool = False,
    random_flip_type: Optional[RandomFlipType] = RandomFlipType.random,
    random_flip_probability: Optional[float] = 0.5,
    padding_eps: float = 0.0001,
    estimate_body_coil_image: bool = False,
    use_seed: bool = True,
    pad_coils: int = None,
) -> object:
    """Build transforms for MRI.

    - Converts input to (complex-valued) tensor.
    - Adds a sampling mask if `mask_func` is defined.
    - Adds coil sensitivities and / or the body coil_image
    - Crops the input data if needed and masks the fully sampled k-space.
    - Add a target.
    - Normalize input data.
    - Pads the coil dimension.

    Parameters
    ----------
    forward_operator : Callable
        The forward operator, e.g. some form of FFT (centered or uncentered).
    backward_operator : Callable
        The backward operator, e.g. some form of inverse FFT (centered or uncentered).
    mask_func : Callable or None
        A function which creates a sampling mask of the appropriate shape.
    crop : Tuple[int, int] or str, Optional
        If not None, this will transform the "kspace" to an image domain, crop it, and transform it back.
        If a tuple of integers is given then it will crop the backprojected kspace to that size. If
        "reconstruction_size" is given, then it will crop the backprojected kspace according to it, but
        a key "reconstruction_size" must be present in the sample. Default: None.
    crop_type : Optional[str]
        Type of cropping, either "gaussian" or "uniform". This will be ignored if `crop` is None. Default: "uniform".
    image_center_crop : bool
        If True the backprojected kspace will be cropped around the center, otherwise randomly.
        This will be ignored if `crop` is None. Default: True.
    random_rotation : bool
        If True, random rotations will be applied of `random_rotation_degrees` degrees, with probability
        `random_rotation_probability`. Default: False.
    random_rotation_degrees : Sequence[int], optional
        Default: (-90, 90).
    random_rotation_probability : float, optional
        Default: 0.5.
    random_flip : bool
        If True, random rotation of `random_flip_type` type, with probability `random_flip_probability`. Default: False.
    random_flip_type : RandomFlipType, optional
        Default: RandomFlipType.random.
    random_flip_probability : float, optional
        Default: 0.5.
    padding_eps: float
        Padding epsilon. Default: 0.0001.
    estimate_body_coil_image : bool
        Estimate body coil image. Default: False.
    use_seed : bool
        If true, a pseudo-random number based on the filename is computed so that every slice of the volume get
        the same mask every time. Default: True.

    Returns
    -------
    object: Callable
        An MRI transformation object.
    """
    # pylint: disable=too-many-locals
    mri_transforms: List[Callable] = [ToTensor()]
    if crop:
        mri_transforms += [
            CropKspace(
                crop=crop,
                forward_operator=forward_operator,
                backward_operator=backward_operator,
                image_space_center_crop=image_center_crop,
                random_crop_sampler_type=crop_type,
                random_crop_sampler_use_seed=use_seed,
            )
        ]
    if random_rotation:
        mri_transforms += [
            RandomRotation(degrees=random_rotation_degrees, p=random_rotation_probability, kspace_key=KspaceKey.kspace)
        ]
    if random_flip:
        mri_transforms += [RandomFlip(flip=random_flip_type, p=random_flip_probability, kspace_key=KspaceKey.kspace)]
    if mask_func:
        mri_transforms += [
            ComputeZeroPadding(KspaceKey.kspace, "padding", padding_eps),
            ApplyZeroPadding(KspaceKey.kspace, "padding"),
            CreateSamplingMask(
                mask_func,
                shape=(None if (isinstance(crop, str)) else crop),
                use_seed=use_seed,
                return_acs=True,
            ),
        ]
    mri_transforms += [PadCoilDimension(pad_coils=pad_coils, key=KspaceKey.kspace)]
    if estimate_body_coil_image and mask_func is not None:
        mri_transforms.append(EstimateBodyCoilImage(mask_func, backward_operator=backward_operator, use_seed=use_seed))

    return Compose(mri_transforms)


def build_post_mri_transforms(
    backward_operator: Callable,
    estimate_sensitivity_maps: bool = True,
    sensitivity_maps_type: SensitivityMapType = SensitivityMapType.rss_estimate,
    sensitivity_maps_gaussian: Optional[float] = None,
    sensitivity_maps_espirit_threshold: Optional[float] = 0.05,
    sensitivity_maps_espirit_kernel_size: Optional[int] = 6,
    sensitivity_maps_espirit_crop: Optional[float] = 0.95,
    sensitivity_maps_espirit_max_iters: Optional[int] = 30,
    delete_acs_mask: bool = True,
    delete_kspace: bool = True,
    image_recon_type: ReconstructionType = ReconstructionType.rss,
    scaling_key: KspaceKey = KspaceKey.masked_kspace,
    scale_percentile: Optional[float] = 0.99,
) -> object:
    """Build transforms for MRI.

    * Converts input to (complex-valued) tensor.
    * Adds a sampling mask if `mask_func` is defined.
    * Adds coil sensitivities and / or the body coil_image
    * Crops the input data if needed and masks the fully sampled k-space.
    * Add a target.
    * Normalize input data.
    * Pads the coil dimension.

    Parameters
    ----------
    backward_operator : Callable
        The backward operator, e.g. some form of inverse FFT (centered or uncentered).
    estimate_sensitivity_maps : bool
        Estimate sensitivity maps using the acs region. Default: True.
    sensitivity_maps_type: sensitivity_maps_type
        Can be SensitivityMapType.rss_estimate, SensitivityMapType.unit or SensitivityMapType.espirit.
        Will be ignored if `estimate_sensitivity_maps` is equal to False. Default: SensitivityMapType.rss_estimate.
    sensitivity_maps_gaussian : float
        Optional sigma for gaussian weighting of sensitivity map.
    sensitivity_maps_espirit_threshold: float, optional
            Threshold for the calibration matrix when `type_of_map` is equal to "espirit". Default: 0.05.
    sensitivity_maps_espirit_kernel_size: int, optional
        Kernel size for the calibration matrix when `type_of_map` is equal to "espirit". Default: 6.
    sensitivity_maps_espirit_crop: float, optional
        Output eigenvalue cropping threshold when `type_of_map` is equal to "espirit". Default: 0.95.
    sensitivity_maps_espirit_max_iters: int, optional
        Power method iterations when `type_of_map` is equal to "espirit". Default: 30.
    delete_acs_mask : bool
        If True will delete key `acs_mask`. Default: True.
    delete_kspace : bool
        If True will delete key `kspace` (fully sampled k-space). Default: True.
    image_recon_type : ReconstructionType
        Type to reconstruct target image. Default: ReconstructionType.rss.
    scaling_key : KspaceKey
        Key in sample to scale scalable items in sample. Default: KspaceKey.masked_kspace.
    scale_percentile : float, optional
        Data will be rescaled with the given percentile. If None, the division is done by the maximum. Default: 0.99
        the same mask every time. Default: True.

    Returns
    -------
    object: Callable
        An MRI transformation object.
    """
    mri_transforms: List[Callable] = []

    if estimate_sensitivity_maps:
        mri_transforms += [
            EstimateSensitivityMapModule(
                kspace_key=KspaceKey.kspace,
                backward_operator=backward_operator,
                type_of_map=sensitivity_maps_type,
                gaussian_sigma=sensitivity_maps_gaussian,
                espirit_threshold=sensitivity_maps_espirit_threshold,
                espirit_kernel_size=sensitivity_maps_espirit_kernel_size,
                espirit_crop=sensitivity_maps_espirit_crop,
                espirit_max_iters=sensitivity_maps_espirit_max_iters,
            )
        ]

    if delete_acs_mask:
        mri_transforms += [DeleteKeysModule(keys=["acs_mask"])]

    mri_transforms += [
        ComputeImageModule(
            kspace_key=KspaceKey.kspace,
            target_key="target",
            backward_operator=backward_operator,
            type_reconstruction=image_recon_type,
        ),
        ApplyMaskModule(
            sampling_mask_key="sampling_mask",
            input_kspace_key=KspaceKey.kspace,
            target_kspace_key=KspaceKey.masked_kspace,
        ),
    ]

    mri_transforms += [
        ComputeScalingFactorModule(
            normalize_key=scaling_key,
            percentile=scale_percentile,
            scaling_factor_key="scaling_factor",
        ),
        NormalizeModule(scaling_factor_key="scaling_factor"),
    ]
    if delete_kspace:
        mri_transforms += [DeleteKeysModule(keys=[KspaceKey.kspace])]

    return Compose(mri_transforms)


def build_mri_transforms(
    forward_operator: Callable,
    backward_operator: Callable,
    mask_func: Optional[Callable],
    crop: Optional[Union[Tuple[int, int], str]] = None,
    crop_type: Optional[str] = "uniform",
    image_center_crop: bool = True,
    random_rotation: bool = False,
    random_rotation_degrees: Optional[Sequence[int]] = (-90, 90),
    random_rotation_probability: Optional[float] = 0.5,
    random_flip: bool = False,
    random_flip_type: Optional[RandomFlipType] = RandomFlipType.random,
    random_flip_probability: Optional[float] = 0.5,
    padding_eps: float = 0.0001,
    estimate_body_coil_image: bool = False,
    estimate_sensitivity_maps: bool = True,
    sensitivity_maps_type: SensitivityMapType = SensitivityMapType.rss_estimate,
    sensitivity_maps_gaussian: Optional[float] = None,
    sensitivity_maps_espirit_threshold: Optional[float] = 0.05,
    sensitivity_maps_espirit_kernel_size: Optional[int] = 6,
    sensitivity_maps_espirit_crop: Optional[float] = 0.95,
    sensitivity_maps_espirit_max_iters: Optional[int] = 30,
    delete_acs_mask: bool = True,
    delete_kspace: bool = True,
    image_recon_type: ReconstructionType = ReconstructionType.rss,
    pad_coils: Optional[int] = None,
    scaling_key: KspaceKey = KspaceKey.masked_kspace,
    scale_percentile: Optional[float] = 0.99,
    use_seed: bool = True,
) -> object:
    """Build transforms for MRI.

    * Converts input to (complex-valued) tensor.
    * Adds a sampling mask if `mask_func` is defined.
    * Adds coil sensitivities and / or the body coil_image
    * Crops the input data if needed and masks the fully sampled k-space.
    * Add a target.
    * Normalize input data.
    * Pads the coil dimension.

    Parameters
    ----------
    forward_operator : Callable
        The forward operator, e.g. some form of FFT (centered or uncentered).
    backward_operator : Callable
        The backward operator, e.g. some form of inverse FFT (centered or uncentered).
    mask_func : Callable or None
        A function which creates a sampling mask of the appropriate shape.
    crop : Tuple[int, int] or str, Optional
        If not None, this will transform the "kspace" to an image domain, crop it, and transform it back.
        If a tuple of integers is given then it will crop the backprojected kspace to that size. If
        "reconstruction_size" is given, then it will crop the backprojected kspace according to it, but
        a key "reconstruction_size" must be present in the sample. Default: None.
    crop_type : Optional[str]
        Type of cropping, either "gaussian" or "uniform". This will be ignored if `crop` is None. Default: "uniform".
    image_center_crop : bool
        If True the backprojected kspace will be cropped around the center, otherwise randomly.
        This will be ignored if `crop` is None. Default: True.
    random_rotation : bool
        If True, random rotations will be applied of `random_rotation_degrees` degrees, with probability
        `random_rotation_probability`. Default: False.
    random_rotation_degrees : Sequence[int], optional
        Default: (-90, 90).
    random_rotation_probability : float, optional
        Default: 0.5.
    random_flip : bool
        If True, random rotation of `random_flip_type` type, with probability `random_flip_probability`. Default: False.
    random_flip_type : RandomFlipType, optional
        Default: RandomFlipType.random.
    random_flip_probability : float, optional
        Default: 0.5.
    padding_eps: float
        Padding epsilon. Default: 0.0001.
    estimate_body_coil_image : bool
        Estimate body coil image. Default: False.
    estimate_sensitivity_maps : bool
        Estimate sensitivity maps using the acs region. Default: True.
    sensitivity_maps_type: sensitivity_maps_type
        Can be SensitivityMapType.rss_estimate, SensitivityMapType.unit or SensitivityMapType.espirit.
        Will be ignored if `estimate_sensitivity_maps` is False. Default: SensitivityMapType.rss_estimate.
    sensitivity_maps_gaussian : float
        Optional sigma for gaussian weighting of sensitivity map.
    sensitivity_maps_espirit_threshold : float, optional
            Threshold for the calibration matrix when `type_of_map` is equal to "espirit". Default: 0.05.
    sensitivity_maps_espirit_kernel_size : int, optional
        Kernel size for the calibration matrix when `type_of_map` is equal to "espirit". Default: 6.
    sensitivity_maps_espirit_crop : float, optional
        Output eigenvalue cropping threshold when `type_of_map` is equal to "espirit". Default: 0.95.
    sensitivity_maps_espirit_max_iters : int, optional
        Power method iterations when `type_of_map` is equal to "espirit". Default: 30.
    delete_acs_mask : bool
        If True will delete key `acs_mask`. Default: True.
    delete_kspace : bool
        If True will delete key `kspace` (fully sampled k-space). Default: True.
    image_recon_type : ReconstructionType
        Type to reconstruct target image. Default: ReconstructionType.rss.
    pad_coils : int
        Number of coils to pad data to.
    scaling_key : KspaceKey
        Key in sample to scale scalable items in sample. Default: KspaceKey.masked_kspace.
    scale_percentile : float, optional
        Data will be rescaled with the given percentile. If None, the division is done by the maximum. Default: 0.99
    use_seed : bool
        If true, a pseudo-random number based on the filename is computed so that every slice of the volume get
        the same mask every time. Default: True.

    Returns
    -------
    object: Callable
        An MRI transformation object.
    """
    # TODO: Use seed

    mri_transforms: List[Callable] = [ToTensor()]
    if crop:
        mri_transforms += [
            CropKspace(
                crop=crop,
                forward_operator=forward_operator,
                backward_operator=backward_operator,
                image_space_center_crop=image_center_crop,
                random_crop_sampler_type=crop_type,
                random_crop_sampler_use_seed=use_seed,
            )
        ]
    if random_rotation:
        mri_transforms += [
            RandomRotation(degrees=random_rotation_degrees, p=random_rotation_probability, kspace_key=KspaceKey.kspace)
        ]
    if random_flip:
        mri_transforms += [RandomFlip(flip=random_flip_type, p=random_flip_probability, kspace_key=KspaceKey.kspace)]
    if mask_func:
        mri_transforms += [
            ComputeZeroPadding(KspaceKey.kspace, "padding", padding_eps),
            ApplyZeroPadding(KspaceKey.kspace, "padding"),
            CreateSamplingMask(
                mask_func,
                shape=(None if (isinstance(crop, str)) else crop),
                use_seed=use_seed,
                return_acs=estimate_sensitivity_maps,
            ),
        ]

    if pad_coils:
        mri_transforms += [PadCoilDimension(pad_coils=pad_coils, key=KspaceKey.kspace)]

    if estimate_body_coil_image and mask_func is not None:
        mri_transforms.append(EstimateBodyCoilImage(mask_func, backward_operator=backward_operator, use_seed=use_seed))

    if estimate_sensitivity_maps:
        mri_transforms += [
            EstimateSensitivityMap(
                kspace_key=KspaceKey.kspace,
                backward_operator=backward_operator,
                type_of_map=sensitivity_maps_type,
                gaussian_sigma=sensitivity_maps_gaussian,
                espirit_threshold=sensitivity_maps_espirit_threshold,
                espirit_kernel_size=sensitivity_maps_espirit_kernel_size,
                espirit_crop=sensitivity_maps_espirit_crop,
                espirit_max_iters=sensitivity_maps_espirit_max_iters,
            )
        ]

    if delete_acs_mask:
        mri_transforms += [DeleteKeys(keys=["acs_mask"])]

    mri_transforms += [
        ComputeImage(
            kspace_key=KspaceKey.kspace,
            target_key="target",
            backward_operator=backward_operator,
            type_reconstruction=image_recon_type,
        ),
        ApplyMask(
            sampling_mask_key="sampling_mask",
            input_kspace_key=KspaceKey.kspace,
            target_kspace_key=KspaceKey.masked_kspace,
        ),
    ]

    mri_transforms += [
        ComputeScalingFactor(
            normalize_key=scaling_key, percentile=scale_percentile, scaling_factor_key="scaling_factor"
        ),
        Normalize(scaling_factor_key="scaling_factor"),
    ]

    if delete_kspace:
        mri_transforms += [DeleteKeys(keys=[KspaceKey.kspace])]

    return Compose(mri_transforms)
