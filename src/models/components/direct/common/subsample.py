# coding=utf-8
# Copyright (c) DIRECT Contributors

"""DIRECT samplers module."""

# Code and comments can be shared with code of FastMRI under the same MIT license:
# https://github.com/facebookresearch/fastMRI/
# The code can have been adjusted to our needs.

import contextlib
import logging
from abc import abstractmethod
from enum import Enum
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import torch

import direct.data.transforms as T
from direct.common._gaussian import gaussian_mask_1d, gaussian_mask_2d  # pylint: disable=no-name-in-module
from direct.common._poisson import poisson as _poisson  # pylint: disable=no-name-in-module
from direct.environment import DIRECT_CACHE_DIR
from direct.types import Number
from direct.utils import str_to_class
from direct.utils.io import download_url

# pylint: disable=arguments-differ

__all__ = (
    "CalgaryCampinasMaskFunc",
    "FastMRIRandomMaskFunc",
    "FastMRIEquispacedMaskFunc",
    "FastMRIMagicMaskFunc",
    "Gaussian1DMaskFunc",
    "Gaussian2DMaskFunc",
    "RadialMaskFunc",
    "SpiralMaskFunc",
    "VariableDensityPoissonMaskFunc",
    "build_masking_function",
)

logger = logging.getLogger(__name__)
GOLDEN_RATIO = (1 + np.sqrt(5)) / 2


@contextlib.contextmanager
def temp_seed(rng, seed):
    state = rng.get_state()
    rng.seed(seed)
    try:
        yield
    finally:
        rng.set_state(state)


class BaseMaskFunc:
    """BaseMaskFunc is the base class to create a sub-sampling mask of a given shape."""

    def __init__(
        self,
        accelerations: Union[List[Number], Tuple[Number, ...]],
        center_fractions: Optional[Union[List[float], Tuple[float, ...]]] = None,
        uniform_range: bool = True,
    ):
        """
        Parameters
        ----------
        accelerations: Union[List[Number], Tuple[Number, ...]]
            Amount of under-sampling_mask. An acceleration of 4 retains 25% of the k-space, the method is given by
            mask_type. Has to be the same length as center_fractions if uniform_range is not True.
        center_fractions: Optional[Union[List[float], Tuple[float, ...]]]
            Fraction of low-frequency columns to be retained.
            If multiple values are provided, then one of these numbers is chosen uniformly each time. If uniform_range
            is True, then two values should be given. Default: None.
        uniform_range: bool
            If True then an acceleration will be uniformly sampled between the two values. Default: True.
        """
        if center_fractions is not None:
            if len([center_fractions]) != len([accelerations]):
                raise ValueError(
                    f"Number of center fractions should match number of accelerations. "
                    f"Got {len([center_fractions])} {len([accelerations])}."
                )

        self.center_fractions = center_fractions
        self.accelerations = accelerations

        self.uniform_range = uniform_range

        self.rng = np.random.RandomState()

    def choose_acceleration(self):
        if not self.accelerations:
            return None

        if not self.uniform_range:
            choice = self.rng.randint(0, len(self.accelerations))
            acceleration = self.accelerations[choice]
            if self.center_fractions is None:
                return acceleration

            center_fraction = self.center_fractions[choice]
            return center_fraction, acceleration
        raise NotImplementedError("Uniform range is not yet implemented.")

    @abstractmethod
    def mask_func(self, *args, **kwargs):
        raise NotImplementedError("This method should be implemented by a child class.")

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        """Produces a sampling mask by calling class method :meth:`mask_func`.

        Parameters
        ----------
        *args
        **kwargs

        Returns
        -------
        mask: torch.Tensor
            Sampling mask.
        """
        mask = self.mask_func(*args, **kwargs)
        return mask


class FastMRIMaskFunc(BaseMaskFunc):
    def __init__(
        self,
        accelerations: Union[List[Number], Tuple[Number, ...]],
        center_fractions: Optional[Union[List[float], Tuple[float, ...]]] = None,
        uniform_range: bool = False,
    ):
        super().__init__(
            accelerations=accelerations,
            center_fractions=center_fractions,
            uniform_range=uniform_range,
        )

    @staticmethod
    def center_mask_func(num_cols: int, num_low_freqs: int) -> np.ndarray:
        # create the mask
        mask = np.zeros(num_cols, dtype=bool)
        pad = (num_cols - num_low_freqs + 1) // 2
        mask[pad : pad + num_low_freqs] = True

        return mask

    @staticmethod
    def _reshape_and_broadcast_mask(shape, mask):
        num_cols = shape[-2]
        num_rows = shape[-3]

        # Reshape the mask
        mask_shape = [1 for _ in shape]
        mask_shape[-2] = num_cols
        mask = mask.reshape(*mask_shape).astype(bool)
        mask_shape[-3] = num_rows

        # Add coil axis, make array writable.
        mask = np.broadcast_to(mask, mask_shape)[np.newaxis, ...].copy()

        return mask


class FastMRIRandomMaskFunc(FastMRIMaskFunc):
    r"""Random vertical line mask function.

    The mask selects a subset of columns from the input k-space data. If the k-space data has :math:`N` columns,
    the mask picks out:

        #.  :math:`N_{\text{low freqs}} = (N \times \text{center_fraction})`  columns in the center corresponding
            to low-frequencies.
        #.  The other columns are selected uniformly at random with a probability equal to:
            :math:`\text{prob} = (N / \text{acceleration} - N_{\text{low freqs}}) / (N - N_{\text{low freqs}})`.
            This ensures that the expected number of columns selected is equal to (N / acceleration).

    It is possible to use multiple center_fractions and accelerations, in which case one possible
    (center_fraction, acceleration) is chosen uniformly at random each time the MaskFunc object is
    called.

    For example, if accelerations = [4, 8] and center_fractions = [0.08, 0.04], then there
    is a 50% probability that 4-fold acceleration with 8% center fraction is selected and a 50%
    probability that 8-fold acceleration with 4% center fraction is selected.

    """

    def __init__(
        self,
        accelerations: Union[List[Number], Tuple[Number, ...]],
        center_fractions: Optional[Union[List[float], Tuple[float, ...]]] = None,
        uniform_range: bool = False,
    ):
        super().__init__(
            accelerations=accelerations,
            center_fractions=center_fractions,
            uniform_range=uniform_range,
        )

    def mask_func(
        self,
        shape: Union[List[int], Tuple[int, ...]],
        return_acs: bool = False,
        seed: Optional[Union[int, Iterable[int]]] = None,
    ) -> torch.Tensor:
        """Creates vertical line mask.

        Parameters
        ----------
        shape: list or tuple of ints
            The shape of the mask to be created. The shape should at least 3 dimensions.
            Samples are drawn along the second last dimension.
        return_acs: bool
            Return the autocalibration signal region as a mask.
        seed: int or iterable of ints or None (optional)
            Seed for the random number generator. Setting the seed ensures the same mask is generated
             each time for the same shape. Default: None.


        Returns
        -------
        mask: torch.Tensor
            The sampling mask.
        """
        if len(shape) < 3:
            raise ValueError("Shape should have 3 or more dimensions")

        with temp_seed(self.rng, seed):
            num_cols = shape[-2]

            center_fraction, acceleration = self.choose_acceleration()
            num_low_freqs = int(round(num_cols * center_fraction))

            mask = self.center_mask_func(num_cols, num_low_freqs)

            if return_acs:
                return torch.from_numpy(self._reshape_and_broadcast_mask(shape, mask))

            # Create the mask
            prob = (num_cols / acceleration - num_low_freqs) / (num_cols - num_low_freqs)
            mask = mask | (self.rng.uniform(size=num_cols) < prob)

        return torch.from_numpy(self._reshape_and_broadcast_mask(shape, mask))


class FastMRIEquispacedMaskFunc(FastMRIMaskFunc):
    r"""Equispaced vertical line mask function.

    :class:`FastMRIEquispacedMaskFunc` creates a sub-sampling mask of given shape. The mask selects a subset of columns
    from the input k-space data. If the k-space data has N columns, the mask picks out:

        #.  :math:`N_{\text{low freqs}} = (N \times \text{center_fraction})` columns in the center corresponding
            to low-frequencies.
        #.  The other columns are selected with equal spacing at a proportion that reaches the desired acceleration
            rate taking into consideration the number of low frequencies. This ensures that the expected number of
            columns selected is equal to :math:`\frac{N}{\text{acceleration}}`.

    It is possible to use multiple center_fractions and accelerations, in which case one possible
    (center_fraction, acceleration) is chosen uniformly at random each time the EquispacedMaskFunc object is called.

    Note that this function may not give equispaced samples (documented in
    https://github.com/facebookresearch/fastMRI/issues/54), which will require modifications to standard GRAPPA
    approaches. Nonetheless, this aspect of the function has been preserved to match the public multicoil data.
    """

    def __init__(
        self,
        accelerations: Union[List[Number], Tuple[Number, ...]],
        center_fractions: Optional[Union[List[float], Tuple[float, ...]]] = None,
        uniform_range: bool = False,
    ):
        super().__init__(
            accelerations=accelerations,
            center_fractions=center_fractions,
            uniform_range=uniform_range,
        )

    def mask_func(
        self,
        shape: Union[List[int], Tuple[int, ...]],
        return_acs: bool = False,
        seed: Optional[Union[int, Iterable[int]]] = None,
    ) -> torch.Tensor:
        """Creates an vertical equispaced vertical line mask.

        Parameters
        ----------
        shape: list or tuple of ints
            The shape of the mask to be created. The shape should at least 3 dimensions.
            Samples are drawn along the second last dimension.
        return_acs: bool
            Return the autocalibration signal region as a mask.
        seed: int or iterable of ints or None (optional)
            Seed for the random number generator. Setting the seed ensures the same mask is generated
             each time for the same shape. Default: None.

        Returns
        -------
        mask: torch.Tensor
            The sampling mask.
        """
        if len(shape) < 3:
            raise ValueError("Shape should have 3 or more dimensions")

        with temp_seed(self.rng, seed):
            num_cols = shape[-2]

            center_fraction, acceleration = self.choose_acceleration()
            num_low_freqs = int(round(num_cols * center_fraction))

            mask = self.center_mask_func(num_cols, num_low_freqs)

            if return_acs:
                return torch.from_numpy(self._reshape_and_broadcast_mask(shape, mask))

            # determine acceleration rate by adjusting for the number of low frequencies
            adjusted_accel = (acceleration * (num_low_freqs - num_cols)) / (num_low_freqs * acceleration - num_cols)
            offset = self.rng.randint(0, round(adjusted_accel))

            accel_samples = np.arange(offset, num_cols - 1, adjusted_accel)
            accel_samples = np.around(accel_samples).astype(np.uint)
            mask[accel_samples] = True

        return torch.from_numpy(self._reshape_and_broadcast_mask(shape, mask))


class FastMRIMagicMaskFunc(FastMRIMaskFunc):
    """Vertical line mask function as implemented in [1]_.

    :class:`FastMRIMagicMaskFunc` exploits the conjugate symmetry via offset-sampling. It is essentially an
    equispaced mask with an offset for the opposite site of the k-space. Since MRI images often exhibit approximate
    conjugate k-space symmetry, this mask is generally more efficient than :class:`FastMRIEquispacedMaskFunc`.

    References
    ----------
    .. [1] Defazio, Aaron. “Offset Sampling Improves Deep Learning Based Accelerated MRI Reconstructions by
        Exploiting Symmetry.” ArXiv:1912.01101 [Cs, Eess], Feb. 2020. arXiv.org, http://arxiv.org/abs/1912.01101.
    """

    def __init__(
        self,
        accelerations: Union[List[Number], Tuple[Number, ...]],
        center_fractions: Optional[Union[List[float], Tuple[float, ...]]] = None,
        uniform_range: bool = False,
    ):
        super().__init__(
            accelerations=accelerations,
            center_fractions=center_fractions,
            uniform_range=uniform_range,
        )

    def mask_func(
        self,
        shape: Union[List[int], Tuple[int, ...]],
        return_acs: bool = False,
        seed: Optional[Union[int, Iterable[int]]] = None,
    ) -> torch.Tensor:
        r"""Creates a vertical equispaced mask that exploits conjugate symmetry.


        Parameters
        ----------
        shape: list or tuple of ints
            The shape of the mask to be created. The shape should at least 3 dimensions.
            Samples are drawn along the second last dimension.
        return_acs: bool
            Return the autocalibration signal region as a mask.
        seed: int or iterable of ints or None (optional)
            Seed for the random number generator. Setting the seed ensures the same mask is generated
             each time for the same shape. Default: None.

        Returns
        -------
        mask: torch.Tensor
            The sampling mask.
        """
        if len(shape) < 3:
            raise ValueError("Shape should have 3 or more dimensions")

        with temp_seed(self.rng, seed):
            num_cols = shape[-2]

            center_fraction, acceleration = self.choose_acceleration()

            num_low_freqs = int(round(num_cols * center_fraction))
            # bound the number of low frequencies between 1 and target columns
            target_cols_to_sample = int(round(num_cols / acceleration))
            num_low_freqs = max(min(num_low_freqs, target_cols_to_sample), 1)

            acs_mask = self.center_mask_func(num_cols, num_low_freqs)

            if return_acs:
                return torch.from_numpy(self._reshape_and_broadcast_mask(shape, acs_mask))

            # adjust acceleration rate based on target acceleration.
            adjusted_target_cols_to_sample = target_cols_to_sample - num_low_freqs
            adjusted_acceleration = 0
            if adjusted_target_cols_to_sample > 0:
                adjusted_acceleration = int(round(num_cols / adjusted_target_cols_to_sample))

            offset = self.rng.randint(0, high=adjusted_acceleration)

            if offset % 2 == 0:
                offset_pos = offset + 1
                offset_neg = offset + 2
            else:
                offset_pos = offset - 1 + 3
                offset_neg = offset - 1 + 0

            poslen = (num_cols + 1) // 2
            neglen = num_cols - (num_cols + 1) // 2
            mask_positive = np.zeros(poslen, dtype=bool)
            mask_negative = np.zeros(neglen, dtype=bool)

            mask_positive[offset_pos::adjusted_acceleration] = True
            mask_negative[offset_neg::adjusted_acceleration] = True
            mask_negative = np.flip(mask_negative)

            mask = np.fft.fftshift(np.concatenate((mask_positive, mask_negative)))
            mask = np.logical_or(mask, acs_mask)

        return torch.from_numpy(self._reshape_and_broadcast_mask(shape, mask))


class CalgaryCampinasMaskFunc(BaseMaskFunc):
    BASE_URL = "https://s3.aiforoncology.nl/direct-project/calgary_campinas_masks/"
    MASK_MD5S = {
        "R10_218x170.npy": "6e1511c33dcfc4a960f526252676f7c3",
        "R10_218x174.npy": "78fe23ae5eed2d3a8ff3ec128388dcc9",
        "R10_218x180.npy": "5039a6c19ac2aa3472a94e4b015e5228",
        "R5_218x170.npy": "6599715103cf3d71d6e87d09f865e7da",
        "R5_218x174.npy": "5bd27d2da3bf1e78ad1b65c9b5e4b621",
        "R5_218x180.npy": "717b51f3155c3a64cfaaddadbe90791d",
    }

    # TODO: Configuration improvements, so no **kwargs needed.
    # pylint: disable=unused-argument
    def __init__(self, accelerations: Union[List[Number], Tuple[Number, ...]], **kwargs):  # noqa
        super().__init__(accelerations=accelerations, uniform_range=False)

        if not all(_ in [5, 10] for _ in accelerations):
            raise ValueError("CalgaryCampinas only provide 5x and 10x acceleration masks.")

        self.masks = {}
        self.shapes: List[Number] = []

        for acceleration in accelerations:
            self.masks[acceleration] = self.__load_masks(acceleration)

    @staticmethod
    def circular_centered_mask(shape, radius):
        center = np.asarray(shape) // 2
        Y, X = np.ogrid[: shape[0], : shape[1]]
        dist_from_center = np.sqrt((X - center[1]) ** 2 + (Y - center[0]) ** 2)
        mask = ((dist_from_center <= radius) * np.ones(shape)).astype(bool)
        return mask[np.newaxis, ..., np.newaxis]

    def mask_func(
        self,
        shape: Union[List[int], Tuple[int, ...]],
        return_acs: bool = False,
        seed: Optional[Union[int, Iterable[int]]] = None,
    ) -> torch.Tensor:
        r"""Downloads and loads pre-computed Poisson masks.

        Currently supports shapes of :math`218 \times 170/174/180` and acceleration factors of `5` or `10`.

        Parameters
        ----------
        shape: list or tuple of ints
            The shape of the mask to be created. The shape should at least 3 dimensions.
            Samples are drawn along the second last dimension.
        return_acs: bool
            Return the autocalibration signal region as a mask.
        seed: int or iterable of ints or None (optional)
            Seed for the random number generator. Setting the seed ensures the same mask is generated
             each time for the same shape. Default: None.

        Returns
        -------
        mask: torch.Tensor
            The sampling mask.
        """
        shape = tuple(shape)[:-1]
        if return_acs:
            return torch.from_numpy(self.circular_centered_mask(shape, 18))

        if shape not in self.shapes:
            raise ValueError(f"No mask of shape {shape} is available in the CalgaryCampinas dataset.")

        with temp_seed(self.rng, seed):
            acceleration = self.choose_acceleration()
            masks = self.masks[acceleration]

            mask, num_masks = masks[shape]
            # Randomly pick one example
            choice = self.rng.randint(0, num_masks)

        return torch.from_numpy(mask[choice][np.newaxis, ..., np.newaxis])

    def __load_masks(self, acceleration):
        masks_path = DIRECT_CACHE_DIR / "calgary_campinas_masks"
        paths = [
            f"R{acceleration}_218x170.npy",
            f"R{acceleration}_218x174.npy",
            f"R{acceleration}_218x180.npy",
        ]

        downloaded = [download_url(self.BASE_URL + _, masks_path, md5=self.MASK_MD5S[_]) is None for _ in paths]
        if not all(downloaded):
            raise RuntimeError(f"Failed to download all Calgary-Campinas masks from {self.BASE_URL}.")

        output = {}
        for path in paths:
            shape = [int(_) for _ in path.split("_")[-1][:-4].split("x")]
            self.shapes.append(tuple(shape))
            mask_array = np.load(masks_path / path)
            output[tuple(shape)] = mask_array, mask_array.shape[0]

        return output


class CIRCUSSamplingMode(str, Enum):
    circus_radial = "circus-radial"
    circus_spiral = "circus-spiral"


class CIRCUSMaskFunc(BaseMaskFunc):
    """Implementation of Cartesian undersampling (radial or spiral) using CIRCUS as shown in [1]_. It creates radial or
    spiral masks for Cartesian acquired data on a grid.

    References
    ----------

    .. [1] Liu J, Saloner D. Accelerated MRI with CIRcular Cartesian UnderSampling (CIRCUS): a variable density
        Cartesian sampling strategy for compressed sensing and parallel imaging. Quant Imaging Med Surg.
        2014 Feb;4(1):57-67. doi: 10.3978/j.issn.2223-4292.2014.02.01. PMID: 24649436; PMCID: PMC3947985.
    """

    def __init__(
        self,
        accelerations: Union[List[Number], Tuple[Number, ...]],
        subsampling_scheme: CIRCUSSamplingMode,
        **kwargs,
    ):
        super().__init__(
            accelerations=accelerations,
            center_fractions=tuple(0 for _ in range(len(accelerations))),
            uniform_range=False,
        )
        if subsampling_scheme not in ["circus-spiral", "circus-radial"]:
            raise NotImplementedError(
                f"Currently CIRCUSMaskFunc is only implemented for 'circus-radial' or 'circus-spiral' "
                f"as a subsampling_scheme. Got subsampling_scheme={subsampling_scheme}."
            )

        self.subsampling_scheme = "circus-radial" if subsampling_scheme is None else subsampling_scheme

    @staticmethod
    def get_square_ordered_idxs(square_side_size: int, square_id: int) -> Tuple[Tuple, ...]:
        """Returns ordered (clockwise) indices of a sub-square of a square matrix.

        Parameters
        ----------
        square_side_size: int
            Square side size. Dim of array.
        square_id: int
            Number of sub-square. Can be 0, ..., square_side_size // 2.

        Returns
        -------
        ordered_idxs: List of tuples.
            Indices of each point that belongs to the square_id-th sub-square
            starting from top-left point clockwise.
        """
        assert square_id in range(square_side_size // 2)

        ordered_idxs = list()

        for col in range(square_id, square_side_size - square_id):
            ordered_idxs.append((square_id, col))

        for row in range(square_id + 1, square_side_size - (square_id + 1)):
            ordered_idxs.append((row, square_side_size - (square_id + 1)))

        for col in range(square_side_size - (square_id + 1), square_id, -1):
            ordered_idxs.append((square_side_size - (square_id + 1), col))

        for row in range(square_side_size - (square_id + 1), square_id, -1):
            ordered_idxs.append((row, square_id))

        return tuple(ordered_idxs)

    def circus_radial_mask(self, shape, acceleration):
        """Implements CIRCUS radial undersampling."""
        max_dim = max(shape) - max(shape) % 2
        min_dim = min(shape) - min(shape) % 2
        num_nested_squares = max_dim // 2
        M = int(np.prod(shape) / (acceleration * (max_dim / 2 - (max_dim - min_dim) * (1 + min_dim / max_dim) / 4)))

        mask = np.zeros((max_dim, max_dim), dtype=np.float32)

        t = self.rng.randint(low=0, high=1e4, size=1, dtype=int).item()

        for square_id in range(num_nested_squares):
            ordered_indices = self.get_square_ordered_idxs(
                square_side_size=max_dim,
                square_id=square_id,
            )
            # J: size of the square, J=2,…,N, i.e., the number of points along one side of the square
            J = 2 * (num_nested_squares - square_id)
            # K: total number of points along the perimeter of the square K=4·J-4;
            K = 4 * (J - 1)

            for m in range(M):
                indices_idx = int(np.floor(np.mod((m + t * M) / GOLDEN_RATIO, 1) * K))
                mask[ordered_indices[indices_idx]] = 1.0

        pad = ((shape[0] % 2, 0), (shape[1] % 2, 0))

        mask = np.pad(mask, pad, constant_values=0)
        mask = T.center_crop(torch.from_numpy(mask.astype(bool)), shape)

        return mask

    def circus_spiral_mask(self, shape, acceleration):
        """Implements CIRCUS spiral undersampling."""
        max_dim = max(shape) - max(shape) % 2
        min_dim = min(shape) - min(shape) % 2

        num_nested_squares = max_dim // 2

        M = int(np.prod(shape) / (acceleration * (max_dim / 2 - (max_dim - min_dim) * (1 + min_dim / max_dim) / 4)))

        mask = np.zeros((max_dim, max_dim), dtype=np.float32)

        c = self.rng.uniform(low=1.1, high=1.3, size=1).item()

        for square_id in range(num_nested_squares):
            ordered_indices = self.get_square_ordered_idxs(
                square_side_size=max_dim,
                square_id=square_id,
            )

            # J: size of the square, J=2,…,N, i.e., the number of points along one side of the square
            J = 2 * (num_nested_squares - square_id)
            # K: total number of points along the perimeter of the square K=4·J-4;
            K = 4 * (J - 1)

            for m in range(M):
                i = np.floor(np.mod(m / GOLDEN_RATIO, 1) * K)
                indices_idx = int(np.mod((i + np.ceil(J**c) - 1), K))

                mask[ordered_indices[indices_idx]] = 1.0

        pad = ((shape[0] % 2, 0), (shape[1] % 2, 0))

        mask = np.pad(mask, pad)
        mask = T.center_crop(torch.from_numpy(mask.astype(bool)), shape)

        return mask

    @staticmethod
    def circular_centered_mask(mask, eps=0.1):
        shape = mask.shape
        center = np.asarray(shape) // 2
        Y, X = np.ogrid[: shape[0], : shape[1]]
        Y, X = torch.tensor(Y), torch.tensor(X)
        radius = 1

        # Finds the maximum (unmasked) disk in mask given a tolerance.
        while True:
            # Creates a disk with R=radius and finds intersection with mask
            disk = (Y - center[0]) ** 2 + (X - center[1]) ** 2 <= radius**2
            intersection = disk & mask
            ratio = disk.sum() / intersection.sum()
            if ratio > 1.0 + eps:
                return intersection
            radius += eps

    def mask_func(
        self,
        shape: Union[List[int], Tuple[int, ...]],
        return_acs: bool = False,
        seed: Optional[Union[int, Iterable[int]]] = None,
    ) -> torch.Tensor:
        """Produces :class:`CIRCUSMaskFunc` sampling masks.

        Parameters
        ----------
        shape: list or tuple of ints
            The shape of the mask to be created. The shape should at least 3 dimensions.
            Samples are drawn along the second last dimension.
        return_acs: bool
            Return the autocalibration signal region as a mask.
        seed: int or iterable of ints or None (optional)
            Seed for the random number generator. Setting the seed ensures the same mask is generated
             each time for the same shape. Default: None.

        Returns
        -------
        mask: torch.Tensor
            The sampling mask.
        """

        if len(shape) < 3:
            raise ValueError("Shape should have 3 or more dimensions")

        with temp_seed(self.rng, seed):
            num_rows = shape[-3]
            num_cols = shape[-2]
            acceleration = self.choose_acceleration()[1]

            if self.subsampling_scheme == "circus-radial":
                mask = self.circus_radial_mask(
                    shape=(num_rows, num_cols),
                    acceleration=acceleration,
                )
            elif self.subsampling_scheme == "circus-spiral":
                mask = self.circus_spiral_mask(
                    shape=(num_rows, num_cols),
                    acceleration=acceleration,
                )

            if return_acs:
                return self.circular_centered_mask(mask).unsqueeze(0).unsqueeze(-1)

            return mask.unsqueeze(0).unsqueeze(-1)


class RadialMaskFunc(CIRCUSMaskFunc):
    """Computes radial masks for Cartesian data."""

    def __init__(
        self,
        accelerations: Union[List[Number], Tuple[Number, ...]],
        **kwargs,
    ):
        super().__init__(
            accelerations=accelerations,
            subsampling_scheme=CIRCUSSamplingMode.circus_radial,
            **kwargs,
        )


class SpiralMaskFunc(CIRCUSMaskFunc):
    """Computes spiral masks for Cartesian data."""

    def __init__(
        self,
        accelerations: Union[List[Number], Tuple[Number, ...]],
        **kwargs,
    ):
        super().__init__(
            accelerations=accelerations,
            subsampling_scheme=CIRCUSSamplingMode.circus_spiral,
            **kwargs,
        )


class VariableDensityPoissonMaskFunc(BaseMaskFunc):
    """Variable Density Poisson sampling mask function. Based on [1]_.

    Notes
    -----

    * Code inspired and modified from [2]_ with BSD-3 licence, Copyright (c) 2016, Frank Ong, Copyright (c) 2016,
        The Regents of the University of California [3]_.

    References
    ----------

    .. [1] Bridson, Robert. “Fast Poisson Disk Sampling in Arbitrary Dimensions.” ACM SIGGRAPH 2007
        Sketches on - SIGGRAPH ’07, ACM Press, 2007, pp. 22-es. DOI.org (Crossref),
        https://doi.org/10.1145/1278780.1278807.
    .. [2] https://github.com/mikgroup/sigpy/blob/1817ff849d34d7cbbbcb503a1b310e7d8f95c242/sigpy/mri/samp.py#L11
    .. [3] https://github.com/mikgroup/sigpy/blob/master/LICENSE

    """

    def __init__(
        self,
        accelerations: Union[List[Number], Tuple[Number, ...]],
        center_fractions: Union[List[float], Tuple[float, ...]],
        crop_corner: Optional[bool] = False,
        max_attempts: Optional[int] = 10,
        tol: Optional[float] = 0.2,
        slopes: Optional[Union[List[float], Tuple[float, ...]]] = None,
        **kwargs,
    ):
        """Inits :class:`VariableDensityPoissonMaskFunc`.

        Parameters
        ----------
        accelerations: list or tuple of positive numbers
            Amount of under-sampling.
        center_fractions: list or tuple of floats
            Must have the same length as `accelerations`. Amount of center fully-sampling.
            For center_scale='r', then a centered disk area with radius equal to
            :math:`R = \sqrt{{n_r}^2 + {n_c}^2} \times r` will be fully sampled, where :math:`n_r` and :math:`n_c`
            denote the input shape.
        crop_corner: bool, optional
            If True mask will be disk. Default: False.
        max_attempts: int, optional
            Maximum rejection samples. Default: 10.
        tol: float, optional
            Maximum deviation between the generated mask acceleration and the desired acceleration. Default: 0.2.
        slopes: Optional[Union[List[float], Tuple[float, ...]]]
            An increasing sequence of non-negative floats (of length 2) to be used
            for the generation of the sampling radius. Default: None.
        """
        super().__init__(
            accelerations=accelerations,
            center_fractions=center_fractions,
            uniform_range=False,
        )
        self.crop_corner = crop_corner
        self.max_attempts = max_attempts
        self.tol = tol
        if slopes is not None:
            assert (
                slopes[0] >= 0 and slopes[0] < slopes[1] and len(slopes) == 2
            ), f"`slopes` must be an increasing sequence of two non-negative floats. Received {slopes}."
        self.slopes = slopes

    def mask_func(
        self,
        shape: Union[List[int], Tuple[int, ...]],
        return_acs: bool = False,
        seed: Optional[Union[int, Iterable[int]]] = None,
    ) -> torch.Tensor:
        """Produces variable Density Poisson sampling masks.

        Parameters
        ----------
        shape: list or tuple of ints
            The shape of the mask to be created. The shape should at least 3 dimensions.
            Samples are drawn along the second last dimension.
        return_acs: bool
            Return the autocalibration signal region as a mask.
        seed: int or iterable of ints or None (optional)
            Seed for the random number generator. Setting the seed ensures the same mask is generated
             each time for the same shape. Default: None.

        Returns
        -------
        mask: torch.Tensor
            The sampling mask of shape (1, shape[0], shape[1], 1).
        """
        if len(shape) < 3:
            raise ValueError("Shape should have 3 or more dimensions")
        num_rows, num_cols = shape[:2]

        with temp_seed(self.rng, seed):
            center_fraction, acceleration = self.choose_acceleration()
            if return_acs:
                return torch.from_numpy(
                    centered_disk_mask((num_rows, num_cols), center_fraction)[np.newaxis, ..., np.newaxis]
                ).bool()
            mask = self.poisson(num_rows, num_cols, center_fraction, acceleration, integerize_seed(seed))
        return torch.from_numpy(mask[np.newaxis, ..., np.newaxis]).bool()

    def poisson(
        self,
        num_rows: int,
        num_cols: int,
        center_fraction: float,
        acceleration: float,
        seed: int = 0,
    ) -> torch.Tensor:
        """Calculates mask by calling the cython `_poisson` method.

        Parameters
        ----------
        num_rows: int
            Number of rows - x-axis size.
        num_cols: int
            Number of columns - y-axis size.
        center_fraction: float
            Amount of center fully-sampling.
        acceleration: float
            Acceleration factor.
        seed: int
            Seed to be used by cython function. Default: 0.

        Returns
        -------
        mask: torch.Tensor
            Sampling mask of shape (`num_rows`, `num_cols`).
        """
        # pylint: disable=too-many-locals
        x, y = np.mgrid[:num_rows, :num_cols]

        x = np.maximum(abs(x - num_rows / 2), 0)
        x /= x.max()
        y = np.maximum(abs(y - num_cols / 2), 0)
        y /= y.max()
        r = np.sqrt(x**2 + y**2)

        if self.slopes is not None:
            slope_min, slope_max = self.slopes
        else:
            slope_min, slope_max = 0, max(num_rows, num_cols)

        while slope_min < slope_max:
            slope = (slope_max + slope_min) / 2
            radius_x = np.clip((1 + r * slope) * num_rows / max(num_rows, num_cols), 1, None)

            radius_y = np.clip((1 + r * slope) * num_cols / max(num_rows, num_cols), 1, None)

            mask = np.zeros((num_rows, num_cols), dtype=int)

            _poisson(num_rows, num_cols, self.max_attempts, mask, radius_x, radius_y, seed)

            mask = mask | centered_disk_mask((num_rows, num_cols), center_fraction)

            if self.crop_corner:
                mask *= r < 1

            actual_acceleration = num_rows * num_cols / mask.sum()

            if abs(actual_acceleration - acceleration) < self.tol:
                break
            if actual_acceleration < acceleration:
                slope_min = slope
            else:
                slope_max = slope

        if abs(actual_acceleration - acceleration) >= self.tol:
            raise ValueError(f"Cannot generate mask to satisfy accel={acceleration}.")

        return mask


class Gaussian1DMaskFunc(FastMRIMaskFunc):
    """Gaussian 1D vertical line mask function."""

    def __init__(
        self,
        accelerations: Union[List[Number], Tuple[Number, ...]],
        center_fractions: Optional[Union[List[float], Tuple[float, ...]]] = None,
        uniform_range: bool = False,
    ):
        super().__init__(
            accelerations=accelerations,
            center_fractions=center_fractions,
            uniform_range=uniform_range,
        )

    def mask_func(
        self,
        shape: Union[List[int], Tuple[int, ...]],
        return_acs: bool = False,
        seed: Optional[Union[int, Iterable[int]]] = None,
    ) -> torch.Tensor:
        """Creates a vertical gaussian mask.

        Parameters
        ----------
        shape: list or tuple of ints
            The shape of the mask to be created. The shape should at least 3 dimensions.
            Samples are drawn along the second last dimension.
        return_acs: bool
            Return the autocalibration signal region as a mask.
        seed: int or iterable of ints or None (optional)
            Seed for the random number generator. Setting the seed ensures the same mask is generated
             each time for the same shape. Default: None.


        Returns
        -------
        mask: torch.Tensor
            The sampling mask.
        """
        if len(shape) < 3:
            raise ValueError("Shape should have 3 or more dimensions")

        with temp_seed(self.rng, seed):
            num_cols = shape[-2]

            center_fraction, acceleration = self.choose_acceleration()
            num_low_freqs = int(round(num_cols * center_fraction))

            mask = self.center_mask_func(num_cols, num_low_freqs).astype(int)

            if return_acs:
                return torch.from_numpy(self._reshape_and_broadcast_mask(shape, mask))

            # Calls cython function
            nonzero_count = int(np.round(num_cols / acceleration - num_low_freqs - 1))
            gaussian_mask_1d(
                nonzero_count, num_cols, num_cols // 2, 6 * np.sqrt(num_cols // 2), mask, integerize_seed(seed)
            )

        return torch.from_numpy(self._reshape_and_broadcast_mask(shape, mask).astype(bool))


class Gaussian2DMaskFunc(BaseMaskFunc):
    """Gaussian 2D mask function."""

    def __init__(
        self,
        accelerations: Union[List[Number], Tuple[Number, ...]],
        center_fractions: Optional[Union[List[float], Tuple[float, ...]]] = None,
        uniform_range: bool = False,
    ):
        super().__init__(
            accelerations=accelerations,
            center_fractions=center_fractions,
            uniform_range=uniform_range,
        )

    def mask_func(
        self,
        shape: Union[List[int], Tuple[int, ...]],
        return_acs: bool = False,
        seed: Optional[Union[int, Iterable[int]]] = None,
    ) -> torch.Tensor:
        """Creates a 2D gaussian mask.

        Parameters
        ----------
        shape: list or tuple of ints
            The shape of the mask to be created. The shape should at least 3 dimensions.
            Samples are drawn along the second last dimension.
        return_acs: bool
            Return the autocalibration signal region as a mask.
        seed: int or iterable of ints or None (optional)
            Seed for the random number generator. Setting the seed ensures the same mask is generated
             each time for the same shape. Default: None.


        Returns
        -------
        mask: torch.Tensor
            The sampling mask.
        """
        if len(shape) < 3:
            raise ValueError("Shape should have 3 or more dimensions")

        num_rows, num_cols = shape[:2]

        with temp_seed(self.rng, seed):
            center_fraction, acceleration = self.choose_acceleration()

            mask = centered_disk_mask((num_rows, num_cols), center_fraction)
            if return_acs:
                return torch.from_numpy(mask[np.newaxis, ..., np.newaxis]).bool()

            # Calls cython function
            nonzero_count = int(np.round(num_cols * num_rows / acceleration - mask.sum() - 1))
            std = 6 * np.array([np.sqrt(num_rows // 2), np.sqrt(num_cols // 2)], dtype=float)
            gaussian_mask_2d(
                nonzero_count, num_rows, num_cols, num_rows // 2, num_cols // 2, std, mask, integerize_seed(seed)
            )

        return torch.from_numpy(mask[np.newaxis, ..., np.newaxis]).bool()


def integerize_seed(seed: Union[None, Tuple[int, ...], List[int]]) -> int:
    """Returns an integer seed.

    If input is integer, will return it. If it's None, it will return a random integer in [0, 1e6). If it's a tuple
    or list of integers, will return the integer part of the average.

    Can be useful for functions that take as input only integer seeds (e.g. cython functions).

    Parameters
    ----------
    seed : int, tuple or list of ints, None
         Input seed to integerize.

    Returns
    -------
    out_seed: int
        Integer seed.
    """
    if seed is None:
        return np.random.randint(0, 1e6)
    if isinstance(seed, int):
        return seed
    if isinstance(seed, (tuple, list)):
        return int(np.mean(seed))
    raise ValueError(f"Invalid seed type. Can be None, integer, or tuple or list of integers. Received {seed}.")


def centered_disk_mask(shape: Union[List[int], Tuple[int, ...]], center_scale: float) -> np.ndarray:
    r"""Creates a mask with a centered disk of radius :math:`R=\sqrt{c_x \cdot c_y \cdot r / \pi}`.

    Parameters
    ----------
    shape : list or tuple of ints
        The shape of the (2D) mask to be created.
    center_scale : float
        Center scale.

    Returns
    -------
    mask : np.ndarray
    """
    center_x = shape[0] // 2
    center_y = shape[1] // 2

    X, Y = np.indices(shape)
    radius = int(np.sqrt(np.prod(shape) * center_scale / np.pi))

    mask = ((X - center_x) ** 2 + (Y - center_y) ** 2) < radius**2
    return mask.astype(int)


class DictionaryMaskFunc(BaseMaskFunc):
    def __init__(self, data_dictionary, **kwargs):  # noqa
        super().__init__(accelerations=None)

        self.data_dictionary = data_dictionary

    def mask_func(self, data, return_acs=False):
        return self.data_dictionary[data]


def build_masking_function(name, accelerations, center_fractions=None, uniform_range=False, **kwargs):
    MaskFunc: BaseMaskFunc = str_to_class("direct.common.subsample", name + "MaskFunc")  # noqa
    mask_func = MaskFunc(
        accelerations=accelerations,
        center_fractions=center_fractions,
        uniform_range=uniform_range,
    )

    return mask_func
