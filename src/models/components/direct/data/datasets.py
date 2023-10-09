# coding=utf-8
# Copyright (c) DIRECT Contributors

"""DIRECT datasets module."""
import bisect
import contextlib
import logging
import pathlib
import sys
import xml.etree.ElementTree as etree  # nosec
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from omegaconf import DictConfig
from torch.utils.data import Dataset, IterableDataset

from src.models.components.direct.data.fake import FakeMRIData
from src.models.components.direct.data.h5_data import H5SliceData
from src.models.components.direct.data.sens import simulate_sensitivity_maps
from src.models.components.direct.data.types import PathOrString
from src.models.components.direct.utils import remove_keys, str_to_class

logger = logging.getLogger(__name__)

__all__ = [
    "build_dataset_from_input",
    "CalgaryCampinasDataset",
    "ConcatDataset",
    "FastMRIDataset",
    "FakeMRIBlobsDataset",
    "SheppLoganDataset",
    "SheppLoganT1Dataset",
    "SheppLoganT2Dataset",
    "SheppLoganProtonDataset",
]


@contextlib.contextmanager
def temp_seed(rng, seed):
    state = rng.get_state()
    rng.seed(seed)
    try:
        yield
    finally:
        rng.set_state(state)


def _et_query(
    root: etree.Element,
    qlist: Sequence[str],
    namespace: str = "http://www.ismrm.org/ISMRMRD",
) -> str:
    """
    ElementTree query function.
    This can be used to query an xml document via ElementTree. It uses qlist
    for nested queries.
    Args:
        root: Root of the xml to search through.
        qlist: A list of strings for nested searches, e.g. ["Encoding",
            "matrixSize"]
        namespace: Optional; xml namespace to prepend query.
    Returns:
        The retrieved data as a string.

    From:
    https://github.com/facebookresearch/fastMRI/blob/13560d2f198cc72f06e01675e9ecee509ce5639a/fastmri/data/mri_data.py#L23

    """
    s = "."
    prefix = "ismrmrd_namespace"

    ns = {prefix: namespace}

    for el in qlist:
        s = s + f"//{prefix}:{el}"

    value = root.find(s, ns)
    if value is None:
        raise RuntimeError("Element not found")

    return str(value.text)


class FakeMRIBlobsDataset(Dataset):
    """A PyTorch Dataset class which outputs random fake k-space images which reconstruct into Gaussian blobs."""

    def __init__(
        self,
        sample_size: int,
        num_coils: int,
        spatial_shape: Union[List[int], Tuple[int]],
        transform: Optional[Callable] = None,
        seed: Optional[int] = None,
        filenames: Optional[Union[List[str], str]] = None,
        pass_attrs: Optional[bool] = None,
        text_description: Optional[str] = None,
        kspace_context: Optional[bool] = None,
        **kwargs,
    ) -> None:
        """Dataset initialisation.

        Parameters
        ----------
        sample_size: int
            Size of the dataset.
        num_coils: int
            Number of coils for the fake k-space data.
        spatial_shape: List or Tuple of ints.
            Shape of the reconstructed fake data. Should be (height, width) or (slice, height, width), corresponding
            to ndim = 2 and ndim = 3.
        transform: Optional[Callable]
            A list of transforms to be performed on the generated samples. Default is None.
        seed: int
            Seed. Default is None.
        filenames: List of strings or string.
            Names for the generated samples. If string is given, a number order starting from "00001" is appended
            to the name of each sample.
        pass_attrs: bool
            Pass the attributes of the generated sample.
        text_description: str
            Description of dataset, can be useful for logging.
        kspace_context: bool
            If true corresponds to 3D reconstruction, else reconstruction is 2D.
        """

        self.logger = logging.getLogger(type(self).__name__)

        if len(spatial_shape) not in [2, 3]:
            raise NotImplementedError(
                f"Currently FakeDataset is implemented only for 2D or 3D data. "
                f"Spatial shape must have 2 or 3 dimensions. Got shape {spatial_shape}."
            )
        self.sample_size = sample_size
        self.num_coils = num_coils
        self.spatial_shape = spatial_shape
        self.transform = transform
        self.pass_attrs = pass_attrs if pass_attrs is not None else True
        self.text_description = text_description
        if self.text_description:
            self.logger.info("Dataset description: %s.", self.text_description)

        self.fake_data: Callable = FakeMRIData(
            ndim=len(self.spatial_shape),
            blobs_n_samples=kwargs.get("blobs_n_samples", None),
            blobs_cluster_std=kwargs.get("blobs_cluster_std", None),
        )
        self.volume_indices: Dict[str, range] = {}

        self.rng = np.random.RandomState()

        with temp_seed(self.rng, seed):
            # size = sample_size * num_slices if data is 3D
            self.data = [
                (filename, slice_no, seed)
                for (filename, seed) in zip(
                    self.parse_filenames_data(filenames),
                    list(self.rng.choice(a=range(int(1e5)), size=self.sample_size, replace=False)),
                )  # ensure reproducibility
                for slice_no in range(self.spatial_shape[0] if len(spatial_shape) == 3 else 1)
            ]
        self.kspace_context = kspace_context if kspace_context else 0
        self.ndim = 2 if self.kspace_context == 0 else 3

        if self.kspace_context != 0:
            raise NotImplementedError("3D reconstruction is not yet supported with FakeMRIBlobsDataset.")

    def parse_filenames_data(self, filenames):
        if filenames is None:
            filenames = ["sample"]

        if isinstance(filenames, str):
            filenames = [filenames]

        if len(filenames) != self.sample_size:
            filenames = [filenames[0] + f"{_:05}" for _ in range(1, self.sample_size + 1)]

        current_slice_number = 0
        for idx, filename in enumerate(filenames):
            if len(filenames) < 5 or idx % (len(filenames) // 5) == 0 or len(filenames) == (idx + 1):
                # pylint: disable=logging-fstring-interpolation
                self.logger.info(f"Parsing: {(idx + 1) / len(filenames) * 100:.2f}%.")

            num_slices = self.spatial_shape[0] if len(self.spatial_shape) == 3 else 1
            self.volume_indices[pathlib.PosixPath(filename)] = range(
                current_slice_number, current_slice_number + num_slices
            )
            current_slice_number += num_slices

        return filenames

    @staticmethod
    def _get_metadata(metadata):
        encoding_size = metadata["encoding_size"]
        reconstruction_size = metadata["reconstruction_size"]
        metadata = {
            "encoding_size": encoding_size,
            "reconstruction_size": reconstruction_size,
        }
        return metadata

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        filename, slice_no, sample_seed = self.data[idx]

        sample = self.fake_data(
            sample_size=1,
            num_coils=self.num_coils,
            spatial_shape=self.spatial_shape,
            name=[filename],
            seed=sample_seed,
        )[0]
        sample["kspace"] = sample["kspace"][slice_no]

        if "attrs" in sample:
            metadata = self._get_metadata(sample["attrs"])
            sample.update(metadata)

            if self.pass_attrs:
                sample["scaling_factor"] = sample["attrs"]["max"]

            del sample["attrs"]

        sample["slice_no"] = slice_no
        if sample["kspace"].ndim == 2:  # Singlecoil data does not always have coils at the first axis.
            sample["kspace"] = sample["kspace"][np.newaxis, ...]

        if self.transform:
            sample = self.transform(sample)

        return sample


def _parse_fastmri_header(xml_header: str) -> Dict:
    # Borrowed from: https://github.com/facebookresearch/\
    # fastMRI/blob/13560d2f198cc72f06e01675e9ecee509ce5639a/fastmri/data/mri_data.py#L23
    et_root = etree.fromstring(xml_header)  # nosec

    encodings = ["encoding", "encodedSpace", "matrixSize"]
    encoding_size = (
        int(_et_query(et_root, encodings + ["x"])),
        int(_et_query(et_root, encodings + ["y"])),
        int(_et_query(et_root, encodings + ["z"])),
    )
    reconstructions = ["encoding", "reconSpace", "matrixSize"]
    reconstruction_size = (
        int(_et_query(et_root, reconstructions + ["x"])),
        int(_et_query(et_root, reconstructions + ["y"])),
        int(_et_query(et_root, reconstructions + ["z"])),
    )

    limits = ["encoding", "encodingLimits", "kspace_encoding_step_1"]
    encoding_limits_center = int(_et_query(et_root, limits + ["center"]))
    encoding_limits_max = int(_et_query(et_root, limits + ["maximum"])) + 1

    padding_left = encoding_size[1] // 2 - encoding_limits_center
    padding_right = padding_left + encoding_limits_max

    metadata = {
        "padding_left": padding_left,
        "padding_right": padding_right,
        "encoding_size": encoding_size,
        "reconstruction_size": reconstruction_size,
    }

    return metadata


class FastMRIDataset(H5SliceData):
    """FastMRI challenge dataset."""

    def __init__(
        self,
        data_root: pathlib.Path,
        transform: Optional[Callable] = None,
        filenames_filter: Optional[List[PathOrString]] = None,
        filenames_lists: Union[List[PathOrString], None] = None,
        filenames_lists_root: Union[PathOrString, None] = None,
        regex_filter: Optional[str] = None,
        pass_mask: bool = False,
        pass_max: bool = True,
        initial_images: Union[List[pathlib.Path], None] = None,
        initial_images_key: Optional[str] = None,
        noise_data: Optional[Dict] = None,
        pass_h5s: Optional[Dict] = None,
        **kwargs,
    ) -> None:
        # TODO: Clean up Dataset class such that only **kwargs need to get parsed.
        # BODY: Additional keysneeded for this dataset can be popped if needed.
        self.pass_mask = pass_mask
        extra_keys = ["mask"] if pass_mask else []
        extra_keys.append("ismrmrd_header")

        super().__init__(
            root=data_root,
            filenames_filter=filenames_filter,
            filenames_lists=filenames_lists,
            filenames_lists_root=filenames_lists_root,
            regex_filter=regex_filter,
            metadata=None,
            extra_keys=tuple(extra_keys),
            pass_attrs=pass_max,
            text_description=kwargs.get("text_description", None),
            pass_h5s=pass_h5s,
            pass_dictionaries=kwargs.get("pass_dictionaries", None),
        )
        if self.sensitivity_maps is not None:
            raise NotImplementedError(
                f"Sensitivity maps are not supported in the current " f"{self.__class__.__name__} class."
            )

        # TODO: Make exclusive or to give error when one of the two keys is not set.
        # TODO: Convert into mixin, and add support to main image
        # TODO: Such a support would also work for the sensitivity maps
        self.initial_images_key = initial_images_key
        self.initial_images = {}

        if initial_images:
            self.initial_images = {k.name: k for k in initial_images}

        self.noise_data = noise_data
        self.transform = transform

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = super().__getitem__(idx)

        if self.pass_attrs:
            sample["scaling_factor"] = sample["attrs"]["max"]
            del sample["attrs"]

        sample.update(_parse_fastmri_header(sample["ismrmrd_header"]))
        del sample["ismrmrd_header"]
        # Some images have strange behavior, e.g. FLAIR 203.
        image_shape = sample["kspace"].shape
        if image_shape[-1] < sample["reconstruction_size"][-2]:  # reconstruction size is (x, y, z)
            sample["reconstruction_size"] = (image_shape[-1], image_shape[-1], 1)

        if self.pass_mask:
            # mask should be shape (1, h, w, 1) mask provided is only w
            kspace_shape = sample["kspace"].shape
            sampling_mask = sample["mask"]

            # Mask needs to be padded.
            sampling_mask[: sample["padding_left"]] = 0
            sampling_mask[sample["padding_right"] :] = 0

            sampling_mask = sampling_mask.reshape(1, -1)
            del sample["mask"]

            sample["sampling_mask"] = self.__broadcast_mask(kspace_shape, sampling_mask)
            sample["acs_mask"] = self.__broadcast_mask(kspace_shape, self.__get_acs_from_fastmri_mask(sampling_mask))

        # Explicitly zero-out the outer parts of kspace which are padded
        sample["kspace"] = self.explicit_zero_padding(
            sample["kspace"], sample["padding_left"], sample["padding_right"]
        )

        if self.transform:
            sample = self.transform(sample)

        if self.noise_data:
            sample["loglikelihood_scaling"] = self.noise_data[sample["slice_no"]]

        return sample

    @staticmethod
    def explicit_zero_padding(kspace, padding_left, padding_right):
        if padding_left > 0:
            kspace[..., 0:padding_left] = 0 + 0 * 1j
        if padding_right > 0:
            kspace[..., padding_right:] = 0 + 0 * 1j

        return kspace

    @staticmethod
    def __get_acs_from_fastmri_mask(mask):
        left = right = mask.shape[-1] // 2
        while mask[..., right]:
            right += 1
        while mask[..., left]:
            left -= 1
        acs_mask = np.zeros_like(mask)
        acs_mask[:, left + 1 : right] = 1
        return acs_mask

    def __broadcast_mask(self, kspace_shape, mask):
        if self.ndim == 2:
            mask = np.broadcast_to(mask, [kspace_shape[1], mask.shape[-1]])
            mask = mask[np.newaxis, ..., np.newaxis]
        elif self.ndim == 3:
            mask = np.broadcast_to(mask, [kspace_shape[2], mask.shape[-1]])
            mask = mask[np.newaxis, np.newaxis, ..., np.newaxis]
        return mask


class CalgaryCampinasDataset(H5SliceData):
    """Calgary-Campinas challenge dataset."""

    def __init__(
        self,
        data_root: pathlib.Path,
        transform: Optional[Callable] = None,
        regex_filter: Optional[str] = None,
        filenames_filter: Optional[List[PathOrString]] = None,
        filenames_lists: Union[List[PathOrString], None] = None,
        filenames_lists_root: Union[PathOrString, None] = None,
        pass_mask: bool = False,
        crop_outer_slices: bool = False,
        pass_h5s: Optional[Dict] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            root=data_root,
            filenames_filter=filenames_filter,
            filenames_lists=filenames_lists,
            filenames_lists_root=filenames_lists_root,
            regex_filter=regex_filter,
            metadata=None,
            extra_keys=None,
            slice_data=slice(50, -50) if crop_outer_slices else None,
            text_description=kwargs.get("text_description", None),
            pass_h5s=pass_h5s,
            pass_dictionaries=kwargs.get("pass_dictionaries", None),
        )

        if self.sensitivity_maps is not None:
            raise NotImplementedError(
                f"Sensitivity maps are not supported in the current " f"{self.__class__.__name__} class."
            )

        # Sampling rate in the slice-encode direction
        self.sampling_rate_slice_encode: float = 0.85
        self.transform = transform
        self.pass_mask: bool = pass_mask

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = super().__getitem__(idx)
        kspace = sample["kspace"]

        # TODO: use broadcasting function.
        if self.pass_mask:
            # # In case the data is already masked, the sampling mask can be recovered by finding the zeros.
            # This needs to be done in the primary function!
            # sampling_mask = ~(np.abs(kspace).sum(axis=(0, -1)) == 0)
            sample["mask"] = (sample["mask"] * np.ones(kspace.shape).astype(np.int32))[..., np.newaxis]

        kspace = kspace[..., ::2] + 1j * kspace[..., 1::2]  # Convert real-valued to complex-valued data.
        num_z = kspace.shape[1]
        kspace[:, int(np.ceil(num_z * self.sampling_rate_slice_encode)) :, :] = 0.0 + 0.0 * 1j

        sample["padding_left"] = 0
        sample["padding_right"] = np.all(np.abs(kspace).sum(-1) == 0, axis=0).nonzero()[0][0]

        # Downstream code expects the coils to be at the first axis.
        sample["kspace"] = np.ascontiguousarray(kspace.transpose(2, 0, 1))

        if self.transform:
            sample = self.transform(sample)
        return sample


class ConcatDataset(Dataset):
    """Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets.
    From pytorch 1.5.1: :class:`torch.data.data.ConcatDataset`.

    Parameters
    ----------
    datasets: sequence
        List of datasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence):
        out_sequence, total = [], 0
        for item in sequence:
            length = len(item)
            out_sequence.append(length + total)
            total += length
        return out_sequence

    def __init__(self, datasets):
        super().__init__()
        if len(datasets) <= 0:
            raise AssertionError("datasets should not be an empty iterable")
        self.datasets = list(datasets)
        for dataset in self.datasets:
            if isinstance(dataset, IterableDataset):
                raise AssertionError("ConcatDataset does not support IterableDataset")
        self.cumulative_sizes = self.cumsum(self.datasets)

        self.logger = logging.getLogger(type(self).__name__)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        sample_idx = idx if dataset_idx == 0 else idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]


class ImageIntensityMode(str, Enum):
    proton = "PROTON"
    t1 = "T1"
    t2 = "T2"


class SheppLoganDataset(Dataset):
    """Shepp Logan Dataset for MRI as implemented in [1]_. Code was adapted from [2]_.

    References
    ----------
    .. [1] Gach, H. Michael, Costin Tanase, and Fernando Boada. "2D & 3D Shepp-Logan phantom standards for MRI."
        2008 19th International Conference on Systems Engineering. IEEE, 2008.
    .. [2] https://github.com/mckib2/phantominator/blob/master/phantominator/mr_shepp_logan.py

    Notes
    -----
    This dataset reconstructs into a single volume.
    """

    GYROMAGNETIC_RATIO: float = 267.52219
    DEFAULT_NUM_ELLIPSOIDS: int = 15
    ELLIPSOID_NUM_PARAMS: int = 13
    IMAGE_INTENSITIES: List[str] = ["PROTON", "T1", "T2"]

    def __init__(
        self,
        shape: Union[int, Union[List[int], Tuple[int, int, int]]],
        num_coils: int,
        intensity: ImageIntensityMode,
        seed: Optional[Union[int, List[int]]] = None,
        ellipsoids: np.ndarray = None,
        B0: float = 3.0,
        T2_star: Optional[bool] = None,
        zlimits: Tuple[float, float] = (-1, 1),
        transform: Optional[Callable] = None,
        text_description: Optional[str] = None,
    ) -> None:
        r"""Inits :class:`SheppLoganDataset`.

        Parameters
        ----------
        shape: Union[int, Union[List[int], Tuple[int, int, int]]]
            Shape of Shepp Logan phantom (3-dimensional).
        num_coils: int
            Number of simulated coils.
        intensity: ImageIntensityMode
            Can be `PROTON` to return the proton density dataset, `T1` or `T2`.
        seed: Optional[Union[int, List[int]]]
            Seed to be used for coil sensitivity maps. Default: None.
        ellipsoids: np.ndarray
            Ellipsoids parameters. If None, it will used the default parameters as per the paper. Default: None.
        B0: float
            Magnetic field. Default: 3.0.
        T2_star: Optional[bool]
            If True, a T2^{*} dataset will be output. Only valid for intensity = `T2`. Default: None.
        zlimits: Tuple[float, float]
            Limits of z-axis. Default: (-1, 1).
        transform: Optional[Callable]
            A list of transforms to be applied on the generated samples. Default is None.
        text_description: Optional[str]
            Description of dataset, can be useful for logging. Default: None.
        """
        self.logger = logging.getLogger(type(self).__name__)

        (self.nx, self.ny, self.nz) = (shape, shape, shape) if isinstance(shape, int) else tuple(shape)
        self.num_coils = num_coils

        assert (
            intensity in self.IMAGE_INTENSITIES
        ), f"Intensity should be in {self.IMAGE_INTENSITIES}. Received {intensity}."
        self.intensity = intensity

        assert len(zlimits) == 2, "`zlimits` must be a tuple with 2 entries: upper and lower " "bounds!"
        assert zlimits[0] <= zlimits[1], "`zlimits`: lower bound must be first entry!"
        self.zlimits = zlimits

        self.shape = shape
        self.B0 = B0
        self.T2_star = T2_star

        self._set_params(ellipsoids)
        self.transform = transform
        self.rng = np.random.RandomState()

        with temp_seed(self.rng, seed):
            self.seed = list(self.rng.choice(a=range(int(1e5)), size=self.nz, replace=False))
        self.text_description = text_description
        if self.text_description:
            self.logger.info("Dataset description: %s.", self.text_description)

        self.name = "shepp_loggan" + "_" + self.intensity
        self.ndim = 2
        self.volume_indices = {}
        self.volume_indices[pathlib.Path(self.name)] = range(self.__len__())

    def _set_params(self, ellipsoids=None) -> None:
        # Get parameters from paper if None provided
        if ellipsoids is None:
            ellipsoids = self.default_mr_ellipsoid_parameters()

        # Extract some parameters so we can use them
        self.center_xs = ellipsoids[:, 0]
        self.center_ys = ellipsoids[:, 1]
        self.center_zs = ellipsoids[:, 2]
        self.half_ax_as = ellipsoids[:, 3]
        self.half_ax_bs = ellipsoids[:, 4]
        self.half_ax_cs = ellipsoids[:, 5]
        self.theta = ellipsoids[:, 6]
        self.M0 = ellipsoids[:, 7]
        self.As = ellipsoids[:, 8]
        self.Cs = ellipsoids[:, 9]
        self.T1 = ellipsoids[:, 10]
        self.T2 = ellipsoids[:, 11]
        self.chis = ellipsoids[:, 12]

        self.ellipsoids = ellipsoids

    def sample_image(self, idx: int) -> np.ndarray:  # pylint: disable=too-many-locals
        # meshgrid does X, Y backwards
        X, Y, Z = np.meshgrid(
            np.linspace(-1, 1, self.ny),
            np.linspace(-1, 1, self.nx),
            np.linspace(self.zlimits[0], self.zlimits[1], self.nz)[idx % self.nz],
        )

        ct = np.cos(self.theta)
        st = np.sin(self.theta)
        sgn = np.sign(self.M0)

        image = np.zeros((self.nx, self.ny, 1))

        # Put ellipses where they need to be
        for j in range(self.ellipsoids.shape[0]):
            center_x, center_y, center_z = self.center_xs[j], self.center_ys[j], self.center_zs[j]
            a, b, c = self.half_ax_as[j], self.half_ax_bs[j], self.half_ax_cs[j]
            ct0, st0 = ct[j], st[j]

            # Find indices falling inside the ellipsoid, ellipses only
            # rotated in xy plane
            indices = ((X - center_x) * ct0 + (Y - center_y) * st0) ** 2 / a**2 + (
                (X - center_x) * st0 - (Y - center_y) * ct0
            ) ** 2 / b**2 + (Z - center_z) ** 2 / c**2 <= 1
            # T1 | Use T1 model if not given explicit T1 value
            if self.intensity == "T1":
                if np.isnan(self.T1[j]):
                    image[indices] += sgn[j] * self.As[j] * (self.B0 ** self.Cs[j])
                else:
                    image[indices] += sgn[j] * self.T1[j]
            # T2
            elif self.intensity == "T2":
                if self.T2_star:
                    image[indices] += sgn[j] / (
                        1 / self.T2[j] + self.GYROMAGNETIC_RATIO * np.abs(self.B0 * self.chis[j])
                    )
                else:
                    image[indices] += sgn[j] * self.T2[j]
            # M0 | Add ellipses together -- subtract of M0 is negative
            else:
                image[indices] += self.M0[j]
        return (image + 0.0j).squeeze()

    def __len__(self) -> int:
        return self.nz

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image = self.sample_image(idx)
        sensitivity_map = simulate_sensitivity_maps((self.nx, self.ny), self.num_coils, seed=self.seed[idx])

        image = image[None] * sensitivity_map

        # Outer slices might be zeros. These will cause nans/infs. Add random normal noise.
        if np.allclose(image, np.zeros(1)):
            image += np.random.randn(*image.shape) * sys.float_info.epsilon

        kspace = self.fft(image)

        sample = {"kspace": kspace, "filename": self.name, "slice_no": idx}

        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    @staticmethod
    def default_mr_ellipsoid_parameters() -> np.ndarray:
        """Returns default parameters of ellipsoids as in [1]_.

        Returns
        -------
        ellipsoids : np.ndarray
            Array containing the parameters for the ellipsoids used to construct the phantom.
            Each row of the form [x, y, z, a, b, c, \theta, m_0, A, C, T1, T2, \chi] represents an ellipsoid, where:
            * (x, y, z): denotes the center of the ellipsoid
            * (a, b, c): denote the lengths of the semi-major axis aligned with the x, y, z-axis, respectively
            * \theta: denotes the rotation angle of the ellipsoid in rads
            * m_0: denotes the spin density
            * (A, C): denote the T1 parameters
            * T1: denotes the T1 value if explicit, otherwise T1 = A \times B_0^{C}
            * T2: denotes the T2 value
            * \chi: denotes the \chi value

        References
        ----------
        .. [1] Gach, H. Michael, Costin Tanase, and Fernando Boada. "2D & 3D Shepp-Logan phantom standards for MRI."
            2008 19th International Conference on Systems Engineering. IEEE, 2008.
        """
        params = _mr_relaxation_parameters()

        ellipsoids = np.zeros((SheppLoganDataset.DEFAULT_NUM_ELLIPSOIDS, SheppLoganDataset.ELLIPSOID_NUM_PARAMS))

        ellipsoids[0, :] = [0, 0, 0, 0.72, 0.95, 0.93, 0, 0.8, *params["scalp"]]
        ellipsoids[1, :] = [0, 0, 0, 0.69, 0.92, 0.9, 0, 0.12, *params["marrow"]]
        ellipsoids[2, :] = [0, -0.0184, 0, 0.6624, 0.874, 0.88, 0, 0.98, *params["csf"]]
        ellipsoids[3, :] = [0, -0.0184, 0, 0.6524, 0.864, 0.87, 0, 0.745, *params["gray-matter"]]
        ellipsoids[4, :] = [-0.22, 0, -0.25, 0.41, 0.16, 0.21, np.deg2rad(-72), 0.98, *params["csf"]]
        ellipsoids[5, :] = [0.22, 0, -0.25, 0.31, 0.11, 0.22, np.deg2rad(72), 0.98, *params["csf"]]
        ellipsoids[6, :] = [0, 0.35, -0.25, 0.21, 0.25, 0.35, 0, 0.617, *params["white-matter"]]
        ellipsoids[7, :] = [0, 0.1, -0.25, 0.046, 0.046, 0.046, 0, 0.95, *params["tumor"]]
        ellipsoids[8, :] = [-0.08, -0.605, -0.25, 0.046, 0.023, 0.02, 0, 0.95, *params["tumor"]]
        ellipsoids[9, :] = [0.06, -0.605, -0.25, 0.046, 0.023, 0.02, np.deg2rad(-90), 0.95, *params["tumor"]]
        ellipsoids[10, :] = [0, -0.1, -0.25, 0.046, 0.046, 0.046, 0, 0.95, *params["tumor"]]
        ellipsoids[11, :] = [0, -0.605, -0.25, 0.023, 0.023, 0.023, 0, 0.95, *params["tumor"]]
        ellipsoids[12, :] = [0.06, -0.105, 0.0625, 0.056, 0.04, 0.1, np.deg2rad(-90), 0.93, *params["tumor"]]
        ellipsoids[13, :] = [0, 0.1, 0.625, 0.056, 0.056, 0.1, 0, 0.98, *params["csf"]]
        ellipsoids[14, :] = [0.56, -0.4, -0.25, 0.2, 0.03, 0.1, np.deg2rad(70), 0.85, *params["blood-clot"]]

        # Need to subtract some ellipses here...
        ellipsoids_neg = np.zeros(ellipsoids.shape)
        for ii in range(ellipsoids.shape[0]):
            # Ellipsoid geometry
            ellipsoids_neg[ii, :7] = ellipsoids[ii, :7]

            # Tissue property differs after 4th subtracted ellipsoid
            if ii > 3:
                ellipsoids_neg[ii, 7:] = ellipsoids[3, 7:]
            else:
                ellipsoids_neg[ii, 7:] = ellipsoids[ii - 1, 7:]

        # Throw out first as we skip this one in the paper's table
        ellipsoids_neg = ellipsoids_neg[1:, :]

        # Spin density is negative for subtraction
        ellipsoids_neg[:, 7] *= -1

        # Paper doesn't use last blood-clot ellipsoid
        ellipsoids = ellipsoids[:-1, :]
        ellipsoids_neg = ellipsoids_neg[:-1, :]

        # Put both ellipsoid groups together
        return np.concatenate((ellipsoids, ellipsoids_neg), axis=0)

    @staticmethod
    def fft(x):
        return np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(x), axes=(1, 2), norm="ortho"))


def _mr_relaxation_parameters():
    r"""Returns MR relaxation parameters for certain tissues as defined in [1]_.

    Returns
    -------
    params : dict
        Tissue properties of scalp, marrow, csf, white/gray matter, tumor and blood clot.
        More specifically, these properties are [A, C, T1, T2, \chi], where:
            * (A, C): denote the T1 parameters
            * T1: denotes the T1 value if explicit, otherwise T1 = A \times B_0^{C}
            * T2: denotes the T2 value
            * \chi: denotes the \chi value

    Notes
    -----
    If T1 is np.nan, T1 = A \times B_0^{C} will be used.

    References
    ----------
    .. [1] Gach, H. Michael, Costin Tanase, and Fernando Boada. "2D & 3D Shepp-Logan phantom standards for MRI."
        2008 19th International Conference on Systems Engineering. IEEE, 2008.
    """

    # params['tissue-name'] = [A, C, (t1 value if explicit), t2, chi]
    params = {}
    params["scalp"] = [0.324, 0.137, np.nan, 0.07, -7.5e-6]
    params["marrow"] = [0.533, 0.088, np.nan, 0.05, -8.85e-6]
    params["csf"] = [np.nan, np.nan, 4.2, 1.99, -9e-6]
    params["blood-clot"] = [1.35, 0.34, np.nan, 0.2, -9e-6]
    params["gray-matter"] = [0.857, 0.376, np.nan, 0.1, -9e-6]
    params["white-matter"] = [0.583, 0.382, np.nan, 0.08, -9e-6]
    params["tumor"] = [0.926, 0.217, np.nan, 0.1, -9e-6]
    return params


class SheppLoganProtonDataset(SheppLoganDataset):
    """Creates an instance of :class:`SheppLoganDataset` with `PROTON` intensity."""

    def __init__(
        self,
        shape: Union[int, Union[List[int], Tuple[int, int, int]]],
        num_coils: int,
        seed: Optional[Union[int, List[int]]] = None,
        ellipsoids: np.ndarray = None,
        B0: float = 3.0,
        zlimits: Tuple[float, float] = (-0.929, 0.929),
        transform: Optional[Callable] = None,
        text_description: Optional[str] = None,
    ) -> None:
        r"""Inits :class:`SheppLoganProtonDataset`.

        Parameters
        ----------
        shape: Union[int, Union[List[int], Tuple[int, int, int]]]
            Shape of Shepp Logan phantom (3-dimensional).
        num_coils: int
            Number of simulated coils.
        seed: Optional[Union[int, List[int]]]
            Seed to be used for coil sensitivity maps. Default: None.
        ellipsoids: np.ndarray
            Ellipsoids parameters. If None, it will used the default parameters as per the paper. Default: None.
        B0: float
            Magnetic field. Default: 3.0.
        zlimits: Tuple[float, float]
            Limits of z-axis. Default: (-0.929, 0.929).
        transform: Optional[Callable]
            A list of transforms to be applied on the generated samples. Default is None.
        text_description: Optional[str]
            Description of dataset, can be useful for logging. Default: None.
        """
        super().__init__(
            shape=shape,
            num_coils=num_coils,
            intensity=ImageIntensityMode.proton,
            seed=seed,
            ellipsoids=ellipsoids,
            B0=B0,
            zlimits=zlimits,
            transform=transform,
            text_description=text_description,
        )


class SheppLoganT1Dataset(SheppLoganDataset):
    """Creates an instance of :class:`SheppLoganDataset` with `T1` intensity."""

    def __init__(
        self,
        shape: Union[int, Union[List[int], Tuple[int, int, int]]],
        num_coils: int,
        seed: Optional[Union[int, List[int]]] = None,
        ellipsoids: np.ndarray = None,
        B0: float = 3.0,
        zlimits: Tuple[float, float] = (-0.929, 0.929),
        transform: Optional[Callable] = None,
        text_description: Optional[str] = None,
    ) -> None:
        r"""Inits :class:`SheppLoganT1Dataset`.

        Parameters
        ----------
        shape: Union[int, Union[List[int], Tuple[int, int, int]]]
            Shape of Shepp Logan phantom (3-dimensional).
        num_coils: int
            Number of simulated coils.
        seed: Optional[Union[int, List[int]]]
            Seed to be used for coil sensitivity maps. Default: None.
        ellipsoids: np.ndarray
            Ellipsoids parameters. If None, it will used the default parameters as per the paper. Default: None.
        B0: float
            Magnetic field. Default: 3.0.
        zlimits: Tuple[float, float]
            Limits of z-axis. Default: (-0.929, 0.929).
        transform: Optional[Callable]
            A list of transforms to be applied on the generated samples. Default is None.
        text_description: Optional[str]
            Description of dataset, can be useful for logging. Default: None.
        """
        super().__init__(
            shape=shape,
            num_coils=num_coils,
            intensity=ImageIntensityMode.t1,
            seed=seed,
            ellipsoids=ellipsoids,
            B0=B0,
            zlimits=zlimits,
            transform=transform,
            text_description=text_description,
        )


class SheppLoganT2Dataset(SheppLoganDataset):
    """Creates an instance of :class:`SheppLoganDataset` with `T2` intensity."""

    def __init__(
        self,
        shape: Union[int, Union[List[int], Tuple[int, int, int]]],
        num_coils: int,
        seed: Optional[Union[int, List[int]]] = None,
        ellipsoids: np.ndarray = None,
        B0: float = 3.0,
        T2_star: Optional[bool] = None,
        zlimits: Tuple[float, float] = (-0.929, 0.929),
        transform: Optional[Callable] = None,
        text_description: Optional[str] = None,
    ) -> None:
        r"""Inits :class:`SheppLoganT2Dataset`.

        Parameters
        ----------
        shape: Union[int, Union[List[int], Tuple[int, int, int]]]
            Shape of Shepp Logan phantom (3-dimensional).
        num_coils: int
            Number of simulated coils.
        seed: Optional[Union[int, List[int]]]
            Seed to be used for coil sensitivity maps. Default: None.
        ellipsoids: np.ndarray
            Ellipsoids parameters. If None, it will used the default parameters as per the paper. Default: None.
        B0: float
            Magnetic field. Default: 3.0.
        T2_star: Optional[bool]
            If True, a T2^{*} dataset will be output. Only valid for intensity = `T2`. Default: None.
        zlimits: Tuple[float, float]
            Limits of z-axis. Default: (-0.929, 0.929).
        transform: Optional[Callable]
            A list of transforms to be applied on the generated samples. Default is None.
        text_description: Optional[str]
            Description of dataset, can be useful for logging. Default: None.
        """
        super().__init__(
            shape=shape,
            num_coils=num_coils,
            intensity=ImageIntensityMode.t2,
            seed=seed,
            ellipsoids=ellipsoids,
            B0=B0,
            T2_star=T2_star,
            zlimits=zlimits,
            transform=transform,
            text_description=text_description,
        )


def build_dataset(
    name: str,
    transforms: Optional[Callable] = None,
    **kwargs: Dict[str, Any],
) -> Dataset:
    """Builds dataset with name :class:`name + "Dataset"` from keyword arguments.

    Only `name` and `transforms` arguments are common for all Datasets.
    ALL other keyword arguments should be passed in **kwargs.

    Parameters
    ----------
    name: str
        Name of dataset class (without `Dataset`) in src.models.components.direct.data.datasets.
    transforms: Callable
        Transformation object. Default: None.
    kwargs: Dict[str, Any]
        Keyword arguments. Can include:
            * data_root: pathlib.Path or str
                Root path to the data for the dataset class (:class:`FastMRIDataset` and :class:`CalgaryCampinasDataset`).
            * filenames_filter: List
                List of filenames to include in the dataset, should be the same as the ones that can be derived from a glob
                on the root. If set, will skip searching for files in the root.
            * sensitivity_maps: pathlib.Path
                Path to sensitivity maps.
            * text_description: str
                Description of dataset, can be used for logging.
            * kspace_context: int
                If set, output will be of shape -kspace_context:kspace_context.

    Returns
    -------
    Dataset
    """
    logger.info("Building dataset for: %s", name)
    dataset_class: Callable = str_to_class("src.models.components.direct.data.datasets", name + "Dataset")
    logger.debug("Dataset class: %s", dataset_class)
    dataset = dataset_class(transform=transforms, **kwargs)

    logger.debug("Dataset: %s", str(dataset))

    return dataset


def build_dataset_from_input(
    transforms: Callable,
    dataset_config: DictConfig,
    **kwargs: Dict[str, Any],
) -> Dataset:
    """Builds dataset from input keyword arguments and configuration file.

    Only `transforms` is common for all Datasets. ALL other keyword arguments should be passed in `**kwargs`.

    Parameters
    ----------
    transforms: object, Callable
        Transformation object.
    dataset_config: DictConfig
        Dataset configuration file.
    kwargs: Dict[str, Any]
        Can include:
            * initial_images: List[pathlib.Path]
                Path to initial_images.
            * initial_kspaces: pathlib.Path
                Path to initial kspace images.
            * filenames_filter: Optional[List[PathOrString]]
                List of filenames to include in the dataset, should be the same as the ones that can be derived from a glob
                on the root. If set, will skip searching for files in the root.
            * data_root: pathlib.Path or str
                Root path to the data for the dataset class.
            * pass_dictionaries: Optional[Dict[str, Dict]]

    Returns
    -------
    Dataset
    """
    # Some datasets require `pass_h5s` argument.
    pass_h5s = None
    if "initial_images" in kwargs and "initial_kspaces" in kwargs:
        raise ValueError(
            f"initial_images and initial_kspaces are mutually exclusive. "
            f"Got {kwargs.get('initial_images')} and {kwargs.get('initial_kspaces')}."
        )
    if "initial_images" in kwargs:
        pass_h5s = {"initial_image": (dataset_config.input_image_key, kwargs.get("initial_images"))}
        del kwargs["initial_images"]
    elif "initial_kspaces" in kwargs:
        pass_h5s = {"initial_kspace": (dataset_config.input_kspace_key, kwargs.get("initial_kspaces"))}
        del kwargs["initial_kspaces"]
    if pass_h5s is not None:
        kwargs.update({"pass_h5s": pass_h5s})

    # This will remove double arguments passed both in kwargs and in the dataset configuration, keeping only in that
    # case the arguments in kwargs.
    # For example, `data_root` can be passed both from the command line and in the configuration file.
    config_kwargs = remove_keys(
        dict(dataset_config), ["name", "transforms"] + list(kwargs.keys() & dict(dataset_config).keys())
    )
    dataset = build_dataset(
        name=dataset_config.name,  # type: ignore
        transforms=transforms,
        **kwargs,
        **config_kwargs,
    )
    return dataset
