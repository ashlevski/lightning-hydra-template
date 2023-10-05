# coding=utf-8
# Copyright (c) DIRECT Contributors

# Code and comments can be shared with code of FastMRI under the same MIT license:
# https://github.com/facebookresearch/fastMRI/
# The code can have been adjusted to our needs.

from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.fft
from numpy.typing import ArrayLike

from direct.data.bbox import crop_to_bbox
from direct.utils import ensure_list, is_complex_data, is_power_of_two
from direct.utils.asserts import assert_complex, assert_same_shape


def to_tensor(data: np.ndarray) -> torch.Tensor:
    """Convert numpy array to PyTorch tensor. Complex arrays will have real and imaginary parts on the last axis.

    Parameters
    ----------
    data: np.ndarray

    Returns
    -------
    torch.Tensor
    """
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)

    return torch.from_numpy(data)


def verify_fft_dtype_possible(data: torch.Tensor, dims: Tuple[int, ...]) -> bool:
    """fft and ifft can only be performed on GPU in float16 if the shapes are powers of 2. This function verifies if
    this is the case.

    Parameters
    ----------
    data: torch.Tensor
    dims: tuple

    Returns
    -------
    bool
    """
    is_complex64 = data.dtype == torch.complex64
    is_complex32_and_power_of_two = (data.dtype == torch.float32) and all(
        is_power_of_two(_) for _ in [data.size(idx) for idx in dims]
    )

    return is_complex64 or is_complex32_and_power_of_two


def view_as_complex(data):
    """Returns a view of input as a complex tensor.

    For an input tensor of size (N, ..., 2) where the last dimension of size 2 represents the real and imaginary
    components of complex numbers, this function returns a new complex tensor of size (N, ...).

    Parameters
    ----------
    data: torch.Tensor
        Input data with torch.dtype torch.float64 and torch.float32 with complex axis (last) of dimension 2
        and of shape (N, \*, 2).

    Returns
    -------
    complex_valued_data: torch.Tensor
        Output complex-valued data of shape (N, \*) with complex torch.dtype.
    """
    return torch.view_as_complex(data)


def view_as_real(data):
    """Returns a view of data as a real tensor.

    For an input complex tensor of size (N, ...) this function returns a new real tensor of size (N, ..., 2) where the
    last dimension of size 2 represents the real and imaginary components of complex numbers.

    Parameters
    ----------
    data: torch.Tensor
        Input data with complex torch.dtype of shape (N, \*).

    Returns
    -------
    real_valued_data: torch.Tensor
        Output real-valued data of shape (N, \*, 2).
    """

    return torch.view_as_real(data)


def fft2(
    data: torch.Tensor,
    dim: Tuple[int, ...] = (1, 2),
    centered: bool = True,
    normalized: bool = True,
    complex_input: bool = True,
) -> torch.Tensor:
    """Apply centered two-dimensional Inverse Fast Fourier Transform. Can be performed in half precision when input
    shapes are powers of two.

    Version for PyTorch >= 1.7.0.

    Parameters
    ----------
    data: torch.Tensor
        Complex-valued input tensor. Should be of shape (\*, 2) and dim is in \*.
    dim: tuple, list or int
        Dimensions over which to compute. Should be positive. Negative indexing not supported
        Default is (1, 2), corresponding to ('height', 'width').
    centered: bool
        Whether to apply a centered fft (center of kspace is in the center versus in the corners).
        For FastMRI dataset this has to be true and for the Calgary-Campinas dataset false.
    normalized: bool
        Whether to normalize the fft. For the FastMRI this has to be true and for the Calgary-Campinas dataset false.
    complex_input:bool
        True if input is complex [real-valued] tensor (complex dim = 2). False if complex-valued tensor is inputted.

    Returns
    -------
    output_data: torch.Tensor
        The Fast Fourier transform of the data.
    """
    if not all((_ >= 0 and isinstance(_, int)) for _ in dim):
        raise TypeError(
            f"Currently fft2 does not support negative indexing. "
            f"Dim should contain only positive integers. Got {dim}."
        )
    if complex_input:
        assert_complex(data, complex_last=True)
        data = view_as_complex(data)

    if centered:
        data = ifftshift(data, dim=dim)
    # Verify whether half precision and if fft is possible in this shape. Else do a typecast.
    if verify_fft_dtype_possible(data, dim):
        data = torch.fft.fftn(
            data,
            dim=dim,
            norm="ortho" if normalized else None,
        )
    else:
        raise ValueError("Currently half precision FFT is not supported.")

    if centered:
        data = fftshift(data, dim=dim)

    if complex_input:
        data = view_as_real(data)
    return data


def ifft2(
    data: torch.Tensor,
    dim: Tuple[int, ...] = (1, 2),
    centered: bool = True,
    normalized: bool = True,
    complex_input: bool = True,
) -> torch.Tensor:
    """Apply centered two-dimensional Inverse Fast Fourier Transform. Can be performed in half precision when input
    shapes are powers of two.

    Version for PyTorch >= 1.7.0.

    Parameters
    ----------
    data: torch.Tensor
        Complex-valued input tensor. Should be of shape (\*, 2) and dim is in \*.
    dim: tuple, list or int
        Dimensions over which to compute. Should be positive. Negative indexing not supported
        Default is (1, 2), corresponding to ( 'height', 'width').
    centered: bool
        Whether to apply a centered ifft (center of kspace is in the center versus in the corners).
        For FastMRI dataset this has to be true and for the Calgary-Campinas dataset false.
    normalized: bool
        Whether to normalize the ifft. For the FastMRI this has to be true and for the Calgary-Campinas dataset false.
    complex_input:bool
        True if input is complex [real-valued] tensor (complex dim = 2). False if complex-valued tensor is inputted.

    Returns
    -------
    output_data: torch.Tensor
        The Inverse Fast Fourier transform of the data.
    """
    if not all((_ >= 0 and isinstance(_, int)) for _ in dim):
        raise TypeError(
            f"Currently ifft2 does not support negative indexing. "
            f"Dim should contain only positive integers. Got {dim}."
        )

    if complex_input:
        assert_complex(data, complex_last=True)
        data = view_as_complex(data)
    if centered:
        data = ifftshift(data, dim=dim)
    # Verify whether half precision and if fft is possible in this shape. Else do a typecast.
    if verify_fft_dtype_possible(data, dim):
        data = torch.fft.ifftn(
            data,
            dim=dim,
            norm="ortho" if normalized else None,
        )
    else:
        raise ValueError("Currently half precision FFT is not supported.")

    if centered:
        data = fftshift(data, dim=dim)
    if complex_input:
        data = view_as_real(data)
    return data


def safe_divide(input_tensor: torch.Tensor, other_tensor: torch.Tensor) -> torch.Tensor:
    """Divide input_tensor and other_tensor safely, set the output to zero where the divisor b is zero.

    Parameters
    ----------
    input_tensor: torch.Tensor
    other_tensor: torch.Tensor

    Returns
    -------
    torch.Tensor: the division.
    """

    data = torch.where(
        other_tensor == 0,
        torch.tensor([0.0], dtype=input_tensor.dtype).to(input_tensor.device),
        input_tensor / other_tensor,
    )
    return data


def modulus(data: torch.Tensor, complex_axis: int = -1) -> torch.Tensor:
    """Compute modulus of complex input data. Assumes there is a complex axis (of dimension 2) in the data.

    Parameters
    ----------
    data: torch.Tensor
    complex_axis: int
        Complex dimension along which the modulus will be calculated. Default: -1.

    Returns
    -------
    output_data: torch.Tensor
        Modulus of data.
    """
    assert_complex(data, complex_axis=complex_axis)

    return (data**2).sum(complex_axis).sqrt()  # noqa


def modulus_if_complex(data: torch.Tensor, complex_axis=-1) -> torch.Tensor:
    """Compute modulus if complex tensor (has complex axis).

    Parameters
    ----------
    data: torch.Tensor
    complex_axis: int
        Complex dimension along which the modulus will be calculated if that dimension is complex. Default: -1.

    Returns
    -------
    torch.Tensor
    """
    if is_complex_data(data, complex_axis=complex_axis):
        return modulus(data=data, complex_axis=complex_axis)
    return data


def roll_one_dim(data: torch.Tensor, shift: int, dim: int) -> torch.Tensor:
    """Similar to roll but only for one dim

    Parameters
    ----------
    data: torch.Tensor
    shift: tuple, int
    dim: int

    Returns
    -------
    torch.Tensor
    """
    shift = shift % data.size(dim)
    if shift == 0:
        return data

    left = data.narrow(dim, 0, data.size(dim) - shift)
    right = data.narrow(dim, data.size(dim) - shift, shift)

    return torch.cat((right, left), dim=dim)


def roll(
    data: torch.Tensor,
    shift: List[int],
    dim: Union[List[int], Tuple[int, ...]],
) -> torch.Tensor:
    """Similar to numpy roll but applies to pytorch tensors.

    Parameters
    ----------
    data: torch.Tensor
    shift: tuple, int
    dim: List or tuple of ints

    Returns
    -------
    torch.Tensor
        Rolled version of data
    """
    if len(shift) != len(dim):
        raise ValueError("len(shift) must match len(dim)")

    for s, d in zip(shift, dim):
        data = roll_one_dim(data, s, d)

    return data


def fftshift(data: torch.Tensor, dim: Union[List[int], Tuple[int, ...], None] = None) -> torch.Tensor:
    """Similar to numpy fftshift but applies to pytorch tensors.

    Parameters
    ----------
    data: torch.Tensor
        Input data.
    dim: List or tuple of ints or None
        Default: None.

    Returns
    -------
    torch.Tensor
    """
    if dim is None:
        # this weird code is necessary for torch.jit.script typing
        dim = [0] * (data.dim())
        for idx in range(1, data.dim()):
            dim[idx] = idx

    # also necessary for torch.jit.script
    shift = [0] * len(dim)
    for idx, dim_num in enumerate(dim):
        shift[idx] = data.shape[dim_num] // 2

    return roll(data, shift, dim)


def ifftshift(data: torch.Tensor, dim: Union[List[int], Tuple[int, ...], None] = None) -> torch.Tensor:
    """Similar to numpy ifftshift but applies to pytorch tensors.

    Parameters
    ----------
    data: torch.Tensor
        Input data.
    dim: List or tuple of ints or None
        Default: None.

    Returns
    -------
    torch.Tensor
    """
    if dim is None:
        # this weird code is necessary for torch.jit.script typing
        dim = [0] * (data.dim())
        for i in range(1, data.dim()):
            dim[i] = i

    # also necessary for torch.jit.script
    shift = [0] * len(dim)
    for i, dim_num in enumerate(dim):
        shift[i] = (data.shape[dim_num] + 1) // 2

    return roll(data, shift, dim)


def complex_multiplication(input_tensor: torch.Tensor, other_tensor: torch.Tensor) -> torch.Tensor:
    """Multiplies two complex-valued tensors. Assumes input tensors are complex (last axis has dimension 2).

    Parameters
    ----------
    input_tensor: torch.Tensor
        Input data
    other_tensor: torch.Tensor
        Input data

    Returns
    -------
    torch.Tensor
    """
    assert_complex(input_tensor, complex_last=True)
    assert_complex(other_tensor, complex_last=True)

    complex_index = -1

    real_part = input_tensor[..., 0] * other_tensor[..., 0] - input_tensor[..., 1] * other_tensor[..., 1]
    imaginary_part = input_tensor[..., 0] * other_tensor[..., 1] + input_tensor[..., 1] * other_tensor[..., 0]

    multiplication = torch.cat(
        [
            real_part.unsqueeze(dim=complex_index),
            imaginary_part.unsqueeze(dim=complex_index),
        ],
        dim=complex_index,
    )
    return multiplication


def complex_dot_product(a: torch.Tensor, b: torch.Tensor, dim: List[int]) -> torch.Tensor:
    r"""Computes the dot product of the complex tensors :math:`a` and :math:`b`: :math:`a^{*}b = <a, b>`.

    Parameters
    ----------
    a : torch.Tensor
        Input :math:`a`.
    b : torch.Tensor
        Input :math:`b`.
    dim : List[int]
        Dimensions which will be suppressed. Useful when inputs are batched.

    Returns
    -------
    complex_dot_product : torch.Tensor
        Dot product of :math:`a` and :math:`b`.
    """
    return complex_multiplication(conjugate(a), b).sum(dim)


def complex_division(input_tensor: torch.Tensor, other_tensor: torch.Tensor) -> torch.Tensor:
    """Divides two complex-valued tensors. Assumes input tensors are complex (last axis has dimension 2).

    Parameters
    ----------
    input_tensor: torch.Tensor
        Input data
    other_tensor: torch.Tensor
        Input data

    Returns
    -------
    torch.Tensor
    """
    assert_complex(input_tensor, complex_last=True)
    assert_complex(other_tensor, complex_last=True)

    complex_index = -1

    denominator = other_tensor[..., 0] ** 2 + other_tensor[..., 1] ** 2
    real_part = safe_divide(
        input_tensor[..., 0] * other_tensor[..., 0] + input_tensor[..., 1] * other_tensor[..., 1], denominator
    )
    imaginary_part = safe_divide(
        input_tensor[..., 1] * other_tensor[..., 0] - input_tensor[..., 0] * other_tensor[..., 1], denominator
    )

    division = torch.cat(
        [
            real_part.unsqueeze(dim=complex_index),
            imaginary_part.unsqueeze(dim=complex_index),
        ],
        dim=complex_index,
    )
    return division


def _complex_matrix_multiplication(
    input_tensor: torch.Tensor, other_tensor: torch.Tensor, mult_func: Callable
) -> torch.Tensor:
    """Perform a matrix multiplication, helper function for complex_bmm and complex_mm.

    Parameters
    ----------
    input_tensor: torch.Tensor
    other_tensor: torch.Tensor
    mult_func: Callable
        Multiplication function e.g. torch.bmm or torch.mm

    Returns
    -------
    torch.Tensor
    """
    if not input_tensor.is_complex() or not other_tensor.is_complex():
        raise ValueError("Both input_tensor and other_tensor have to be complex-valued torch tensors.")

    output = (
        mult_func(input_tensor.real, other_tensor.real)
        - mult_func(input_tensor.imag, other_tensor.imag)
        + 1j * mult_func(input_tensor.real, other_tensor.imag)
        + 1j * mult_func(input_tensor.imag, other_tensor.real)
    )
    return output


def complex_mm(input_tensor: torch.Tensor, other_tensor: torch.Tensor) -> torch.Tensor:
    """Performs a matrix multiplication of the 2D complex matrices input_tensor and other_tensor. If input_tensor is a
    (n×m) tensor, other_tensor is a (m×p) tensor, out will be a (n×p) tensor.

    Parameters
    ----------
    input_tensor: torch.Tensor
        Input 2D tensor.
    other_tensor: torch.Tensor
        Other 2D tensor.

    Returns
    -------
    out: torch.Tensor
        Complex-multiplied 2D output tensor.
    """
    return _complex_matrix_multiplication(input_tensor, other_tensor, torch.mm)


def complex_bmm(input_tensor: torch.Tensor, other_tensor: torch.Tensor) -> torch.Tensor:
    """Complex batch multiplication.

    Parameters
    ----------
    input_tensor: torch.Tensor
        Input tensor.
    other_tensor: torch.Tensor
        Other tensor.

    Returns
    -------
    out: torch.Tensor
        Batch complex-multiplied output tensor.
    """
    return _complex_matrix_multiplication(input_tensor, other_tensor, torch.bmm)


def conjugate(data: torch.Tensor) -> torch.Tensor:
    """Compute the complex conjugate of a torch tensor where the last axis denotes the real and complex part (last axis
    has dimension 2).

    Parameters
    ----------
    data: torch.Tensor

    Returns
    -------
    conjugate_tensor: torch.Tensor
    """
    assert_complex(data, complex_last=True)
    data = data.clone()  # Clone is required as the data in the next line is changed in-place.
    data[..., 1] = data[..., 1] * -1.0

    return data


def apply_padding(
    data: torch.Tensor,
    padding: Union[None, torch.Tensor],
) -> torch.Tensor:
    """Applies zero padding to `data`.

    Parameters
    ----------
    data : torch.Tensor
        Batched or not input to be padded of shape (`batch`, \*, `height`, `width`, \*).
    padding : torch.Tensor or None
        Binary tensor of shape (`batch`, 1, `height`, `width`, 1). Entries in `padding` with non-zero value
        point to samples in `data` that will be zero-padded. If None, `data` will be returned.

    Returns
    -------
    data : torch.Tensor
        Padded data.
    """
    if padding is None:
        return data
    return torch.where(padding == 1, torch.tensor([0.0], dtype=data.dtype, device=data.device), data)


def apply_mask(
    kspace: torch.Tensor,
    mask_func: Union[Callable, torch.Tensor],
    seed: Optional[int] = None,
    return_mask: bool = True,
) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Subsample kspace by setting kspace to zero as given by a binary mask.

    Parameters
    ----------
    kspace: torch.Tensor
        k-space as a complex-valued tensor.
    mask_func: callable or torch.tensor
        Masking function, taking a shape and returning a mask with this shape or can be broadcast as such
        Can also be a sampling mask.
    seed: int
        Seed for the random number generator
    return_mask: bool
        If true, mask will be returned

    Returns
    -------
    masked data, mask: (torch.Tensor, torch.Tensor)
    """
    # TODO: Split the function to apply_mask_func and apply_mask

    assert_complex(kspace, complex_last=True)

    if not isinstance(mask_func, torch.Tensor):
        shape = np.array(kspace.shape)[1:]  # The first dimension is always the coil dimension.
        mask = mask_func(shape=shape, seed=seed)
    else:
        mask = mask_func

    masked_kspace = torch.where(mask == 0, torch.tensor([0.0], dtype=kspace.dtype, device=kspace.device), kspace)

    if not return_mask:
        return masked_kspace

    return masked_kspace, mask


def tensor_to_complex_numpy(data: torch.Tensor) -> np.ndarray:
    """Converts a complex pytorch tensor to a complex numpy array. The last axis denote the real and imaginary parts
    respectively.

    Parameters
    ----------
    data: torch.Tensor
        Input data

    Returns
    -------
    out: np.array
        Complex valued np.ndarray
    """
    assert_complex(data, complex_last=True)
    data_numpy = data.detach().cpu().numpy()
    return data_numpy[..., 0] + 1j * data_numpy[..., 1]


def root_sum_of_squares(data: torch.Tensor, dim: int = 0, complex_dim: int = -1) -> torch.Tensor:
    r"""Compute the root sum of squares (RSS) transform along a given dimension of the input tensor:

    .. math::
        x_{\textrm{RSS}} = \sqrt{\sum_{i \in \textrm{coil}} |x_i|^2}

    Parameters
    ----------
    data: torch.Tensor
        Input tensor
    dim: int
        Coil dimension. Default is 0 as the first dimension is always the coil dimension.
    complex_dim: int
        Complex channel dimension. Default is -1. If data not complex this is ignored.

    Returns
    -------
    torch.Tensor: RSS of the input tensor.

    """
    if is_complex_data(data):
        return torch.sqrt((data**2).sum(complex_dim).sum(dim))

    return torch.sqrt((data**2).sum(dim))


def center_crop(data: torch.Tensor, shape: Union[List[int], Tuple[int, ...]]) -> torch.Tensor:
    """Apply a center crop along the last two dimensions.

    Parameters
    ----------
    data: torch.Tensor
    shape: List or tuple of ints
        The output shape, should be smaller than the corresponding data dimensions.

    Returns
    -------
    torch.Tensor: The center cropped data.
    """
    # TODO: Make dimension independent.
    if not (0 < shape[0] <= data.shape[-2]) or not (0 < shape[1] <= data.shape[-1]):
        raise ValueError(f"Crop shape should be smaller than data. Requested {shape}, got {data.shape}.")

    width_lower = (data.shape[-2] - shape[0]) // 2
    width_upper = width_lower + shape[0]
    height_lower = (data.shape[-1] - shape[1]) // 2
    height_upper = height_lower + shape[1]

    return data[..., width_lower:width_upper, height_lower:height_upper]


def complex_center_crop(
    data_list: Union[List[torch.Tensor], torch.Tensor],
    crop_shape: Union[List[int], Tuple[int, ...]],
    offset: int = 1,
    contiguous: bool = False,
) -> Union[List[torch.Tensor], torch.Tensor]:
    """Apply a center crop to the input data, or to a list of complex images.

    Parameters
    ----------
    data_list: Union[List[torch.Tensor], torch.Tensor]
        The complex input tensor to be center cropped. It should have at least 3 dimensions
         and the cropping is applied along dimensions didx and didx+1 and the last dimensions should have a size of 2.
    crop_shape: List[int] or Tuple[int, ...]
        The output shape. The shape should be smaller than the corresponding dimensions of data.
        If one value is None, this is filled in by the image shape.
    offset: int
        Starting dimension for cropping.
    contiguous: bool
        Return as a contiguous array. Useful for fast reshaping or viewing.

    Returns
    -------
    Union[List[torch.Tensor], torch.Tensor]
        The center cropped input_image(s).
    """
    data_list = ensure_list(data_list)
    assert_same_shape(data_list)

    image_shape = list(data_list[0].shape)
    ndim = data_list[0].ndim
    bbox = [0] * ndim + image_shape

    # Allow for False in crop directions
    shape = [_ if _ else image_shape[idx + offset] for idx, _ in enumerate(crop_shape)]
    for idx, _ in enumerate(shape):
        bbox[idx + offset] = (image_shape[idx + offset] - shape[idx]) // 2
        bbox[len(image_shape) + idx + offset] = shape[idx]

    if not all(_ >= 0 for _ in bbox[:ndim]):
        raise ValueError(
            f"Bounding box requested has negative values, "
            f"this is likely to data size being smaller than the crop size. Got {bbox} with image_shape {image_shape} "
            f"and requested shape {shape}."
        )

    output = [crop_to_bbox(data, bbox) for data in data_list]

    if contiguous:
        output = [_.contiguous() for _ in output]

    if len(output) == 1:  # Only one element:
        return output[0]
    return output


def complex_random_crop(
    data_list: Union[List[torch.Tensor], torch.Tensor],
    crop_shape: Union[List[int], Tuple[int, ...]],
    offset: int = 1,
    contiguous: bool = False,
    sampler: str = "uniform",
    sigma: Union[float, List[float], None] = None,
    seed: Union[None, int, ArrayLike] = None,
) -> Union[List[torch.Tensor], torch.Tensor]:
    """Apply a random crop to the input data tensor or a list of complex.

    Parameters
    ----------
    data_list: Union[List[torch.Tensor], torch.Tensor]
        The complex input tensor to be center cropped. It should have at least 3 dimensions and the cropping is applied
        along dimensions -3 and -2 and the last dimensions should have a size of 2.
    crop_shape: List[int] or Tuple[int, ...]
        The output shape. The shape should be smaller than the corresponding dimensions of data.
    offset: int
        Starting dimension for cropping.
    contiguous: bool
        Return as a contiguous array. Useful for fast reshaping or viewing.
    sampler: str
        Select the random indices from either a `uniform` or `gaussian` distribution (around the center)
    sigma: float or list of float or None
        Standard variance of the gaussian when sampler is `gaussian`. If not set will take 1/3th of image shape
    seed: None, int or ArrayLike

    Returns
    -------
    Union[List[torch.Tensor], torch.Tensor]
        The center cropped input tensor or list of tensors.
    """
    if sampler == "uniform" and sigma is not None:
        raise ValueError(f"sampler `uniform` is incompatible with sigma {sigma}, has to be None.")

    data_list = ensure_list(data_list)
    assert_same_shape(data_list)

    image_shape = list(data_list[0].shape)

    ndim = data_list[0].ndim
    bbox = [0] * ndim + image_shape

    crop_shape = [_ if _ else image_shape[idx + offset] for idx, _ in enumerate(crop_shape)]
    crop_shape = np.asarray(crop_shape)

    limits = np.zeros(len(crop_shape), dtype=int)
    for idx, _ in enumerate(limits):
        limits[idx] = image_shape[offset + idx] - crop_shape[idx]

    if not all(_ >= 0 for _ in limits):
        raise ValueError(
            f"Bounding box limits have negative values, "
            f"this is likely to data size being smaller than the crop size. Got {limits}"
        )
    if seed is not None:
        np.random.seed(seed)
    if sampler == "uniform":
        lower_point = np.random.randint(0, limits + 1).tolist()
    elif sampler == "gaussian":
        data_shape = np.asarray(image_shape[offset : offset + len(crop_shape)])
        if not sigma:
            sigma = data_shape / 6  # w, h
        else:
            if isinstance(sigma, float) or isinstance(sigma, list) and len(sigma) == 1:
                sigma = [sigma for _ in range(len(crop_shape))]
            elif len(sigma) != len(crop_shape):  # type: ignore
                raise ValueError(
                    f"Either one sigma has to be set or same as the length of the bounding box. Got {sigma}."
                )
        lower_point = (
            np.random.normal(loc=data_shape / 2, scale=sigma, size=len(data_shape)) - crop_shape / 2
        ).astype(int)
        lower_point = np.clip(lower_point, 0, limits)
    else:
        raise ValueError(f"Sampler is either `uniform` or `gaussian`. Got {sampler}.")

    for idx, _ in enumerate(crop_shape):
        bbox[offset + idx] = lower_point[idx]
        bbox[offset + ndim + idx] = crop_shape[idx]

    output = [crop_to_bbox(data, bbox) for data in data_list]

    if contiguous:
        output = [_.contiguous() for _ in output]

    if len(output) == 1:
        return output[0]
    return output


def crop_to_acs(acs_mask: torch.Tensor, kspace: torch.Tensor) -> torch.Tensor:
    """Crops k-space to autocalibration region given the acs_mask.

    Parameters
    ----------
    acs_mask : torch.Tensor
        Autocalibration mask of shape (height, width).
    kspace : torch.Tensor
        K-space of shape (coil, height, width, \*).

    Returns
    -------
    torch.Tensor
        Cropped k-space of shape (coil, height', width', \*), where height' and width' are the new dimensions derived
        from the acs_mask.
    """
    nonzero_idxs = torch.nonzero(acs_mask)
    x, y = nonzero_idxs[..., 0], nonzero_idxs[..., 1]
    xl, xr = x.min(), x.max()
    yl, yr = y.min(), y.max()
    return kspace[:, xl : xr + 1, yl : yr + 1]


def reduce_operator(
    coil_data: torch.Tensor,
    sensitivity_map: torch.Tensor,
    dim: int = 0,
) -> torch.Tensor:
    r"""
    Given zero-filled reconstructions from multiple coils :math:`\{x_i\}_{i=1}^{N_c}` and
    coil sensitivity maps :math:`\{S_i\}_{i=1}^{N_c}` it returns:

        .. math::
            R(x_{1}, .., x_{N_c}, S_1, .., S_{N_c}) = \sum_{i=1}^{N_c} {S_i}^{*} \times x_i.

    Adapted from [1]_.

    Parameters
    ----------
    coil_data: torch.Tensor
        Zero-filled reconstructions from coils. Should be a complex tensor (with complex dim of size 2).
    sensitivity_map: torch.Tensor
        Coil sensitivity maps. Should be complex tensor (with complex dim of size 2).
    dim: int
        Coil dimension. Default: 0.

    Returns
    -------
    torch.Tensor:
        Combined individual coil images.

    References
    ----------

    .. [1] Sriram, Anuroop, et al. “End-to-End Variational Networks for Accelerated MRI Reconstruction.”
        ArXiv:2004.06688 [Cs, Eess], Apr. 2020. arXiv.org, http://arxiv.org/abs/2004.06688.

    """

    assert_complex(coil_data, complex_last=True)
    assert_complex(sensitivity_map, complex_last=True)

    return complex_multiplication(conjugate(sensitivity_map), coil_data).sum(dim)


def expand_operator(
    data: torch.Tensor,
    sensitivity_map: torch.Tensor,
    dim: int = 0,
) -> torch.Tensor:
    r"""
    Given a reconstructed image :math:`x` and coil sensitivity maps :math:`\{S_i\}_{i=1}^{N_c}`, it returns

        .. math::
            E(x) = (S_1 \times x, .., S_{N_c} \times x) = (x_1, .., x_{N_c}).

    Adapted from [1]_.

    Parameters
    ----------
    data: torch.Tensor
        Image data. Should be a complex tensor (with complex dim of size 2).
    sensitivity_map: torch.Tensor
        Coil sensitivity maps. Should be complex tensor (with complex dim of size 2).
    dim: int
        Coil dimension. Default: 0.

    Returns
    -------
    torch.Tensor:
        Zero-filled reconstructions from each coil.

    References
    ----------

    .. [1] Sriram, Anuroop, et al. “End-to-End Variational Networks for Accelerated MRI Reconstruction.”
        ArXiv:2004.06688 [Cs, Eess], Apr. 2020. arXiv.org, http://arxiv.org/abs/2004.06688.

    """

    assert_complex(data, complex_last=True)
    assert_complex(sensitivity_map, complex_last=True)

    return complex_multiplication(sensitivity_map, data.unsqueeze(dim))
