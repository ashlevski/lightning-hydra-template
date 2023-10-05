# coding=utf-8
# Copyright (c) DIRECT Contributors
import inspect
from typing import List, Optional

import torch

from direct.utils import is_complex_data


def assert_positive_integer(*variables, strict: bool = False) -> None:
    """Assert if given variables are positive integer.

    Parameters
    ----------
    variables: Any
    strict: bool
        If true, will allow zero values.
    """
    if not strict:
        type_name = "positive integer"
    else:
        type_name = "positive integer larger than zero"

    for variable in variables:
        if not isinstance(variable, int) or (variable <= 0 and strict) or (variable < 0 and not strict):
            callers_local_vars = inspect.currentframe().f_back.f_locals.items()  # type: ignore
            variable_name = [var_name for var_name, var_val in callers_local_vars if var_val is variable][0]

            raise ValueError(f"{variable_name} has to be a {type_name}. " f"Got {variable} of type {type(variable)}.")


def assert_same_shape(data_list: List[torch.Tensor]):
    """Check if all tensors in the list have the same shape.

    Parameters
    ----------
    data_list: list
        List of tensors
    """
    shape_list = set(_.shape for _ in data_list)
    if not len(shape_list) == 1:
        raise ValueError(f"All inputs are expected to have the same shape. Got {shape_list}.")


def assert_complex(data: torch.Tensor, complex_axis: int = -1, complex_last: Optional[bool] = None) -> None:
    """Assert if a tensor is complex (has complex dimension of size 2 corresponding to real and imaginary channels).

    Parameters
    ----------
    data: torch.Tensor
    complex_axis: int
        Complex dimension along which the assertion will be done. Default: -1 (last).
    complex_last: Optional[bool]
        If true, will override complex_axis with -1 (last). Default: None.
    """
    # TODO: This is because ifft and fft or torch expect the last dimension to represent the complex axis.
    if complex_last:
        complex_axis = -1
    assert is_complex_data(
        data, complex_axis
    ), f"Complex dimension assumed to be 2 (complex valued), but not found in shape {data.shape}."
