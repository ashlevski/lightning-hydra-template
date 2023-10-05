# coding=utf-8
# Copyright (c) DIRECT Contributors
import abc
import ast
import functools
import importlib
import logging
import os
import pathlib
import random
import subprocess
import sys
from collections import OrderedDict
from typing import Any, Callable, Dict, KeysView, List, Optional, Tuple, Union

import numpy as np
import torch
from omegaconf.omegaconf import DictConfig

from direct.types import DictOrDictConfig

logger = logging.getLogger(__name__)


COMPLEX_DIM = 2


def is_complex_data(data: torch.Tensor, complex_axis: int = -1) -> bool:
    """Returns True if data is a complex tensor at a specified dimension, i.e. complex_axis of data is of size 2,
    corresponding to real and imaginary channels..

    Parameters
    ----------
    data: torch.Tensor
    complex_axis: int
        Complex dimension along which the check will be done. Default: -1 (last).

    Returns
    -------
    bool
        True if data is a complex tensor.
    """

    return data.size(complex_axis) == COMPLEX_DIM


def is_power_of_two(number: int) -> bool:
    """Check if input is a power of 2.

    Parameters
    ----------
    number: int

    Returns
    -------
    bool
    """
    return number != 0 and ((number & (number - 1)) == 0)


def ensure_list(data: Any) -> List:
    """Ensure input is a list.

    Parameters
    ----------
    data: object

    Returns
    -------
    list
    """
    if data is None:
        return []

    if not isinstance(data, (list, tuple)):
        return [data]

    return list(data)


def cast_as_path(data: Optional[Union[pathlib.Path, str]]) -> Optional[pathlib.Path]:
    """Ensure the the input is a path.

    Parameters
    ----------
    data: str or pathlib.Path

    Returns
    -------
    pathlib.Path
    """
    if data is None:
        return None

    return pathlib.Path(data)


def str_to_class(module_name: str, function_name: str) -> Callable:
    """Convert a string to a class Base on: https://stackoverflow.com/a/1176180/576363.

    Also support function arguments, e.g. ifft(dim=2) will be parsed as a partial and return ifft where dim has been
    set to 2.


    Examples
    --------
    >>> def mult(f, mul=2):
    >>>    return f*mul

    >>> str_to_class(".", "mult(mul=4)")
    >>> str_to_class(".", "mult(mul=4)")
    will return a function which multiplies the input times 4, while

    >>> str_to_class(".", "mult")
    just returns the function itself.

    Parameters
    ----------
    module_name: str
        e.g. direct.data.transforms
    function_name: str
        e.g. Identity
    Returns
    -------
    object
    """
    tree = ast.parse(function_name)
    func_call = tree.body[0].value  # type: ignore
    args = [ast.literal_eval(arg) for arg in func_call.args] if hasattr(func_call, "args") else []
    kwargs = (
        {arg.arg: ast.literal_eval(arg.value) for arg in func_call.keywords} if hasattr(func_call, "keywords") else {}
    )

    # Load the module, will raise ModuleNotFoundError if module cannot be loaded.
    module = importlib.import_module(module_name)

    if not args and not kwargs:
        return getattr(module, function_name)
    return functools.partial(getattr(module, func_call.func.id), *args, **kwargs)


def dict_to_device(
    data: Dict[str, torch.Tensor],
    device: Union[torch.device, str, None],
    keys: Union[List, Tuple, KeysView, None] = None,
) -> Dict:
    """Copy tensor-valued dictionary to device. Only torch.Tensor is copied.

    Parameters
    ----------
    data: Dict[str, torch.Tensor]
    device: torch.device, str
    keys: List, Tuple
        Subselection of keys to copy.

    Returns
    -------
    Dictionary at the new device.
    """
    if keys is None:
        keys = data.keys()
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items() if k in keys}


def detach_dict(data: Dict[str, torch.Tensor], keys: Optional[Union[List, Tuple, KeysView]] = None) -> Dict:
    """Return a detached copy of a dictionary. Only torch.Tensor's are detached.

    Parameters
    ----------
    data: Dict[str, torch.Tensor]
    keys: List, Tuple
        Subselection of keys to detach

    Returns
    -------
    Dictionary at the new device.
    """
    if keys is None:
        keys = data.keys()
    return {k: v.detach() for k, v in data.items() if k in keys if isinstance(v, torch.Tensor)}


def reduce_list_of_dicts(data: List[Dict[str, torch.Tensor]], mode="average", divisor=None) -> Dict[str, torch.Tensor]:
    """Average a list of dictionary mapping keys to Tensors.

    Parameters
    ----------
    data: List[Dict[str, torch.Tensor]])
    mode: str
        Which reduction mode, average reduces the dictionary, sum just adds while average computes the average.
    divisor: None or int
        If given values are divided by this factor.

    Returns
    -------
    Dict[str, torch.Tensor]: Reduced dictionary.
    """
    if not data:
        return {}

    if mode not in ["average", "sum"]:
        raise ValueError("Reduction can only be `sum` or `average`.")

    if not divisor:
        divisor = 1.0

    result_dict = {k: torch.zeros_like(v) for k, v in data[0].items()}

    for elem in data:
        result_dict = {k: result_dict[k] + v for k, v in elem.items()}

    if mode == "average":
        divisor *= len(data)

    return {k: v / divisor for k, v in result_dict.items()}


def merge_list_of_dicts(list_of_dicts: List[Dict]) -> Dict:
    """A list of dictionaries is merged into one dictionary.

    Parameters
    ----------
    list_of_dicts: List[Dict]

    Returns
    -------
    Dict
    """
    if not list_of_dicts:
        return {}

    return functools.reduce(lambda a, b: {**dict(a), **dict(b)}, list_of_dicts)


def evaluate_dict(
    fns_dict: Dict[str, Callable], source: torch.Tensor, target: torch.Tensor, reduction: str = "mean"
) -> Dict:
    """Evaluate a dictionary of functions.

    Examples
    --------
    > evaluate_dict({'l1_loss: F.l1_loss, 'l2_loss': F.l2_loss}, a, b)

    Will return
    > {'l1_loss', F.l1_loss(a, b, reduction='mean'), 'l2_loss': F.l2_loss(a, b, reduction='mean')

    Parameters
    ----------
    fns_dict: Dict[str, Callable]
    source: torch.Tensor
    target: torch.Tensor
    reduction: str

    Returns
    -------
    Dict[str, torch.Tensor]
        Evaluated dictionary.
    """
    return {k: fns_dict[k](source, target, reduction=reduction) for k, v in fns_dict.items()}


def prefix_dict_keys(data: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """Append a prefix to a dictionary keys.

    Parameters
    ----------
    data: Dict[str, Any]
    prefix: str

    Returns
    -------
    Dict[str, Any]
    """
    return {prefix + k: v for k, v in data.items()}


def git_hash() -> str:
    """Returns the current git hash.

    Returns
    -------
    _git_hash: str
        The current git hash.
    """
    try:
        _git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.PIPE).decode().strip()
    except FileNotFoundError:
        _git_hash = "git not installed."
    except subprocess.CalledProcessError as e:
        exit_code = e.returncode
        stdout = e.output.decode(sys.getfilesystemencoding())
        stderr = e.stderr.decode(sys.getfilesystemencoding())
        _git_hash = f"cannot get git hash: git returned {exit_code}\n" f"stdout: {stdout}.\n" f"stderr: {stderr}."

    return _git_hash


def normalize_image(image: torch.Tensor, eps: float = 0.00001) -> torch.Tensor:
    r"""Normalize image to range [0,1] for visualization.

    Given image :math:`x` and :math:`\epsilon`, it returns:

    .. math::
        \frac{x - \min{x}}{\max{x} + \epsilon}.

    Parameters
    ----------
    image: torch.Tensor
        Image to scale.
    eps: float

    Returns
    -------
    image: torch.Tensor
        Scaled image.
    """

    image = image - image.min()
    image = image / (image.max() + eps)
    return image


def multiply_function(multiplier: float, func: Callable) -> Callable:
    """Create a function which multiplier another one with a multiplier.

    Parameters
    ----------
    multiplier: float
        Number to multiply with.
    func: callable
        Function to multiply.

    Returns
    -------
    return_func: Callable
    """

    def return_func(*args, **kwargs):
        return multiplier * func(*args, **kwargs)

    return return_func


class DirectTransform:
    """Direct transform class.

    Defines :meth:`__repr__` method for Direct transforms.
    """

    def __init__(self):
        """Inits DirectTransform."""
        super().__init__()
        self.coil_dim = 1
        self.spatial_dims = (2, 3)
        self.complex_dim = -1

    def __repr__(self):
        """Representation of DirectTransform."""
        repr_string = self.__class__.__name__ + "("
        for k, v in self.__dict__.items():
            if k == "logger":
                continue
            repr_string += f"{k}="
            if callable(v):
                if hasattr(v, "__class__"):
                    repr_string += type(v).__name__ + ", "
                else:
                    # TODO(jt): better way to log functions
                    repr_string += str(v) + ", "
            elif isinstance(v, (dict, OrderedDict)):
                repr_string += f"{k}=dict(len={len(v)}), "
            elif isinstance(v, list):
                repr_string += f"{k}=list(len={len(v)}), "
            elif isinstance(v, tuple):
                repr_string += f"{k}=tuple(len={len(v)}), "
            else:
                repr_string += str(v) + ", "

        if repr_string[-2:] == ", ":
            repr_string = repr_string[:-2]
        return repr_string + ")"


class DirectModule(DirectTransform, abc.ABC, torch.nn.Module):
    @abc.abstractmethod
    def __init__(self):
        super().__init__()

    def forward(self, sample: Dict):
        pass  # This comment passes "Function/method with an empty body PTC-W0049" error.


def count_parameters(models: Dict) -> None:
    """Count the number of parameters of a dictionary of models.

    Parameters
    ----------
    models: Dict
        Dictionary mapping model name to model.
    """
    total_number_of_parameters = 0
    for model_name in models:
        n_params = sum(p.numel() for p in models[model_name].parameters())
        logger.info(f"Number of parameters model {model_name}: {n_params} ({n_params / 10.0 ** 3:.2f}k).")
        logger.debug(models[model_name])
        total_number_of_parameters += n_params
    logger.info(
        f"Total number of parameters model: {total_number_of_parameters} "
        f"({total_number_of_parameters / 10.0 ** 3:.2f}k)."
    )


def _select_random_seed(min_seed_value: int = 1, max_seed_value: int = 2**32) -> int:
    """Selects random seed.

    Parameters
    ----------
    min_seed_value: int
        Minimum seed value. Default: 1.
    max_seed_value: int
        Maximum seed value. Default: 2**32.

    Returns
    -------
    seed: int
        Random integer in range(min_seed_value, max_seed_value).
    """
    return random.randint(min_seed_value, max_seed_value)  # nosec


def set_all_seeds(seed: int) -> None:
    """Sets seed for deterministic runs.

    Parameters
    ----------
    seed:  int
        Seed for random module.

    Returns
    -------
    """
    # Global seed.
    random.seed(seed)

    # Set individual seeds
    torch.manual_seed(_select_random_seed())
    torch.cuda.manual_seed(_select_random_seed())
    np.random.seed(_select_random_seed())
    os.environ["PYTHONHASHSEED"] = str(_select_random_seed())
    os.environ["PL_GLOBAL_SEED"] = str(_select_random_seed())


def chunks(list_to_chunk: List, number_of_chunks: int):
    """Yield `number_of_chunks number` of sequential chunks from `list_to_chunk`. Adapted from [1]_.

    Parameters
    ----------
    list_to_chunk: List
    number_of_chunks: int

    References
    ----------

    .. [1] https://stackoverflow.com/a/54802737
    """
    d, r = divmod(len(list_to_chunk), number_of_chunks)
    for idx in range(number_of_chunks):
        si = (d + 1) * (idx if idx < r else r) + d * (0 if idx < r else idx - r)
        yield list_to_chunk[si : si + (d + 1 if idx < r else d)]


def remove_keys(input_dict: Dict, keys: Union[str, List[str], Tuple[str]]) -> Dict:
    """Removes `keys` from `input_dict`.

    Parameters
    ----------
    input_dict: Dict
    keys: Union[str, List[str], Tuple[str]]

    Returns
    -------
    Dict
    """
    input_dict = dict(input_dict).copy()
    if not isinstance(keys, (list, tuple)):
        keys = [keys]
    for key in keys:
        if key not in input_dict:
            continue
        del input_dict[key]
    return input_dict


def dict_flatten(in_dict: DictOrDictConfig, dict_out: Optional[DictOrDictConfig] = None) -> Dict[str, Any]:
    """Flattens a nested dictionary (or DictConfig) and returns a new flattened dictionary.

    If a `dict_out` is provided, the flattened dictionary will be added to it.

    Parameters
    ----------
    in_dict : DictOrDictConfig
        The nested dictionary or DictConfig to flatten.
    dict_out : Optional[DictOrDictConfig], optional
        An existing dictionary to add the flattened dictionary to. Default: None.

    Returns
    -------
    Dict[str, Any]
        The flattened dictionary.

    Notes
    -----
    * This function only keeps the final keys, and discards the intermediate ones.

    Examples
    --------
    >>> dictA = {"a": 1, "b": {"c": 2, "d": 3, "e": {"f": 4, 6: "a", 5: {"g": 6}, "l": [1, "two"]}}}
    >>> dict_flatten(dictA)
    {'a': 1, 'c': 2, 'd': 3, 'f': 4, 6: 'a', 'g': 6, 'l': [1, 'two']}
    """
    if dict_out is None:
        dict_out = {}
    for k, v in in_dict.items():
        if isinstance(v, (dict, DictConfig)):
            dict_flatten(in_dict=v, dict_out=dict_out)
            continue
        dict_out[k] = v
    return dict_out
