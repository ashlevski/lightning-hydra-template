# coding=utf-8
# Copyright (c) DIRECT Contributors

# Taken from Detectron 2, licensed under Apache 2.0.
# https://github.com/facebookresearch/detectron2/blob/989f52d67d05445ccd030d8f13d6cc53e297fb91/detectron2/utils/comm.py
# Changes:
# - Docstring to match the rest of the library.
# - Calls to other subroutines which do not exist in DIRECT.
# - Extra logging.

# TODO: Mypy has a lot of difficulties parsing this file
# Several issues:
# - Likely size_list needs to be properly cast as after int(...) it remains to be a tensor?
# - torch.distributed does not exist in os x

import functools
import logging
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


_LOCAL_PROCESS_GROUP = None
"""
A torch process group which only includes processes that on the same machine as the current process.
This variable is set when processes are spawned by `launch()` in "engine/launch.py".
"""


def synchronize():
    """Synchronize processes between GPUs.

    Wait until all devices are available. Does nothing in a non-distributed setting.
    """
    if not torch.distributed.is_available():
        logger.info("torch.distributed: not available.")
        return

    if not torch.distributed.is_initialized():
        logger.info("torch.distributed: not initialized.")
        return

    if torch.distributed.get_world_size() == 1:
        logger.info("torch distributed: world size is 1")
        return

    torch.distributed.barrier()


def get_rank() -> int:
    """Get rank of the process, even when torch.distributed is not initialized.

    Returns
    -------
    int
    """
    if not torch.distributed.is_available():
        return 0
    if not torch.distributed.is_initialized():
        return 0

    return torch.distributed.get_rank()


def get_local_rank() -> int:
    """Get rank of the process within the same machine, even when torch.distributed is not initialized.

    Returns
    -------
    int: The rank of the current process within the local (per-machine) process group.
    """
    if not torch.distributed.is_available():
        return 0
    if not torch.distributed.is_initialized():
        return 0
    if _LOCAL_PROCESS_GROUP is None:
        raise ValueError(f"{_LOCAL_PROCESS_GROUP} needs to be set.")

    return torch.distributed.get_rank(group=_LOCAL_PROCESS_GROUP)


def get_local_size() -> int:
    """Number of compute units in local machine.

    Returns
    -------
    int
    """
    if not torch.distributed.is_available():
        return 1
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size(group=_LOCAL_PROCESS_GROUP)


def is_main_process() -> bool:
    """Simple wrapper around get_rank().

    Returns
    -------
    bool
    """
    return get_rank() == 0


def get_world_size() -> int:
    """Get number of compute device in the world, returns 1 in case multi device is not initialized.

    Returns
    -------
    int
    """
    if not torch.distributed.is_available():
        return 1
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size()


@functools.lru_cache()
def _get_global_gloo_group() -> torch.distributed.group:
    """Return a process group based on gloo backend, containing all the ranks The result is cached."""
    if torch.distributed.get_backend() == "nccl":
        return torch.distributed.new_group(backend="gloo")
    return torch.distributed.group.WORLD  # type: ignore


def _serialize_to_tensor(data: object, group: torch.distributed.group) -> torch.Tensor:
    backend = torch.distributed.get_backend(group)
    if backend not in ["gloo", "nccl"]:
        raise AssertionError
    device = torch.device("cpu" if backend == "gloo" else "cuda")

    # Pickeling goes through the normal pickle interface, the current torch.save also zips data.
    buffer = pickle.dumps(data)
    if len(buffer) > 1024**3:
        logger.warning(
            f"Rank {get_rank()} trying to all-gather {len(buffer) / (1024 ** 3):.2f} GB of data on device {device}"
        )
    storage = torch.ByteStorage.from_buffer(buffer)  # type: ignore
    tensor = torch.ByteTensor(storage).to(device=device)  # type: ignore
    return tensor


def _pad_to_largest_tensor(
    tensor: torch.Tensor, group: torch.distributed.group
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """
    Parameters
    ----------
    tensor: torch.Tensor
    group: torch.distributed.group

    Returns
    -------
    list[int]: size of the tensor, on each rank
    Tensor: padded tensor that has the max size
    """
    world_size = torch.distributed.get_world_size(group=group)

    if not world_size > 1:
        raise ValueError("communication.gather/all_gather must be called from ranks within the given group!")
    local_size = torch.tensor([tensor.numel()], dtype=torch.int64, device=tensor.device)
    size_list = [torch.zeros([1], dtype=torch.int64, device=tensor.device) for _ in range(world_size)]
    torch.distributed.all_gather(size_list, local_size, group=group)  # type: ignore

    # Cast list to integers
    size_list = [int(size.item()) for size in size_list]  # type: ignore
    max_size = max(size_list)  # type: ignore

    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    if local_size != max_size:
        padding = torch.zeros((max_size - local_size,), dtype=torch.uint8, device=tensor.device)  # type: ignore
        tensor = torch.cat((tensor, padding), dim=0)
    return size_list, tensor


def all_gather(data: object, group: Optional[torch.distributed.group] = None):
    """Run all_gather on arbitrary picklable data (not necessarily tensors).

    Parameters
    ----------
    data: object
        Any pickleable object.
    group :
        A torch process group. By default, will use a group which contains all ranks on gloo backend.

    Returns
    -------
    list: list of data gathered for each rank
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()
    if torch.distributed.get_world_size(group) == 1:
        return [data]

    tensor = _serialize_to_tensor(data, group)  # type: ignore

    size_list, tensor = _pad_to_largest_tensor(tensor, group)
    max_size = max(size_list)  # type: ignore

    # receiving Tensor from all ranks
    tensor_list = [torch.empty((max_size,), dtype=torch.uint8, device=tensor.device) for _ in size_list]  # type: ignore
    torch.distributed.all_gather(tensor_list, tensor, group=group)  # type: ignore

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]  # type: ignore
        data_list.append(pickle.loads(buffer))

    return data_list


def gather(
    data: object,
    destination_rank: int = 0,
    group: Optional[torch.distributed.group] = None,
) -> List:
    """Run gather on arbitrary picklable data (not necessarily tensors).

    Parameters
    ----------
    data: object
        Any pickleable object
    destination_rank: int
        Destination rank
    group :
        A torch process group. By default, will use a group which contains all ranks on gloo backend.

    Returns
    -------
    list[data]: on destination_rank, a list of data gathered from each rank. Otherwise, an empty list.
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()
    if torch.distributed.get_world_size(group=group) == 1:
        return [data]
    rank = torch.distributed.get_rank(group=group)

    tensor = _serialize_to_tensor(data, group)
    size_list, tensor = _pad_to_largest_tensor(tensor, group)

    # receiving Tensor from all ranks
    if rank == destination_rank:
        max_size = max(size_list)  # type: ignore
        tensor_list = [torch.empty((max_size,), dtype=torch.uint8, device=tensor.device) for _ in size_list]  # type: ignore
        # Ignore type, as torch.distributed is not implemented on OS X.
        torch.distributed.gather(tensor, tensor_list, dst=destination_rank, group=group)  # type: ignore

        data_list = []
        for size, tensor in zip(size_list, tensor_list):
            buffer = tensor.cpu().numpy().tobytes()[:size]  # type: ignore
            data_list.append(pickle.loads(buffer))
        return data_list
    # Ignore type, as torch.distributed is not implemented on OS X.
    torch.distributed.gather(tensor, [], dst=destination_rank, group=group)  # type: ignore
    return []


def shared_random_seed() -> int:
    """All workers must call this function, otherwise it will deadlock.

    Returns
    -------
    A random number that is the same across all workers. If workers need a shared RNG, they can use this shared seed to
    create one.
    """
    ints = np.random.randint(2**31)
    all_ints = all_gather(ints)
    return all_ints[0]


def reduce_tensor_dict(tensors_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Reduce the tensor dictionary from all processes so that process with rank 0 has the averaged results. Returns a
    dict with the same fields as tensors_dict, after reduction.

    Parameters
    ----------
    tensors_dict: dict
        dictionary with str keys mapping to torch tensors.
    Returns
    -------
    dict: the reduced dict.
    """
    if not tensors_dict:
        return tensors_dict

    world_size = get_world_size()
    if world_size <= 1:
        return tensors_dict
    with torch.no_grad():
        tensor_names = []
        all_tensors = []
        for k in sorted(tensors_dict.keys()):  # sorted to make sure this is consistent across processes.
            tensor_names.append(k)
            all_tensors.append(tensors_dict[k])
        all_tensors = torch.stack(all_tensors, dim=0)  # type: ignore
        torch.distributed.reduce(all_tensors, dst=0)  # type: ignore
        if torch.distributed.get_rank() == 0:
            # Only accumulate in main process
            all_tensors /= world_size  # type: ignore
        reduced_tensor_dict = dict(zip(tensor_names, all_tensors))
    return reduced_tensor_dict
