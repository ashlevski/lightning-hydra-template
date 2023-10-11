# coding=utf-8
# Copyright (c) DIRECT Contributors

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Taken from Detectron 2, licensed under Apache 2.0.
# https://github.com/facebookresearch/detectron2/blob/903d28b63c02dffc81935a38a85ab5a16450a445/detectron2/engine/launch.py
# Changes:
# - Docstring to match the rest of the library.
# - Calls to other subroutines which do not exist in DIRECT.
# - Stylistic changes.

import logging
import sys
from datetime import timedelta
from typing import Callable, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from src.utils.direct.utils import communication

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

__all__ = ["launch", "launch_distributed", "DEFAULT_TIMEOUT"]

DEFAULT_TIMEOUT = timedelta(minutes=30)


def _find_free_port():
    """Finds ans returns a free port."""
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def launch_distributed(
    main_func: Callable,
    num_gpus_per_machine: int,
    num_machines: int = 1,
    machine_rank: int = 0,
    dist_url: str = "auto",
    args: Tuple = (),
    timeout: timedelta = DEFAULT_TIMEOUT,
) -> None:
    """Launch multi-gpu or distributed training.

    This function must be called on all machines involved in the training and it will spawn
    child processes (defined by `num_gpus_per_machine`) on each machine.

    Parameters
    ----------
    main_func: Callable
        A function that will be called by `main_func(*args)`.
    num_gpus_per_machine: int
        The number of GPUs per machine.
    num_machines : int
        The number of machines.
    machine_rank: int
        The rank of this machine (one per machine).
    dist_url: str
        URL to connect to for distributed training, including protocol e.g. "tcp://127.0.0.1:8686".
        Can be set to auto to automatically select a free port on localhost
    args: Tuple
        arguments passed to main_func.
    timeout: timedelta
        Timeout of the distributed workers.
    """
    world_size = num_machines * num_gpus_per_machine
    if world_size > 1:
        # https://github.com/pytorch/pytorch/pull/14391
        # TODO prctl in spawned processes

        if dist_url == "auto":
            if num_machines != 1:
                raise ValueError("dist_url=auto not supported in multi-machine jobs.")
            port = _find_free_port()
            dist_url = f"tcp://127.0.0.1:{port}"
        if num_machines > 1 and dist_url.startswith("file://"):
            logger.warning("file:// is not a reliable init_method in multi-machine jobs. Prefer tcp://")

        mp.spawn(
            _distributed_worker,
            nprocs=num_gpus_per_machine,
            args=(
                main_func,
                world_size,
                num_gpus_per_machine,
                machine_rank,
                dist_url,
                args,
                timeout,
            ),
            daemon=False,
        )
    else:
        main_func(*args)


def _distributed_worker(
    local_rank: int,
    main_func: Callable,
    world_size: int,
    num_gpus_per_machine: int,
    machine_rank: int,
    dist_url: str,
    args: Tuple,
    timeout: timedelta = DEFAULT_TIMEOUT,
) -> None:
    """Sets up `init_process_group`.

    Parameters
    ----------
    local_rank: int
        Local rank.
    main_func: Callable
        A function that will be called by `main_func(*args)`.
    world_size: int
        World size equal to `num_machines * num_gpus_per_machine`.
    machine_rank: int
        The rank of this machine (one per machine).
    num_gpus_per_machine: int
        The number of GPUs per machine.
    dist_url: str
        URL to connect to for distributed training, including protocol e.g. "tcp://127.0.0.1:8686".
        Can be set to auto to automatically select a free port on localhost
    args: Tuple
        arguments passed to main_func.
    timeout: timedelta
        Timeout of the distributed workers.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your installation.")

    global_rank = machine_rank * num_gpus_per_machine + local_rank
    try:
        dist.init_process_group(
            backend="NCCL",
            init_method=dist_url,
            world_size=world_size,
            rank=global_rank,
            timeout=timeout,
        )
    except Exception as e:
        logger.error(f"Process group URL: {dist_url}")
        raise e
    # synchronize is needed here to prevent a possible timeout after calling init_process_group
    # See: https://github.com/facebookresearch/maskrcnn-benchmark/issues/172
    communication.synchronize()
    logger.info(f"Global rank {global_rank}.")
    logger.info("Synchronized GPUs.")

    if num_gpus_per_machine > torch.cuda.device_count():
        raise RuntimeError
    torch.cuda.set_device(local_rank)

    # Setup the local process group (which contains ranks within the same machine)
    if communication._LOCAL_PROCESS_GROUP is not None:
        raise RuntimeError
    num_machines = world_size // num_gpus_per_machine
    for idx in range(num_machines):
        ranks_on_i = list(range(idx * num_gpus_per_machine, (idx + 1) * num_gpus_per_machine))
        pg = dist.new_group(ranks_on_i)
        if idx == machine_rank:
            communication._LOCAL_PROCESS_GROUP = pg

    main_func(*args)


def launch(
    func: Callable,
    num_machines: int,
    num_gpus: int,
    machine_rank: int,
    dist_url: str,
    *args: Tuple,
) -> None:
    """Launch the training, in case there is only one GPU available the function can be called directly.

    Parameters
    ----------
    func: Callable
        Function to launch.
    num_machines : int
        The number of machines.
    num_gpus: int
        The number of GPUs.
    machine_rank: int
        The machine rank.
    dist_url: str
        URL to connect to for distributed training, including protocol.
    args: Tuple
        Arguments to pass to func.
    """
    # There is no need for the launch script within one node and at most one GPU.
    if num_machines == 1 and num_gpus <= 1:
        if torch.cuda.device_count() > 1:
            logger.warning(
                f"Device count is {torch.cuda.device_count()}, "
                f"but num_machines is set to {num_machines} and num_gpus is {num_gpus}."
            )
        func(*args)
    elif torch.cuda.device_count() > 1 and num_gpus <= 1:
        print(
            f"Device count is {torch.cuda.device_count()}, yet number of GPUs is {num_gpus}. "
            f"Unexpected behavior will occur. Consider exposing less GPUs (e.g. through docker). Exiting."
        )
        sys.exit()

    else:
        launch_distributed(
            func,
            num_gpus,
            num_machines=num_machines,
            machine_rank=machine_rank,
            dist_url=dist_url,
            args=args,
        )
