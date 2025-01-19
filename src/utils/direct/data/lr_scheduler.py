# coding=utf-8
# Copyright (c) DIRECT Contributors

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Taken from Detectron 2, licensed under Apache 2.0.
# https://github.com/facebookresearch/detectron2/blob/60d7a1fd33cc48e58968659cd3301f3300b2786b/detectron2/solver/lr_scheduler.py
# Changes:
# - Docstring to match the rest of the library.
# - Calls to other subroutines which do not exist in DIRECT.
# - Stylistic changes.

import logging
import math
from bisect import bisect_right
from typing import List

import torch

# NOTE: PyTorch's LR scheduler interface uses names that assume the LR changes
# only on epoch boundaries. We typically use iteration based schedules instead.
# As a result, "epoch" (e.g., as in self.last_epoch) should be understood to mean
# "iteration" instead.

# FIXME: ideally this would be achieved with a CombinedLRScheduler, separating
# MultiStepLR with WarmupLR but the current LRScheduler design doesn't allow it.


class LRScheduler(torch.optim.lr_scheduler._LRScheduler):  # pylint: disable=protected-access
    def __init__(self, optimizer, last_epoch=-1, verbose=False):
        super().__init__(optimizer, last_epoch, verbose)
        self.logger = logging.getLogger(type(self).__name__)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which is not the optimizer or logger.
        """
        state_dict = {key: value for key, value in self.__dict__.items() if key not in ["optimizer", "logger"]}
        return state_dict


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):  # pylint: disable=protected-access
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        milestones: List[int],
        gamma: float = 0.1,
        warmup_factor: float = 0.001,
        warmup_iterations: int = 1000,
        warmup_method: str = "linear",
        last_epoch: int = -1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {milestones}",
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iterations = warmup_iterations
        self.warmup_method = warmup_method
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:  # type: ignore
        warmup_factor = _get_warmup_factor_at_iter(
            self.warmup_method,
            self.last_epoch,  # type: ignore
            self.warmup_iterations,
            self.warmup_factor,
        )
        return [
            base_lr * warmup_factor * self.gamma ** bisect_right(self.milestones, self.last_epoch)  # type: ignore
            for base_lr in self.base_lrs  # type: ignore
        ]

    def _compute_values(self) -> List[float]:
        # The new interface
        return self.get_lr()


class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):  # pylint: disable=protected-access
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_iters: int,
        warmup_factor: float = 0.001,
        warmup_iterations: int = 1000,
        warmup_method: str = "linear",
        last_epoch: int = -1,
    ):
        self.max_iters = max_iters
        self.warmup_factor = warmup_factor
        self.warmup_iterations = warmup_iterations
        self.warmup_method = warmup_method
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:  # type: ignore
        warmup_factor = _get_warmup_factor_at_iter(
            self.warmup_method,
            self.last_epoch,  # type: ignore
            self.warmup_iterations,
            self.warmup_factor,
        )
        # Different definitions of half-cosine with warmup are possible. For
        # simplicity we multiply the standard half-cosine schedule by the warmup
        # factor. An alternative is to start the period of the cosine at warmup_iterations
        # instead of at 0. In the case that warmup_iterations << max_iters the two are
        # very close to each other.
        return [
            base_lr * warmup_factor * 0.5 * (1.0 + math.cos(math.pi * self.last_epoch / self.max_iters))  # type: ignore
            for base_lr in self.base_lrs  # type: ignore
        ]

    def _compute_values(self) -> List[float]:
        # The new interface
        return self.get_lr()


def _get_warmup_factor_at_iter(method: str, curr_iter: int, warmup_iters: int, warmup_factor: float) -> float:
    """Return the learning rate warmup factor at a specific iteration.

    Parameters
    ----------
    method: str
        Warmup method; either "constant" or "linear".
    curr_iter: int
        Iteration at which to calculate the warmup factor.
    warmup_iters: int
        The length of the warmup phases.
    warmup_factor: float
        The base warmup factor (the meaning changes according to the method used).

    Returns
    -------
    float: The effective warmup factor at the given iteration.
    """
    if curr_iter >= warmup_iters:
        return 1.0

    if method == "constant":
        return warmup_factor
    if method == "linear":
        alpha = curr_iter / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    raise ValueError(f"Unknown warmup method: {method}")
