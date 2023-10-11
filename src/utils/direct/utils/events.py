# coding=utf-8
# Copyright (c) DIRECT Contributors

# Inspired on
# https://github.com/facebookresearch/detectron2/blob/45808c0ed68332cdb4c55801f1e2934d58231d35/detectron2/utils/events.py
# https://github.com/facebookresearch/fvcore/blob/master/fvcore/common/history_buffer.py
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Licensed under Apache 2.0
# Changes make here:
# - removed PathManager
# - changed formatting to coding style of the rest of this library
# - Added typing, changed put to add.

import datetime
import json
import logging
import os
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch

_CURRENT_STORAGE_STACK: List[Any] = []


def get_event_storage():
    """
    Returns:
        The :class:`EventStorage` object that's currently being used.
        Throws an error if no :class`EventStorage` is currently enabled.
    """
    if len(_CURRENT_STORAGE_STACK) == 0:
        raise ValueError("get_event_storage() has to be called inside a 'with EventStorage(...)' context!")
    return _CURRENT_STORAGE_STACK[-1]


class EventWriter:
    """Base class for writers that obtain events from :class:`EventStorage` and process them."""

    def write(self):
        raise NotImplementedError

    def close(self):
        pass


class JSONWriter(EventWriter):
    """Write scalars to a json file.

    It saves scalars as one json per line (instead of a big json) for easy parsing.

    Examples parsing such a json file:

    .. code-block:: none

        $ cat metrics.json | jq -s '.[0:2]'
        [
          {
            "data_time": 0.008433341979980469,
            "iteration": 20,
            "loss": 1.9228371381759644,
            "loss_box_reg": 0.050025828182697296,
            "loss_classifier": 0.5316952466964722,
            "loss_mask": 0.7236229181289673,
            "loss_rpn_box": 0.0856662318110466,
            "loss_rpn_cls": 0.48198649287223816,
            "lr": 0.007173333333333333,
            "time": 0.25401854515075684
          },
          {
            "data_time": 0.007216215133666992,
            "iteration": 40,
            "loss": 1.282649278640747,
            "loss_box_reg": 0.06222952902317047,
            "loss_classifier": 0.30682939291000366,
            "loss_mask": 0.6970193982124329,
            "loss_rpn_box": 0.038663312792778015,
            "loss_rpn_cls": 0.1471673548221588,
            "lr": 0.007706666666666667,
            "time": 0.2490077018737793
          }
        ]

        $ cat metrics.json | jq '.loss_mask'
        0.7126231789588928
        0.689423680305481
        0.6776131987571716
        ...
    """

    def __init__(self, json_file: Union[Path, str], window_size: int = 2):
        """

        Parameters
        ----------
        json_file: Union[Path, str]
            Path to the JSON file. Data will be appended if it exists
        window_size: int
            Window size of median smoothing for variables for which `smoothing_hint` is True.
        validation: bool
            If true, will only log keys starting with val_
        """

        self._file_handle = open(json_file, "a", encoding="utf-8")
        self._window_size = window_size

    def write(self):
        storage = get_event_storage()
        to_save = {"iteration": storage.iter}
        to_save.update(storage.latest_with_smoothing_hint(self._window_size))
        self._file_handle.write(json.dumps(to_save, sort_keys=True) + "\n")
        self._file_handle.flush()
        try:
            os.fsync(self._file_handle.fileno())
        except AttributeError:
            pass

    def close(self):
        self._file_handle.close()


class TensorboardWriter(EventWriter):
    """Write all scalars to a tensorboard file."""

    def __init__(self, log_dir: Union[Path, str], window_size: int = 20, **kwargs):
        """
        Parameters
        ----------
        log_dir: Union[Path, str]
            The directory to save the output events.
        window_size: int
            The scalars will be median-smoothed by this window size.
        kwargs: dict
            other arguments passed to `torch.utils.tensorboard.SummaryWriter(...)`
        """
        self._window_size = window_size
        from torch.utils.tensorboard import SummaryWriter

        self._writer = SummaryWriter(str(log_dir), **kwargs)

    def write(self):
        storage = get_event_storage()
        for k, v in storage.latest_with_smoothing_hint(self._window_size).items():
            self._writer.add_scalar(k, v, storage.iter)

        if len(storage.vis_data) >= 1:
            for img_name, img, step_num in storage.vis_data:
                self._writer.add_image(img_name, img, step_num)
            storage.clear_images()

    def close(self):
        if hasattr(self, "_writer"):  # doesn't exist when the code fails at import
            self._writer.close()


class CommonMetricPrinter(EventWriter):
    """Print **common** metrics to the terminal, including iteration time, ETA, memory, all losses, and the learning
    rate.

    To print something different, please implement a similar printer by yourself.
    """

    def __init__(self, max_iter):
        """
        Args:
            max_iter (int): the maximum number of iterations to train.
                Used to compute ETA.
        """
        self.logger = logging.getLogger(type(self).__name__)
        self._max_iter = max_iter
        self._last_write = None

    def write(self):
        storage = get_event_storage()
        iteration = storage.iter

        try:
            data_time = storage.history("data_time").avg(20)
        except KeyError:
            # they may not exist in the first few iterations (due to warmup)
            # or when SimpleTrainer is not used
            data_time = None

        eta_string = "N/A"
        try:
            iter_time = storage.history("time").global_avg()
            eta_seconds = storage.history("time").median(1000) * (self._max_iter - iteration)
            storage.add_scalar("eta_seconds", eta_seconds, smoothing_hint=False)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        except KeyError:
            iter_time = None
            # estimate eta on our own - more noisy
            if self._last_write is not None:
                estimate_iter_time = (time.perf_counter() - self._last_write[1]) / (iteration - self._last_write[0])
                eta_seconds = estimate_iter_time * (self._max_iter - iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            self._last_write = (iteration, time.perf_counter())

        try:
            lr = f"{storage.history('lr').latest():.6f}"
        except KeyError:
            lr = "N/A"

        if torch.cuda.is_available():
            max_mem_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
        else:
            max_mem_mb = None

        metrics_and_losses_string = "  ".join(
            [
                f"{k}: {v.median(20):.6f}"
                for k, v in storage.histories().items()
                if ("loss" in k or "metric" in k or "reg" in k)
            ]
        )

        time_string = f"time: {iter_time:.4f}  " if iter_time is not None else ""
        data_time_string = f"data_time: {data_time:.4f}  " if data_time is not None else ""
        memory_string = f"max_mem: {max_mem_mb:.0f}M" if max_mem_mb is not None else ""

        # no logger here, the code already saves the iterations to json.
        self.logger.info(
            f"eta: {eta_string}  iter: {iteration}  "
            f"{metrics_and_losses_string}  {time_string}{data_time_string}lr: {lr}  {memory_string}"
        )


class EventStorage:
    """The user-facing class that provides metric storage functionalities.

    In the future we may add support for storing / logging other types of data if needed.
    """

    def __init__(self, start_iter=0):
        """
        Parameters
        ----------
        start_iter: int
            The index to start with.
        """
        self._history = defaultdict(HistoryBuffer)
        self._smoothing_hints = {}
        self._latest_scalars = {}
        self._iter = start_iter
        self._current_prefix = ""
        self._vis_data = []

    def add_image(self, img_name, img_tensor):
        """Add an `img_tensor` to the `_vis_data` associated with `img_name`.

        Parameters
        ----------
        img_name: str
            The name of the input_image to put into tensorboard.
        img_tensor: torch.Tensor or numpy.array
            An `uint8` or `float` Tensor of shape `[channel, height, width]` where `channel` is 3. The input_image format should be RGB. The elements in img_tensor can either have values in [0, 1] (float32) or [0, 255] (uint8). The `img_tensor` will be visualized in tensorboard.
        """
        self._vis_data.append((img_name, img_tensor, self._iter))

    def clear_images(self):
        """Delete all the stored images for visualization.

        This should be called after images are written to tensorboard.
        """
        self._vis_data = []

    def add_scalar(self, name, value, smoothing_hint=True):
        """Add a scalar `value` to the `HistoryBuffer` associated with `name`.

        Parameters
        ----------
        name: str
        value: float
        smoothing_hint: bool
            A 'hint' on whether this scalar is noisy and should be smoothed when logged. The hint will be accessible through `EventStorage.smoothing_hints`. A writer may ignore the hint and apply custom smoothing rule. It defaults to True because most scalars we save need to be smoothed to provide any useful signal.

        Returns
        -------
        """
        name = self._current_prefix + name
        history = self._history[name]
        value = float(value)
        history.update(value, self._iter)
        self._latest_scalars[name] = value

        existing_hint = self._smoothing_hints.get(name)
        if existing_hint is not None:
            if existing_hint != smoothing_hint:
                raise AssertionError(f"Scalar {name} was put with a different smoothing_hint!")
        else:
            self._smoothing_hints[name] = smoothing_hint

    def add_scalars(self, *, smoothing_hint=True, **kwargs):
        """Put multiple scalars from keyword arguments.

        Examples
        --------
            storage.add_scalars(loss=my_loss, accuracy=my_accuracy, smoothing_hint=True)
        """
        for k, v in kwargs.items():
            self.add_scalar(k, v, smoothing_hint=smoothing_hint)

    def add_graph(self, img_name, img_tensor):
        """Add an `img_tensor` to the `_vis_data` associated with `img_name`.

        Parameters
        ----------
        img_name: str
            The name of the input_image to put into tensorboard.
        img_tensor: torch.Tensor or numpy.array
            An `uint8` or `float` Tensor of shape `[channel, height, width]` where `channel` is 3. The input_image format should be RGB. The elements in img_tensor can either have values in [0, 1] (float32) or [0, 255] (uint8). The `img_tensor` will be visualized in tensorboard.
        """
        self._vis_data.append((img_name, img_tensor, self._iter))

    def history(self, name):
        """
        Returns:
            HistoryBuffer: the scalar history for name
        """
        ret = self._history.get(name, None)
        if ret is None:
            raise KeyError(f"No history metric available for {name}!")
        return ret

    def histories(self):
        """
        Returns:
            dict[name -> HistoryBuffer]: the HistoryBuffer for all scalars
        """
        return self._history

    def latest(self):
        """
        Returns:
            dict[name -> number]: the scalars that's added in the current iteration.
        """
        return self._latest_scalars

    def latest_with_smoothing_hint(self, window_size=20):
        """Similar to :meth:`latest`, but the returned values are either the un-smoothed original latest value, or a
        median of the given window_size, depend on whether the smoothing_hint is True.

        This provides a default behavior that other writers can use.
        """
        result = {}
        for k, v in self._latest_scalars.items():
            result[k] = self._history[k].median(window_size) if self._smoothing_hints[k] else v
        return result

    def smoothing_hints(self):
        """
        Returns:
            dict[name -> bool]: the user-provided hint on whether the scalar
                is noisy and needs smoothing.
        """
        return self._smoothing_hints

    def step(self):
        """User should call this function at the beginning of each iteration, to notify the storage of the start of a
        new iteration.

        The storage will then be able to associate the new data with the correct iteration number.
        """
        self._iter += 1
        # TODO: This clears validation metrics.
        # self._latest_scalars = {}

    @property
    def vis_data(self):
        return self._vis_data

    @property
    def iter(self):
        return self._iter

    def __enter__(self):
        _CURRENT_STORAGE_STACK.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if _CURRENT_STORAGE_STACK[-1] != self:
            raise AssertionError
        _CURRENT_STORAGE_STACK.pop()

    @contextmanager
    def name_scope(self, name):
        """
        Yields:
            A context within which all the events added to this storage
            will be prefixed by the name scope.
        """
        old_prefix = self._current_prefix
        self._current_prefix = name.rstrip("/") + "/"
        yield
        self._current_prefix = old_prefix


class HistoryBuffer:
    """Track a series of scalar values and provide access to smoothed values over a window or the global average of the
    series."""

    def __init__(self, max_length: int = 1000000) -> None:
        """
        Args:
            max_length: maximal number of values that can be stored in the
                buffer. When the capacity of the buffer is exhausted, old
                values will be removed.
        """
        self._max_length: int = max_length
        self._data: List[Tuple[float, float]] = []  # (value, iteration) pairs
        self._count: int = 0
        self._global_avg: float = 0

    def update(self, value: float, iteration: Optional[float] = None) -> None:
        """Add a new scalar value produced at certain iteration.

        If the length of the buffer exceeds self._max_length, the oldest element will be removed from the buffer.
        """
        if iteration is None:
            iteration = self._count
        if len(self._data) == self._max_length:
            self._data.pop(0)
        self._data.append((value, iteration))

        self._count += 1
        self._global_avg += (value - self._global_avg) / self._count

    def latest(self) -> float:
        """Return the latest scalar value added to the buffer."""
        return self._data[-1][0]

    def median(self, window_size: int) -> float:
        """Return the median of the latest `window_size` values in the buffer."""
        return np.median([x[0] for x in self._data[-window_size:]])

    def avg(self, window_size: int) -> float:
        """Return the mean of the latest `window_size` values in the buffer."""
        return float(np.mean([x[0] for x in self._data[-window_size:]]))

    def global_avg(self) -> float:
        """Return the mean of all the elements in the buffer.

        Note that this includes those getting removed due to limited buffer storage.
        """
        return self._global_avg

    def values(self) -> List[Tuple[float, float]]:
        """
        Returns:
            list[(number, iteration)]: content of the current buffer.
        """
        return self._data
