# coding=utf-8
# Copyright (c) DIRECT Contributors

"""Main engine of DIRECT.

Implements all the main training, testing and validation logic.
"""

import functools
import gc
import logging
import pathlib
import signal
import sys
import warnings
from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
from torch import nn
from torch.cuda.amp import GradScaler
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.utils import make_grid

import direct
from src.utils.direct.checkpointer import Checkpointer
from src.utils.direct.config.defaults import BaseConfig
from src.utils.direct.data import transforms as T
from src.utils.direct.data.bbox import crop_to_largest
from src.utils.direct.data.datasets import ConcatDataset
from src.utils.direct.data.samplers import ConcatDatasetBatchSampler
from src.utils.direct.exceptions import ProcessKilledException, TrainingException
from src.utils.direct.types import PathOrString
from src.utils.direct.utils import communication, normalize_image, prefix_dict_keys, reduce_list_of_dicts, str_to_class
from src.utils.direct.utils.events import CommonMetricPrinter, EventStorage, JSONWriter, TensorboardWriter, get_event_storage
from src.utils.direct.utils.io import write_json

DoIterationOutput = namedtuple(
    "DoIterationOutput",
    ["output_image", "sensitivity_map", "data_dict"],
)


class DataDimensionality:
    def __init__(self):
        self._ndim = None

    @property
    def ndim(self):
        if not self._ndim:
            raise ValueError("ndim needs to be set before it can be called.")

        return self._ndim

    @ndim.setter
    def ndim(self, ndim):
        if not isinstance(ndim, int) or ndim <= 0:
            raise ValueError(f"ndim has to be an integer larger than 0. Got {ndim}.")

        self._ndim = ndim


class Engine(ABC, DataDimensionality):
    def __init__(
        self,
        cfg: BaseConfig,
        model: nn.Module,
        device: str,
        forward_operator: Optional[Callable] = None,
        backward_operator: Optional[Callable] = None,
        mixed_precision: bool = False,
        **models: nn.Module,
    ):
        """Inits :class:`Engine`.

        Parameters
        ----------
        cfg: BaseConfig
            Configuration file.
        model: nn.Module
            Model.
        device: str
            Device. Can be "cuda" or "cpu".
        forward_operator: Callable, optional
            The forward operator. Default: None.
        backward_operator: Callable, optional
            The backward operator. Default: None.
        mixed_precision: bool
            Use mixed precision. Default: False.
        **models: nn.Module
            Additional models.
        """
        self.logger = logging.getLogger(type(self).__name__)

        self.cfg = cfg
        self.model = model
        self.models = models
        self.device = device

        # Operators can be useful in some operations
        self.forward_operator = forward_operator
        self.backward_operator = backward_operator

        self.mixed_precision = mixed_precision
        self.checkpointer: Union[Checkpointer, None] = None

        self.__optimizer: Union[torch.optim.Optimizer, None] = None
        self.__lr_scheduler = None
        self._scaler = GradScaler(enabled=self.mixed_precision)
        self.__writers = None
        self.__bind_sigint_signal()

        DataDimensionality.__init__(self)

    @abstractmethod
    def build_loss(self) -> Dict:
        pass

    @staticmethod
    def _build_function_class(functions_list, root_module, postfix) -> Dict:
        if not functions_list:
            return {}

        # _postfix is added as only keys containing loss, metric or reg are logged.
        functions_dict = {
            curr_func.split("(")[0] + f"_{postfix}": str_to_class(root_module, curr_func)
            for curr_func in functions_list
        }
        return functions_dict

    def build_metrics(self, metrics_list) -> Dict:
        return self._build_function_class(metrics_list, "direct.functionals", "metric")

    def build_regularizers(self, regularizers_list) -> Dict:
        return self._build_function_class(regularizers_list, "direct.functionals", "reg")

    @abstractmethod
    def _do_iteration(
        self,
        data: Dict[str, torch.Tensor],
        loss_fns: Optional[Dict[str, Callable]] = None,
        regularizer_fns: Optional[Dict[str, Callable]] = None,
    ) -> DoIterationOutput:
        """This is a placeholder for the iteration function.

        This needs to perform the backward pass. If using mixed-precision you need to implement `autocast` as well in
        this function. It is recommended you raise an error if `self.mixed_precision` is true but mixed precision is not
        available.
        """

    @torch.no_grad()
    def predict(
        self,
        dataset: Dataset,
        experiment_directory: pathlib.Path,
        checkpoint: Union[int, str, pathlib.Path, None] = -1,
        num_workers: int = 6,
        batch_size: int = 1,
        crop: Optional[str] = None,
    ) -> List[np.ndarray]:
        self.logger.info("Predicting...")
        torch.cuda.empty_cache()
        self.ndim = dataset.ndim  # type: ignore
        self.logger.info("Data dimensionality: %s.", self.ndim)

        self.checkpointer = Checkpointer(
            save_directory=experiment_directory, save_to_disk=False, model=self.model, **self.models  # type: ignore
        )
        # If integer, latest or None
        if isinstance(checkpoint, int) or checkpoint == "latest" or checkpoint is None:
            # Do not load again if we already have loaded the checkpoint.
            if self.checkpointer.checkpoint_loaded is not checkpoint:
                self.checkpointer.load(iteration=checkpoint, checkpointable_objects=None)
        # Otherwise it's a path or a url
        else:
            self.checkpointer.load_models_from_file(checkpoint)

        batch_sampler = self.build_batch_sampler(
            dataset,
            batch_size=batch_size,
            sampler_type="sequential",
            limit_number_of_volumes=None,
        )
        # TODO: Batch size can be much larger, perhaps have a different batch size during evaluation.
        data_loader = self.build_loader(dataset, batch_sampler=batch_sampler, num_workers=num_workers)
        output = list(self.reconstruct_volumes(data_loader, add_target=False, crop=crop))

        return output

    @staticmethod
    def build_loader(
        dataset: Dataset,
        batch_sampler: Optional[Sampler] = None,
        num_workers: int = 6,
    ) -> DataLoader:
        # TODO(jt): Custom memory pinning.
        loader = DataLoader(
            dataset=dataset,
            sampler=None,
            batch_size=1,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            drop_last=False,
            shuffle=False,
            pin_memory=False,  # This can do strange things, and needs a custom implementation.
            # prefetch_factor=1,
            # persistent_workers=True,
        )
        return loader

    @staticmethod
    def build_batch_sampler(
        dataset: Union[Dataset, List[Dataset]],
        batch_size: int,
        sampler_type: str,
        **kwargs,
    ) -> Sampler:
        if sampler_type == "random":
            if not isinstance(dataset, List) or any(not isinstance(_, Dataset) for _ in dataset):
                raise ValueError("Random sampler requires a list of datasets as input.")
            batch_sampler = ConcatDatasetBatchSampler(datasets=dataset, batch_size=batch_size)
        elif sampler_type == "sequential":
            sampler = src.utils.direct.data.samplers.DistributedSequentialSampler(dataset, **kwargs)
            batch_sampler = src.utils.direct.data.samplers.BatchVolumeSampler(
                sampler,
                batch_size=batch_size,
            )
        else:
            raise ValueError(f"Sampler type {sampler_type} not supported.")

        return batch_sampler

    def training_loop(
        self,
        training_datasets: List,  # TODO(jt): Improve typing
        start_iter: int,
        validation_datasets: Optional[List] = None,
        experiment_directory: Optional[pathlib.Path] = None,
        num_workers: int = 6,
        start_with_validation: bool = False,
    ):
        self.logger.info(f"Local rank: {communication.get_local_rank()}.")
        self.models_training_mode()

        loss_fns = self.build_loss()
        metric_fns = self.build_metrics(self.cfg.training.metrics)  # type: ignore
        regularizer_fns = self.build_regularizers(self.cfg.training.regularizers)  # type: ignore
        storage = get_event_storage()

        self.ndim = training_datasets[0].ndim
        self.logger.info("Data dimensionality: %s.", self.ndim)

        try:
            training_data = ConcatDataset(training_datasets)
            if len(training_data) <= 0:
                raise AssertionError("No training data available")
        except AssertionError as err:
            self.logger.info("%s: Terminating training...", err)
            sys.exit(-1)

        self.logger.info("Concatenated dataset length: %s.", str(len(training_data)))
        self.logger.info(
            "Building batch sampler for training set with batch size %s.", self.cfg.training.batch_size  # type: ignore
        )

        training_sampler = self.build_batch_sampler(
            training_datasets,
            self.cfg.training.batch_size,  # type: ignore
            "random",
        )
        data_loader = self.build_loader(
            training_data,
            batch_sampler=training_sampler,
            num_workers=num_workers,
        )

        # Convenient shorthand
        validation_func = functools.partial(
            self.validation_loop,
            validation_datasets,
            None,
            experiment_directory,
            num_workers=num_workers,
        )

        total_iter = self.cfg.training.num_iterations  # type: ignore
        fail_counter = 0
        for data, iter_idx in zip(data_loader, range(start_iter, total_iter)):
            if iter_idx == 0:
                self.log_first_training_example_and_model(data)

            if start_with_validation and iter_idx == start_iter:
                self.logger.info(f"Starting with validation at iteration: {iter_idx}.")
                validation_func(iter_idx)
            try:
                iteration_output = self._do_iteration(data, loss_fns, regularizer_fns=regularizer_fns)
                loss_dict = iteration_output.data_dict
            except (ProcessKilledException, TrainingException) as e:
                # If the process is killed, the DoIterationOutput
                # if saved at state iter_idx, which is the current state,
                # so the computation can restart from the last iteration.
                self.logger.exception(f"Exiting with exception: {e}.")
                self.checkpoint_and_write_to_logs(iter_idx)
                sys.exit(-1)
            except RuntimeError as e:
                # Maybe string can change
                if "out of memory" in str(e):
                    if fail_counter == 3:
                        self.checkpoint_and_write_to_logs(iter_idx)
                        raise TrainingException(f"OOM, had three exceptions in a row tries: {e}.")
                    fail_counter += 1
                    self.logger.info(f"OOM Error: {e}. Skipping batch. Retry {fail_counter}/3.")
                    self.__optimizer.zero_grad()  # type: ignore
                    gc.collect()
                    torch.cuda.empty_cache()
                    continue

                self.checkpoint_and_write_to_logs(iter_idx)
                self.logger.info(f"Cannot recover from exception {e}. Exiting.")
                raise RuntimeError(e)

            if fail_counter > 0:
                self.logger.info("Recovered from OOM, skipped batch.")
            fail_counter = 0
            # Gradient accumulation
            if (iter_idx + 1) % self.cfg.training.gradient_steps == 0:  # type: ignore
                if self.cfg.training.gradient_steps > 1:  # type: ignore
                    for parameter in self.model.parameters():
                        if parameter.grad is not None:
                            # In-place division
                            parameter.grad.div_(self.cfg.training.gradient_steps)  # type: ignore
                if self.cfg.training.gradient_clipping > 0.0:  # type: ignore
                    self._scaler.unscale_(self.__optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.training.gradient_clipping  # type: ignore
                    )

                # Gradient norm
                if self.cfg.training.gradient_debug:  # type: ignore
                    warnings.warn(
                        "Gradient debug set. This will affect training performance. Only use for debugging."
                        "This message will only be displayed once."
                    )
                    parameters = list(filter(lambda p: p.grad is not None, self.model.parameters()))
                    gradient_norm = sum([parameter.grad.data**2 for parameter in parameters]).sqrt()  # type: ignore
                    storage.add_scalar("train/gradient_norm", gradient_norm)

                # Same as self.__optimizer.step() for mixed precision.
                self._scaler.step(self.__optimizer)
                # Updates the scale for next iteration.
                self._scaler.update()

            # TODO: Optimizer is only set in case of training, mypy inference does not seem to be correct.
            # Perhaps this has to be written differently, though. Related to #83
            self.__lr_scheduler.step()  # type: ignore # noqa
            storage.add_scalar("lr", self.__optimizer.param_groups[0]["lr"], smoothing_hint=False)  # type: ignore

            self.__optimizer.zero_grad()  # type: ignore

            # Reduce the loss over all devices
            loss_dict_reduced = communication.reduce_tensor_dict(loss_dict)
            loss_reduced = sum(loss_dict_reduced.values())

            storage.add_scalars(loss=loss_reduced, **loss_dict_reduced)
            # Maybe not needed.
            del data

            self.checkpoint_model_at_interval(iter_idx, total_iter)
            self.write_to_logs_at_interval(iter_idx, total_iter)
            self.validate_model_at_interval(validation_func, iter_idx, total_iter)

            storage.step()

    def validate_model_at_interval(self, func, iter_idx, total_iter):
        if iter_idx >= 5:  # No validation or anything needed
            if iter_idx % self.cfg.training.validation_steps == 0 or (iter_idx + 1) == total_iter:
                func(iter_idx)

    def checkpoint_model_at_interval(self, iter_idx, total_iter):
        if iter_idx >= 5:
            if iter_idx % self.cfg.training.checkpointer.checkpoint_steps == 0 or (iter_idx + 1) == total_iter:
                self.logger.info(f"Checkpointing at iteration {iter_idx}.")
                self.checkpointer.save(iter_idx)

    def write_to_logs_at_interval(self, iter_idx, total_iter):
        if iter_idx >= 5:
            # Log every 20 iterations, or at a validation step or at the end of training.
            if (
                iter_idx % 20 == 0
                or iter_idx % self.cfg.training.validation_steps == 0
                or (iter_idx + 1) == total_iter
            ):
                self.write_to_logs()

    def checkpoint_and_write_to_logs(self, iter_idx):
        if iter_idx >= 5:
            self.checkpointer.save(iter_idx)  # Save checkpoint at kill. # noqa
        self.write_to_logs()

    def validation_loop(
        self,
        validation_datasets,
        loss_fns,
        experiment_directory,
        iter_idx,
        num_workers: int = 6,
    ):
        if not validation_datasets:
            return

        storage = get_event_storage()

        for curr_validation_dataset in validation_datasets:
            curr_dataset_name = curr_validation_dataset.text_description
            self.logger.info("Evaluating: %s...", curr_dataset_name)
            self.logger.info("Building dataloader for dataset: %s.", curr_dataset_name)
            curr_batch_sampler = self.build_batch_sampler(
                curr_validation_dataset,
                batch_size=self.cfg.validation.batch_size,  # type: ignore
                sampler_type="sequential",
                limit_number_of_volumes=None,
            )
            curr_data_loader = self.build_loader(
                curr_validation_dataset,
                batch_sampler=curr_batch_sampler,
                num_workers=num_workers,
            )

            (
                curr_loss_dict,
                curr_metrics_per_case,
                visualize_slices,
                visualize_target,
            ) = self.evaluate(
                curr_data_loader,
                loss_fns,
            )

            if experiment_directory:
                json_output_fn = experiment_directory / f"metrics_val_{curr_dataset_name}_{iter_idx}.json"
                json_output_fn.parent.mkdir(exist_ok=True, parents=True)  # A / in the filename can create a folder
                if communication.is_main_process():
                    write_json(
                        json_output_fn,
                        curr_metrics_per_case,
                    )
                self.logger.info("Wrote per image logs to: %s.", str(json_output_fn))

            # Metric dict still needs to be reduced as it gives values *per* data
            curr_metric_dict = reduce_list_of_dicts(list(curr_metrics_per_case.values()), mode="average")

            key_prefix = "val/" if not curr_dataset_name else f"val/{curr_dataset_name}/"
            loss_reduced = sum(curr_loss_dict.values())
            storage.add_scalars(
                **{key_prefix + "loss": loss_reduced},
                **{
                    **prefix_dict_keys(curr_metric_dict, key_prefix),
                    **prefix_dict_keys(curr_loss_dict, key_prefix),
                },
                smoothing_hint=False,
            )
            visualize_slices = self.process_slices_for_visualization(visualize_slices, visualize_target)
            storage.add_image(f"{key_prefix}prediction", visualize_slices)

            if iter_idx // self.cfg.training.validation_steps - 1 == 0:  # type: ignore
                visualize_target = [normalize_image(image) for image in visualize_target]
                visualize_target = make_grid(
                    crop_to_largest(visualize_target, pad_value=0),
                    nrow=self.cfg.logging.tensorboard.num_images,  # type: ignore
                    scale_each=True,
                )
                storage.add_image(f"{key_prefix}target", visualize_target)

            self.logger.info("Done evaluation of %s at iteration %s.", str(curr_dataset_name), str(iter_idx))
        self.model.train()

    def process_slices_for_visualization(self, visualize_slices, visualize_target):
        # Log slices.
        # Compute the difference as well, and normalize for visualization
        difference_slices = [a - b for a, b in zip(visualize_slices, visualize_target)]
        # Normalize slices
        difference_slices = [(d / np.abs(d)) * 0.5 + 0.5 for d in difference_slices]
        visualize_slices = [normalize_image(image) for image in visualize_slices]

        # Visualize slices, and crop to the largest volume
        visualize_slices = make_grid(
            crop_to_largest(visualize_slices + difference_slices, pad_value=0),
            nrow=self.cfg.logging.tensorboard.num_images,
            scale_each=True,
        )
        return visualize_slices

    def models_training_mode(self):
        self.model.train()
        for curr_model in self.models:
            self.models[curr_model].train()

    def models_validation_mode(self):
        self.model.eval()
        for curr_model in self.models:
            self.models[curr_model].eval()

    def models_to_device(self):
        self.model = self.model.to(self.device)
        for curr_model_name in self.models:
            self.models[curr_model_name] = self.models[curr_model_name].to(self.device)

    def train(
        self,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler,  # noqa
        training_datasets: List[Dataset],
        experiment_directory: pathlib.Path,
        validation_datasets: Optional[Dataset] = None,
        resume: bool = False,
        start_with_validation: bool = False,
        initialization: Optional[PathOrString] = None,
        num_workers: int = 6,
    ) -> None:
        self.logger.info("Starting training.")
        # Can consider not to make this a member of self, but that requires that optimizer is passed to
        # training_loop()
        self.__optimizer = optimizer
        self.__lr_scheduler = lr_scheduler

        self.models_to_device()

        # Optimizer
        self.__optimizer.zero_grad()  # type: ignore

        # Mixed precision setup. This requires the model to be on the gpu.
        git_hash = src.utils.direct.utils.git_hash()
        checkpointing_metadata = {
            "__author__": git_hash if git_hash else "N/A",
            "__version__": src.utils.direct.__version__,
            "__mixed_precision__": self.mixed_precision,
        }
        if self.mixed_precision:
            # TODO(jt): Check if on GPU
            self.logger.info("Using mixed precision training.")

        self.checkpointer = Checkpointer(
            save_directory=experiment_directory,
            save_to_disk=False if not communication.is_main_process() else True,
            model=self.model,  # type: ignore
            optimizer=optimizer,  # type: ignore
            lr_scheduler=lr_scheduler,  # type: ignore
            scaler=self._scaler,  # type: ignore
            **checkpointing_metadata,  # type: ignore
            **self.models,  # type: ignore
        )

        # Load checkpoint
        start_iter = 0
        checkpoint = {}

        if resume:
            self.logger.info("Attempting to resume...")
            # This changes the model inplace
            checkpoint = self.checkpointer.load(iteration="latest")
            if not checkpoint:
                self.logger.info("No checkpoint found. Starting from scratch.")
            else:
                start_iter = checkpoint["iteration"] + 1
                self.logger.info(f"Starting from iteration: {start_iter}.")

        if start_iter > 0 and initialization:
            self.logger.warning(
                f"Initialization checkpoint set to {initialization},"
                f" but model will resume training from previous checkpoint. Initialization ignored."
            )
        elif initialization:
            self.logger.info(f"Initializing from {initialization}...")
            self.checkpointer.load_models_from_file(initialization)
            start_with_validation = True
            self.logger.info("Setting start_with_validation to True.")

        if "__version__" in checkpoint:
            self.logger.info(f"DIRECT version of checkpoint: {checkpoint['__version__']}.")
            if checkpoint["__version__"] != src.utils.direct.__version__:
                self.logger.warning(
                    f"Current DIRECT version {direct.__version__} is different from the one "
                    f"this checkpoint is saved with: {checkpoint['__version__']}. This can be fine, "
                    f"but beware that this can be a source of confusion."
                )

        if "__author__" in checkpoint:
            self.logger.info(f"Git hash of checkpoint: {checkpoint['__author__']}.")
            if checkpoint["__author__"] != src.utils.direct.utils.git_hash():
                self.logger.warning(
                    f"Current git hash {direct.utils.git_hash()} is different from the one "
                    f"this checkpoint is saved with: {checkpoint['__author__']}. This can be fine, "
                    f"but beware that this can be a source of confusion."
                )

        if "__datetime__" in checkpoint:
            self.logger.info("Checkpoint created at: %s.", checkpoint["__datetime__"])

        if "__mixed_precision__" in checkpoint:
            if (not self.mixed_precision) and checkpoint["__mixed_precision__"]:
                self.logger.warning(
                    "Mixed precision training is not enabled, yet saved checkpoint requests this. "
                    "Will now enable mixed precision."
                )
                self.mixed_precision = True
            elif not checkpoint["__mixed_precision__"] and self.mixed_precision:
                self.logger.warning(
                    "Mixed precision levels of training and loading checkpoint do not match. "
                    "Requested mixed precision but checkpoint is saved without. "
                    "This will almost surely lead to performance degradation."
                )

        if start_with_validation:
            self.logger.info("Requested to start with validation.")

        self.logger.info(f"World size: {communication.get_world_size()}.")
        self.logger.info(f"Device count: {torch.cuda.device_count()}.")
        if communication.get_world_size() > 1:
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[communication.get_local_rank()],
                broadcast_buffers=False,
                find_unused_parameters=False,
            )

        # World size > 1 if distributed mode, else allow a DataParallel fallback, can be convenient for debugging.
        elif torch.cuda.device_count() > 1 and communication.get_world_size() == 1:
            self.model = DataParallel(self.model)

        self.__writers = (
            [
                JSONWriter(experiment_directory / "metrics.json"),
                CommonMetricPrinter(self.cfg.training.num_iterations),  # type: ignore
                TensorboardWriter(experiment_directory / "tensorboard"),
            ]
            if communication.is_main_process()
            else []
        )

        with EventStorage(start_iter):
            self.training_loop(
                training_datasets,
                start_iter,
                validation_datasets,
                experiment_directory=experiment_directory,
                num_workers=num_workers,
                start_with_validation=start_with_validation,
            )

        self.logger.info("Training completed.")

    @abstractmethod
    def reconstruct_volumes(self, *args, **kwargs):  # noqa
        pass

    @abstractmethod
    def evaluate(self, *args, **kwargs):  # noqa
        pass

    def log_process(self, idx, total):
        if idx % (total // 10) == 0 or total == (idx + 1):
            self.logger.info(f"Progress: {(idx + 1) / total * 100:.2f}%.")

    def log_first_training_example_and_model(self, data):
        storage = get_event_storage()
        self.logger.info(f"First case: slice_no: {data['slice_no'][0]}, filename: {data['filename'][0]}.")

        # TODO(jt): Cleaner, loop over types of images
        first_sampling_mask = data["sampling_mask"][0][0]
        first_target = data["target"][0]

        if self.ndim == 3:
            first_sampling_mask = first_sampling_mask[0]
            slice_dim = -4
            num_slices = first_target.shape[slice_dim]
            first_target = first_target[num_slices // 2]
        elif self.ndim > 3:
            raise NotImplementedError

        storage.add_image("train/mask", first_sampling_mask[..., 0].unsqueeze(0))
        storage.add_image(
            "train/target",
            normalize_image(first_target.unsqueeze(0)),
        )

        if "initial_image" in data:
            storage.add_image(
                "train/initial_image",
                normalize_image(T.modulus(data["initial_image"][0]).unsqueeze(0)),
            )

        # TODO: Add graph

        self.write_to_logs()

    def write_to_logs(self):
        if self.__writers is not None:
            for writer in self.__writers:
                writer.write()

    def __bind_sigint_signal(self):
        """Bind SIGINT signal to handle preemption or other kill of the process."""

        # pylint: disable = E1101
        def raise_process_killed_error(signal_id, _):
            """Raise the ProcessKilledError."""
            self.logger.info(f"Received {signal.Signals(signal_id).name} Shutting down...")
            raise ProcessKilledException(signal_id, signal.Signals(signal_id).name)

        signal.signal(signalnum=signal.SIGINT, handler=raise_process_killed_error)
