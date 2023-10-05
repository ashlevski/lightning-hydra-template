# coding=utf-8
# Copyright (c) DIRECT Contributors

"""MRI model engine of DIRECT."""

import gc
import pathlib
import time
from collections import defaultdict
from os import PathLike
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.cuda.amp import autocast
from torch.nn import functional as F
from torch.utils.data import DataLoader

import direct.data.transforms as T
from direct.config import BaseConfig
from direct.engine import DoIterationOutput, Engine
from direct.functionals import NMAELoss, NMSELoss, NRMSELoss, SobelGradL1Loss, SobelGradL2Loss, SSIMLoss
from direct.types import TensorOrNone
from direct.utils import (
    communication,
    detach_dict,
    dict_to_device,
    merge_list_of_dicts,
    multiply_function,
    reduce_list_of_dicts,
)
from direct.utils.communication import reduce_tensor_dict


class MRIModelEngine(Engine):
    """Engine for MRI models.

    Each child class should implement their own :meth:`forward_function`.
    """

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
        """Inits :class:`MRIModelEngine`.

        Parameters
        ----------
        cfg: BaseConfig
            Configuration file.
        model: nn.Module
            Model.
        device: str
            Device. Can be "cuda:{idx}" or "cpu".
        forward_operator: Callable, optional
            The forward operator. Default: None.
        backward_operator: Callable, optional
            The backward operator. Default: None.
        mixed_precision: bool
            Use mixed precision. Default: False.
        **models: nn.Module
            Additional models.
        """
        super().__init__(
            cfg,
            model,
            device,
            forward_operator=forward_operator,
            backward_operator=backward_operator,
            mixed_precision=mixed_precision,
            **models,
        )
        self._spatial_dims = (2, 3)
        self._coil_dim = 1
        self._complex_dim = -1

    def forward_function(self, data: Dict[str, Any]) -> Tuple[TensorOrNone, TensorOrNone]:
        """This method performs the model's forward method given `data` which contains all tensor inputs.

        Must be implemented by child classes.
        """
        raise NotImplementedError("Must be implemented by child class.")

    def _do_iteration(
        self,
        data: Dict[str, Any],
        loss_fns: Optional[Dict[str, Callable]] = None,
        regularizer_fns: Optional[Dict[str, Callable]] = None,
    ) -> DoIterationOutput:
        """Performs forward method and calculates loss functions.

        Parameters
        ----------
        data : Dict[str, Any]
            Data containing keys with values tensors such as k-space, image, sensitivity map, etc.
        loss_fns : Optional[Dict[str, Callable]]
            Callable loss functions.
        regularizer_fns : Optional[Dict[str, Callable]]
            Callable regularization functions.

        Returns
        -------
        DoIterationOutput
            Contains outputs.
        """

        # loss_fns can be None, e.g. during validation
        if loss_fns is None:
            loss_fns = {}

        if regularizer_fns is None:
            regularizer_fns = {}

        data = dict_to_device(data, self.device)

        output_image: TensorOrNone
        output_kspace: TensorOrNone

        with autocast(enabled=self.mixed_precision):
            data["sensitivity_map"] = self.compute_sensitivity_map(data["sensitivity_map"])

            output_image, output_kspace = self.forward_function(data)
            output_image = T.modulus_if_complex(output_image, complex_axis=self._complex_dim)

            loss_dict = {k: torch.tensor([0.0], dtype=data["target"].dtype).to(self.device) for k in loss_fns.keys()}
            regularizer_dict = {
                k: torch.tensor([0.0], dtype=data["target"].dtype).to(self.device) for k in regularizer_fns.keys()
            }
            loss_dict = self.compute_loss_on_data(loss_dict, loss_fns, data, output_image, output_kspace)
            regularizer_dict = self.compute_loss_on_data(
                regularizer_dict, regularizer_fns, data, output_image, output_kspace
            )

            loss = sum(loss_dict.values()) + sum(regularizer_dict.values())  # type: ignore

        if self.model.training:
            self._scaler.scale(loss).backward()

        loss_dict = detach_dict(loss_dict)  # Detach dict, only used for logging.
        regularizer_dict = detach_dict(regularizer_dict)

        return DoIterationOutput(
            output_image=output_image,
            sensitivity_map=data["sensitivity_map"],
            data_dict={**loss_dict, **regularizer_dict},
        )

    def build_loss(self) -> Dict:
        def get_resolution(reconstruction_size):
            return _compute_resolution(self.cfg.training.loss.crop, reconstruction_size)  # type: ignore

        def nmae_loss(
            source: torch.Tensor,
            target: torch.Tensor,
            reduction: str = "mean",
            reconstruction_size: Optional[Tuple] = None,
        ) -> torch.Tensor:
            """Calculate NMAE loss given source and target.

            Parameters
            ----------
            source: torch.Tensor
                Has shape (batch, *).
            target: torch.Tensor
                Has shape (batch, *).
            reduction: str
                Reduction type. Can be "sum" or "mean".

            Returns
            -------
            nmae_loss: torch.Tensor
                NMAE loss.
            """
            if reconstruction_size is not None:
                resolution = get_resolution(reconstruction_size)
                source, target = _crop_volume(source, target, resolution)

            nmae_loss = NMAELoss(reduction=reduction).forward(source, target)

            return nmae_loss

        def nmse_loss(
            source: torch.Tensor,
            target: torch.Tensor,
            reduction: str = "mean",
            reconstruction_size: Optional[Tuple] = None,
        ) -> torch.Tensor:
            """Calculate NMSE loss given source and target.

            Parameters
            ----------
            source: torch.Tensor
                Has shape (batch, *).
            target: torch.Tensor
                Has shape (batch, *).
            reduction: str
                Reduction type. Can be "sum" or "mean".

            Returns
            -------
            nmse_loss: torch.Tensor
                NMSE loss.
            """
            if reconstruction_size is not None:
                resolution = get_resolution(reconstruction_size)
                source, target = _crop_volume(source, target, resolution)
            nmse_loss = NMSELoss(reduction=reduction).forward(source, target)

            return nmse_loss

        def nrmse_loss(
            source: torch.Tensor,
            target: torch.Tensor,
            reduction: str = "mean",
            reconstruction_size: Optional[Tuple] = None,
        ) -> torch.Tensor:
            """Calculate NRMSE loss given source and target.

            Parameters
            ----------
            source: torch.Tensor
                Has shape (batch, *).
            target: torch.Tensor
                Has shape (batch, *).
            reduction: str
                Reduction type. Can be "sum" or "mean".

            Returns
            -------
            nrmse_loss: torch.Tensor
                NRMSE loss.
            """
            if reconstruction_size is not None:
                resolution = get_resolution(reconstruction_size)
                source, target = _crop_volume(source, target, resolution)
            nrmse_loss = NRMSELoss(reduction=reduction).forward(source, target)

            return nrmse_loss

        def l1_loss(
            source: torch.Tensor,
            target: torch.Tensor,
            reduction: str = "mean",
            reconstruction_size: Optional[Tuple] = None,
        ) -> torch.Tensor:
            """Calculate L1 loss given source image and target.

            Parameters
            ----------
            source: torch.Tensor
                Source tensor of shape (batch, *).
            target: torch.Tensor
                Target tensor of shape (batch, *).
            reduction: str
                Reduction type. Can be "sum" or "mean".
            reconstruction_size: Optional[Tuple]
                Reconstruction size to center crop. Default: None.

            Returns
            -------
            l1_loss: torch.Tensor
                L1 loss.
            """
            if reconstruction_size is not None:
                resolution = get_resolution(reconstruction_size)
                source, target = _crop_volume(source, target, resolution)
            l1_loss = F.l1_loss(source, target, reduction=reduction)

            return l1_loss

        def l2_loss(
            source: torch.Tensor,
            target: torch.Tensor,
            reduction: str = "mean",
            reconstruction_size: Optional[Tuple] = None,
        ) -> torch.Tensor:
            """Calculate L2 loss (MSE) given source image and and `data` containing target.

            Parameters
            ----------
            source: torch.Tensor
                Source tensor of shape (batch, *).
            target: torch.Tensor
                Target tensor of shape (batch, *).
            reduction: str
                Reduction type. Can be "sum" or "mean".
            reconstruction_size: Optional[Tuple]
                Reconstruction size to center crop. Default: None.

            Returns
            -------
            l2_loss: torch.Tensor
                L2 loss.
            """
            if reconstruction_size is not None:
                resolution = get_resolution(reconstruction_size)
                source, target = _crop_volume(source, target, resolution)
            l2_loss = F.mse_loss(source, target, reduction=reduction)

            return l2_loss

        def ssim_loss(
            source: torch.Tensor,
            target: torch.Tensor,
            reduction: str = "mean",
            reconstruction_size: Optional[Tuple] = None,
        ) -> torch.Tensor:
            """Calculate SSIM loss given source image and target image.

            Parameters
            ----------
            source: torch.Tensor
                Source tensor of shape (batch, height, width, [complex=2]).
            target: torch.Tensor
                Target tensor of shape (batch, height, width, [complex=2]).
            reduction: str
                Reduction type. Can be "sum" or "mean".
            reconstruction_size: Optional[Tuple]
                Reconstruction size to center crop. Default: None.

            Returns
            -------
            ssim_loss: torch.Tensor
                SSIM loss.
            """
            resolution = get_resolution(reconstruction_size)
            if reduction != "mean":
                raise AssertionError(
                    f"SSIM loss can only be computed with reduction == 'mean'." f" Got reduction == {reduction}."
                )

            source_abs, target_abs = _crop_volume(source, target, resolution)
            data_range = torch.tensor([target_abs.max()], device=target_abs.device)

            ssim_loss = SSIMLoss().to(source_abs.device).forward(source_abs, target_abs, data_range=data_range)

            return ssim_loss

        def grad_l1_loss(
            source: torch.Tensor,
            target: torch.Tensor,
            reduction: str = "mean",
            reconstruction_size: Optional[Tuple] = None,
        ) -> torch.Tensor:
            """Calculate Sobel gradient L1 loss given source image and target image.

            Parameters
            ----------
            source: torch.Tensor
                Source tensor of shape (batch, height, width, [complex=2]).
            target: torch.Tensor
                Target tensor of shape (batch, height, width, [complex=2]).
            reduction: str
                Reduction type. Can be "sum" or "mean".
            reconstruction_size: Optional[Tuple]
                Reconstruction size to center crop. Default: None.

            Returns
            -------
            grad_loss: torch.Tensor
                Sobel grad L1 loss.
            """
            resolution = get_resolution(reconstruction_size)
            source_abs, target_abs = _crop_volume(source, target, resolution)
            grad_l1_loss = SobelGradL1Loss(reduction).to(source_abs.device).forward(source_abs, target_abs)

            return grad_l1_loss

        def grad_l2_loss(
            source: torch.Tensor,
            target: torch.Tensor,
            reduction: str = "mean",
            reconstruction_size: Optional[Tuple] = None,
        ) -> torch.Tensor:
            """Calculate Sobel gradient L2 loss given source image and target image.

            Parameters
            ----------
            source: torch.Tensor
                Source tensor of shape (batch, height, width, [complex=2]).
            target: torch.Tensor
                Target tensor of shape (batch, height, width, [complex=2]).
            reduction: str
                Reduction type. Can be "sum" or "mean".
            reconstruction_size: Optional[Tuple]
                Reconstruction size to center crop. Default: None.

            Returns
            -------
            grad_loss: torch.Tensor
                Sobel grad L1 loss.
            """
            resolution = get_resolution(reconstruction_size)
            source_abs, target_abs = _crop_volume(source, target, resolution)
            grad_l2_loss = SobelGradL2Loss(reduction).to(source_abs.device).forward(source_abs, target_abs)

            return grad_l2_loss

        # Build losses
        loss_dict = {}
        for curr_loss in self.cfg.training.loss.losses:  # type: ignore
            loss_fn = curr_loss.function
            if loss_fn in ["l1_loss", "kspace_l1_loss"]:
                loss_dict[loss_fn] = multiply_function(curr_loss.multiplier, l1_loss)
            elif loss_fn in ["l2_loss", "kspace_l2_loss"]:
                loss_dict[loss_fn] = multiply_function(curr_loss.multiplier, l2_loss)
            elif loss_fn == "ssim_loss":
                loss_dict[loss_fn] = multiply_function(curr_loss.multiplier, ssim_loss)
            elif loss_fn == "grad_l1_loss":
                loss_dict[loss_fn] = multiply_function(curr_loss.multiplier, grad_l1_loss)
            elif loss_fn == "grad_l2_loss":
                loss_dict[loss_fn] = multiply_function(curr_loss.multiplier, grad_l2_loss)
            elif loss_fn in ["nmse_loss", "kspace_nmse_loss"]:
                loss_dict[loss_fn] = multiply_function(curr_loss.multiplier, nmse_loss)
            elif loss_fn in ["nrmse_loss", "kspace_nrmse_loss"]:
                loss_dict[loss_fn] = multiply_function(curr_loss.multiplier, nrmse_loss)
            elif loss_fn in ["nmae_loss", "kspace_nmae_loss"]:
                loss_dict[loss_fn] = multiply_function(curr_loss.multiplier, nmae_loss)
            else:
                raise ValueError(f"{loss_fn} not permissible.")

        return loss_dict

    def compute_sensitivity_map(self, sensitivity_map: torch.Tensor) -> torch.Tensor:
        r"""Computes sensitivity maps :math:`\{S^k\}_{k=1}^{n_c}` if `sensitivity_model` is available.

        :math:`\{S^k\}_{k=1}^{n_c}` are normalized such that

        .. math::
            \sum_{k=1}^{n_c}S^k {S^k}^* = I.

        Parameters
        ----------
        sensitivity_map: torch.Tensor
            Sensitivity maps of shape (batch, coil, height,  width, complex=2).

        Returns
        -------
        sensitivity_map: torch.Tensor
            Normalized and refined sensitivity maps of shape (batch, coil, height,  width, complex=2).
        """
        # Some things can be done with the sensitivity map here, e.g. apply a u-net
        if "sensitivity_model" in self.models:
            # Move channels to first axis
            sensitivity_map = sensitivity_map.permute(
                (0, 1, 4, 2, 3)
            )  # shape (batch, coil, complex=2, height,  width)

            sensitivity_map = self.compute_model_per_coil("sensitivity_model", sensitivity_map).permute(
                (0, 1, 3, 4, 2)
            )  # has channel last: shape (batch, coil, height,  width, complex=2)

        # The sensitivity map needs to be normalized such that
        # So \sum_{i \in \text{coils}} S_i S_i^* = 1

        sensitivity_map_norm = torch.sqrt(
            ((sensitivity_map**2).sum(self._complex_dim)).sum(self._coil_dim)
        )  # shape (batch, height, width)
        sensitivity_map_norm = sensitivity_map_norm.unsqueeze(self._coil_dim).unsqueeze(self._complex_dim)

        return T.safe_divide(sensitivity_map, sensitivity_map_norm)

    @torch.no_grad()
    def reconstruct_volumes(  # type: ignore
        self,
        data_loader: DataLoader,
        loss_fns: Optional[Dict[str, Callable]] = None,
        regularizer_fns: Optional[Dict[str, Callable]] = None,
        add_target: bool = True,
        crop: Optional[str] = None,
    ):
        """Validation process. Assumes that each batch only contains slices of the same volume *AND* that these are
        sequentially ordered.

        Parameters
        ----------
        data_loader: DataLoader
        loss_fns: Dict[str, Callable], optional
        regularizer_fns: Dict[str, Callable], optional
        add_target: bool
            If true, will add the target to the output
        crop: str, optional
            Crop type.

        Yields
        ------
        (curr_volume, [curr_target,] loss_dict_list, filename): torch.Tensor, [torch.Tensor,], dict, pathlib.Path
        """
        # pylint: disable=too-many-locals, arguments-differ
        self.models_to_device()
        self.models_validation_mode()
        torch.cuda.empty_cache()

        # Let us inspect this data
        all_filenames = list(data_loader.dataset.volume_indices.keys())  # type: ignore
        num_for_this_process = len(list(data_loader.batch_sampler.sampler.volume_indices.keys()))  # type: ignore
        self.logger.info(
            "Reconstructing a total of %s volumes. This process has %s volumes (world size: %s).",
            len(all_filenames),
            num_for_this_process,
            communication.get_world_size(),
        )

        last_filename = None  # At the start of evaluation, there are no filenames.
        curr_volume = None
        curr_target = None
        slice_counter = 0
        filenames_seen = 0

        # Loop over dataset. This requires the use of direct.data.sampler.DistributedSequentialSampler as this sampler
        # splits the data over the different processes, and outputs the slices linearly. The implicit assumption here is
        # that the slices are outputted from the Dataset *sequentially* for each volume one by one, and each batch only
        # contains data from one volume.
        time_start = time.time()
        loss_dict_list = []
        # TODO: Use iter_idx to keep track of volume
        for _, data in enumerate(data_loader):
            torch.cuda.empty_cache()
            gc.collect()
            filename = _get_filename_from_batch(data)
            if last_filename is None:
                last_filename = filename  # First iteration last_filename is not set.
            if last_filename != filename:
                curr_volume = None
                curr_target = None
                slice_counter = 0
                last_filename = filename

            scaling_factors = data["scaling_factor"].clone()
            resolution = _compute_resolution(
                key=crop,
                reconstruction_size=data.get("reconstruction_size", None),
            )
            # Compute output
            iteration_output = self._do_iteration(data, loss_fns=loss_fns, regularizer_fns=regularizer_fns)
            output = iteration_output.output_image
            loss_dict = iteration_output.data_dict

            # Output can be complex-valued, and has to be cropped. This holds for both output and target.
            output_abs = _process_output(
                output,
                scaling_factors,
                resolution=resolution,
                complex_axis=self._complex_dim,
            )

            if add_target:
                target_abs = _process_output(
                    data["target"],
                    scaling_factors,
                    resolution=resolution,
                    complex_axis=self._complex_dim,
                )

            if curr_volume is None:
                volume_size = len(data_loader.batch_sampler.sampler.volume_indices[filename])  # type: ignore
                curr_volume = torch.zeros(*(volume_size, *output_abs.shape[1:]), dtype=output_abs.dtype)
                loss_dict_list.append(loss_dict)
                if add_target:
                    curr_target = curr_volume.clone()

            curr_volume[slice_counter : slice_counter + output_abs.shape[0], ...] = output_abs.cpu()
            if add_target:
                curr_target[slice_counter : slice_counter + output_abs.shape[0], ...] = target_abs.cpu()  # type: ignore

            slice_counter += output_abs.shape[0]

            # Check if we had the last batch
            if slice_counter == volume_size:
                filenames_seen += 1

                self.logger.info(
                    "%i of %i volumes reconstructed: %s (shape = %s) in %.3fs.",
                    filenames_seen,
                    num_for_this_process,
                    last_filename,
                    list(curr_volume.shape),
                    time.time() - time_start,
                )
                # Maybe not needed.
                del data
                yield (curr_volume, curr_target, reduce_list_of_dicts(loss_dict_list), filename) if add_target else (
                    curr_volume,
                    reduce_list_of_dicts(loss_dict_list),
                    filename,
                )

    @torch.no_grad()
    def evaluate(  # type: ignore
        self,
        data_loader: DataLoader,
        loss_fns: Optional[Dict[str, Callable]],
    ):
        """Validation process.

        Assumes that each batch only contains slices of the same volume *AND* that these are sequentially ordered.

        Parameters
        ----------
        data_loader: DataLoader
        loss_fns: Dict[str, Callable], optional

        Returns
        -------
        loss_dict, all_gathered_metrics, visualize_slices, visualize_target
        """
        # TODO(jt): visualization should be a namedtuple or a dict or so
        # TODO(gy): Implement visualization of extra keys. E.g. sensitivity_map.
        # pylint: disable=arguments-differ, too-many-locals

        self.models_to_device()
        self.models_validation_mode()
        torch.cuda.empty_cache()

        volume_metrics = self.build_metrics(self.cfg.validation.metrics)  # type: ignore
        val_losses = []
        val_volume_metrics: Dict[PathLike, Dict] = defaultdict(dict)

        # Container to for the slices which can be visualized in TensorBoard.
        visualize_slices: List[np.ndarray] = []
        visualize_target: List[np.ndarray] = []

        for _, output in enumerate(
            self.reconstruct_volumes(
                data_loader, loss_fns=loss_fns, add_target=True, crop=self.cfg.validation.crop  # type: ignore
            )
        ):
            volume, target, volume_loss_dict, filename = output
            curr_metrics = {
                metric_name: metric_fn(target, volume).clone() for metric_name, metric_fn in volume_metrics.items()
            }
            curr_metrics_string = ", ".join([f"{x}: {float(y)}" for x, y in curr_metrics.items()])
            self.logger.info("Metrics for %s: %s", filename, curr_metrics_string)
            # TODO: Path can be tricky if it is not unique (e.g. image.h5)
            val_volume_metrics[filename.name] = curr_metrics
            val_losses.append(volume_loss_dict)

            # Log the center slice of the volume
            if len(visualize_slices) < self.cfg.logging.tensorboard.num_images:  # type: ignore
                visualize_slices.append(volume[volume.shape[0] // 2])
                visualize_target.append(target[target.shape[0] // 2])

        # Average loss dict
        loss_dict = reduce_list_of_dicts(val_losses)
        reduce_tensor_dict(loss_dict)

        communication.synchronize()
        torch.cuda.empty_cache()

        # TODO: Does not work yet with normal gather.
        all_gathered_metrics = merge_list_of_dicts(communication.all_gather(val_volume_metrics))
        return loss_dict, all_gathered_metrics, visualize_slices, visualize_target

    def compute_model_per_coil(self, model_name: str, data: torch.Tensor) -> torch.Tensor:
        """Performs forward pass of model `model_name` in `self.models` per coil.

        Parameters
        ----------
        model_name: str
            Model to run.
        data: torch.Tensor
            Multi-coil data of shape (batch, coil, complex=2, height, width).

        Returns
        -------
        output: torch.Tensor
            Computed output per coil.
        """
        output = []
        for idx in range(data.size(self._coil_dim)):
            subselected_data = data.select(self._coil_dim, idx)
            output.append(self.models[model_name](subselected_data))

        return torch.stack(output, dim=self._coil_dim)

    def compute_loss_on_data(
        self,
        loss_dict: Dict[str, torch.Tensor],
        loss_fns: Dict[str, Callable],
        data: Dict[str, Any],
        output_image: Optional[torch.Tensor] = None,
        output_kspace: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if output_image is None and output_kspace is None:
            raise ValueError("Inputs for `output_image` and `output_kspace` cannot be both None.")
        for key, value in loss_dict.items():
            if "kspace" in key:
                if output_kspace is not None:
                    output, target, reconstruction_size = output_kspace, data["kspace"], None
                else:
                    raise ValueError(f"Requested to compute `{key}` loss but received None for `output_kspace`.")
            else:
                if output_image is not None:
                    output, target, reconstruction_size = (
                        output_image,
                        data["target"],
                        data.get("reconstruction_size", None),
                    )
                else:
                    raise ValueError(f"Requested to compute `{key}` loss but received None for `output_image`.")
            loss_dict[key] = value + loss_fns[key](output, target, "mean", reconstruction_size)
        return loss_dict

    def _forward_operator(self, image, sensitivity_map, sampling_mask):
        return T.apply_mask(
            self.forward_operator(
                T.expand_operator(image, sensitivity_map, dim=self._coil_dim),
                dim=self._spatial_dims,
            ),
            sampling_mask,
            return_mask=False,
        )

    def _backward_operator(self, kspace, sensitivity_map, sampling_mask):
        return T.reduce_operator(
            self.backward_operator(T.apply_mask(kspace, sampling_mask, return_mask=False), dim=self._spatial_dims),
            sensitivity_map,
            dim=self._coil_dim,
        )


def _crop_volume(
    source: torch.Tensor, target: torch.Tensor, resolution: Union[List[int], Tuple[int, ...]]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """2D source/target cropper.

    Parameters
    ----------
    source: torch.Tensor
        Has shape (batch, height, width)
    target: torch.Tensor
        Has shape (batch, height, width)
    resolution: list of ints or tuple of ints
        Target resolution.

    Returns
    -------
    (torch.Tensor, torch.Tensor)
    """

    if not resolution or all(_ == 0 for _ in resolution):
        return source.unsqueeze(1), target.unsqueeze(1)  # Added channel dimension.

    source_abs = T.center_crop(source, resolution).unsqueeze(1)  # Added channel dimension.
    target_abs = T.center_crop(target, resolution).unsqueeze(1)  # Added channel dimension.

    return source_abs, target_abs


def _process_output(
    data: torch.Tensor,
    scaling_factors: Optional[torch.Tensor] = None,
    resolution: Optional[Union[List[int], Tuple[int]]] = None,
    complex_axis: Optional[int] = -1,
) -> torch.Tensor:
    """Crops and scales input tensor.

    Parameters
    ----------
    data: torch.Tensor
    scaling_factors: Optional[torch.Tensor]
        Scaling factor. Default: None.
    resolution: Optional[Union[List[int], Tuple[int]]]
        Resolution. Default: None.
    complex_axis: Optional[int]
        Dimension along which modulus of `data` will be computed (if it's complex). Default: -1 (last).

    Returns
    -------
    torch.Tensor
    """
    # data is of shape (batch, complex=2, height, width)
    if scaling_factors is not None:
        data = data * scaling_factors.view(-1, *((1,) * (len(data.shape) - 1))).to(data.device)

    data = T.modulus_if_complex(data, complex_axis=complex_axis)

    if len(data.shape) == 3:  # (batch, height, width)
        data = data.unsqueeze(1)  # Added channel dimension.

    if resolution is not None:
        data = T.center_crop(data, resolution).contiguous()

    return data


def _compute_resolution(
    key: Optional[str], reconstruction_size: Optional[Union[List[int], Tuple[int]]] = None
) -> Union[List[int], None]:
    """Computes resolution.

    Parameters
    ----------
    key: str
        Can be `header` or None.
    reconstruction_size: Optional[Union[List[int], Tuple[int]]]
        Reconstruction size. Default: None.

    Returns
    -------
    resolution: Union[str, List[int], None]
        Resolution of reconstruction.
    """

    if key == "header":
        # This will be of the form [tensor(x_0, x_1, ...), tensor(y_0, y_1,...), tensor(z_0, z_1, ...)] over
        # batches.
        resolution = [_.detach().cpu().numpy().tolist() for _ in reconstruction_size]  # type: ignore
        # The volume sampler should give validation indices belonging to the *same* volume, so it should be
        # safe taking the first element, the matrix size are in x,y,z (we work in z,x,y).
        resolution = [_[0] for _ in resolution][:-1]
        return resolution
    elif not key:
        return None
    else:
        raise ValueError("Cropping should be either set to `header` to get the values from the header or None.")


def _get_filename_from_batch(data: dict) -> pathlib.Path:
    filenames = data["filename"]
    if len(set(filenames)) != 1:
        raise ValueError(
            f"Expected a batch during validation to only contain filenames of one case. " f"Got {set(filenames)}."
        )
    # This can be fixed when there is a custom collate_fn
    return pathlib.Path(filenames[0])
