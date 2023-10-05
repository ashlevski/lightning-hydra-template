# coding=utf-8
# Copyright (c) DIRECT Contributors

from typing import Callable, Dict, Optional

import torch
from torch import nn
from torch.cuda.amp import autocast

from direct.config import BaseConfig
from direct.engine import DoIterationOutput
from direct.nn.mri_models import MRIModelEngine
from direct.utils import detach_dict, dict_to_device, reduce_list_of_dicts


class CIRIMEngine(MRIModelEngine):
    """Cascades of Independently Recurrent Inference Machines Engine."""

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
        """Inits :class:`CIRIMEngine."""
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

    def _do_iteration(
        self,
        data: Dict[str, torch.Tensor],
        loss_fns: Optional[Dict[str, Callable]] = None,
        regularizer_fns: Optional[Dict[str, Callable]] = None,
    ) -> DoIterationOutput:
        # loss_fns can be done, e.g. during validation
        if loss_fns is None:
            loss_fns = {}

        if regularizer_fns is None:
            regularizer_fns = {}

        loss_dicts = []
        regularizer_dicts = []

        data = dict_to_device(data, self.device)

        # sensitivity_map of shape (batch, coil, height,  width, complex=2)
        sensitivity_map = data["sensitivity_map"].clone()
        data["sensitivity_map"] = self.compute_sensitivity_map(sensitivity_map)

        with autocast(enabled=self.mixed_precision):
            output_image = next(
                self.model(
                    masked_kspace=data["masked_kspace"],
                    sampling_mask=data["sampling_mask"],
                    sensitivity_map=data["sensitivity_map"],
                )
            )

            loss_dict = {k: torch.tensor([0.0], dtype=data["target"].dtype).to(self.device) for k in loss_fns.keys()}
            regularizer_dict = {
                k: torch.tensor([0.0], dtype=data["target"].dtype).to(self.device) for k in regularizer_fns.keys()
            }

            # Compute a weighting factor for the loss on each iteration
            iter_loss_weights = torch.logspace(-1, 0, steps=len(output_image[0])).to(output_image[0][0])
            # Iterate through the cascades
            for output_image_cascade in output_image:
                # Iterate through the iterations of the model
                for i, output_image_iter in enumerate(output_image_cascade):
                    for key, value in loss_dict.items():
                        loss_dict[key] = (
                            value
                            + loss_fns[key](
                                output_image_iter,
                                data["target"],
                                reduction="mean",
                                reconstruction_size=data.get("reconstruction_size", None),
                            )
                            * iter_loss_weights[i]
                        )

                    for key, value in regularizer_dict.items():
                        loss_dict[key] = (
                            value
                            + loss_fns[key](
                                output_image_iter,
                                data["target"],
                                reconstruction_size=data.get("reconstruction_size", None),
                            )
                            * iter_loss_weights[i]
                        )

            # Total length of the number of cascades and the number of iterations
            len_output_image = len(output_image) + len(output_image[0])

            loss_dict = {k: v / len_output_image for k, v in loss_dict.items()}
            regularizer_dict = {k: v / len_output_image for k, v in regularizer_dict.items()}

            loss = sum(loss_dict.values()) + sum(regularizer_dict.values())

        if self.model.training:
            self._scaler.scale(loss).backward()

        loss_dicts.append(detach_dict(loss_dict))
        regularizer_dicts.append(
            detach_dict(regularizer_dict)
        )  # Need to detach dict as this is only used for logging.

        # Add the loss dicts.
        loss_dict = reduce_list_of_dicts(loss_dicts, mode="sum")
        regularizer_dict = reduce_list_of_dicts(regularizer_dicts, mode="sum")

        return DoIterationOutput(
            output_image=output_image[-1][-1],  # Prediction of the last iteration of the last cascade
            sensitivity_map=data["sensitivity_map"],
            data_dict={**loss_dict, **regularizer_dict},
        )
