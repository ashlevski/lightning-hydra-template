# coding=utf-8
# Copyright (c) DIRECT Contributors

from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch import nn
from torch.cuda.amp import autocast

import direct.data.transforms as T
from direct.config import BaseConfig
from direct.engine import DoIterationOutput
from direct.nn.mri_models import MRIModelEngine
from direct.utils import detach_dict, dict_to_device, reduce_list_of_dicts


class MultiDomainNetEngine(MRIModelEngine):
    """Multi Domain Network Engine."""

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
        """Inits :class:`MultiDomainNetEngine."""
        super().__init__(
            cfg,
            model,
            device,
            forward_operator=forward_operator,
            backward_operator=backward_operator,
            mixed_precision=mixed_precision,
            **models,
        )

    def forward_function(self, data: Dict[str, Any]) -> Tuple[torch.Tensor, None]:
        output_multicoil_image = self.model(
            masked_kspace=data["masked_kspace"],
            sensitivity_map=data["sensitivity_map"],
        )
        output_image = T.root_sum_of_squares(
            output_multicoil_image, self._coil_dim, self._complex_dim
        )  # shape (batch, height,  width)

        output_kspace = None

        return output_image, output_kspace
