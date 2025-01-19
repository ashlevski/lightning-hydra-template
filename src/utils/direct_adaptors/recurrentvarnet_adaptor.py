import functools
from typing import Tuple, Any, Dict, Callable, Optional

import torch
from torch import nn

from src.utils.direct.constants import COMPLEX_SIZE
from src.utils.direct.data.mri_transforms import EstimateSensitivityMapModule
from src.utils.direct.data.transforms import fft2, ifft2
from src.utils.direct.nn.recurrentvarnet.recurrentvarnet import RecurrentVarNet
import src.utils.direct.data.transforms as T
from src.utils.direct.nn.types import InitType


class RecurrentVarNetAdaptor(nn.Module):
    """Recurrent Variational Network Adaptor. adopted from direct tk"""

    def __init__(
        self,
        in_channels: int = COMPLEX_SIZE,
        num_steps: int = 15,
        recurrent_hidden_channels: int = 64,
        recurrent_num_layers: int = 4,
        no_parameter_sharing: bool = True,
        learned_initializer: bool = False,
        initializer_initialization: Optional[InitType] = None,
        initializer_channels: Optional[Tuple[int, ...]] = (32, 32, 64, 64),
        initializer_dilations: Optional[Tuple[int, ...]] = (1, 1, 2, 4),
        initializer_multiscale: int = 1,
        normalized: bool = False,
        sensitivity_model: nn.Module = None,
    ):
        super().__init__()
        self.sensitivity_model = sensitivity_model
        self.forward_operator = functools.partial(fft2, centered=True)
        self.backward_operator = functools.partial(ifft2, centered=True)
        """Inits :class:`RecurrentVarNetEngine."""
        self.model = RecurrentVarNet(
            forward_operator = self.forward_operator,
            backward_operator = self.backward_operator,
            in_channels = in_channels,
            num_steps = num_steps,
            recurrent_hidden_channels = recurrent_hidden_channels,
            recurrent_num_layers = recurrent_num_layers,
            no_parameter_sharing = no_parameter_sharing,
            learned_initializer = learned_initializer,
            initializer_initialization = initializer_initialization,
            initializer_channels = initializer_channels,
            initializer_dilations = initializer_dilations,
            initializer_multiscale = initializer_multiscale,
            normalized = normalized,
        )
        self._complex_dim = -1
        self.sens_model = EstimateSensitivityMapModule()

    def forward(self, data: Tuple) -> Tuple[torch.Tensor, torch.Tensor]:
        # data is tuple of #1 masked_kspace #2 sampling_mask #3 sensitivity_map
        data = self.compute_sensitivity(data)
        output_kspace = self.model(
            masked_kspace=data[list(data)[0]],
            sampling_mask=data[list(data)[1]],
            sensitivity_map=self.compute_sensitivity_map(data[list(data)[2]]),
        )
        # output_kspace = T.apply_padding(output_kspace, data.get("padding", None))

        output_image = T.root_sum_of_squares(
            self.backward_operator(output_kspace, dim=self.model._spatial_dims, centered=True),
            dim=self.model._coil_dim,
        )  # shape (batch, height,  width)

        return output_image, output_kspace

    def compute_sensitivity(self,data):
        with torch.no_grad():
            input = {}
            input[self.sens_model.kspace_key] = torch.view_as_real(data[0]).type(dtype=torch.float32)
            input["acs_mask"] = data[1].unsqueeze(-1).type(dtype=torch.float32)
            input["sensitivity_map"] = torch.view_as_real(data[2]).type(dtype=torch.float32)
            return self.sens_model(input)
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
        # ToDo: add transform per coil
        # Some things can be done with the sensitivity map here, e.g. apply a u-net
        if True:#"sensitivity_model" in self.models:
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
            ((sensitivity_map**2).sum(self._complex_dim)).sum(self.model._coil_dim)
        )  # shape (batch, height, width)
        sensitivity_map_norm = sensitivity_map_norm.unsqueeze(self.model._coil_dim).unsqueeze(self._complex_dim)

        return T.safe_divide(sensitivity_map, sensitivity_map_norm)

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
        for idx in range(data.size(self.model._coil_dim)):
            subselected_data = data.select(self.model._coil_dim, idx)
            output.append(self.sensitivity_model(subselected_data))

        return torch.stack(output, dim=self.model._coil_dim)