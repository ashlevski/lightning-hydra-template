# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any

import torch
from torch import Tensor, tensor

from torchmetrics.functional.image.vif import _vif_per_channel
from torchmetrics.metric import Metric


class VisualInformationFidelity(Metric):
    """Compute Pixel Based Visual Information Fidelity (VIF_).

    As input to ``forward`` and ``update`` the metric accepts the following input

    - ``preds`` (:class:`~torch.Tensor`): Predictions from model of shape ``(N,C,H,W)`` with H,W ≥ 41
    - ``target`` (:class:`~torch.Tensor`): Ground truth values of shape ``(N,C,H,W)`` with H,W ≥ 41

    As output of `forward` and `compute` the metric returns the following output

    - ``vif-p`` (:class:`~torch.Tensor`): Tensor with vif-p score

    Args:
        sigma_n_sq: variance of the visual noise
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> import torch
        >>> _ = torch.manual_seed(42)
        >>> from torchmetrics.image import VisualInformationFidelity
        >>> preds = torch.randn([32, 3, 41, 41])
        >>> target = torch.randn([32, 3, 41, 41])
        >>> vif = VisualInformationFidelity()
        >>> vif(preds, target)
        tensor(0.0032)

    """

    is_differentiable = True
    higher_is_better = True
    full_state_update = False

    vif_score: Tensor
    total: Tensor

    def __init__(self, sigma_n_sq: float = 2.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        if not isinstance(sigma_n_sq, float) and not isinstance(sigma_n_sq, int):
            raise ValueError(f"Argument `sigma_n_sq` is expected to be a positive float or int, but got {sigma_n_sq}")

        if sigma_n_sq < 0:
            raise ValueError(f"Argument `sigma_n_sq` is expected to be a positive float or int, but got {sigma_n_sq}")

        self.add_state("vif_score", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0.0), dist_reduce_fx="sum")
        self.sigma_n_sq = sigma_n_sq

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        channels = preds.size(1)
        vif_per_channel = [
            _vif_per_channel(preds[:, i, :, :], target[:, i, :, :], self.sigma_n_sq) for i in range(channels)
        ]
        vif_per_channel = torch.mean(torch.stack(vif_per_channel), 0) if channels > 1 else torch.cat(vif_per_channel)
        self.vif_score += torch.sum(vif_per_channel)
        self.total += preds.shape[0]

    def compute(self) -> Tensor:
        """Compute vif-p over state."""
        return self.vif_score / self.total