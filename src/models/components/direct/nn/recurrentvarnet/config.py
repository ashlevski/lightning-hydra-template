# coding=utf-8
# Copyright (c) DIRECT Contributors

from dataclasses import dataclass
from typing import Optional, Tuple

from direct.config.defaults import ModelConfig
from direct.nn.types import InitType


@dataclass
class RecurrentVarNetConfig(ModelConfig):
    num_steps: int = 15  # :math:`T`
    recurrent_hidden_channels: int = 64
    recurrent_num_layers: int = 4  # :math:`n_l`
    no_parameter_sharing: bool = True
    learned_initializer: bool = True
    initializer_initialization: Optional[str] = InitType.sense
    initializer_channels: Optional[Tuple[int, ...]] = (32, 32, 64, 64)  # :math:`n_d`
    initializer_dilations: Optional[Tuple[int, ...]] = (1, 1, 2, 4)  # :math:`p`
    initializer_multiscale: int = 1
    normalized: bool = False
