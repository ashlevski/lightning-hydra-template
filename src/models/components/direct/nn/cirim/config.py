# coding=utf-8
# Copyright (c) DIRECT Contributors

from dataclasses import dataclass

from direct.config.defaults import ModelConfig


@dataclass
class CIRIMConfig(ModelConfig):
    time_steps: int = 8  # :math:`T`
    depth: int = 2
    recurrent_hidden_channels: int = 64
    num_cascades: int = 8
    no_parameter_sharing: bool = True
