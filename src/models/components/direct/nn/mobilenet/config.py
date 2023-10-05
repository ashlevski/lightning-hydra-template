# coding=utf-8
# Copyright (c) DIRECT Contributors
from dataclasses import dataclass
from typing import List, Optional

from direct.config.defaults import ModelConfig


@dataclass
class MobileNetV2Config(ModelConfig):
    num_channels: int = 2
    num_classes: int = 1000
    width_mult: float = 1.0
    inverted_residual_setting: Optional[List] = None
    round_nearest: int = 8
    # block = None
    norm_layer: Optional[str] = None
