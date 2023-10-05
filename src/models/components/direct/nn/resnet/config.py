# coding=utf-8
# Copyright (c) DIRECT Contributors

from dataclasses import dataclass
from typing import Optional

from direct.config.defaults import ModelConfig


@dataclass
class ResNetConfig(ModelConfig):
    in_channels: int = 2
    out_channels: Optional[int] = None
    hidden_channels: int = 32
    num_blocks: int = 15
    batchnorm: bool = True
    scale: Optional[float] = 0.1
    image_init: str = "sense"
