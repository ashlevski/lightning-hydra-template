# coding=utf-8
# Copyright (c) DIRECT Contributors
from dataclasses import dataclass

from direct.config.defaults import ModelConfig


@dataclass
class JointICNetConfig(ModelConfig):
    num_iter: int = 10
    use_norm_unet: bool = False
    image_unet_num_filters: int = 8
    image_unet_num_pool_layers: int = 4
    image_unet_dropout: float = 0.0
    kspace_unet_num_filters: int = 8
    kspace_unet_num_pool_layers: int = 4
    kspace_unet_dropout: float = 0.0
    sens_unet_num_filters: int = 8
    sens_unet_num_pool_layers: int = 4
    sens_unet_dropout: float = 0.0
