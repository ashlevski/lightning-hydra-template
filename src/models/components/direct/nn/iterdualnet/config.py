# coding=utf-8
# Copyright (c) DIRECT Contributors
from dataclasses import dataclass

from direct.config.defaults import ModelConfig


@dataclass
class IterDualNetConfig(ModelConfig):
    num_iter: int = 10
    image_normunet: bool = False
    kspace_normunet: bool = False
    image_unet_num_filters: int = 8
    image_unet_num_pool_layers: int = 4
    image_unet_dropout: float = 0.0
    kspace_unet_num_filters: int = 8
    kspace_unet_num_pool_layers: int = 4
    kspace_unet_dropout: float = 0.0
    image_no_parameter_sharing: bool = True
    kspace_no_parameter_sharing: bool = False
    compute_per_coil: bool = True
