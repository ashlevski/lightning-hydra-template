# coding=utf-8
# Copyright (c) DIRECT Contributors

from dataclasses import dataclass
from typing import Optional

from direct.config.defaults import ModelConfig
from direct.nn.conjgradnet.conjgrad import CGUpdateType
from direct.nn.conjgradnet.conjgradnet import ConjGradNetInitType
from direct.nn.types import ActivationType, ModelName


@dataclass
class ConjGradNetConfig(ModelConfig):
    num_steps: int = 8
    image_init: str = ConjGradNetInitType.zeros
    no_parameter_sharing: bool = True
    cg_tol: float = 1e-7
    cg_iters: int = 10
    cg_param_update_type: str = CGUpdateType.FR
    denoiser_architecture: str = ModelName.resnet
    resnet_hidden_channels: int = 128
    resnet_num_blocks: int = 15
    resenet_batchnorm: bool = True
    resenet_scale: Optional[float] = 0.1
    unet_num_filters: Optional[int] = 32
    unet_num_pool_layers: Optional[int] = 4
    unet_dropout: Optional[float] = 0.0
    didn_hidden_channels: Optional[int] = 16
    didn_num_dubs: Optional[int] = 6
    didn_num_convs_recon: Optional[int] = 9
    conv_hidden_channels: Optional[int] = 64
    conv_n_convs: Optional[int] = 15
    conv_activation: Optional[str] = ActivationType.relu
    conv_batchnorm: Optional[bool] = False
