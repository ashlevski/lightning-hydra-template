# coding=utf-8
# Copyright (c) DIRECT Contributors
from dataclasses import dataclass

from direct.config.defaults import ModelConfig


@dataclass
class KIKINetConfig(ModelConfig):
    num_iter: int = 10
    image_model_architecture: str = "MWCNN"
    kspace_model_architecture: str = "UNET"
    image_mwcnn_hidden_channels: int = 16
    image_mwcnn_num_scales: int = 4
    image_mwcnn_bias: bool = True
    image_mwcnn_batchnorm: bool = False
    image_unet_num_filters: int = 8
    image_unet_num_pool_layers: int = 4
    image_unet_dropout_probability: float = 0.0
    kspace_conv_hidden_channels: int = 16
    kspace_conv_n_convs: int = 4
    kspace_conv_batchnorm: bool = False
    kspace_didn_hidden_channels: int = 64
    kspace_didn_num_dubs: int = 6
    kspace_didn_num_convs_recon: int = 9
    kspace_unet_num_filters: int = 8
    kspace_unet_num_pool_layers: int = 4
    kspace_unet_dropout_probability: float = 0.0
    normalize: bool = False
