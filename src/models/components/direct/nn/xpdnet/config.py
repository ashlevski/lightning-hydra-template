# coding=utf-8
# Copyright (c) DIRECT Contributors

from dataclasses import dataclass

from direct.config.defaults import ModelConfig


@dataclass
class XPDNetConfig(ModelConfig):
    num_primal: int = 5
    num_dual: int = 1
    num_iter: int = 10
    use_primal_only: bool = True
    kspace_model_architecture: str = "CONV"
    dual_conv_hidden_channels: int = 16
    dual_conv_n_convs: int = 4
    dual_conv_batchnorm: bool = False
    dual_didn_hidden_channels: int = 64
    dual_didn_num_dubs: int = 6
    dual_didn_num_convs_recon: int = 9
    mwcnn_hidden_channels: int = 16
    mwcnn_num_scales: int = 4
    mwcnn_bias: bool = True
    mwcnn_batchnorm: bool = False
    normalize: bool = False
