# coding=utf-8
# Copyright (c) DIRECT Contributors

from torch import nn

from direct.constants import COMPLEX_SIZE
from direct.nn.conv.conv import Conv2d
from direct.nn.didn.didn import DIDN
from direct.nn.resnet.resnet import ResNet
from direct.nn.types import ActivationType, ModelName
from direct.nn.unet.unet_2d import NormUnetModel2d, UnetModel2d


def _get_model_config(
    model_architecture_name: ModelName, in_channels: int = COMPLEX_SIZE, out_channels: int = COMPLEX_SIZE, **kwargs
) -> nn.Module:
    model_kwargs = {"in_channels": in_channels, "out_channels": out_channels}
    if model_architecture_name in ["unet", "normunet"]:
        model_architecture = UnetModel2d if model_architecture_name == "unet" else NormUnetModel2d
        model_kwargs.update(
            {
                "num_filters": kwargs.get("unet_num_filters", 32),
                "num_pool_layers": kwargs.get("unet_num_pool_layers", 4),
                "dropout_probability": kwargs.get("unet_dropout", 0.0),
            }
        )
    elif model_architecture_name == "resnet":
        model_architecture = ResNet
        model_kwargs.update(
            {
                "in_channels": in_channels,
                "hidden_channels": kwargs.get("resnet_hidden_channels", 64),
                "num_blocks": kwargs.get("resnet_num_blocks", 15),
                "batchnorm": kwargs.get("resnet_batchnorm", True),
                "scale": kwargs.get("resnet_scale", 0.1),
            }
        )
    elif model_architecture_name == "didn":
        model_architecture = DIDN
        model_kwargs.update(
            {
                "hidden_channels": kwargs.get("didn_hidden_channels", 16),
                "num_dubs": kwargs.get("didn_num_dubs", 6),
                "num_convs_recon": kwargs.get("didn_num_convs_recon", 9),
            }
        )
    else:
        model_architecture = Conv2d
        model_kwargs.update(
            {
                "hidden_channels": kwargs.get("conv_hidden_channels", 64),
                "n_convs": kwargs.get("conv_n_convs", 15),
                "activation": nn.PReLU()
                if kwargs.get("conv_activation", "prelu") == ActivationType.prelu
                else nn.ReLU()
                if kwargs.get("conv_activation", "relu") == ActivationType.relu
                else nn.LeakyReLU(),
                "batchnorm": kwargs.get("conv_batchnorm", False),
            }
        )

    return model_architecture, model_kwargs
