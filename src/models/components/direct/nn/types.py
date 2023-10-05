# coding=utf-8
# Copyright (c) DIRECT Contributors

from direct.types import DirectEnum


class ActivationType(DirectEnum):
    relu = "relu"
    prelu = "prelu"
    leaky_rely = "leaky_relu"


class ModelName(DirectEnum):
    unet = "unet"
    normunet = "normunet"
    resnet = "resnet"
    didn = "didn"
    conv = "conv"


class InitType(DirectEnum):
    sense = "sense"
    zero_filled = "zero_filled"
    input_image = "input_image"
