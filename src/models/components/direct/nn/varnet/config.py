# coding=utf-8
# Copyright (c) DIRECT Contributors
from dataclasses import dataclass

from direct.config.defaults import ModelConfig


@dataclass
class EndToEndVarNetConfig(ModelConfig):
    num_layers: int = 8
    regularizer_num_filters: int = 18
    regularizer_num_pull_layers: int = 4
    regularizer_dropout: float = 0.0
