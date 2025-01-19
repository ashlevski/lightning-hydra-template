# coding=utf-8
# Copyright (c) DIRECT Contributors
from dataclasses import dataclass

from src.utils.direct.config.defaults import ModelConfig


@dataclass
class MultiDomainNetConfig(ModelConfig):
    standardization: bool = True
    num_filters: int = 16
    num_pool_layers: int = 4
    dropout_probability: float = 0.0
