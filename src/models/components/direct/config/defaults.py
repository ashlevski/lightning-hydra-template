# coding=utf-8
# Copyright (c) DIRECT Contributors

from dataclasses import dataclass, field
from typing import Any, List, Optional

from omegaconf import MISSING

from direct.config import BaseConfig
from direct.data.datasets_config import DatasetConfig


@dataclass
class TensorboardConfig(BaseConfig):
    num_images: int = 8


@dataclass
class LoggingConfig(BaseConfig):
    log_as_image: Optional[List[str]] = None
    tensorboard: TensorboardConfig = TensorboardConfig()


@dataclass
class FunctionConfig(BaseConfig):
    function: str = MISSING
    multiplier: float = 1.0


@dataclass
class CheckpointerConfig(BaseConfig):
    checkpoint_steps: int = 500


@dataclass
class LossConfig(BaseConfig):
    crop: Optional[str] = None
    losses: List[Any] = field(default_factory=lambda: [FunctionConfig()])


@dataclass
class TrainingConfig(BaseConfig):
    # Dataset
    datasets: List[Any] = field(default_factory=lambda: [DatasetConfig()])

    # model_checkpoint gives the checkpoint from which we can load the *model* weights.
    model_checkpoint: Optional[str] = None

    # Optimizer
    optimizer: str = "Adam"
    lr: float = 5e-4
    weight_decay: float = 1e-6
    batch_size: int = 2

    # LR Scheduler
    lr_step_size: int = 5000
    lr_gamma: float = 0.5
    lr_warmup_iter: int = 500

    # Stochastic weight averaging
    swa_start_iter: Optional[int] = None

    num_iterations: int = 50000

    # Validation
    validation_steps: int = 1000

    # Gradient
    gradient_steps: int = 1
    gradient_clipping: float = 0.0
    gradient_debug: bool = False

    # Loss
    loss: LossConfig = LossConfig()

    # Checkpointer
    checkpointer: CheckpointerConfig = CheckpointerConfig()

    # Metrics
    metrics: List[str] = field(default_factory=lambda: [])

    # Regularizers
    regularizers: List[str] = field(default_factory=lambda: [])


@dataclass
class ValidationConfig(BaseConfig):
    datasets: List[Any] = field(default_factory=lambda: [DatasetConfig()])
    batch_size: int = 8
    metrics: List[str] = field(default_factory=lambda: [])
    regularizers: List[str] = field(default_factory=lambda: [])
    crop: Optional[str] = "training"


@dataclass
class InferenceConfig(BaseConfig):
    dataset: DatasetConfig = DatasetConfig()
    batch_size: int = 1
    crop: Optional[str] = None


@dataclass
class ModelConfig(BaseConfig):
    model_name: str = MISSING


@dataclass
class PhysicsConfig(BaseConfig):
    forward_operator: str = "fft2"
    backward_operator: str = "ifft2"
    use_noise_matrix: bool = False
    noise_matrix_scaling: Optional[float] = 1.0


@dataclass
class DefaultConfig(BaseConfig):
    model: ModelConfig = MISSING
    additional_models: Optional[Any] = None

    physics: PhysicsConfig = PhysicsConfig()

    training: TrainingConfig = TrainingConfig()  # This should be optional.
    validation: ValidationConfig = ValidationConfig()  # This should be optional.
    inference: Optional[InferenceConfig] = None

    logging: LoggingConfig = LoggingConfig()
