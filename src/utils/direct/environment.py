# coding=utf-8
# Copyright (c) DIRECT Contributors

import argparse
import logging
import os
import pathlib
import sys
from collections import namedtuple
from typing import Callable, Dict, Optional, Tuple, Union

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.utils import collect_env

import src.utils.direct.utils.logging
from src.utils.direct.config.defaults import DefaultConfig, InferenceConfig, PhysicsConfig, TrainingConfig, ValidationConfig
from src.utils.direct.utils import communication, count_parameters, str_to_class
from src.utils.direct.utils.io import check_is_valid_url, read_text_from_url
from src.utils.direct.utils.logging import setup

logger = logging.getLogger(__name__)

# Environmental variables
DIRECT_ROOT_DIR = pathlib.Path(pathlib.Path(__file__).resolve().parent.parent)
DIRECT_CACHE_DIR = pathlib.Path(os.environ.get("DIRECT_CACHE_DIR", str(DIRECT_ROOT_DIR)))
DIRECT_MODEL_DOWNLOAD_DIR = (
    pathlib.Path(os.environ.get("DIRECT_MODEL_DOWNLOAD_DIR", str(DIRECT_ROOT_DIR))) / "downloaded_models"
)


def load_model_config_from_name(model_name: str) -> Callable:
    """Load specific configuration module for models based on their name.

    Parameters
    ----------
    model_name: str
        Path to model relative to direct.nn.

    Returns
    -------
    model_cfg: Callable
        Model configuration.
    """
    module_path = f"direct.nn.{model_name.split('.')[0].lower()}.config"
    model_name += "Config"
    config_name = model_name.split(".")[-1]
    try:
        model_cfg = str_to_class(module_path, config_name)
    except (AttributeError, ModuleNotFoundError) as e:
        logger.error(f"Path {module_path} for config_name {config_name} does not exist (err = {e}).")
        sys.exit(-1)
    return model_cfg


def load_model_from_name(model_name: str) -> Callable:
    """Load model based on `model_name`.

    Parameters
    ----------
    model_name: str
        Model name as in direct.nn.

    Returns
    -------
    model: Callable
        Model class.
    """
    module_path = f"direct.nn.{'.'.join([_.lower() for _ in model_name.split('.')[:-1]])}"
    module_name = model_name.split(".")[-1]
    try:
        model = str_to_class(module_path, module_name)
    except (AttributeError, ModuleNotFoundError) as e:
        logger.error(f"Path {module_path} for model_name {module_name} does not exist (err = {e}).")
        sys.exit(-1)

    return model


def load_dataset_config(dataset_name: str) -> Callable:
    """Load specific dataset configuration for dataset based on `dataset_name`.

    Parameters
    ----------
    dataset_name: str
        Name of dataset.

    Returns
    -------
    dataset_config: Callable
        Dataset configuration.
    """
    dataset_config = str_to_class("direct.data.datasets_config", dataset_name + "Config")
    return dataset_config


def build_operators(cfg: PhysicsConfig) -> Tuple[Callable, Callable]:
    """Builds operators from configuration."""
    # Get the operators
    forward_operator = str_to_class("direct.data.transforms", cfg.forward_operator)
    backward_operator = str_to_class("direct.data.transforms", cfg.backward_operator)
    return forward_operator, backward_operator


def setup_logging(
    machine_rank: int,
    output_directory: pathlib.Path,
    run_name: str,
    cfg_filename: Union[pathlib.Path, str],
    cfg: DefaultConfig,
    debug: bool,
) -> None:
    """Logs environment information.

    Parameters
    ----------
    machine_rank: int
        Machine rank.
    output_directory: pathlib.Path
        Path to output directory.
    run_name: str
        Name of run.
    cfg_filename: Union[pathlib.Path, str]
        Name of configuration file.
    cfg: DefaultConfig
        Configuration file.
    debug: bool
        Whether the debug mode is enabled.
    """
    # Setup logging
    log_file = output_directory / f"log_{machine_rank}_{communication.get_local_rank()}.txt"

    setup(
        use_stdout=communication.is_main_process() or debug,
        filename=log_file,
        log_level=("INFO" if not debug else "DEBUG"),
    )
    logger.info("Machine rank: %s", machine_rank)
    logger.info("Local rank: %s", communication.get_local_rank())
    logger.info("Logging: %s", log_file)
    logger.info("Saving to: %s", output_directory)
    logger.info("Run name: %s", run_name)
    logger.info("Config file: %s", cfg_filename)
    logger.info("CUDA %s - cuDNN %s", torch.version.cuda, torch.backends.cudnn.version())
    logger.info("Environment information: %s", collect_env.get_pretty_env_info())
    logger.info("DIRECT version: %s", direct.__version__)
    git_hash = direct.utils.git_hash()
    logger.info("Git hash: %s", git_hash if git_hash else "N/A")
    logger.info("Configuration: %s", OmegaConf.to_yaml(cfg))


def load_models_into_environment_config(cfg_from_file: DictConfig) -> Tuple[dict, DictConfig]:
    """Load the configuration for the models.

    Parameters
    ----------
    cfg_from_file: DictConfig
        Omegaconf configuration.

    Returns
    -------
    (models, models_config): (dict, DictConfig)
        Models dictionary and models configuration dictionary.
    """
    cfg = {"model": cfg_from_file.model}

    if "additional_models" in cfg_from_file:
        cfg = {**cfg, **cfg_from_file.additional_models}
    # Parse config of additional models
    # TODO: Merge this with the normal model config loading.
    models_config = {}
    models = {}
    for curr_model_name in cfg:
        if "model_name" not in cfg[curr_model_name]:
            logger.error(f"Model {curr_model_name} has no model_name.")
            sys.exit(-1)

        curr_model_cfg = cfg[curr_model_name]
        model_name = curr_model_cfg.model_name
        models[curr_model_name] = load_model_from_name(model_name)

        models_config[curr_model_name] = OmegaConf.merge(load_model_config_from_name(model_name), curr_model_cfg)

    return models, OmegaConf.merge(models_config)  # type: ignore


def initialize_models_from_config(
    cfg: DictConfig, models: dict, forward_operator: Callable, backward_operator: Callable, device: str
) -> Tuple[torch.nn.Module, Dict]:
    """Creates models from config.

    Parameters
    ----------
    cfg: DictConfig
        Configuration.
    models: dict
        Models dictionary including configurations.
    forward_operator: Callable
        Forward operator.
    backward_operator: Callable
        Backward operator.
    device: str
        Type of device.

    Returns
    -------
    model: torch.nn.Module
        Model.
    additional_models: Dict
        Additional models.
    """
    # Create the model
    logger.info("Building models.")
    # TODO(jt): Model name is not used here.
    additional_models = {}
    for k, v in cfg.additional_models.items():
        # Remove model_name key
        curr_model = models[k]
        curr_model_cfg = {kk: vv for kk, vv in v.items() if kk != "model_name"}
        additional_models[k] = curr_model(**curr_model_cfg)

    model = models["model"](
        forward_operator=forward_operator,
        backward_operator=backward_operator,
        **{k: v for (k, v) in cfg.model.items()},
    ).to(device)

    # Log total number of parameters
    count_parameters({"model": model, **additional_models})
    return model, additional_models


def setup_engine(
    cfg: DictConfig,
    device: str,
    model: torch.nn.Module,
    additional_models: dict,
    forward_operator: Optional[Union[Callable, object]] = None,
    backward_operator: Optional[Union[Callable, object]] = None,
    mixed_precision: bool = False,
):
    """Setups engine.

    Parameters
    ----------
    cfg: DictConfig
        Configuration.
    device: str
        Type of device.
    model: torch.nn.Module
        Model.
    additional_models: dict
        Additional models.
    forward_operator: Callable
        Forward operator.
    backward_operator: Callable
        Backward operator.
    mixed_precision: bool
        Whether to enable mixed precision or not. Default: False.

    Returns
    -------
    engine
        Experiment Engine.
    """

    # There is a bit of repetition here, but the warning provided is more descriptive
    # TODO(jt): Try to find a way to combine this with the setup above.
    model_name_short = cfg.model.model_name.split(".")[0]
    engine_name = cfg.model.model_name.split(".")[-1] + "Engine"

    try:
        engine_class = str_to_class(
            f"direct.nn.{model_name_short.lower()}.{model_name_short.lower()}_engine",
            engine_name,
        )
    except (AttributeError, ModuleNotFoundError) as e:
        logger.error(f"Engine does not exist for {cfg.model.model_name} (err = {e}).")
        sys.exit(-1)

    engine = engine_class(  # noqa
        cfg,
        model,
        device=device,
        forward_operator=forward_operator,
        backward_operator=backward_operator,
        mixed_precision=mixed_precision,
        **additional_models,
    )
    return engine


def extract_names(cfg):
    cfg = cfg.copy()
    if isinstance(cfg, DictConfig):
        if "name" not in cfg:
            raise ValueError("`name` needs to be present in config.")
        curr_name = cfg["name"]

    elif isinstance(cfg, ListConfig):
        return [extract_names(v) for v in cfg]

    else:
        raise ValueError(f"Expected DictConfig or ListConfig. Got {type(cfg)}.")

    return curr_name, cfg


def setup_common_environment(
    run_name: str,
    base_directory: pathlib.Path,
    cfg_pathname: Union[pathlib.Path, str],
    device: str,
    machine_rank: int,
    mixed_precision: bool,
    debug: bool = False,
):
    """Setup environment.

    Parameters
    ----------
    run_name: str
        Run name.
    base_directory: pathlib.Path
        Base directory path.
    cfg_pathname: Union[pathlib.Path, str]
        Path or url to configuratio file.
    device: str
        Device type.
    machine_rank: int
        Machine rank.
    mixed_precision: bool
        Whether to enable mixed precision or not. Default: False.
    debug: bool
        Whether the debug mode is enabled.

    Returns
    -------
    environment
        Common Environment.
    """

    logger = logging.getLogger()

    experiment_dir = base_directory / run_name
    if communication.get_local_rank() == 0:
        # Want to prevent multiple workers from trying to write a directory
        # This is required in the logging below
        experiment_dir.mkdir(parents=True, exist_ok=True)
    communication.synchronize()  # Ensure folders are in place.

    # Load configs from YAML file to check which model needs to be loaded.
    # Can also be loaded from a URL
    if check_is_valid_url(cfg_pathname):
        cfg_from_external_source = OmegaConf.create(read_text_from_url(cfg_pathname))
    else:
        cfg_from_external_source = OmegaConf.load(cfg_pathname)

    # Load the default configs to ensure type safety
    cfg = OmegaConf.structured(DefaultConfig)

    models, models_config = load_models_into_environment_config(cfg_from_external_source)
    cfg.model = models_config.model
    del models_config["model"]
    cfg.additional_models = models_config

    # Setup everything for training
    cfg.training = TrainingConfig
    cfg.validation = ValidationConfig
    cfg.inference = InferenceConfig

    cfg_from_file_new = cfg_from_external_source.copy()
    for key in cfg_from_external_source:
        # TODO: This does not really do a full validation.
        # BODY: This will be handeled once Hydra is implemented.
        if key in ["models", "additional_models"]:  # Still handled separately
            continue

        if key in ["training", "validation", "inference"]:
            if not cfg_from_external_source[key]:
                logger.info(f"key {key} missing in config.")
                continue

            if key in ["training", "validation"]:
                dataset_cfg_from_file = extract_names(cfg_from_external_source[key].datasets)
                for idx, (dataset_name, dataset_config) in enumerate(dataset_cfg_from_file):
                    cfg_from_file_new[key].datasets[idx] = dataset_config
                    cfg[key].datasets.append(load_dataset_config(dataset_name))  # pylint: disable = E1136
            else:
                dataset_name, dataset_config = extract_names(cfg_from_external_source[key].dataset)
                cfg_from_file_new[key].dataset = dataset_config
                cfg[key].dataset = load_dataset_config(dataset_name)  # pylint: disable = E1136

        cfg[key] = OmegaConf.merge(cfg[key], cfg_from_file_new[key])  # pylint: disable = E1136, E1137

    # Make configuration read only.
    # TODO(jt): Does not work when indexing config lists.
    # OmegaConf.set_readonly(cfg, True)
    setup_logging(machine_rank, experiment_dir, run_name, cfg_pathname, cfg, debug)
    forward_operator, backward_operator = build_operators(cfg.physics)

    model, additional_models = initialize_models_from_config(cfg, models, forward_operator, backward_operator, device)

    engine = setup_engine(
        cfg,
        device,
        model,
        additional_models,
        forward_operator=forward_operator,
        backward_operator=backward_operator,
        mixed_precision=mixed_precision,
    )

    environment = namedtuple(
        "environment",
        ["cfg", "experiment_dir", "engine"],
    )
    return environment(cfg, experiment_dir, engine)


def setup_training_environment(
    run_name: str,
    base_directory: pathlib.Path,
    cfg_filename: Union[pathlib.Path, str],
    device: str,
    machine_rank: int,
    mixed_precision: bool,
    debug: bool = False,
):
    """Setup training environment.

    Parameters
    ----------
    run_name: str
        Run name.
    base_directory: pathlib.Path
        Base directory path.
    cfg_filename: Union[pathlib.Path, str]
        Path or url to configuratio file.
    device: str
        Device type.
    machine_rank: int
        Machine rank.
    mixed_precision: bool
        Whether to enable mixed precision or not. Default: False.
    debug: bool
        Whether the debug mode is enabled.

    Returns
    -------
    environment
        Training Environment.
    """

    env = setup_common_environment(
        run_name,
        base_directory,
        cfg_filename,
        device,
        machine_rank,
        mixed_precision,
        debug=debug,
    )
    # Write config file to experiment directory.
    config_file_in_project_folder = env.experiment_dir / "config.yaml"
    logger.info("Writing configuration file to: %s", config_file_in_project_folder)
    if communication.is_main_process():
        with open(config_file_in_project_folder, "w", encoding="utf-8") as f:
            f.write(OmegaConf.to_yaml(env.cfg))
    communication.synchronize()

    return env


def setup_testing_environment(
    run_name: str,
    base_directory: pathlib.Path,
    device: str,
    machine_rank: int,
    mixed_precision: bool,
    cfg_pathname: Optional[Union[pathlib.Path, str]] = None,
    debug: bool = False,
):
    """Setup testing environment.

    Parameters
    ----------
    run_name: str
        Run name.
    base_directory: pathlib.Path
        Base directory path.
    device: str
        Device type.
    machine_rank: int
        Machine rank.
    mixed_precision: bool
        Whether to enable mixed precision or not. Default: False.
    cfg_pathname: Union[pathlib.Path, str], optional
        Path or url to configuration file.
    debug: bool
        Whether the debug mode is enabled.

    Returns
    -------
    environment
        Testing Environment.
    """
    if cfg_pathname is None:  # If None, try to load from base experiment directory
        cfg_pathname = base_directory / run_name / "config.yaml"

    # If not an URL, check if it exists
    if not check_is_valid_url(cfg_pathname):
        if not pathlib.Path(cfg_pathname).exists():
            raise FileNotFoundError(f"Config file {cfg_pathname} does not exist.")

    env = setup_common_environment(
        run_name,
        base_directory,
        cfg_pathname,
        device,
        machine_rank,
        mixed_precision,
        debug=debug,
    )

    environment = namedtuple(
        "environment",
        ["cfg", "engine"],
    )
    return environment(env.cfg, env.engine)


def setup_inference_environment(
    run_name: str,
    base_directory: pathlib.Path,
    device: str,
    machine_rank: int,
    mixed_precision: bool,
    cfg_file: Optional[Union[pathlib.Path, str]] = None,
    debug: bool = False,
):
    """Setup inference environment.

    Parameters
    ----------
    run_name: str
        Run name.
    base_directory: pathlib.Path
        Base directory path.
    device: str
        Device type.
    machine_rank: int
        Machine rank.
    mixed_precision: bool
        Whether to enable mixed precision or not. Default: False.
    cfg_file: Union[pathlib.Path, str], optional
        Path or url to configuration file.
    debug: bool
        Whether the debug mode is enabled.

    Returns
    -------
    environment
        Inference Environment.
    """
    env = setup_testing_environment(
        run_name, base_directory, device, machine_rank, mixed_precision, cfg_file, debug=debug
    )

    environment = namedtuple(
        "environment",
        ["cfg", "engine"],
    )
    return environment(env.cfg, env.engine)


class Args(argparse.ArgumentParser):
    """Defines global default arguments."""

    def __init__(self, epilog=None, add_help=True, **overrides):
        """Inits Args.

        Parameters
        ----------
        epilog: str
            Text to display after the argument help. Default: None.
        add_help: bool
            Add a -h/--help option to the parser. Default: True.
        **overrides: (dict, optional)
            Keyword arguments used to override default argument values
        """
        super().__init__(epilog=epilog, formatter_class=argparse.RawDescriptionHelpFormatter, add_help=add_help)

        self.add_argument(
            "--device",
            type=str,
            default="cuda",
            help='Which device to train on. Set to "cuda" to use the GPU.',
        )
        self.add_argument("--seed", default=42, type=int, help="Seed for random number generators.")
        self.add_argument("--num-workers", type=int, default=4, help="Number of workers.")
        self.add_argument("--mixed-precision", help="Use mixed precision.", action="store_true")
        self.add_argument("--debug", help="Set debug mode true.", action="store_true")

        self.add_argument("--num-gpus", type=int, default=1, help="# GPUs per machine.")
        self.add_argument("--num-machines", type=int, default=1, help="# of machines.")
        self.add_argument(
            "--machine-rank",
            type=int,
            default=0,
            help="the rank of this machine (unique per machine).",
        )
        self.add_argument(
            "--initialization-images",
            help="Path to images which will be used as initialization to the model. "
            "The filenames assumed to be the same as the images themselves. If these are h5 files, "
            "the key to read in the h5 has to be set in the configuration in the dataset.input_image_key.",
            required=False,
            nargs="+",
            type=pathlib.Path,
        )
        self.add_argument(
            "--initialization-kspace",
            help="Path to kspace which will be used as initialization to the model. "
            "The filenames assumed to be the same as the images themselves. If these are h5 files, "
            "the key to read in the h5 has to be set in the configuration in the dataset.input_image_key.",
            required=False,
            nargs="+",
            type=pathlib.Path,
        )
        self.add_argument(
            "--noise",
            help="Path to json file mapping relative filename to noise estimates. "
            "Path to training and validation data",
            required=False,
            nargs="+",
            type=pathlib.Path,
        )

        # Taken from: https://github.com/facebookresearch/detectron2/blob/bd2ea475b693a88c063e05865d13954d50242857/detectron2/engine/defaults.py#L49 # noqa
        # PyTorch still may leave orphan processes in multi-gpu training. Therefore we use a deterministic way
        # to obtain port, so that users are aware of orphan processes by seeing the port occupied.
        port = 2**15 + 2**14 + hash(os.getuid()) % 2**14
        self.add_argument(
            "--dist-url",
            default=f"tcp://127.0.0.1:{port}",
            help="initialization URL for pytorch distributed backend. See "
            "https://pytorch.org/docs/stable/distributed.html for details.",
        )

        self.set_defaults(**overrides)
