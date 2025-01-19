# coding=utf-8
# Copyright (c) DIRECT Contributors
import argparse
import pathlib

from src.utils.direct.cli.utils import file_or_url
from src.utils.direct.environment import Args
from src.utils.direct.train import train_from_argparse


def register_parser(parser: argparse._SubParsersAction):
    """Register wsi commands to a root parser."""

    epilog = """
        Examples:
        ---------
        Run on single machine:
            $ direct train experiment_dir --num-gpus 8 --cfg cfg.yaml [--training-root training_set --validation-root validation_set]
        Run on multiple machines:
            (machine0)$ direct train experiment_dir --machine-rank 0 --num-machines 2 --dist-url <URL> [--training-root training_set --validation-root validation_set] [--other-flags]
            (machine1)$ direct train experiment_dir --machine-rank 1 --num-machines 2 --dist-url <URL> [--training-root training_set --validation-root validation_set] [--other-flags]
        """
    common_parser = Args(add_help=False)
    train_parser = parser.add_parser(
        "train",
        help="Train models using direct.",
        parents=[common_parser],
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    train_parser.add_argument(
        "experiment_dir",
        type=pathlib.Path,
        help="Path to the experiment directory.",
    )
    train_parser.add_argument("--training-root", type=pathlib.Path, help="Path to the training data.", required=False)
    train_parser.add_argument(
        "--validation-root", type=pathlib.Path, help="Path to the validation data.", required=False
    )
    train_parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Config file for training. Can be either a local file or a remote URL.",
        required=True,
        type=file_or_url,
    )
    train_parser.add_argument(
        "--initialization-checkpoint",
        type=file_or_url,
        help="If this value is set to a proper checkpoint when training starts, "
        "the model will be initialized with the weights given. "
        "No other keys in the checkpoint will be loaded. "
        "When another checkpoint would be available and the --resume flag is used, "
        "this flag is ignored. This can be a path to a file or an URL. "
        "If a URL is given the checkpoint will first be downloaded to the environmental variable "
        "`DIRECT_MODEL_DOWNLOAD_DIR` (default=current directory). Be aware that if `model_checkpoint` is "
        "set in the configuration that this flag will overwrite the configuration value, also in the dumped config.",
    )
    train_parser.add_argument("--resume", help="Resume training if possible.", action="store_true")
    train_parser.add_argument(
        "--force-validation",
        help="Start with a validation round, when recovering from a crash. "
        "If you use this option, be aware that when combined with --resume, "
        "each new run will start with a validation round.",
        action="store_true",
    )
    train_parser.add_argument("--name", help="Run name.", required=False, type=str)

    train_parser.set_defaults(subcommand=train_from_argparse)
