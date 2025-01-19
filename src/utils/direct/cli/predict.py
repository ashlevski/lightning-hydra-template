# coding=utf-8
# Copyright (c) DIRECT Contributors

import argparse
import pathlib

from src.utils.direct.cli.utils import file_or_url
from src.utils.direct.environment import Args
from src.utils.direct.predict import predict_from_argparse


def register_parser(parser: argparse._SubParsersAction):
    """Register wsi commands to a root parser."""

    epilog = """
        Examples:
        ---------
        Run on single machine:
            $ direct predict <output_directory> --cfg <cfg_path_or_url> --checkpoint <checkpoint_path_or_url> \
                            --num-gpus <num_gpus> [--data-root <data_root>] [--other-flag-args <other_flags>]

        Run on multiple machines:
            (machine0)$ direct predict <output_directory> --cfg <cfg_path_or_url> --checkpoint \
                            <checkpoint_path_or_url> --machine-rank 0 --num-machines 2 [--data-root <data_root>] \
                                [--dist-url <URL> --other-flag-args <other_flags>]
            (machine1)$ direct predict <output_directory> --cfg <cfg_path_or_url> --checkpoint \
                            <checkpoint_path_or_url> --machine-rank 1 --num-machines 2 [--data-root <data_root>] \
                                [--dist-url <URL> --other-flag-args <other_flags>]
        """
    common_parser = Args(add_help=False)
    predict_parser = parser.add_parser(
        "predict",
        help="Run inference using direct.",
        parents=[common_parser],
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    predict_parser.add_argument("output_directory", type=pathlib.Path, help="Path to the output directory.")
    predict_parser.add_argument(
        "--data-root",
        type=pathlib.Path,
        help="Path to the inference data directory (if required by inference dataset).",
        required=False,
    )
    predict_parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Config file for inference. Can be either a local file or a remote URL."
        "Only use it to overwrite the standard loading of the config in the project directory.",
        required=True,
        type=file_or_url,
    )
    predict_parser.add_argument(
        "--checkpoint",
        dest="checkpoint",
        type=file_or_url,
        required=True,
        help="Checkpoint to a model. This can be a path to a local file or an URL. "
        "If a URL is given the checkpoint will first be downloaded to the environmental variable "
        "`DIRECT_MODEL_DOWNLOAD_DIR` (default=current directory).",
    )
    predict_parser.add_argument(
        "--filenames-filter",
        type=pathlib.Path,
        help="Path to list of filenames to parse.",
    )
    predict_parser.add_argument(
        "--experiment_directory",
        dest="experiment_directory",
        help="Path to the experiment directory where inference logs will be saved. This is optional. "
        "If not passed, output_directory will be used as the experiment_directory.",
        required=False,
        type=pathlib.Path,
    )
    predict_parser.add_argument(
        "--name",
        dest="name",
        help="Run name if this is different than the experiment directory.",
        required=False,
        type=str,
        default="",
    )

    predict_parser.set_defaults(subcommand=predict_from_argparse)
