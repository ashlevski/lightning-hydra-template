# coding=utf-8
# Copyright (c) DIRECT Contributors

import argparse
import pathlib
import sys

from src.utils.direct.types import FileOrUrl, PathOrString
from src.utils.direct.utils.io import check_is_valid_url


def is_file(path):
    path = pathlib.Path(path)
    if path.is_file():
        return path
    raise argparse.ArgumentTypeError(f"{path} is not a valid file or url.")


def file_or_url(path: PathOrString) -> FileOrUrl:
    if check_is_valid_url(path):
        return FileOrUrl(path)
    path = pathlib.Path(path)
    if path.is_file():
        return FileOrUrl(path)
    raise argparse.ArgumentTypeError(f"{path} is not a valid file or url.")


def check_train_val(key, name):
    if key is not None and len(key) != 2:
        sys.exit(f"--{name} has to be of the form `train_folder, validation_folder` if a validation folder is set.")
