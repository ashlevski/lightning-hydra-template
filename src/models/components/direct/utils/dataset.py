# coding=utf-8
# Copyright (c) DIRECT Contributors
import pathlib
import urllib.parse
from typing import List

from direct.types import PathOrString
from direct.utils.io import check_is_valid_url, read_list


def get_filenames_for_datasets_from_config(cfg, files_root: PathOrString, data_root: pathlib.Path):
    """Given a configuration object it returns a list of filenames.

    Parameters
    ----------
    cfg: cfg-object
        cfg object having property lists having the relative paths compared to files root.
    files_root: Union[str, pathlib.Path]
    data_root: pathlib.Path

    Returns
    -------
    list of filenames or None
    """
    if "filenames_lists" not in cfg:
        return None
    lists = cfg.filenames_lists
    return get_filenames_for_datasets(lists, files_root, data_root)


def get_filenames_for_datasets(lists: List[PathOrString], files_root: PathOrString, data_root: pathlib.Path):
    """Given lists of filenames of data points, concatenate these into a large list of full filenames.

    Parameters
    ----------
    lists: List[PathOrString]
    files_root: PathOrString
    data_root: pathlib.Path

    Returns
    -------
    list of filenames or None
    """
    # Build the path, know that files_root can also be a URL
    is_url = check_is_valid_url(files_root)

    filter_filenames = []
    for curr_list in lists:
        if not is_url:
            path_to_list = pathlib.Path(files_root) / curr_list
        else:
            # The path needs to be extended / and '...' needs to be parsed. The urljoin handles this correctly
            # Note: any query arguments are dropped. So any temporary keys such as ?Q=XYZ will not be added to the URL.
            path_to_list = urllib.parse.urljoin(files_root, curr_list)

        filter_filenames += [data_root / pathlib.Path(_) for _ in read_list(path_to_list)]

    return filter_filenames
