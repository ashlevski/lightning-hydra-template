# coding=utf-8
# Copyright (c) DIRECT Contributors
from collections import OrderedDict


def fix_state_dict_module_prefix(state_dict):
    """If models are saved after being wrapped in e.g. DataParallel, the keys of the state dict are prefixed with
    `module.`. This function removes this prefix.

    Parameters
    ----------
    state_dict: dict
        state_dict of a network module
    Returns
    -------
    dict
    """
    if list(state_dict.keys())[0].startswith("module."):
        new_ordered_dict = OrderedDict()
        for _, (k, v) in enumerate(state_dict.items()):
            name = k[7:]
            new_ordered_dict[name] = v
        state_dict = new_ordered_dict

    return state_dict
