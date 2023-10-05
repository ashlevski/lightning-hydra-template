# coding=utf-8
# Copyright (c) DIRECT Contributors

from __future__ import annotations

import pathlib
from enum import Enum
from typing import NewType, Union

import torch
from omegaconf.omegaconf import DictConfig
from torch import nn as nn
from torch.cuda.amp import GradScaler

DictOrDictConfig = Union[dict, DictConfig]
Number = Union[float, int]
PathOrString = Union[pathlib.Path, str]
FileOrUrl = NewType("FileOrUrl", PathOrString)
HasStateDict = Union[nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler, GradScaler]
TensorOrNone = Union[None, torch.Tensor]


class DirectEnum(str, Enum):
    """Type of any enumerator with allowed comparison to string invariant to cases."""

    @classmethod
    def from_str(cls, value: str) -> DirectEnum | None:
        statuses = cls.__members__.keys()
        for st in statuses:
            if st.lower() == value.lower():
                return cls[st]
        return None

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Enum):
            _other = str(other.value)
        else:
            _other = str(other)
        return bool(self.value.lower() == _other.lower())

    def __hash__(self) -> int:
        # re-enable hashtable so it can be used as a dict key or in a set
        return hash(self.value.lower())


class KspaceKey(DirectEnum):
    kspace = "kspace"
    masked_kspace = "masked_kspace"


class IntegerListOrTupleStringMeta(type):
    """Metaclass for the :class:`IntegerListOrTupleString` class.

    Returns
    -------
    bool
        True if the instance is a valid representation of IntegerListOrTupleString, False otherwise.
    """

    def __instancecheck__(cls, instance):
        """Check if the given instance is a valid representation of an IntegerListOrTupleString.

        Parameters
        ----------
        cls : type
            The class being checked, i.e., IntegerListOrTupleStringMeta.
        instance : object
            The instance being checked.

        Returns
        -------
        bool
            True if the instance is a valid representation of IntegerListOrTupleString, False otherwise.
        """
        if isinstance(instance, str):
            try:
                assert (instance.startswith("[") and instance.endswith("]")) or (
                    instance.startswith("(") and instance.endswith(")")
                )
                elements = instance.strip()[1:-1].split(",")
                integers = [int(element) for element in elements]
                return all(isinstance(num, int) for num in integers)
            except (AssertionError, ValueError, AttributeError):
                pass
        return False


class IntegerListOrTupleString(metaclass=IntegerListOrTupleStringMeta):
    """IntegerListOrTupleString class represents a list or tuple of integers based on a string representation.

    Examples
    --------
    s1 = "[1, 2, 45, -1, 0]"
    print(isinstance(s1, IntegerListOrTupleString))  # True
    print(IntegerListOrTupleString(s1))  # [1, 2, 45, -1, 0]
    print(type(IntegerListOrTupleString(s1)))  # <class 'list'>
    print(type(IntegerListOrTupleString(s1)[0]))  # <class 'int'>

    s2 = "(10, -9, 20)"
    print(isinstance(s2, IntegerListOrTupleString))  # True
    print(IntegerListOrTupleString(s2))  # (10, -9, 20)
    print(type(IntegerListOrTupleString(s2)))  # <class 'tuple'>
    print(type(IntegerListOrTupleString(s2)[0]))  # <class 'int'>

    s3 = "[a, we, 2]"
    print(isinstance(s3, IntegerListOrTupleString))  # False

    s4 = "(1, 2, 3]"
    print(isinstance(s4 IntegerListOrTupleString))  # False
    """

    def __new__(cls, string):
        """
        Create a new instance of IntegerListOrTupleString based on the given string representation.

        Parameters
        ----------
        string : str
            The string representation of the integer list or tuple.

        Returns
        -------
        list or tuple
            A new instance of IntegerListOrTupleString.
        """
        list_or_tuple = list if string.startswith("[") else tuple
        string = string.strip()[1:-1]  # Remove outer brackets
        elements = string.split(",")
        integers = [int(element) for element in elements]
        return list_or_tuple(integers)
