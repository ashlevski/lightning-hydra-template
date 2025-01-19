# coding=utf-8
# Copyright (c) DIRECT Contributors
"""Checkpointer module.

Handles all logic related to checkpointing.
"""
import datetime
import logging
import pathlib
import re
import urllib.parse
import warnings
from pickle import UnpicklingError
from typing import Dict, Mapping, Optional, Union, get_args

import torch
import torch.nn as nn
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel

from src.utils.direct.environment import DIRECT_MODEL_DOWNLOAD_DIR
from src.utils.direct.types import HasStateDict, PathOrString
from src.utils.direct.utils.io import check_is_valid_url, download_url

logger = logging.getLogger(__name__)

# TODO: Rewrite Checkpointer
# There are too many issues with typing and mypy in the checkpointer.
# What is required:
# - [ ] All models should be treated on the same footing, everything with a state dict.
# - [ ] Metadata should preferably be a different key, perhaps most of it should be returned as NamedTuple?
# - [ ] Can also choose to have metadata in the dict! Then we need to do something more fancy, and multiple tests.
# - [ ] Perhaps drop the save path from Constructor?


class Checkpointer:
    """Main Checkpointer module.

    Handles writing and restoring from checkpoints of modules and submodules.
    """

    def __init__(
        self,
        save_directory: pathlib.Path,
        save_to_disk: bool = True,
        model_regex: str = "^.*model$",
        **checkpointables: Mapping[str, Union[str, bool, HasStateDict]],
    ):
        self.logger = logging.getLogger(type(self).__name__)
        self.save_directory = save_directory
        self.model_regex = model_regex

        model = checkpointables["model"]
        del checkpointables["model"]

        self.model = self._remove_module_attribute(model)
        for key in checkpointables.copy():
            if re.match(model_regex, key):
                checkpointables[key] = self._remove_module_attribute(checkpointables[key])

        self.save_to_disk = save_to_disk
        self.checkpoint_loaded: Union[int, str, None] = None
        self.checkpointables = checkpointables

    @staticmethod
    def _remove_module_attribute(model):
        if hasattr(model, "module"):
            if not isinstance(model, (DistributedDataParallel, DataParallel)):
                warnings.warn(
                    "Model has a `.module` property and is not derived from DistributeDataParallel or DataParallel. "
                    "This is strange, but assuming the model is in `.module`."
                )

            model = model.module
        return model

    def load(
        self,
        iteration: Union[int, str, None],
        checkpointable_objects: Optional[Dict[str, nn.Module]] = None,
    ) -> Dict:
        if iteration is not None and not isinstance(iteration, int) and iteration != "latest":
            raise ValueError("Value `iteration` is expected to be either None, an integer or `latest`.")

        if iteration is None:
            return {}

        if iteration in ("latest", -1):
            last_model_text_path = self.save_directory / "last_model.txt"
            self.logger.info("Attempting to load latest model.")
            if last_model_text_path.exists():
                with open(pathlib.Path(last_model_text_path), "r", encoding="utf-8") as f:
                    iteration = int(f.readline())
                    self.logger.info("Loading last saved iteration: %s", iteration)

            else:
                self.logger.info(
                    "Latest model not found. Perhaps `last_model.txt` (path = %s) "
                    "is missing? You can try to set an explicit iteration number, or create this file if "
                    "you believe this is an error. Will not load any model.",
                    last_model_text_path,
                )
                return {}

        checkpoint_path = self.save_directory / f"model_{iteration}.pt"
        checkpoint = self.load_from_path(checkpoint_path, checkpointable_objects)
        checkpoint["iteration"] = iteration

        self.checkpoint_loaded = iteration
        # Return whatever is left
        return checkpoint

    def load_from_path(
        self,
        checkpoint_path: PathOrString,
        checkpointable_objects: Optional[Dict[str, nn.Module]] = None,
        only_models: bool = False,
    ) -> Dict:
        """Load a checkpoint from a path.

        Parameters
        ----------
        checkpoint_path: Path or str
            Path to checkpoint, either a path to a file or a path to a URL where the file can be downloaded
        checkpointable_objects: dict
            Dictionary mapping names to nn.Module's
        only_models: bool
            If true will only load the models and no other objects in the checkpoint

        Returns
        -------
        Dictionary with loaded models.
        """
        checkpoint = self._load_checkpoint(checkpoint_path)
        checkpointables = self.checkpointables if not checkpointable_objects else checkpointable_objects

        self.logger.info("Loading model...")
        self._load_model(self.model, checkpoint["model"])

        for key in checkpointables:  # type: ignore
            if only_models and not re.match(self.model_regex, key):
                continue

            if key not in checkpoint:
                self.logger.warning("Requested to load %s, but this was not stored.", key)
                continue

            if key.endswith("__") and key.startswith("__"):
                continue

            self.logger.info("Loading %s...", key)
            obj = self.checkpointables[key]
            state_dict = checkpoint.pop(key)
            if re.match(self.model_regex, key):
                self.logger.debug("key %s matches regex %s.", key, self.model_regex)
                self._load_model(obj, state_dict)  # type: ignore
            else:
                obj.load_state_dict(state_dict)  # type: ignore

        return checkpoint

    def _load_model(self, obj, state_dict):
        # https://github.com/facebookresearch/fvcore/blob/master/fvcore/common/checkpoint.py
        # Link has more elaborate checking for incompatibles in _log_incompatible_keys
        incompatible = obj.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys:
            raise NotImplementedError
        if incompatible.unexpected_keys:
            self.logger.warning("Unexpected keys provided which cannot be loaded: %s.", incompatible.unexpected_keys)

    def load_models_from_file(self, checkpoint_path: PathOrString) -> None:
        _ = self.load_from_path(checkpoint_path, only_models=True)

    def save(self, iteration: int, **kwargs: Dict[str, str]) -> None:
        # For instance useful to only have the rank 0 process write to disk.
        if not self.save_to_disk:
            return

        data: Dict[str, Union[nn.Module, str]] = {"model": self.model.state_dict()}

        for key, obj in self.checkpointables.items():
            if key.endswith("__") and key.startswith("__"):
                # Keys of the form __TEXT__ do should not have state
                data[key] = obj

            elif isinstance(obj, get_args(HasStateDict)):
                data[key] = obj.state_dict()  # type: ignore
            else:
                self.logger.warning("Value of key %s has no state_dict.", key)

        data.update(kwargs)

        checkpoint_path = self.save_directory / f"model_{iteration}.pt"
        self.logger.info("Saving checkpoint to: %s.", checkpoint_path)

        data["__datetime__"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(str(checkpoint_path), "wb") as f:
            torch.save(data, f)

        # noinspection PyTypeChecker
        with open(self.save_directory / "last_model.txt", "w", encoding="utf-8") as f:  # type: ignore
            f.write(str(iteration))  # type: ignore

    def _load_checkpoint(self, checkpoint_path: PathOrString) -> Dict:
        """Load a checkpoint from path or string.

        Parameters
        ----------
        checkpoint_path: Path or str
            Path to checkpoint, either a path to a file or a path to a URL where the file can be downloaded
        Returns
        -------
        Dict loaded from checkpoint.
        """
        # Check if the path is an URL
        if check_is_valid_url(str(checkpoint_path)):
            self.logger.info("Initializing from remote checkpoint %s...", checkpoint_path)
            checkpoint_path = self._download_or_load_from_cache(str(checkpoint_path))
            self.logger.info("Loading downloaded checkpoint %s.", checkpoint_path)

        checkpoint_path = pathlib.Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Requested to load {checkpoint_path}, but does not exist.")

        self.logger.info("Loaded checkpoint path: %s.", checkpoint_path)

        try:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))

        except UnpicklingError as exc:
            self.logger.exception("Tried to load %s, but was unable to unpickle: %s.", checkpoint_path, exc)
            raise

        return checkpoint

    @staticmethod
    def _download_or_load_from_cache(url: str) -> pathlib.Path:
        # Get final part of url.
        file_path = urllib.parse.urlparse(url).path
        filename = pathlib.Path(file_path).name

        cache_path = DIRECT_MODEL_DOWNLOAD_DIR / filename
        download_url(url, DIRECT_MODEL_DOWNLOAD_DIR, max_redirect_hops=3)

        return cache_path
