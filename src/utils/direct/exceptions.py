# coding=utf-8
# Copyright (c) DIRECT Contributors
import logging


class DirectException(BaseException):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.logger = logging.getLogger(__name__)


class ProcessKilledException(DirectException):
    """The process received SIGINT signal."""

    def __init__(self, signal_id: int, signal_name: str):
        """
        Parameters
        ----------
        signal_id: str
        signal_name: str
        """
        super().__init__()
        self.logger.exception(
            f"Received signal (signal_id = {signal_id} - signal_name = {signal_name}). " "Critical. Process will stop."
        )
        self.signal_id = signal_id
        self.signal_name = signal_name


class TrainingException(DirectException):
    def __init__(self, message=None):
        super().__init__()
        if message:
            self.logger.exception("TrainingException")
        else:
            self.logger.exception(f"TrainingException: {message}")


class ItemNotFoundException(DirectException):
    def __init__(self, item_name, message=None):
        super().__init__()
        error_name = "".join([s.capitalize() for s in item_name.split(" ")]) + "Exception"
        if message:
            self.logger.exception(error_name)
        else:
            self.logger.exception("%s: %s", error_name, message)
