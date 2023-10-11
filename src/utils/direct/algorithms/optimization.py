# coding=utf-8
# Copyright (c) DIRECT Contributors

"""General mathematical optimization techniques."""

from abc import ABC, abstractmethod
from typing import Callable, Optional

import torch


class Algorithm(ABC):
    """Base class for implementing mathematical optimization algorithms."""

    def __init__(self, max_iter: int = 30):
        self.max_iter = max_iter
        self.iter = 0

    @abstractmethod
    def _update(self):
        """Abstract method for updating the algorithm's parameters."""
        raise NotImplementedError

    @abstractmethod
    def _fit(self, *args, **kwargs):
        """Abstract method for fitting the algorithm.

        Parameters
        ----------
        *args : tuple
            Tuple of arguments.
        **kwargs : dict
            Keyword arguments.
        """
        raise NotImplementedError

    @abstractmethod
    def _done(self) -> bool:
        """Abstract method for checking if the algorithm has ran for `max_iter`.

        Returns
        -------
        bool
        """
        raise NotImplementedError

    def update(self) -> None:
        """Update the algorithm's parameters and increment the iteration count."""
        self._update()
        self.iter += 1

    def done(self) -> bool:
        """Check if the algorithm has converged.

        Returns
        -------
        bool
            Whether the algorithm has converged or not.
        """
        return self._done()

    def fit(self, *args, **kwargs) -> None:
        """Fit the algorithm.

        Parameters
        ----------
        *args : tuple
            Tuple of arguments for `_fit` method.
        **kwargs : dict
            Keyword arguments for `_fit` method.
        """
        self._fit(*args, **kwargs)
        while not self.done():
            self.update()


class MaximumEigenvaluePowerMethod(Algorithm):
    """A class for solving the maximum eigenvalue problem using the Power Method."""

    def __init__(
        self,
        forward_operator: Callable,
        norm_func: Optional[Callable] = None,
        max_iter: int = 30,
    ):
        """Inits :class:`MaximumEigenvaluePowerMethod`.

        Parameters
        ----------
        forward_operator : Callable
            The forward operator for the problem.
        norm_func : Callable, optional
            An optional function for normalizing the eigenvector. Default: None.
        max_iter : int, optional
            Maximum number of iterations to run the algorithm. Default: 30.
        """
        self.forward_operator = forward_operator
        self.norm_func = norm_func
        super().__init__(max_iter)

    def _update(self) -> None:
        """Perform a single update step of the algorithm.

        Updates maximum eigenvalue guess and corresponding eigenvector.
        """
        y = self.forward_operator(self.x)
        if self.norm_func is None:
            self.max_eig = (y * self.x.conj()).sum() / (self.x * self.x.conj()).sum()
        else:
            self.max_eig = self.norm_func(y)
        self.x = y / self.max_eig

    def _done(self) -> bool:
        """Check if the algorithm is done.

        Returns
        -------
        bool
            Whether the algorithm has converged or not.
        """
        return self.iter >= self.max_iter

    def _fit(self, x: torch.Tensor) -> None:
        """Sets initial maximum eigenvector guess.

        Parameters
        ----------
        x : torch.Tensor
            Initial guess for the eigenvector.
        """
        # pylint: disable=arguments-differ
        self.x = x
