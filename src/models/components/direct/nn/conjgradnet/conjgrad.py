# coding=utf-8
# Copyright (c) DIRECT Contributors

from typing import Callable, List

import torch
from torch import nn

from direct.data.transforms import (
    complex_division,
    complex_dot_product,
    complex_multiplication,
    expand_operator,
    reduce_operator,
)
from direct.types import DirectEnum


class CGUpdateType(DirectEnum):
    FR = "FR"
    PRP = "PRP"
    DY = "DY"
    BAN = "BAN"


class ConjGrad(nn.Module):
    r"""Performs the Conjugate Gradient (CG) algorithm to approach a solution to:

    .. math ::

        \min_{x} f(x) = \min_{x} \fraq{1}{2} \big( ||\mathcal{A}(x) - y||_2^2 + \lambda ||x - z||_2^2 \big)

    or equivalently solving the normal equation of the above:

    .. math ::

        \mathcal{B}(x): = \big(\mathcal{A} \circ \mathcal{A}^{*} + \lambda I\big) (x)
        = \mathcal{A}^{*}(y) + \lambda z =: b.

    Notes
    -----
    :class:`ConjGrad` has no trainable parameters. However, PyTorch ensures that gradients are computed.
    """

    def __init__(
        self,
        forward_operator: Callable,
        backward_operator: Callable,
        num_iters: int = 10,
        tol: float = 1e-6,
        bk_update_type: CGUpdateType = CGUpdateType.FR,
    ):
        r"""Inits :class:`ConjGrad`.

        Parameters
        ----------
        forward_operator : Callable
            Forward operator :math:`\mathcal{A}` (e.g. fft).
        backward_operator : Callable
            Backward/adjoint operator :math:`\mathcal{A}^{*}` (e.g. ifft).
        num_iters : int
            Convergence criterion 1: number of CG iterations. Default: 10.
        tol : float
            Convergence criterion 2: checks if CG has converged by checking `r_k` norm. Default: 1e-6.
        bk_update_type : CGUpdateType
            How to compute :math:`b_k`. Can be "FR", "PRP", "DY" and "BAN". Default "FR".
        """
        super().__init__()
        self.num_iters = num_iters
        self.tol = tol
        self.bk_update_type = bk_update_type

        self.forward_operator = forward_operator
        self.backward_operator = backward_operator

        self._spatial_dims = (2, 3)
        self._coil_dim = 1

    def _A_star_op(
        self, kspace: torch.Tensor, sensitivity_map: torch.Tensor, sampling_mask: torch.Tensor
    ) -> torch.Tensor:
        r"""Computes :math:`\mathcal{A}^{*}(y)`.

        Parameters
        ----------
        kspace : torch.Tensor
            K-space of shape (N, coil, height, width, complex=2).
        sensitivity_map : torch.Tensor
            Coil sensitivities of shape (N, coil, height, width, complex=2).
        sampling_mask : torch.Tensor
            Sampling mask of shape (N, 1, height, width, 1).

        Returns
        -------
        torch.Tensor
            Projected multi-coil k-space to image domain.
        """
        return reduce_operator(
            self.backward_operator(
                torch.where(sampling_mask == 0, torch.tensor([0.0], dtype=kspace.dtype).to(kspace.device), kspace),
                dim=self._spatial_dims,
            ),
            sensitivity_map,
            dim=self._coil_dim,
        )

    def _A_star_A_op(
        self, image: torch.Tensor, sensitivity_map: torch.Tensor, sampling_mask: torch.Tensor
    ) -> torch.Tensor:
        r"""Computes :math:`\mathcal{A}^{*} \circ \mathcal{A}(x)`.

        Parameters
        ----------
        image : torch.Tensor
            Image of shape (N, height, width, complex=2).
        sensitivity_map : torch.Tensor
            Coil sensitivities of shape (N, coil, height, width, complex=2).
        sampling_mask : torch.Tensor
            Sampling mask of shape (N, 1, height, width, 1).

        Returns
        -------
        torch.Tensor
        """
        k = self.forward_operator(expand_operator(image, sensitivity_map, dim=self._coil_dim), dim=self._spatial_dims)
        return self._A_star_op(k, sensitivity_map, sampling_mask)

    def B_op(
        self, x: torch.Tensor, sensitivity_map: torch.Tensor, sampling_mask: torch.Tensor, lambd: torch.Tensor
    ) -> torch.Tensor:
        r"""Computes :math:`\mathcal{B}(x) = (\mathcal{A}^{*} \circ \mathcal{A}+ \lambda I) (x)`

        Parameters
        ----------
        x : torch.Tensor
            Image of shape (N, height, width, complex=2).
        sensitivity_map : torch.Tensor
            Coil sensitivities of shape (N, coil, height, width, complex=2).
        sampling_mask : torch.Tensor
            Sampling mask of shape (N, 1, height, width, 1).
        lambd : torch.Tensor
            Regularaziation parameter of shape (1).

        Returns
        -------
        torch.Tensor
        """
        return self._A_star_A_op(x, sensitivity_map, sampling_mask) + lambd * x

    def cg(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        sensitivity_map: torch.Tensor,
        sampling_mask: torch.Tensor,
        lambd: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        r"""Computes the conjugate gradient algorithm.

        Parameters
        ----------
        x : torch.Tensor
            Guess for :math:`x_0` of shape (N, height, width, complex=2).
        y : torch.Tensor
            Initial/masked k-space of shape (N, coil, height, width, complex=2).
        sensitivity_map : torch.Tensor
            Sensitivity map of shape (N, coil, height, width, complex=2).
        sampling_mask : torch.Tensor
            Sampling mask of shape (N, 1, height, width, 1).
        lambd : torch.Tensor
            Regularaziation parameter of shape (1).
        z : torch.Tensor
            Denoised input of shape (N, height, width, complex=2).

        Returns
        -------
        torch.Tensor
            `x_K`.
        """
        # pylint: disable=too-many-locals
        dim = torch.arange(1, x.ndim - 1).tolist()
        shape = [x.shape[0]] + [1 for _ in range(len(x.shape[1:]) - 1)] + [2]

        b = self._A_star_op(y, sensitivity_map, sampling_mask) + lambd * z
        rk_old = b - self.B_op(x, sensitivity_map, sampling_mask, lambd)
        pk = rk_old.clone()

        rk_norm_sq_old = complex_dot_product(rk_old, rk_old, dim)
        for _ in range(self.num_iters):
            Bpk = self.B_op(pk, sensitivity_map, sampling_mask, lambd)

            ak = complex_division(rk_norm_sq_old, complex_dot_product(rk_old, Bpk, dim)).reshape(shape)

            x = x + complex_multiplication(ak, pk)
            rk_new = rk_old - complex_multiplication(ak, Bpk)

            rk_norm_sq_new = complex_dot_product(rk_new, rk_new, dim)
            if rk_norm_sq_new.abs().sqrt().mean() < self.tol:
                break

            if self.bk_update_type == "FR":
                bk = complex_division(rk_norm_sq_new, rk_norm_sq_old)
            elif self.bk_update_type == "PRP":
                bk = _PRP(rk_new, rk_old, dim)
            elif self.bk_update_type == "DY":
                bk = _DY(rk_new, rk_old, pk, dim)
            else:
                bk = _BAN(rk_new, rk_old, dim)
            bk = bk.reshape(shape)

            pk = rk_new + complex_multiplication(bk, pk)

            rk_norm_sq_old = rk_norm_sq_new.clone()
            rk_old = rk_new.clone()

        return x

    def forward(
        self,
        masked_kspace: torch.Tensor,
        sensitivity_map: torch.Tensor,
        sampling_mask: torch.Tensor,
        z: torch.Tensor,
        lambd: torch.Tensor,
    ) -> torch.Tensor:
        """Performs forward pass of :class:`ConjGrad`.

        Parameters
        ----------
        masked_kspace : torch.Tensor
            Masked k-space of shape (N, coil, height, width, complex=2).
        sensitivity_map : torch.Tensor
            Coil sensitivities of shape (N, coil, height, width, complex=2).
        sampling_mask : torch.Tensor
            Sampling mask of shape (N, 1, height, width, 1).
        z : torch.Tensor
            Prediction of image of shape (N, height, width, complex=2).
        lambd : torch.Tensor
            Regularaziation (trainable or not) parameter of shape (1).

        Returns
        -------
        torch.Tensor
        """
        return self.cg(z, masked_kspace, sensitivity_map, sampling_mask, lambd, z)


def _PRP(rk_new: torch.Tensor, rk_old: torch.Tensor, dim: List[int]) -> torch.Tensor:
    r"""Polak-Ribiere-Polyak (PRP) update method for :math:`b_k`:

    .. math ::

        b_k = \frac{ r_{k+1}^{*}(r_{k+1} - r_k) }{ ||r_k||_2^2 } ,

    Parameters
    ----------
    rk_new : torch.Tensor
        Input for :math:`r_{k+1}`.
    rk_old : torch.Tensor
        Input for :math:`r_k`.
    dim : List[int]
        Dimensions which will be suppressed. Useful when inputs are batched.

    Returns
    -------
    bk : torch.Tensor
        PRP computation for :math:`b_k`.
    """
    yk = rk_new - rk_old
    return complex_division(complex_dot_product(rk_new, yk, dim), complex_dot_product(rk_old, rk_old, dim))


def _DY(rk_new: torch.Tensor, rk_old: torch.Tensor, pk: torch.Tensor, dim: List[int]) -> torch.Tensor:
    r"""Dai-Yuan (DY) update method for :math:`b_k`:

    .. math ::

        b_k = \frac{ ||r_{k+1}||_2^2 }{ p_{k}^{*} y_k } , y_k = r_{k+1} - r_k.

    Parameters
    ----------
    rk_new : torch.Tensor
        Input for :math:`r_{k+1}`.
    rk_old : torch.Tensor
        Input for :math:`r_k`.
    pk : torch.Tensor
        Input fot :math:`p_k`.
    dim : List[int]
        Dimensions which will be suppressed. Useful when inputs are batched.

    Returns
    -------
    bk : torch.Tensor
        DY computation for :math:`b_k`.
    """
    yk = rk_new - rk_old
    return complex_division(complex_dot_product(rk_new, rk_new, dim), complex_dot_product(pk, yk, dim))


def _BAN(rk_new: torch.Tensor, rk_old: torch.Tensor, dim: List[int]) -> torch.Tensor:
    r"""Bamigbola-Ali-Nwaeze (BAN) update method for :math:`b_k`:

    .. math ::

        b_k = \frac{ r_{k+1}^{*} y_k }{ r_{k}^{*} y_k }, y_k = r_{k+1} - r_k.

    Parameters
    ----------
    rk_new : torch.Tensor
        Input for :math:`r_{k+1}`.
    rk_old : torch.Tensor
        Input for :math:`r_k`.
    dim : List[int]
        Dimensions which will be suppressed. Useful when inputs are batched.

    Returns
    -------
    bk : torch.Tensor
        BAN computation for :math:`b_k`.
    """
    yk = rk_new - rk_old
    return complex_division(complex_dot_product(rk_new, yk, dim), complex_dot_product(rk_old, yk, dim))
