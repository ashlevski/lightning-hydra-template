import torch
from torch import Tensor

from projects.longitudinal.torchir.utils import identity_grid


def bending_energy_3d(
    coord_grid: Tensor, vector_dim: int = -1, dvf_input: bool = False
) -> Tensor:
    """Calculates bending energy penalty for a 3D coordinate grid.

    For further details regarding this regularization please read the work by `Rueckert 1999`_.

    Args:
        coord_grid: 3D coordinate grid, i.e. a 5D Tensor with standard dimensions
        (n_samples, 3, z, y, x).
        vector_dim: Specifies the location of the vector dimension. Default: -1
        dvf_input: If ``True``, coord_grid is assumed a displacement vector field and
        an identity_grid will be added. Default: ``False``

    Returns:
        Bending energy per instance in the batch.

    .. _Rueckert 1999: https://ieeexplore.ieee.org/document/796284

    """
    assert coord_grid.ndim == 5, "Input tensor should be 5D, i.e. 3D images."

    if vector_dim != 1:
        coord_grid = coord_grid.movedim(vector_dim, -1)

    if dvf_input:
        coord_grid = coord_grid + identity_grid(coord_grid.shape[2:], stackdim=0)

    d_z = torch.diff(coord_grid, dim=1)
    d_y = torch.diff(coord_grid, dim=2)
    d_x = torch.diff(coord_grid, dim=3)

    d_zz = torch.diff(d_z, dim=1)[:, :, :-2, :-2]
    d_zy = torch.diff(d_z, dim=2)[:, :-1, :-1, :-2]
    d_zx = torch.diff(d_z, dim=3)[:, :-1, :-2, :-1]
    d_yy = torch.diff(d_y, dim=2)[:, :-2, :, :-2]
    d_yx = torch.diff(d_y, dim=3)[:, :-2, :-1, :-1]
    d_xx = torch.diff(d_x, dim=3)[:, :-2, :-2, :]

    return torch.mean(
        d_zz ** 2 + d_yy ** 2 + d_xx ** 2 + 2 * (d_zy ** 2 + d_zx ** 2 + d_yx ** 2),
        axis=(1, 2, 3, 4),
    )


def bending_energy_2d(
    coord_grid: Tensor, vector_dim: int = -1, dvf_input: bool = False
) -> Tensor:
    """Calculates bending energy penalty for a 2D coordinate grid.

    For further details regarding this regularization please read the work by `Rueckert 1999`_.

    Args:
        coord_grid: 2D coordinate grid, i.e. a 4D Tensor with standard dimensions
        (n_samples, 2, y, x).
        vector_dim: Specifies the location of the vector dimension. Default: -1
        dvf_input: If ``True``, coord_grid is assumed a displacement vector field and
        an identity_grid will be added. Default: ``False``

    Returns:
        Bending energy per instance in the batch.

    .. _Rueckert 1999: https://ieeexplore.ieee.org/document/796284

    """
    assert coord_grid.ndim == 4, "Input tensor should be 4D, i.e. 2D images."

    if vector_dim != 1:
        coord_grid = coord_grid.movedim(vector_dim, -1)

    if dvf_input:
        coord_grid = coord_grid + identity_grid(coord_grid.shape[2:], stackdim=0)

    d_y = torch.diff(coord_grid, dim=1)
    d_x = torch.diff(coord_grid, dim=2)

    d_yy = torch.diff(d_y, dim=1)[:, :, :-2]
    d_yx = torch.diff(d_y, dim=2)[:, :-1, :-1]
    d_xx = torch.diff(d_x, dim=2)[:, :-2, :]

    return torch.mean(d_yy ** 2 + d_xx ** 2 + 2 * d_yx ** 2, axis=(1, 2, 3))
