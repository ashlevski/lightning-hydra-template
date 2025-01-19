import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset


def identity_grid(shape, stackdim, dtype=torch.float32, device="cpu"):
    """Create an nd identity grid."""
    tensors = (torch.arange(s, dtype=dtype, device=device) for s in shape)
    return torch.stack(
        torch.meshgrid(*tensors)[::-1], stackdim
    )  # z,y,x shape and flip for x, y, z coords


class Resampler(nn.Module):
    """
    Generic resampler for 2D and 3D images.
    Expects voxel coordinates as coord_field
    Args:
        input (Tensor): input batch (N x C x IH x IW) or (N x C x ID x IH x IW)
        grid (Tensor): flow-field of size (N x OH x OW x 2) or (N x OD x OH x OW x 3)
        padding_mode (str): padding mode for outside grid values
            'zeros' | 'border'. Default: 'zeros'
    """

    def __init__(
        self, coord_dim: int = 1, mode: str = "bilinear", padding_mode: str = "border"
    ):
        super().__init__()
        self.mode = mode
        self.padding_mode = padding_mode
        self.coord_dim = coord_dim

    def forward(self, input, coord_grid):
        im_shape = input.shape[2:]
        assert coord_grid.shape[self.coord_dim] == len(
            im_shape
        )  # number of coordinates should match image dimension

        coord_grid = coord_grid.movedim(self.coord_dim, -1)

        # scale for pytorch grid_sample function
        max_extent = (
            torch.tensor(
                im_shape[::-1], dtype=coord_grid.dtype, device=coord_grid.device
            )
            - 1
        )
        coord_grid = 2 * (coord_grid / max_extent) - 1
        return F.grid_sample(
            input,
            coord_grid,
            mode=self.mode,
            padding_mode=self.padding_mode,
            align_corners=True,
        )


class StableStd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        assert tensor.numel() > 1
        ctx.tensor = tensor.detach()
        res = torch.std(tensor).detach()
        ctx.result = res.detach()
        return res

    @staticmethod
    def backward(ctx, grad_output):
        tensor = ctx.tensor.detach()
        result = ctx.result.detach()
        e = 1e-6
        assert tensor.numel() > 1
        return (
            (2.0 / (tensor.numel() - 1.0))
            * (grad_output.detach() / (result.detach() * 2 + e))
            * (tensor.detach() - tensor.mean().detach())
        )


stablestd = StableStd.apply


class ScaledTanH(nn.Module):
    def __init__(self, scaling):
        super().__init__()
        self.scaling = scaling

    def forward(self, input):
        return torch.tanh(input) * self.scaling

    def __repr__(self):
        return self.__class__.__name__ + "(" + "scaling = " + str(self.scaling) + ")"


class BiasedTanh(nn.Module):
    def __init__(self, scale_in=1.0, scale_out=1.0, bias=0.0):
        super().__init__()
        self.scale_in = scale_in
        self.scale_out = scale_out
        self.bias = bias

    def forward(self, input):
        return torch.tanh(input * self.scale_in) * self.scale_out + self.bias

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "scale_in="
            + str(self.scale_in)
            + ", "
            + "scale_out="
            + str(self.scale_out)
            + ", "
            + "bias="
            + str(self.bias)
            + ")"
        )


class ScalingAF(nn.Module):
    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, input):
        return self.scale_factor ** torch.tanh(input)

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "scale_factor="
            + str(self.scale_factor)
            + ")"
        )


class IRDataSet(Dataset):
    """
    Wrapper to convert a dataset into a dataset suitable for image registration experiments.
    """

    def __init__(self, ds, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ds = ds

    def __len__(self):
        return len(self.ds) * len(self.ds)

    def __getitem__(self, idx):
        fixed_idx = idx // len(self.ds)
        moving_idx = idx % len(self.ds)
        return {"fixed": self.ds[fixed_idx], "moving": self.ds[moving_idx]}


# Bspline functions
def bspline_kernel_nd(t, order, dtype=float, **kwargs):
    tpowers = t ** torch.arange(order, 0 - 1, -1, dtype=dtype)
    if order == 1:
        return tpowers @ torch.tensor(((-1, 1), (1, 0)), dtype=dtype)
    elif order == 2:
        return (
            tpowers
            @ torch.tensor(((1, -2, 1), (-2, 2, 0), (1, 1, 0)), dtype=dtype)
            / 2.0
        )
    elif order == 3:
        return (
            tpowers
            @ torch.tensor(
                ((-1, 3, -3, 1), (3, -6, 3, 0), (-3, 0, 3, 0), (1, 4, 1, 0)),
                dtype=dtype,
            )
            / 6.0
        )


def bspline_convolution_kernel(upsampling_factors, order, dtype=float):
    ndim = len(upsampling_factors)
    for i, us_factor in enumerate(upsampling_factors):
        t = torch.linspace(1 - (1 / us_factor), 0, us_factor)
        ker1D = bspline_kernel_nd(t[:, None], order, dtype).T.flatten()
        shape = (1,) * i + ker1D.shape + (1,) * (ndim - 1 - i)
        try:
            kernel = kernel * ker1D.view(shape)
        except NameError:
            kernel = ker1D.view(shape)
    return kernel


# transform
def scaling_matrix_eff(x, ndim=2):
    assert (
        x.ndim == 2
    ), "Input should be a tensor of (m, n), where m is number of instances, and n number of parameters."
    assert ndim in (2, 3), "Only 2D and 3D implemented."
    assert ndim == x.shape[-1], f"Number of parameters for {ndim}D should be {ndim}."
    dtype, device = x.dtype, x.device
    T = torch.zeros((len(x), ndim, ndim), dtype=dtype, device=device)
    sel_mask = torch.eye(ndim, device=device, dtype=torch.bool)
    T[:, sel_mask] = x
    return T


def rotation_matrix_eff(x, axis=0, ndim=2):
    """
    For 3D axis = x: 2, y: 1, z: 0.
    """
    assert (
        x.ndim == 2 and x.shape[-1] == 1
    ), "Input should be a tensor of (m, 1), where m is number of instances."
    assert ndim in (2, 3), "Only 2D and 3D implemented."
    dtype, device = x.dtype, x.device
    T = torch.eye(ndim, dtype=dtype, device=device)[None].repeat(len(x), 1, 1)
    lidx, hidx = ((0, 1), (0, 2), (1, 2))[axis]
    c = torch.cos(x)
    s = torch.sin(x)
    T[:, lidx, lidx] = c.squeeze()
    T[:, lidx, hidx] = s.squeeze()
    T[:, hidx, lidx] = -s.squeeze()
    T[:, hidx, hidx] = c.squeeze()
    return T


def shearing_matrix_eff(x, ndim=2):
    assert (
        x.ndim == 2
    ), "Input should be a tensor of (m, n), where m is number of instances, and n number of parameters."
    assert ndim in (2, 3), "Only 2D and 3D implemented."
    assert (ndim - 1) * ndim == x.shape[
        -1
    ], f"Number of parameters for {ndim}D should be {(ndim - 1) * ndim}"
    dtype, device = x.dtype, x.device
    T = torch.eye(ndim, dtype=dtype, device=device)[None].repeat(len(x), 1, 1)
    T[:, ~torch.eye(ndim, device=device, dtype=torch.bool)] = torch.tan(x)
    return T


def translation_matrix(x, ndim=2):  # not used for efficient transforms
    assert (
        x.ndim == 2
    ), "Input should be a tensor of (m, n), where m is number of instances, and n number of parameters."
    assert ndim in (2, 3), "Only 2D and 3D implemented."
    assert ndim == x.shape[-1], f"Number of parameters for {ndim}D should be {ndim}."
    dtype, device = x.dtype, x.device
    T = torch.eye(ndim + 1, dtype=dtype, device=device)[None].repeat(len(x), 1, 1)
    T[:, :ndim, ndim] = x
    return T


def tmat_2d_affine(rotation, scale, shear):
    c = torch.cos(rotation[:, 0])
    s = torch.sin(rotation[:, 0])
    sx = scale[:, 0]
    sy = scale[:, 1]
    sh = torch.tan(shear[:, 0])
    T = torch.stack((sx * (c - s * sh), -s * sy, (sx * (s + c * sh)), sy * c)).view(
        -1, 2, 2
    )
    return T


shearmat2d = shearing_matrix_eff
rotmat2d = rotation_matrix_eff
scalemat2d = scaling_matrix_eff


def batch_transform_efficient(
    coord_grid, coord_dim, Tmat, translation, Forigin, Morigin
):
    # print(coord_grid.shape)
    coord_grid = coord_grid.movedim(coord_dim, -1)
    # print(coord_grid.shape)
    shape = coord_grid.shape
    coord_grid = coord_grid.view(shape[0], -1, shape[-1])  # flatten
    # print(coord_grid.shape, Forigin.shape)
    coord_grid = coord_grid + Forigin[:, None]
    coord_grid = coord_grid @ Tmat.transpose(
        2, 1
    )  # transpose because of switched order
    coord_grid = coord_grid + (translation[:, None] - Morigin[:, None])
    shape = (len(coord_grid),) + shape[1:]  # enable broadcasting
    coord_grid = coord_grid.view(shape)  # recover original shape
    coord_grid = coord_grid.movedim(-1, coord_dim)
    return coord_grid


def batch_transform(coord_grid, coord_dim, Tmat, Forigin, Morigin):
    coord_grid = coord_grid.movedim(coord_dim, -1)
    ndim = coord_grid.shape[-1]
    Tmat = translation_matrix(-Morigin) @ Tmat @ translation_matrix(Forigin)
    coord_grid = torch.cat(
        (
            coord_grid,
            torch.ones(
                coord_grid.shape[:-1] + (1,),
                dtype=coord_grid.dtype,
                device=coord_grid.device,
            ),
        ),
        dim=ndim + 1,
    )  # add extra dimension for MM
    shape = coord_grid.shape
    coord_grid = coord_grid.view(shape[0], -1, shape[-1])  # flatten
    coord_grid = coord_grid @ Tmat.transpose(2, 1)  # MM
    coord_grid = coord_grid.view(shape)  # recover shape
    coord_grid = coord_grid.narrow(ndim + 1, 0, ndim)  # remove extra dimension
    return coord_grid.movedim(-1, coord_dim)
