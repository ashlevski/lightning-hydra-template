# coding=utf-8
# Copyright (c) DIRECT Contributors
from typing import List, Union

import numpy as np
import torch


def crop_to_bbox(
    data: Union[np.ndarray, torch.Tensor], bbox: List[int], pad_value: int = 0
) -> Union[np.ndarray, torch.Tensor]:
    """Extract bbox from images, coordinates can be negative.

    Parameters
    ----------
    data: np.ndarray or torch.tensor
       nD array or torch tensor.
    bbox: list or tuple
       bbox of the form (coordinates, size),
       for instance (4, 4, 2, 1) is a patch starting at row 4, col 4 with height 2 and width 1.
    pad_value: number
       if bounding box would be out of the image, this is value the patch will be padded with.

    Returns
    -------
    ndarray
        Numpy array of data cropped to BoundingBox
    """
    if not isinstance(data, (np.ndarray, torch.Tensor)):
        raise ValueError(f"Expected `data` to be ndarray or tensor. Got {type(data)}.")

    # Coordinates, size
    ndim = len(bbox) // 2
    if len(bbox) % 2 != 0:
        raise ValueError(f"Bounding box should have the form of [x_0, x_1, ..., h_0, h_1], but got length {ndim}.")
    bbox_coords, bbox_size = np.asarray(bbox[:ndim]), np.asarray(bbox[ndim:])
    # Offsets
    l_offset = -bbox_coords.copy()
    l_offset[l_offset < 0] = 0

    r_offset = (bbox_coords + bbox_size) - np.array(data.shape)
    r_offset[r_offset < 0] = 0

    region_idx = [slice(i, j) for i, j in zip(bbox_coords + l_offset, bbox_coords + bbox_size - r_offset)]

    if isinstance(data, torch.Tensor):
        # TODO(jt): Investigate if clone is needed
        out = data[tuple(region_idx)].clone()
    else:
        out = data[tuple(region_idx)].copy()

    if np.all(l_offset == 0) and np.all(r_offset == 0):
        return out

    # If we have a positive offset, we need to pad the patch.
    if isinstance(data, torch.Tensor):
        patch = pad_value * torch.ones(bbox_size.tolist(), dtype=data.dtype)
    else:
        patch = pad_value * np.ones(bbox_size.tolist(), dtype=data.dtype)

    patch_idx = [slice(i, j) for i, j in zip(l_offset, bbox_size - r_offset)]
    patch[tuple(patch_idx)] = out

    return patch


def crop_to_largest(
    data: List[Union[np.ndarray, torch.Tensor]], pad_value: int = 0
) -> List[Union[np.ndarray, torch.Tensor]]:
    """Given a list of arrays or tensors, return the same list with the data padded to the largest in the set. Can be
    convenient for e.g. logging and tiling several images as with torchvision's `make_grid'`

    Parameters
    ----------
    data: List[Union[np.ndarray, torch.Tensor]]
    pad_value: int

    Returns
    -------
    List[Union[np.ndarray, torch.Tensor]]
    """
    if not data:
        return data

    shapes = np.asarray([_.shape for _ in data])
    max_shape = shapes.max(axis=0)

    crop_start_per_shape = [-(max_shape - np.asarray(_)) // 2 for _ in shapes]
    crop_boxes = [_.tolist() + max_shape.tolist() for _ in crop_start_per_shape]

    return [crop_to_bbox(curr_data, bbox, pad_value=pad_value) for curr_data, bbox in zip(data, crop_boxes)]
