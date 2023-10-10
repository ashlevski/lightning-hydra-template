# MRI Data Loader and Transforms

This directory provides a reference data loader to read the fastMRI data one
slice at a time and some useful data transforms to work with the data in
PyTorch.

Each partition (train, validation or test) of the fastMRI data is distributed
as a set of HDF5 files, such that each HDF5 file contains data from one MR
acquisition. The set of fields and attributes in these HDF5 files depends on
the track (single-coil or multi-coil) and the data partition.

## 2020 fastMRI Challenge (Brain Data)

For the 2020 fastMRI challenge, all data will be on the brain data with a
multi-coil track.

### Multi-Coil Track (2020)

* Training & Validation data:
  * `kspace`: Multi-coil k-space data. The shape of the kspace tensor is
  (number of slices, number of coils, height, width).
  * `reconstruction_rss`: Root-sum-of-squares reconstruction of the multi-coil
  k-space data cropped to the center.
* Test data:
  * `kspace`: Undersampled multi-coil k-space. The shape of the kspace tensor
  is (number of slices, number of coils, height, width).
  * `mask` Defines the undersampled Cartesian k-space trajectory. The number of
  elements in the mask tensor is the same as the width of k-space.

## 2019 fastMRI Challenge (Knee Data)

### Single-Coil Track

* Training & Validation data:
  * `kspace`: Emulated single-coil k-space data. The shape of the kspace tensor
  is (number of slices, height, width).
  * `reconstruction_rss`: Root-sum-of-squares reconstruction of the multi-coil
  k-space that was
    used to derive the emulated single-coil k-space cropped to the center 320
    × 320 region.
    The shape of the reconstruction rss tensor is (number of slices, 320, 320).
  * `reconstruction_esc`: The inverse Fourier transform of the single-coil
  k-space data cropped to the center 320 × 320 region. The shape of the
  reconstruction esc tensor is (number of slices, 320, 320).
* Test data:
  * `kspace`: Undersampled emulated single-coil k-space. The shape of the
  kspace tensor is (number of slices, height, width).
  * `mask`: Defines the undersampled Cartesian k-space trajectory. The number
  of elements in the mask tensor is the same as the width of k-space.

### Multi-Coil Track (2019)

* Training & Validation data:
  * `kspace`: Multi-coil k-space data. The shape of the kspace tensor is
  (number of slices, number of coils, height, width).
  * `reconstruction_rss`: Root-sum-of-squares reconstruction of the multi-coil
  k-space data cropped to the center 320 × 320 region. The shape of the
  reconstruction rss tensor is (number of slices, 320, 320).
* Test data:
  * `kspace`: Undersampled multi-coil k-space. The shape of the kspace tensor
  is (number of slices, number of coils, height, width).
  * `mask` Defines the undersampled Cartesian k-space trajectory. The number
  of elements in the mask tensor is the same as the width of k-space.

## Data Transforms

`data.transforms` provides a number of useful data transformation functions
that work with PyTorch tensors.

## Data Loader

`fastmri.data.mri_data` provides a `SliceDataset` class to read one MR slice at
a time. It takes as input a `transform` function or callable object that can be
used transform the data into the format that you need. This makes the data
loader versatile and can be used to run different kinds of reconstruction
methods.

The following is a simple example for how to use the data loader. For more
concrete examples, please look at the baseline model code in the
`fastmri_examples` directory.

```python
import pathlib
from fastmri.data import subsample
from fastmri.data import transforms, mri_data

# Create a mask function
mask_func = subsample.RandomMaskFunc(
    center_fractions=[0.08, 0.04],
    accelerations=[4, 8]
)

def data_transform(kspace, mask, target, data_attributes, filename, slice_num):
    # Transform the data into appropriate format
    # Here we simply mask the k-space and return the result
    kspace = transforms.to_tensor(kspace)
    masked_kspace = transforms.apply_mask(kspace, mask_func)
    return masked_kspace

dataset = mri_data.SliceDataset(
    root=pathlib.Path('/path/to/data'),
    transform=data_transform,
    challenge='singlecoil'
)

for masked_kspace in dataset:
    # Do reconstruction
    pass
```
