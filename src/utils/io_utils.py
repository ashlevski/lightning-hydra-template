import nibabel as nib
import torch
import numpy as np
def save_tensor_to_nifti(tensor, file_path):
    # Convert the PyTorch tensor to a numpy array
    # Assume that the tensor is a 3D volume
    tensor_np = tensor.cpu().numpy()

    # Create a NIfTI image using the numpy array
    # For this, we need to have a suitable affine matrix. Here we use an identity matrix.
    # You may need to use an appropriate affine matrix for your specific data.
    affine = np.eye(4)
    nifti_img = nib.Nifti1Image(tensor_np, affine)

    # Save the NIfTI image
    nib.save(nifti_img, file_path)

def save_to_nifti(data, file_path):
    # Convert the PyTorch tensor to a numpy array
    # Assume that the tensor is a 3D volume


    # Create a NIfTI image using the numpy array
    # For this, we need to have a suitable affine matrix. Here we use an identity matrix.
    # You may need to use an appropriate affine matrix for your specific data.
    affine = np.eye(4)
    nifti_img = nib.Nifti1Image(data, affine)

    # Save the NIfTI image
    nib.save(nifti_img, file_path)