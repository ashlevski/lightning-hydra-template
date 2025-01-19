import os
import shutil
import nibabel as nib
import numpy as np
# Dictionary of paths for each acceleration
paths_dict = {
    'R5': [

    ],
    'R10': [

    ],
    'R15': [

    ],
    'R20': [
        "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-07-16_18-49-40/",
        "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-07-16_19-18-17/",
    ]
}

methods = [
    'R_20',
    'R_20_per',
]


# Define the destination path
dest_path = "/home/amirmohammad.shamaei/MRI_REC_template/projects/uncertainty/"

# Loop through each acceleration rate and its corresponding paths
for acceleration, paths in paths_dict.items():
    # Create a directory for the acceleration rate if it doesn't exist
    acceleration_dir = os.path.join(dest_path, acceleration)
    os.makedirs(acceleration_dir, exist_ok=True)

    # Loop through each path for the acceleration rate
    for i, path in enumerate(paths):
        # Get the method name from the methods list
        method = methods[i]

        # Create a directory for the method if it doesn't exist
        method_dir = os.path.join(acceleration_dir, method)
        os.makedirs(method_dir, exist_ok=True)

        # Copy files from the path to the method directory
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if os.path.isfile(file_path) and ".nii" in file:  # Check if it's a file and contains ".nii"
                                # Read NIfTI file
                nifti_image = nib.load(file_path)
                nifti_data = nifti_image.get_fdata()

                # Normalize the values to the range of 0 to 255
                nifti_norm = (nifti_data) 
                # nifti_norm = nifti_norm.astype('uint8')

                # Create a new NIfTI image with the normalized data
                normalized_nifti = nib.Nifti1Image(np.flip(nifti_norm.transpose([2, 1, 0])), nifti_image.affine, nifti_image.header)

                output_file = os.path.join(method_dir, file)
                nib.save(normalized_nifti, output_file)
                # shutil.copy2(file_path, method_dir)