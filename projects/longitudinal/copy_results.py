import os
import shutil
import nibabel as nib
import numpy as np
# Dictionary of paths for each acceleration
paths_dict = {
    'R5': [
        "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-05-01_00-25-49/",
        "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-04-29_19-15-43/",
        "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-05-01_00-14-57/",
        "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-04-29_13-26-36/",
        "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-04-30_11-05-35/",
        "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-06-11_09-31-49/",
        "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-06-12_20-15-42/"
    ],
    'R10': [
        "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-05-01_00-27-51/",
        "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-04-23_13-57-04/",
        "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-05-01_00-17-23/",
        "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-04-27_09-49-21/",
        "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-04-26_09-43-42/",
        "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-06-10_13-57-11/",
        "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-06-12_20-16-09/"
    ],
    'R15': [
        "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-05-01_00-33-50/",
        "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-04-29_19-21-45/",
        "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-05-01_00-20-21/",
        "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-04-29_14-09-19/",
        "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-04-30_11-06-35/",
        "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-05-31_16-31-59/",
        "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-06-03_17-59-04/"
    ],
    'R20': [
        "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-05-01_00-36-21/",
        "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-04-29_22-59-31/",
        "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-05-01_00-21-50/",
        "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-04-29_23-15-39/",
        "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-04-30_01-44-07/",
        "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-06-11_09-41-25/",
        "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-06-12_20-17-09/"
    ]
}

methods = [
    'Baseline_Souza_IKIK',
    'Baseline_Ours_E2EVarNet',
    'Enhanced_Souza_UNet',
    'Enhanced_Ours_UNet',
    'Enhanced_Ours_Transformers',
    'Enhanced_Trained_and_tested_on_the_atlas',
    'Enhanced_Trained_on_previous_scan_and_tested_on_the_atlas'
]


# Define the destination path
dest_path = "/home/amirmohammad.shamaei/imri_result/"

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