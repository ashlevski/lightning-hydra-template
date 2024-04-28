import csv

import numpy as np

import os
import h5py
import nibabel as nib


def convert_nifti_to_h5(in_folder_path,out_folder_path):
    # Create HDF5 file
        if not os.path.exists(out_folder_path):
            os.makedirs(out_folder_path)
        for root, dirs, files in os.walk(in_folder_path):
            for file in files:
                if file.endswith(".nii") and "preds" not in file:  # Assuming NIfTI files are in gzip format
                    file_path = os.path.join(root, file)

                    # Read NIfTI file
                    nifti_image = nib.load(file_path)
                    nifti_data = nifti_image.get_fdata()

                    output_file = os.path.join(out_folder_path, file.replace("_targets.nii", ""))+".h5"
                    with h5py.File(output_file, 'w') as h5_file:
                    # Create a dataset in HDF5 file with the "image" keyword
                        h5_file.create_dataset("image", data=nifti_data)


if __name__ == "__main__":
    in_folder_path = "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-04-20_18-15-19/"
    out_folder_path = "/work/souza_lab/amir/Data/roberto/Reference_e2ev/"

    convert_nifti_to_h5(in_folder_path,out_folder_path)

print('Text file converted to CSV successfully!')