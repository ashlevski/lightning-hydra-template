import csv

# # Open the text file for reading
# with open('/home/ai2lab/datasets/roberto/test.txt') as f:
#     # Read all lines
#     lines = f.readlines()

import os
import h5py
import nibabel as nib


def convert_nifti_to_h5(folder_path):
    # Create HDF5 file

        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".nii.gz"):  # Assuming NIfTI files are in gzip format
                    file_path = os.path.join(root, file)

                    # Read NIfTI file
                    nifti_image = nib.load(file_path)
                    nifti_data = nifti_image.get_fdata()

                    output_file = os.path.join(root, file[:-9])+".h5"
                    with h5py.File(output_file, 'w') as h5_file:
                    # Create a dataset in HDF5 file with the "image" keyword
                        h5_file.create_dataset("image", data=nifti_data)


if __name__ == "__main__":
    folder_path = "/home/ai2lab/datasets/roberto/Reference/"
    folder_path = "/home/ai2lab/datasets/roberto/PS-reg/10x/"
    folder_path = "/home/ai2lab/datasets/roberto/Non-enhanced/10x/"
    folder_path = "/home/ai2lab/datasets/roberto/Norm-baseline-nifti-reorient/"
    folder_path = "/work/souza_lab/amir/Data/roberto/Non-enhanced/15x/"
    # output_file = "output.h5"

    convert_nifti_to_h5(folder_path)

print('Text file converted to CSV successfully!')