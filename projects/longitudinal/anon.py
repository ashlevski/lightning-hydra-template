import os
import csv
import random
import string
import shutil
from pathlib import Path
import hashlib

def generate_deterministic_string(input_string, length=3):
    """Generate a deterministic string based on the input string"""
    hash_object = hashlib.md5(input_string.encode())
    hash_hex = hash_object.hexdigest()
    return hash_hex[:length].upper()

def rename_nifti_file(original_name):
    """Rename a NIfTI file according to the specified pattern"""
    random_string = generate_deterministic_string(original_name.split("_")[0] +original_name.split("_")[1]+  original_name.split("_")[2]+  original_name.split("_")[3] +  original_name.split("_")[4] )
    if original_name.endswith("_preds.nii"):
        return f"{random_string}_" + original_name.split("_")[-3] + "_" + original_name.split("_")[-2] + "_preds.nii"
    elif original_name.endswith("_preds_seg.nii"):
        return f"{random_string}_"  + original_name.split("_")[-4] + "_" + original_name.split("_")[-3] + "_preds_seg.nii"
    elif original_name.endswith("_targets.nii"):
        return original_name.split("_")[-3] + "_" + original_name.split("_")[-2] + "_targets.nii"
    elif original_name.endswith("_targets_seg.nii"):
        return original_name.split("_")[-4] + "_" + original_name.split("_")[-3] + "_targetss_seg.nii"
    else:
        return original_name

def copy_and_rename_nifti_files(folder_path, output_folder):
    """Copy NIfTI files to the output folder with renamed filenames and create a CSV with original and new names"""
    input_folder = Path(folder_path)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    csv_file = output_folder / "filename_mapping.csv"
    
    # List to store the original and new filenames
    filename_mapping = []
    
    # Iterate through all files in the input folder
    for file in input_folder.glob("*.nii*"):  # This will match .nii and .nii.gz files
        original_name = file.name
        new_name = rename_nifti_file(original_name)
        
        # Copy the file to the output folder with the new name
        shutil.copy2(file, output_folder / new_name)
        
        # Add the mapping to our list
        filename_mapping.append((original_name, new_name))
    
    # Write the mapping to a CSV file
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Original Name', 'New Name'])  # Header
        writer.writerows(filename_mapping)
    
    print(f"Copied and renamed {len(filename_mapping)} files.")
    print(f"Mapping saved to {csv_file}")

dist = 'sample_results_toR'
# Example usage
input_folder = f"/home/amirmohammad.shamaei/imri_result/{dist}/"
output_folder = f"/home/amirmohammad.shamaei/imri_result/{dist}_renamed/"
copy_and_rename_nifti_files(input_folder, output_folder)