import os
import h5py
import numpy as np

def normalize_and_save_data(directory):
    normalized_directory = directory + "_norm"
    os.makedirs(normalized_directory, exist_ok=True)

    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.h5'):
            file_path = os.path.join(directory, filename)
            with h5py.File(file_path, 'r') as file:
                # Assuming the dataset name inside the h5 file is 'kspace'
                # Adjust the dataset name based on your actual file structure
                data = file['kspace'][:]

                # Calculate the maximum of the absolute values
                max_abs_value = np.abs(data).max()

                # Normalize the data
                normalized_data = data / max_abs_value

                # Save the normalized data to a new h5 file
                normalized_file_path = os.path.join(normalized_directory, filename)
                with h5py.File(normalized_file_path, 'w') as normalized_file:
                    normalized_file.create_dataset('kspace', data=normalized_data)

# Example usage
directory = "/work/souza_lab/amir/Data/h5/"
normalize_and_save_data(directory)