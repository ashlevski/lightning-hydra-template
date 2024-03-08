import os.path
from typing import Optional, Callable

import h5py
import nibabel
import numpy as np
import random

import pandas as pd
import torch
from torch.utils.data import Dataset
from os import path
import src.utils.direct.data.transforms as T
from src.utils.direct.data.transforms import ifft2
import nibabel as nib
class SliceDataset(Dataset):
    def __init__(self,
                data_dir: str,
                target_dir: str,
                baseline_dir: str,
                metadata_dir: str,
                input_transforms: Optional[Callable],
                crop_slice_idx = 0,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.target_dir = target_dir
        self.baseline_dir = baseline_dir
        self.metadata = pd.read_csv(metadata_dir, header=None)
        self.len = 0#self.metadata['k space X res'].sum()
        self.metadata_temp = pd.DataFrame(columns=["Baseline", "Slice Number","File name"])
        self.crop_slice_idx= crop_slice_idx#+8
        for index, row in self.metadata.iterrows():
            value = 256
            self.start = crop_slice_idx
            self.end = value-crop_slice_idx
            i=0
            # for i in range(self.start,self.end):
            row_ = [row.iloc[0], i, row.iloc[1]]
            self.metadata_temp.loc[len(self.metadata_temp)] = row_
            self.len += 1
        self.input_transforms = input_transforms
        # self.target_transforms = target_transforms



    def __len__(self):
        return int(self.len)


    def __getitem__(self, idx):
        metadata = self.metadata_temp.iloc[idx]
        # path_2_data = os.path.join(self.data_dir,f'{metadata["File name"]}.nii.gz')
        # path_2_target = os.path.join(self.target_dir,f'{metadata["File name"]}.nii.gz')
        # prev_file_name = f'{metadata["Baseline"]}_{metadata["File name"]}.nii.gz'
        # path_2_baseline = os.path.join(self.baseline_dir, f'{prev_file_name}')
        #
        # data = nib.load(path_2_data).get_fdata()[metadata["Slice Number"]]
        # target = nib.load(path_2_target).get_fdata()[metadata["Slice Number"]]
        # baseline = nib.load(path_2_baseline).get_fdata()[metadata["Slice Number"]]

        path_2_data = os.path.join(self.data_dir,f'{metadata["File name"][:-2]}.h5')
        path_2_target = os.path.join(self.target_dir,f'{metadata["File name"][:-2]}.h5')
        # prev_file_name = f'{metadata["Baseline"]}_{metadata["File name"][:-2]}.h5'
        prev_file_name = f'{metadata["Baseline"]}.h5'
        path_2_baseline = os.path.join(self.baseline_dir, f'{prev_file_name}')

        with h5py.File(path_2_data, "r") as hf:
            data = hf["image"][self.crop_slice_idx:-self.crop_slice_idx]
        with h5py.File(path_2_target, "r") as hf:
            target = hf["image"][self.crop_slice_idx:-self.crop_slice_idx]
        with h5py.File(path_2_baseline, "r") as hf:
            baseline = hf["image"][self.crop_slice_idx:-self.crop_slice_idx]

        data = np.transpose(data, (1, 2, 0))
        target = np.transpose(target, (1, 2, 0))
        baseline = np.transpose(baseline, (1, 2, 0))
        if self.input_transforms is not None:
            data = self.input_transforms(data)
            target = self.input_transforms(target)
            baseline = self.input_transforms(baseline)
            # target = self.target_transforms(target)

        sample = {}
        sample["data"] = data.type(dtype=torch.float32)
        sample["target"] = target.type(dtype=torch.float32)
        sample["baseline"] = baseline.type(dtype=torch.float32)
        sample["metadata"] = metadata.to_dict()

        return sample

