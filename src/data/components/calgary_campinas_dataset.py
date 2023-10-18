import os.path
from typing import Optional, Callable

import h5py
import numpy as np
import random

import pandas as pd
import torch
from torch.utils.data import Dataset
from os import path

class SliceDataset(Dataset):
    def __init__(self,
                data_dir: str,
                metadata_dir: str,
                mask_dir: str,
                target_dir: str,
                input_transforms: Optional[Callable],
                target_transforms: Optional[Callable],
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.metadata = pd.read_csv(metadata_dir)
        self.mask_file = np.load(mask_dir)
        self.target_dir = target_dir
        self.len = self.metadata['Slices'].sum()
        self.metadata_temp = pd.DataFrame(columns=["File name", "Slice Number"])
        for index, row in self.metadata.iterrows():
            value = row['Slices']
            for i in range(value):
                row_ = [row['File name'], i]
                self.metadata_temp.loc[len(self.metadata_temp)] = row_
        self.input_transforms = input_transforms
        self.target_transforms = target_transforms


    def __len__(self):
        return int(self.len)


    def __getitem__(self, idx):
        metadata = self.metadata_temp.iloc[idx]
        path_2_input = os.path.join(self.data_dir,f'{metadata["File name"]}.h5')
        path_2_target = os.path.join(self.data_dir,f'{metadata["File name"]}.h5')

        with h5py.File(path_2_input, "r") as hf:
            kspace = hf["kspace"][metadata["Slice Number"]]

        with h5py.File(path_2_target, "r") as hf:
            target = hf["kspace"][metadata["Slice Number"]]

        if self.data_transforms is not None:
            kspace = self.data_transforms(kspace)
        if self.target_transforms is not None:
            target = self.target_transforms(target)
        sample = kspace, self.mask_file, target, metadata
        return sample

