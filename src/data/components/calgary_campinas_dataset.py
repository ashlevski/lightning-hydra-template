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
                data_transforms: Optional[Callable],
                target_transforms: Optional[Callable],
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.metadata = pd.read_csv(metadata_dir)
        self.mask_file = np.load(mask_dir)
        self.target_dir = target_dir


    def __len__(self):
        return len(self.file_paths)


    def __getitem__(self, idx):
        metadata = self.metadata_files.iloc[idx]
        path_2_input = os.path.join(self.data_dir,self.metadata)
        path_2_target = os.path.join(self.data_dir, self.metadata)

        with h5py.File(path_2_input, "r") as hf:
            kspace = hf["kspace"][:]

        with h5py.File(path_2_target, "r") as hf:
            target = hf["kspace"][:]

        if self.transform is None:
            sample = (kspace, self.mask_file, target, metadata)
        else:
            sample = self.transform(kspace, self.mask_file, target, metadata)

        return sample

