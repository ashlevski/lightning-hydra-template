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
                input_transforms: Optional[Callable],
                target_transforms: Optional[Callable],
                crop_slice_idx = 0,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.metadata = pd.read_csv(metadata_dir)
        self.mask_file = np.load(mask_dir)
        self.len = 0#self.metadata['k space X res'].sum()
        self.metadata_temp = pd.DataFrame(columns=["File name", "Slice Number"])
        for index, row in self.metadata.iterrows():
            value = row['k space X res']
            start = crop_slice_idx
            end = value-crop_slice_idx
            for i in range(start,end):
                row_ = [row['File name'], i]
                self.metadata_temp.loc[len(self.metadata_temp)] = row_
                self.len += 1
        self.input_transforms = input_transforms
        self.target_transforms = target_transforms



    def __len__(self):
        return int(self.len)


    def __getitem__(self, idx):
        metadata = self.metadata_temp.iloc[idx]
        path_2_data = os.path.join(self.data_dir,f'{metadata["File name"]}.h5')


        with (h5py.File(path_2_data, "r") as hf):
            kspace = hf["kspace"][metadata["Slice Number"]-1:metadata["Slice Number"]+2]
            z, x, y, c = kspace.shape
            kspace = kspace.transpose(1, 2, 3, 0).reshape(x, y, -1)
            # target = hf["target"][metadata["Slice Number"]]

        if self.input_transforms is not None:
            kspace = self.input_transforms(kspace)
        # if self.target_transforms is not None:
        #     target = self.target_transforms(target)
        # if self.target_transforms is not None:
        #     target = self.target_transforms(target)

        sample = {}
        y = 170
        sample["acs_mask"] = torch.from_numpy(self.mask_file[0]).type(dtype=torch.float32).unsqueeze(-1).unsqueeze(0)
        sample["kspace"] = kspace.type(dtype=torch.float32).view(c, z, x, y, 2)
        sample["metadata"] = metadata.to_dict()
        # sample["target"] = target.type(dtype=torch.float32)
        # sample = kspace, sample["acs_mask"].squeeze(), target, sample['sensitivity_map'].squeeze(), metadata.to_dict()
        return sample

