import os.path
from typing import Optional, Callable

import h5py
import numpy as np
import random

import pandas as pd
import torch
from torch.utils.data import Dataset
from os import path
import src.utils.direct.data.transforms as T
from src.utils.direct.data.transforms import ifft2
class SliceDataset(Dataset):
    def __init__(self,
                data_dir: str,
                metadata_dir: str,
                metadata_prev_dir: str,
                mask_dir: str,
                input_transforms: Optional[Callable],
                target_transforms: Optional[Callable],
                crop_slice_idx = 0,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.metadata = pd.read_csv(metadata_dir)
        self.metadata_prev = pd.read_csv(metadata_prev_dir)
        self.mask_file = np.load(mask_dir)
        self.len = 0#self.metadata['k space X res'].sum()
        self.metadata_temp = pd.DataFrame(columns=["File name", "Slice Number", "Patient ID"])
        for index, row in self.metadata.iterrows():
            value = row['k space X res']
            self.start = crop_slice_idx
            self.end = value-crop_slice_idx
            for i in range(self.start,self.end):
                row_ = [row['File name'], i, row['Patient ID']]
                self.metadata_temp.loc[len(self.metadata_temp)] = row_
                self.len += 1
        self.input_transforms = input_transforms
        self.target_transforms = target_transforms



    def __len__(self):
        return int(self.len)


    def __getitem__(self, idx):
        metadata = self.metadata_temp.iloc[idx]
        path_2_data = os.path.join(self.data_dir,f'{metadata["File name"]}.h5')

        prev_file_name = self.metadata_prev[self.metadata_prev['Patient ID'] == metadata['Patient ID']]["File name"].item()
        path_2_prev_data = os.path.join(self.data_dir, f'{prev_file_name}.h5')
        with h5py.File(path_2_data, "r") as hf:
            kspace = hf["kspace"][metadata["Slice Number"]]
            # target = hf["target"][metadata["Slice Number"]]
        with h5py.File(path_2_prev_data, "r") as hf:
            kspace_pre = hf["kspace"][:]
            z, x, y, c = kspace_pre.shape
            kspace_pre = kspace_pre.transpose(1, 2, 3, 0).reshape(x, y, -1)
        if self.input_transforms is not None:
            kspace = self.input_transforms(kspace)
            kspace_pre = self.input_transforms(kspace_pre)

        # if self.target_transforms is not None:
        #     target = self.target_transforms(target)
        # if self.target_transforms is not None:
        #     target = self.target_transforms(target)
        kspace_pre= kspace_pre.view(c, z, x, 170, 2)
        img_pred = T.root_sum_of_squares(
            ifft2(kspace_pre.type(dtype=torch.float32), dim=(2, 3),centered=False, normalized=False),
            dim=0,
        )  # shape (batch, height,  width)
        sample = {}

        sample["acs_mask"] = torch.from_numpy(self.mask_file[0]).type(dtype=torch.float32).unsqueeze(-1).unsqueeze(0)
        sample["kspace"] = kspace.type(dtype=torch.float32)
        sample["img_pre"] = img_pred.type(dtype=torch.float32)
        sample["metadata"] = metadata.to_dict()
        # sample["target"] = target.type(dtype=torch.float32)
        # sample = kspace, sample["acs_mask"].squeeze(), target, sample['sensitivity_map'].squeeze(), metadata.to_dict()
        return sample

