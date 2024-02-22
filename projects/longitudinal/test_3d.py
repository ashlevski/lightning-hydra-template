import os
from os.path import join
from pathlib import Path

import h5py
import hydra
import lightning as L
import numpy as np
import pandas as pd
import torch
from torchvision import transforms

from src.utils.io_utils import save_tensor_to_nifti
from src.utils.transforms import Nan_to_num


class test_3d():
    def __int__(self):
        super(test_3d,self).__init__()

    def forward(self,model):
        device = 'cuda'
        model.to(device)
        data_dir = "/home/ai2lab/datasets/roberto/Non-enhanced/10x/"
        metadata_dir = "/home/ai2lab/datasets/roberto/test.csv"
        target_dir = "/home/ai2lab/datasets/roberto/Reference/"
        baseline_dir = "/home/ai2lab/datasets/roberto/PS-reg/10x/"

        directory_path = "/home/amir/imri_result/3d/"
        directory_path = Path(directory_path)

        # Create the directory, even if it already exists
        directory_path.mkdir(parents=True, exist_ok=True)

        metadata = pd.read_csv(metadata_dir, header=None)
        len = 0  # metadata['k space X res'].sum()
        metadata_temp = pd.DataFrame(columns=["Baseline", "Slice Number", "File name"])
        input_transforms = transforms.Compose([
            transforms.ToTensor(),
            # Scale(),  # Assuming Scale is a custom transform in src.utils.transforms
            Nan_to_num(),  # Assuming Nan_to_num is a custom transform in src.utils.transforms
            transforms.CenterCrop(size=[218, 170]),
        ])

        for index, row in metadata.iterrows():
            preds = []
            targets = []
            value = 256 - 50
            for i in range(50, value + 1, 16):
                path_2_data = os.path.join(data_dir, f'{row.iloc[1][:-2]}.h5')
                path_2_target = os.path.join(target_dir, f'{row.iloc[1][:-2]}.h5')
                prev_file_name = f'{row.iloc[0]}_{row.iloc[1][:-2]}.h5'
                path_2_baseline = os.path.join(baseline_dir, f'{prev_file_name}')

                with h5py.File(path_2_data, "r") as hf:
                    data = hf["image"][i - 8:i + 8]
                with h5py.File(path_2_target, "r") as hf:
                    target = hf["image"][i - 8:i + 8]
                with h5py.File(path_2_baseline, "r") as hf:
                    baseline = hf["image"][i - 8:i + 8]

                data = np.transpose(data, (1, 2, 0))
                target = np.transpose(target, (1, 2, 0))
                baseline = np.transpose(baseline, (1, 2, 0))

                if input_transforms is not None:
                    data = input_transforms(data)
                    target = input_transforms(target)
                    baseline = input_transforms(baseline)

                sample = {}
                sample["data"] = data.type(dtype=torch.float32).unsqueeze(0).to(device)
                sample["target"] = target.type(dtype=torch.float32).unsqueeze(0).to(device)
                sample["baseline"] = baseline.type(dtype=torch.float32).unsqueeze(0).to(device)
                sample["metadata"] = metadata.to_dict()
                with torch.no_grad():
                    output_image, _, target_img, _, _ = model.forward(sample)
                    preds.append(output_image)
                    targets.append(target_img.unsqueeze(0))

            preds = torch.cat(preds, dim=1).squeeze()
            targets = torch.cat(targets, dim=1).squeeze()
            save_tensor_to_nifti(preds, join(directory_path, f"{row.iloc[1]}_preds.nii"))
            save_tensor_to_nifti(targets, join(directory_path, f"{row.iloc[1]}_targets.nii"))
