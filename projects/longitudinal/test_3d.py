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
from src.utils.transforms import Nan_to_num, Scale


class test_3d():
    def __int__(self):
        super(test_3d,self).__init__()

    def forward(self,model):
        ckpt_path = "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-02-28_00-21-45/checkpoints/epoch_198.ckpt"
        # ckpt_path = "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-02-27_21-45-39/checkpoints/epoch_119.ckpt"
        device = 'cuda'
        model.to(device)
        data_dir = "/work/souza_lab/amir/Data/roberto/Non-enhanced/10x/"
        metadata_dir = "/work/souza_lab/amir/Data/roberto/test.csv"
        target_dir = "/work/souza_lab/amir/Data/roberto/Reference/"
        baseline_dir = "/work/souza_lab/amir/Data/roberto/PS-reg/10x/"

        directory_path = "/home/amirmohammad.shamaei/imri_result/3d_/"
        directory_path = Path(directory_path)

        # Create the directory, even if it already exists
        directory_path.mkdir(parents=True, exist_ok=True)
        model = model.load_from_checkpoint(ckpt_path)
        model.eval()
        metadata = pd.read_csv(metadata_dir, header=None)
        len = 0  # metadata['k space X res'].sum()
        metadata_temp = pd.DataFrame(columns=["Baseline", "Slice Number", "File name"])
        input_transforms = transforms.Compose([
            transforms.ToTensor(),
            Nan_to_num(),  # Assuming Nan_to_num is a custom transform in src.utils.transforms
            transforms.CenterCrop(size=[218, 170]),
            Scale(),  # Assuming Scale is a custom transform in src.utils.transforms
        ])

        for index, row in metadata.iterrows():
            preds = []
            targets = []
            value = 256 - 48 
            # for i in range(54, value + 1, 16):
            #     path_2_data = os.path.join(data_dir, f'{row.iloc[1][:-2]}.h5')
            #     path_2_target = os.path.join(target_dir, f'{row.iloc[1][:-2]}.h5')
            #     prev_file_name = f'{row.iloc[0]}_{row.iloc[1][:-2]}.h5'
            #     path_2_baseline = os.path.join(baseline_dir, f'{prev_file_name}')

            #     with h5py.File(path_2_data, "r") as hf:
            #         data = hf["image"][i - 8:i + 8]
            #     with h5py.File(path_2_target, "r") as hf:
            #         target = hf["image"][i - 8:i + 8]
            #     with h5py.File(path_2_baseline, "r") as hf:
            #         baseline = hf["image"][i - 8:i + 8]

            #     data = np.transpose(data, (1, 2, 0))
            #     target = np.transpose(target, (1, 2, 0))
            #     baseline = np.transpose(baseline, (1, 2, 0))

            #     if input_transforms is not None:
            #         data = input_transforms(data)
            #         target = input_transforms(target)
            #         baseline = input_transforms(baseline)

            #     # scale_data = (torch.abs(data).max())
            #     # scale_target = (torch.abs(target).max())
            #     # scale_baseline = (torch.abs(baseline).max())

            #     # data = data / scale_data
            #     # target = target / scale_target
            #     # baseline = baseline / scale_baseline

            #     sample = {}
            #     sample["data"] = data.type(dtype=torch.float32).unsqueeze(0).to(device)
            #     sample["target"] = target.type(dtype=torch.float32).unsqueeze(0).to(device)
            #     sample["baseline"] = baseline.type(dtype=torch.float32).unsqueeze(0).to(device)
            #     sample["metadata"] = metadata.to_dict()
            #     with torch.no_grad():
            #         # output_image, _, target_img, _, _ = model.forward(sample)
            #         # preds.append(sample["data"])
            #         # targets.append(target_img)
            #         print(target.shape)
            #         losses, pred, target, output_image_mv, _ = model.model_step(sample)
            #         # pred, _, target, _, _ = model.forward(sample)
            #         preds.append(pred)
            #         targets.append(target)

            # print(preds[0].shape)
            # preds = torch.cat(preds, dim=1).squeeze()
            # targets = torch.cat(targets, dim=1).squeeze()

            path_2_data = os.path.join(data_dir, f'{row.iloc[1][:-2]}.h5')
            path_2_target = os.path.join(target_dir, f'{row.iloc[1][:-2]}.h5')
            prev_file_name = f'{row.iloc[0]}_{row.iloc[1][:-2]}.h5'
            path_2_baseline = os.path.join(baseline_dir, f'{prev_file_name}')

            with h5py.File(path_2_data, "r") as hf:
                data = hf["image"][48:value]
            with h5py.File(path_2_target, "r") as hf:
                target = hf["image"][48:value]
            with h5py.File(path_2_baseline, "r") as hf:
                baseline = hf["image"][48:value]

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
                losses, pred, target, output_image_mv, _ = model.model_step(sample)
                # pred, _, target, _, _ = model.forward(sample)
                print(pred.shape)
                print(target.shape)
                # preds.append(pred)
                # targets.append(target)
            
            # preds = torch.cat(preds, dim=1).squeeze()
            # targets = torch.cat(targets, dim=1).squeeze()
            print(pred.shape)
            save_tensor_to_nifti(pred.squeeze(), join(directory_path, f"{row.iloc[1]}_preds.nii"))
            save_tensor_to_nifti(target.squeeze(), join(directory_path, f"{row.iloc[1]}_targets.nii"))
