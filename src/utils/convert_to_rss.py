
import csv
import glob
from os.path import join

import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
from pathlib import Path

import torchvision

import src.utils.direct.data.transforms as T
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.data.components.transforms import ComplexToReal
from src.utils.direct.data.transforms import ifft2
import pandas as pd

from src.utils.io_utils import save_tensor_to_nifti
from src.utils.transforms import NormalizeSampleTransform

data_dir= "/home/ai2lab/datasets/Dataset_C2/12_channel"
metadata_train_dir= "/home/amir/PycharmProjects/MRI_REC_template/data/Calgary_Campinas/train_mv1.csv"
metadata_val_dir= "/home/amir/PycharmProjects/MRI_REC_template/data/Calgary_Campinas/val_mv1.csv"
metadata_test_dir= "/home/amir/PycharmProjects/MRI_REC_template/data/Calgary_Campinas/test_mv1.csv"
metadata_prev_train_dir= "/home/amir/PycharmProjects/MRI_REC_template/data/Calgary_Campinas/meta_unique.csv"
metadata_prev_val_dir= "/home/amir/PycharmProjects/MRI_REC_template/data/Calgary_Campinas/meta_unique.csv"
metadata_prev_test_dir= "/home/amir/PycharmProjects/MRI_REC_template/data/Calgary_Campinas/meta_unique.csv"
def rss(kspace_pre,input_transforms):

    z, x, y, c = kspace_pre.shape
    kspace_pre = kspace_pre.transpose(1, 2, 3, 0).reshape(x, y, -1)
    if input_transforms is not None:
        kspace_pre = input_transforms(kspace_pre)

    kspace_pre = kspace_pre.view(c, z, x, 170, 2)
    img_pred = T.root_sum_of_squares(
        ifft2(kspace_pre.type(dtype=torch.float32), dim=(2, 3), centered=False, normalized=False),
        dim=0,
    )  # shape (batch, height,  width)
    return img_pred
def _process_volume(data_dir, id):
    path_2_input = os.path.join(data_dir, id)
    input_transforms = torchvision.transforms.Compose(
    [torchvision.transforms.transforms.ToTensor(),
    NormalizeSampleTransform(),
    torchvision.transforms.transforms.CenterCrop(size=[218, 170]),
    ComplexToReal()])
    with h5py.File(path_2_input, 'r') as hf:
        kspace = hf['kspace'][:]
        img_rss = rss(kspace,input_transforms)

    return img_rss


folder = data_dir
folder_to_h5 = folder + "/h5/"
path_to_meta = metadata_test_dir
folder_to_target = folder + "/target_rss/"
Path(folder_to_target).mkdir(exist_ok=True)


metadata = pd.read_csv(path_to_meta)
for i, row in tqdm(metadata.iterrows(), position=0, leave=True):
    img_nlinv = _process_volume(folder_to_h5, f'{row["File name"]}.h5')
    # plt.imshow(img_nlinv[100])
    # plt.show()
    save_tensor_to_nifti(img_nlinv, join(folder_to_target, f'{row["File name"]}.nii'))
