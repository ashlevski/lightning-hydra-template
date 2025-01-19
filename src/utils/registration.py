from os.path import join
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import templateflow.api as tflow
import nibabel as nib
from dipy.align import affine_registration
from dipy.align.imaffine import AffineRegistration, MutualInformationMetric
from matplotlib import pyplot as plt
from tqdm import tqdm
from dipy.align.transforms import AffineTransform3D, RigidTransform3D, TranslationTransform3D

from src.utils.io_utils import save_to_nifti


# mni_img = nib.load(tflow.get('MNI152NLin2009cAsym', resolution=1, suffix="T1w", desc=None))
# mni_data = mni_img.get_fdata()
# from ndslib.data import download_bids_dataset
# download_bids_dataset()
# t1_img = nib.load("ds001233/sub-17/ses-pre/anat/sub-17_ses-pre_T1w.nii.gz")
#
# from nibabel.processing import resample_from_to
# t1_resampled = resample_from_to(t1_img, (mni_img.shape, mni_img.affine))
# t1_resamp_data = t1_resampled.get_fdata()
#
# fig, axes = plt.subplots(1, 3, figsize=(8, 4))
# ax = axes.ravel()
#
# ax[0].imshow(mni_data[:, :, 85])
# ax[1].imshow(t1_resamp_data[:, :, 85])
#
# stereo = np.zeros((193, 229, 3), dtype=np.uint8)
# stereo[..., 0] = 255 * mni_data[:, :, 85]/np.max(mni_data)
# stereo[..., 1] = 255 * t1_resamp_data[:, :, 85]/np.max(t1_resamp_data)
# ax[2].imshow(stereo)
# fig.tight_layout()
#
# plt.show()
#
# from dipy.align.transforms import AffineTransform3D
# affreg = AffineRegistration()
# affine3d = affreg.optimize(mni_data, t1_resamp_data, AffineTransform3D(), params0=None)
# t1_xform = affine3d.transform(t1_resamp_data)
#
# fig, axes = plt.subplots(1, 3, figsize=(8, 4))
# ax = axes.ravel()
#
# ax[0].imshow(mni_data[:, :, 85]/np.max(mni_data))
# ax[1].imshow(t1_xform[:, :, 85]/np.max(t1_xform))
#
# stereo = np.zeros((193, 229, 3), dtype=np.uint8)
# stereo[..., 0] = 255 * mni_data[:, :, 85]/np.max(mni_data)
# stereo[..., 1] = 255 * t1_xform[:, :, 85]/np.max(t1_xform)
# ax[2].imshow(stereo)
# fig.tight_layout()
# plt.show()
def plot_reg(fixed, moved, title= "REG", i=85, view='a'):
    fig, axes = plt.subplots(1, 3, figsize=(8, 4))
    ax = axes.ravel()
    if view == 'a':
        ax[0].imshow(fixed[i, :, :])
        ax[1].imshow(moved[i, :, :])

        stereo = np.zeros((218, 170, 3), dtype=np.uint8)
        stereo[..., 0] = 255 * fixed[i, :, :]/np.max(fixed)
        stereo[..., 1] = 255 * moved[i, :, :]/np.max(moved)
        ax[2].imshow(stereo)
        fig.tight_layout()
        plt.title(title)
        plt.show()
    if view == 'c':
        ax[0].imshow(fixed[:, i, :])
        ax[1].imshow(moved[:, i, :])

        stereo = np.zeros((256, 170, 3), dtype=np.uint8)
        stereo[..., 0] = 255 * fixed[:, i, :] / np.max(fixed)
        stereo[..., 1] = 255 * moved[:, i, :] / np.max(moved)
        ax[2].imshow(stereo)
        fig.tight_layout()
        plt.title(title)
        plt.show()
    if view == 's':
        ax[0].imshow(fixed[:, :, i])
        ax[1].imshow(moved[:, :, i])

        stereo = np.zeros((256, 218, 3), dtype=np.uint8)
        stereo[..., 0] = 255 * fixed[:, :, i] / np.max(fixed)
        stereo[..., 1] = 255 * moved[:, :, i] / np.max(moved)
        ax[2].imshow(stereo)
        fig.tight_layout()
        plt.title(title)
        plt.show()

data_dir= "/home/ai2lab/datasets/Dataset_C2/12_channel"
metadata_train_dir= "/home/amir/PycharmProjects/MRI_REC_template/data/Calgary_Campinas/train_mv1.csv"
metadata_val_dir= "/home/amir/PycharmProjects/MRI_REC_template/data/Calgary_Campinas/val_mv1.csv"
metadata_test_dir= "/home/amir/PycharmProjects/MRI_REC_template/data/Calgary_Campinas/test_mv1.csv"
metadata_prev_train_dir= "/home/amir/PycharmProjects/MRI_REC_template/data/Calgary_Campinas/meta_unique.csv"
metadata_prev_val_dir= "/home/amir/PycharmProjects/MRI_REC_template/data/Calgary_Campinas/meta_unique.csv"
metadata_prev_test_dir= "/home/amir/PycharmProjects/MRI_REC_template/data/Calgary_Campinas/meta_unique.csv"
folder = data_dir
folder_to_h5 = folder + "/h5/"
path_to_meta = metadata_train_dir
folder_to_target = folder + "/target_rss/"
Path(folder_to_target).mkdir(exist_ok=True)
folder_to_target_aligned = folder + "/target_rss_aligned/"
Path(folder_to_target_aligned).mkdir(exist_ok=True)
metadata = pd.read_csv(metadata_test_dir)
path_to_meta_prev = pd.read_csv(metadata_prev_test_dir)
# nbins = 32
# sampling_prop = None
# metric = MutualInformationMetric(nbins, sampling_prop)
level_iters = [10000, 1000, 1000]
sigmas = [3.0, 1.0, 0.0]
factors = [4, 2, 1]
# affreg = AffineRegistration(metric=metric,
#                             level_iters=level_iters,
#                             sigmas=sigmas,
#                             factors=factors)

pipeline = ["center_of_mass", "translation", "rigid", "affine"]
identity = np.eye(4)
list_ = ["e15865s3_P55296.nii","e19277s5_P29696.nii",'e16673s3_P24576.nii','e19219s14_P56832.nii','e19197s3_P52224.nii','e15812s13_P51712.nii','e15812s3_P44544.nii','e19197s13_P59392.nii','e19219s14_P56832.nii','e19229s4_P49664.nii','e15812s13_P51712.nii','e15812s3_P44544.nii','e19197s13_P59392.nii','e19197s3_P52224.nii','e19219s14_P56832.nii']
for i, row in (metadata.iterrows()):
    # if f'{row["File name"]}.nii' in list_:
    if row["Patient ID"]=='NORM100':
        path = join(folder_to_target, f'{row["File name"]}.nii')
        prev_file_name = path_to_meta_prev[path_to_meta_prev['Patient ID'] == row['Patient ID']]["File name"].item()
        path_to_prev = join(folder_to_target, f'{prev_file_name}.nii')
        moving = nib.load(path_to_prev).get_fdata()
        fixed = nib.load(path).get_fdata()
        # moving = np.flip(moving, 2)
        plot_reg(fixed,moving,row["File name"]+"before",120)
        print(row.tolist())

        # affreg = AffineRegistration()
        # affine3d = affreg.optimize(fixed, moving, AffineTransform3D(), params0=None)
        # moved = affine3d.transform(moved)
        # plot_reg(fixed, moved, row["File name"],120)
        #
        # rigid = affreg.optimize(fixed, moved, TranslationTransform3D(), params0=None)
        # moved = rigid.transform(moved)

        moved, reg_affine = affine_registration(
            moving,
            fixed,
            moving_affine=identity,
            static_affine=identity,
            nbins=32,
            metric='MI',
            pipeline=pipeline,
            level_iters=level_iters,
            sigmas=sigmas,
            factors=factors)
        plot_reg(fixed, moved, row["File name"]+"after", 120)
        save_to_nifti(moved,f"{folder_to_target_aligned}/{row['File name']}.nii")
        # h5_file = h5py.File(f"{folder_to_target_aligned}/{row['File name']}.h5", "w")
        # h5_file.create_dataset('image', data=moved)
        # h5_file.close()
        pass
        # plt.imshow(img_nlinv[100])
        # plt.show()
        # save_tensor_to_nifti(img_nlinv, join(folder_to_target, f'{row["File name"]}.nii'))