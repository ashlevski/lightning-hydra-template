from os.path import join
from pathlib import Path
import SimpleITK as sitk
import h5py
import numpy as np
import pandas as pd
import templateflow.api as tflow
import nibabel as nib
from dipy.align.imaffine import AffineRegistration
from matplotlib import pyplot as plt
from tqdm import tqdm
from dipy.align.transforms import AffineTransform3D

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
def histogram_match(source_image, target_image):
    matched_image = sitk.HistogramMatching(source_image, target_image, numberOfHistogramLevels=1024,
                                           numberOfMatchPoints=7)
    return matched_image
def rigid_registration(fixed_image, moving_image):
    registration_method = sitk.ImageRegistrationMethod()

    # Set the similarity metric
    registration_method.SetMetricAsMeanSquares()

    # Set the optimizer
    registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate=2.0, minStep=1e-4, numberOfIterations=100)

    # Set the initial transformation to identity
    initial_transform = sitk.TranslationTransform(fixed_image.GetDimension())
    registration_method.SetInitialTransform(initial_transform)

    # Perform the registration
    final_transform = registration_method.Execute(fixed_image, moving_image)

    # Transform the moving image
    registered_image = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())
    return registered_image, final_transform

def affine_registration(fixed_image, moving_image):
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations=100, maximumNumberOfCorrections=5)
    initial_transform = sitk.AffineTransform(fixed_image.GetDimension())
    registration_method.SetInitialTransform(initial_transform)
    final_transform = registration_method.Execute(fixed_image, moving_image)
    registered_image = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())
    return registered_image, final_transform
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
folder_to_target_aligned = folder + "/target_rss_aligned_itk/"
Path(folder_to_target_aligned).mkdir(exist_ok=True)
metadata = pd.read_csv(metadata_test_dir)
path_to_meta_prev = pd.read_csv(metadata_prev_test_dir)
for i, row in (metadata.iterrows()):
    path = join(folder_to_target, f'{row["File name"]}.nii')
    prev_file_name = path_to_meta_prev[path_to_meta_prev['Patient ID'] == row['Patient ID']]["File name"].item()
    path_to_prev = join(folder_to_target, f'{prev_file_name}.nii')
    moved = nib.load(path_to_prev).get_fdata()
    fixed = nib.load(path).get_fdata()
    # moved = np.flip(moved,2)

    # Here we convert the moving and fixed images to SimpleITK format
    fixed_sitk = sitk.GetImageFromArray(fixed)
    moving_sitk = sitk.GetImageFromArray(moved)

    # We perform the histogram matching
    matched_moving_sitk = histogram_match(moving_sitk, fixed_sitk)

    # We convert the matched image back to numpy array
    matched_moving_image_data = sitk.GetArrayFromImage(matched_moving_sitk)
    plot_reg(fixed,moved,row["File name"])
    rigid_registered_image, rigid_transform = affine_registration(fixed_sitk, matched_moving_sitk)
    # Convert the registered image to numpy array for visualization
    rigid_registered_image_data = sitk.GetArrayFromImage(rigid_registered_image)
    # affreg = AffineRegistration()
    # affine3d = affreg.optimize(fixed, moved, AffineTransform3D(), params0=None)
    # moved = affine3d.transform(moved)
    plot_reg(fixed, rigid_registered_image_data, row["File name"])

    save_to_nifti(rigid_registered_image_data, f"{folder_to_target_aligned}/{row['File name']}.nii")
    # h5_file = h5py.File(f"{folder_to_target_aligned}/{row['File name']}.h5", "w")
    # h5_file.create_dataset('image', data=moved)
    # h5_file.close()
    pass
    # plt.imshow(img_nlinv[100])
    # plt.show()
    # save_tensor_to_nifti(img_nlinv, join(folder_to_target, f'{row["File name"]}.nii'))