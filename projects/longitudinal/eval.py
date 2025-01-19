from os import listdir
import nibabel
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from evaluate import ssim, nmse, psnr, Metrics, METRIC_FUNCS
import numpy as np
from pathlib import Path

save_dir = "/home/amirmohammad.shamaei/imri_result/comp_2d_vs_3d/"
directory_path = Path(save_dir)
directory_path.mkdir(parents=True, exist_ok=True)


# name_ = "R20"
# paths = [
#     "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-04-29_22-59-31/",
#     "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-04-30_19-14-34/",
#     "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-04-30_19-05-08/",
#     "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-04-29_23-15-39/",
#     "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-04-30_01-44-07/",
# ]
# methods = ['Baseline e2e', "UNET", "Transformers"]

name_ = "R15"
paths = [
    "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-05-01_00-33-50/",
    "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-04-29_19-21-45/",
    "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-05-01_00-20-21/",
    "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-04-29_14-09-19/",
    "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-04-30_11-06-35/",
    "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-06-03_17-59-04/",
    "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-05-31_16-31-59/",
]
methods = [
    'Non-enhanced [Souza et al., WW-net IKIK]',
    'Non-enhanced [Ours, E2EVarNet]',
    'Enhanced [Souza et al., UNet]',
    'Enhanced [Ours, UNet]',
    'Enhanced [Ours, Transformers]',
    'Enhanced by atlas [Ours, Transformers, tested]',
    'Enhanced by atlas [Ours, Transformers, trained]',
]

# '''name_ = "R5"
# paths = [
#     "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-04-29_19-15-43/",
#     "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-04-29_13-26-36/",
#     "???",
# ]
# methods = ['Baseline e2e', "UNET", "Transformers"]'''

#test
# name_ = "R10"
# paths = [
#     "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-03-05_01-00-20/",
#     "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-04-23_13-57-04/",
#     "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-04-30_18-25-07/",
#     "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-04-27_09-49-21/",
#     "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-04-26_09-43-42/",
# ]
# methods = [
#     'Baseline [Souza et al., WW-net IKIK]',
#     'Baseline [Ours, E2EVarNet]',
#     'Enhanced [Souza et al., UNet]',
#     'Enhanced [Our, UNet]',
#     'Enhanced [Our, Transformers]'
# ]


"""# 'roberto data ieee paper 
name_ = "with_axial_normalization_DiT_with_att"
paths = [
    "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-03-05_01-00-20/",
    "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-04-22_16-06-12/",
    "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-03-04_17-31-39/",
    "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-04-14_19-19-06/",
    "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-04-14_21-22-11/",
    "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-04-23_09-42-38/",
    
    # "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-03-04_20-58-41/",
    # "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-03-06_23-41-38/",
    # "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-03-07_12-02-59/",
    # "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-03-08_11-12-56/",
    # "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-03-08_18-48-50/",
    # "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-03-12_09-12-41/",
    # "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-03-25_20-10-16/",
    # "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-04-12_18-34-44/",
    # "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-04-12_18-40-54/"
]
methods = ['Baseline', 'Baseline e2e', '2D', "2D DiT", "2D DiT + easy_reg", "2D DiT + e2e"]"""


# name_ = "single visit"
# paths = [
#     "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-04-20_18-10-17/",
#     "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-04-20_18-15-19/",
#     "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-04-20_18-21-05/",
# ]
# methods = ['R5', 'R10', "R15", ]

ssims_list, nmse_list, psnr_list, metrics_list = [], [], [], []


for path, method in zip(paths, methods):
    ssims = []
    nmses = []
    psnrs = []
    metric = Metrics(METRIC_FUNCS)
    file_tar_list = []
    for file in listdir(path):
        if file.endswith("preds.nii"):
            pred = nibabel.load(path + file).get_fdata()
            file_tar = file.split("preds.nii")[0]
            file_tar_list.append(file_tar)
            target = nibabel.load(path + file_tar + "targets.nii").get_fdata()
            pred = pred
            target = target
            pred = pred / pred.max(axis=(1,2),keepdims=True)
            target = target / target.max(axis=(1,2),keepdims=True)
            metric.push(target, pred)
            ssims.append(float(ssim(target, pred)[0]))
            nmses.append(float(nmse(target, pred)))
            psnrs.append(float(psnr(target, pred)))

    ssims_list.append(ssims)
    nmse_list.append(nmses)
    psnr_list.append(psnrs)
    metrics_list.append(metric)

data_ssim = pd.DataFrame(dict(zip(methods, ssims_list)))
data_nmse = pd.DataFrame(dict(zip(methods, nmse_list)))
data_psnr = pd.DataFrame(dict(zip(methods, psnr_list)))

# Create box plots using Seaborn
# Create box plots using Seaborn
# Setting up a more appealing style
sns.set(style="whitegrid", palette="pastel")

# Create a larger figure to improve readability
plt.figure(figsize=(12, 8))

# Creating the boxplot with enhanced color and style
box = sns.boxplot(data=data_ssim, width=0.5, fliersize=5, linewidth=2.5, whis=1.5)

# Adding title with increased font size for better visibility
plt.title('SSIM Distribution', fontsize=20, fontweight='bold', color='navy')

# Labeling axes with increased font size
plt.xlabel('Category', fontsize=15)
plt.ylabel('SSIM Value', fontsize=15)

# Customizing tick parameters for better readability
plt.xticks(fontsize=12, rotation=45)  # Rotate x-ticks if necessary
plt.yticks(fontsize=12)

# Optional: Adding a grid for better readability of the plot
plt.grid(True, linestyle='--', linewidth=0.8)
plt.tight_layout()
plt.savefig(f"{save_dir}{name_}_ssim.png", dpi=300)  # Save as high-res image


# Create a larger figure to improve readability
plt.figure(figsize=(12, 8))

# Creating the boxplot with enhanced color and style
box = sns.boxplot(data=data_psnr, width=0.5, fliersize=5, linewidth=2.5, whis=1.5)

# Adding title with increased font size for better visibility
plt.title('SSIM Distribution', fontsize=20, fontweight='bold', color='navy')

# Labeling axes with increased font size
plt.xlabel('Category', fontsize=15)
plt.ylabel('PSNR Value', fontsize=15)

# Customizing tick parameters for better readability
plt.xticks(fontsize=12, rotation=45)  # Rotate x-ticks if necessary
plt.yticks(fontsize=12)

# Optional: Adding a grid for better readability of the plot
plt.grid(True, linestyle='--', linewidth=0.8)
plt.tight_layout()
plt.savefig(f"{save_dir}{name_}_psnr.png", dpi=300)  # Save as high-res image

# Create a larger figure to improve readability
plt.figure(figsize=(12, 8))

# Creating the boxplot with enhanced color and style
box = sns.boxplot(data=data_nmse, width=0.5, fliersize=5, linewidth=2.5, whis=1.5)

# Adding title with increased font size for better visibility
plt.title('SSIM Distribution', fontsize=20, fontweight='bold', color='navy')

# Labeling axes with increased font size
plt.xlabel('Category', fontsize=15)
plt.ylabel('SSIM Value', fontsize=15)

# Customizing tick parameters for better readability
plt.xticks(fontsize=12, rotation=45)  # Rotate x-ticks if necessary
plt.yticks(fontsize=12)

# Optional: Adding a grid for better readability of the plot
plt.grid(True, linestyle='--', linewidth=0.8)
plt.tight_layout()
plt.savefig(f"{save_dir}{name_}_nmse.png", dpi=300)  # Save as high-res image

# plt.figure(figsize=(10, 6))
# sns.boxplot(data=data_nmse)
# plt.title('NMSE')
# plt.savefig(save_dir + name_ + '_nmse.png')

# plt.figure(figsize=(10, 6))
# sns.boxplot(data=data_psnr)
# plt.title('PSNR')
# plt.savefig(save_dir + name_ + '_psnr.png')


# Concatenate the DataFrames
data = pd.concat([data_ssim, data_nmse, data_psnr], axis=1)

# Add the 'file_tar' column
data['file_tar'] = file_tar_list

# Save to CSV
data.to_csv(save_dir + name_ + ".csv", index=False)


open(save_dir + name_ + "_output.txt", "w")
for method, metric in zip(methods, metrics_list):
    print(f'{method} : mean is {metric.means()} and std is {metric.stddevs()}')
    with open(save_dir + name_ + "_output.txt", "a") as f:
        f.write(f'{method} : mean is {metric.means()} and std is {metric.stddevs()}\n')