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

# '''test
name_ = "test"
paths = [
    "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-04-23_13-57-04/",
    "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-04-27_09-49-21/",
    "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-04-26_09-43-42/",
    # "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-04-26_17-37-57/",
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
methods = ['Baseline e2e', "UNET", "Transformers"]


''' roberto data ieee paper 
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
methods = ['Baseline', 'Baseline e2e', '2D', "2D DiT", "2D DiT + easy_reg", "2D DiT + e2e"]
'''

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
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.boxplot(data=data_ssim)
plt.title('SSIM')
plt.savefig(save_dir + name_ + '_ssim.png')

plt.figure(figsize=(10, 6))
sns.boxplot(data=data_nmse)
plt.title('NMSE')
plt.savefig(save_dir + name_ + '_nmse.png')

plt.figure(figsize=(10, 6))
sns.boxplot(data=data_psnr)
plt.title('PSNR')
plt.savefig(save_dir + name_ + '_psnr.png')


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