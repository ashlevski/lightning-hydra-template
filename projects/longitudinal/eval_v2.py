from os import listdir
import nibabel
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from evaluate import ssim, nmse, psnr, Metrics, METRIC_FUNCS
import numpy as np
from pathlib import Path
import os

def plot_boxplot(metric):
    plt.figure(figsize=(12, 8))
    # sns.set_style("whitegrid")  # Set the style of the plot

    # Customize the colors of the plot
    # colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    # sns.set_palette(colors)
    sns.set_theme(style="darkgrid")
    sns.set_palette ('Set2')
    # Create the box plot
    sns.boxplot(x='Acceleration', y='Values', hue='Method', data=df[df['Metric'] == metric])

    # Set the title and labels
    plt.title(f'{metric} Comparison by Acceleration and Method', fontsize=16)
    plt.xlabel('Acceleration', fontsize=12)
    plt.ylabel(metric, fontsize=12)

    # Customize the legend
    plt.legend(title='Method', title_fontsize=12, fontsize=10,
               bbox_to_anchor=(1.05, 1), loc='upper left')

    # Adjust the layout
    plt.tight_layout()

    # Remove the top and right spines
    sns.despine()

    # plt.show()


save_dir = "/home/amirmohammad.shamaei/imri_result/comp_2d_vs_3d/"
directory_path = Path(save_dir)
directory_path.mkdir(parents=True, exist_ok=True)
name_ = "total"

# Dictionary of paths for each acceleration
paths_dict = {
    'R5': [
        "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-05-01_00-25-49/",
        "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-04-29_19-15-43/",
        "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-05-01_00-14-57/",
        "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-04-29_13-26-36/",
        "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-04-30_11-05-35/",
    ],
    'R10': [
        "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-05-01_00-27-51/",
        "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-04-23_13-57-04/",
        "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-05-01_00-17-23/",
        "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-04-27_09-49-21/",
        "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-04-26_09-43-42/",
    ],
    'R15': [
        "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-05-01_00-33-50/",
        "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-04-29_19-21-45/",
        "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-05-01_00-20-21/",
        "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-04-29_14-09-19/",
        "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-04-30_11-06-35/",
    ],
    'R20': [
        "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-05-01_00-36-21/",
        "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-04-29_22-59-31/",
        "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-05-01_00-21-50/",
        "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-04-29_23-15-39/",
        "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-04-30_01-44-07/",
    ]
}

methods = [
    'Non-enhanced [Souza et al., WW-net IKIK]',
    'Non-enhanced [Ours, E2EVarNet]',
    'Enhanced [Souza et al., UNet]',
    'Enhanced [Ours, UNet]',
    'Enhanced [Ours, Transformers]',
]

# Initialize data storage
data = []

# Loop through each acceleration and its paths
for acc, paths in paths_dict.items():
    for path, method in zip(paths,methods):
        # for method in methods:
        ssims, nmses, psnrs = [], [], []
        for file in os.listdir(path):
            if file.endswith("preds.nii"):
                pred = nibabel.load(os.path.join(path, file)).get_fdata()
                file_tar = file.split("preds.nii")[0]
                target = nibabel.load(os.path.join(path, file_tar + "targets.nii")).get_fdata()
                pred_norm = pred / np.max(pred, axis=(1, 2), keepdims=True)
                target_norm = target / np.max(target, axis=(1, 2), keepdims=True)
                ssims.append(float(ssim(target_norm, pred_norm)[0]))
                nmses.append(float(nmse(target_norm, pred_norm)))
                psnrs.append(float(psnr(target_norm, pred_norm)))

        # Append results to data storage
        for metric_values, metric_name in zip([ssims, nmses, psnrs], ['SSIM', 'NMSE', 'PSNR']):
            data.append({'Acceleration': acc, 'Method': method, 'Metric': metric_name, 'Values': metric_values})

# Create DataFrame
df = pd.DataFrame(data)

# Explode lists into rows
df = df.explode('Values')

# Convert values to float for plotting
df['Values'] = df['Values'].astype(float)


df.to_csv(f"{save_dir}{name_}_df.csv")

plot_boxplot("SSIM")
plt.savefig(f"{save_dir}{name_}_ssim.png", dpi=300)  # Save as high-res image

plot_boxplot("NMSE")
plt.savefig(f"{save_dir}{name_}_NMSE.png", dpi=300)  # Save as high-res image

plot_boxplot("PSNR")
plt.savefig(f"{save_dir}{name_}_PSNR.png", dpi=300)  # Save as high-res image


