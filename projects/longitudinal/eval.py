# from os import listdir

# import nibabel
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from evaluate import ssim, nmse,psnr,Metrics,METRIC_FUNCS
# import numpy as np
# from pathlib import Path
# save_dir = "/home/amirmohammad.shamaei/imri_result/comp_2d_vs_3d/"
# directory_path = Path(save_dir)
# directory_path.mkdir(parents=True, exist_ok=True)



# path_1 = "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-03-05_01-00-20/"
# path_2 = "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-03-04_17-31-39/"
# path_3 = "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-03-04_20-58-41/"
# # path_4 = "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-03-06_23-41-38/"
# path_4 = "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-03-07_12-02-59/"
# # path_4 = "C:\\Users\\amsha\\Desktop\\temp\\i mri result\\nl_reg\\"

# ssims_1 = []
# nmse_1 = []
# psnr_1 = []
# metric_1 = Metrics(METRIC_FUNCS)
# for file in listdir(path_1):
#     if file.endswith("preds.nii"):
#         pred = nibabel.load(path_1+file).get_fdata()
#         file_tar = file.split("preds.nii")[0]
#         target = nibabel.load(path_1+file_tar+"targets.nii").get_fdata()
#         # pred = pred/pred.max(axis=(1,2),keepdims=True)
#         # target = target / target.max(axis=(1,2),keepdims=True)
        
#         metric_1.push(target, pred)
#         ssims_1.append(float(ssim(target,pred)[0]))
#         nmse_1.append(float(nmse(target, pred)))
#         psnr_1.append(float(psnr(target, pred)))


# ssims_2 = []
# nmse_2 = []
# psnr_2 = []
# metric_2 = Metrics(METRIC_FUNCS)
# for file in listdir(path_2):
#     if file.endswith("preds.nii"):
#         pred = nibabel.load(path_2+file).get_fdata()
#         file_tar = file.split("preds.nii")[0]
#         target = nibabel.load(path_2+file_tar+"targets.nii").get_fdata()
#         # pred = pred/pred.max(axis=(1,2),keepdims=True)
#         # target = target / target.max(axis=(1,2),keepdims=True)
#         metric_2.push(target, pred)
#         ssims_2.append(float(ssim(target,pred)[0]))
#         nmse_2.append(float(nmse(target, pred)))
#         psnr_2.append(float(psnr(target, pred)))

# ssims_3 = []
# nmse_3 = []
# psnr_3 = []
# metric_3 = Metrics(METRIC_FUNCS)
# for file in listdir(path_3):
#     if file.endswith("preds.nii"):
#         pred = nibabel.load(path_3+file).get_fdata()[2:-2]
#         file_tar = file.split("preds.nii")[0]
#         target = nibabel.load(path_3+file_tar+"targets.nii").get_fdata()[2:-2]
#         # pred = pred/pred.max(axis=(1,2),keepdims=True)
#         # target = target / target.max(axis=(1,2),keepdims=True)
        
#         metric_3.push(target, pred)
#         ssims_3.append(float(ssim(target,pred)[0]))
#         nmse_3.append(float(nmse(target, pred)))
#         psnr_3.append(float(psnr(target, pred)))

# #
# ssims_4 = []
# nmse_4 = []
# psnr_4 = []
# metric_4 = Metrics(METRIC_FUNCS)
# for file in listdir(path_4):
#     if file.endswith("preds.nii"):
#         pred = nibabel.load(path_4+file).get_fdata()[2:-2]
#         file_tar = file.split("preds.nii")[0]
#         target = nibabel.load(path_4+file_tar+"targets.nii").get_fdata()[2:-2]
#         # pred = pred/pred.max(axis=(1,2),keepdims=True)
#         # target = target / target.max(axis=(1,2),keepdims=True)

#         metric_4.push(target, pred)
#         ssims_4.append(float(ssim(target,pred)[0]))
#         nmse_4.append(float(nmse(target, pred)))
#         psnr_4.append(float(psnr(target, pred)))

# data_ssim = pd.DataFrame({
#     'Baseline': ssims_1,
#     '2D': ssims_2,
#     '3D non-reg': ssims_4,
#     '3D': ssims_3
# })

# name_ = "DiT"

# # # Create a box plot using Seaborn
# sns.set(style="whitegrid")
# plt.figure(figsize=(10, 6))  # Adjust the figure size if needed
# sns.boxplot(data=data_ssim)
# plt.title('SSIM')
# plt.savefig(save_dir+name_+'_ssim.png')

# data_nmse = pd.DataFrame({
#     'Baseline': nmse_1,
#     '2D': nmse_2,
#     '3D non-reg': nmse_4,
#     '3D': nmse_3
# })

# # # Create a box plot using Seaborn
# sns.set(style="whitegrid")
# plt.figure(figsize=(10, 6))  # Adjust the figure size if needed
# sns.boxplot(data=data_nmse)
# plt.title('NMSE')
# plt.savefig(save_dir+name_+'_nmse.png')


# data_psnr = pd.DataFrame({
#     'Baseline': psnr_1,
#     '2D': psnr_2,
#     '3D non-reg': psnr_4,
#     '3D': psnr_3
# })

# # Create a box plot using Seaborn
# sns.set(style="whitegrid")
# plt.figure(figsize=(10, 6))  # Adjust the figure size if needed
# sns.boxplot(data=data_psnr)
# plt.title('PSNR')
# plt.savefig(save_dir+name_+'_psnr.png')


# for method, i in zip(['Baseline', '2D', '3D', '3D non-reg'],[metric_1,metric_2,metric_3, metric_4]):
#     print(f"{method} : mean is {i.means()} and std is {i.stddevs()}")
#     with open(save_dir+name_+"_output.txt", "w") as f:  # Use 'a' to append to the file
#         f.write(f"{method} : mean is {i.means()} and std is {i.stddevs()}\n")
# pass

# # print(sum(ssims_1) / len(ssims_1))
# # print(sum(ssims_2) / len(ssims_2))


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
name_ = "with_axial_normalization_DiT"
paths = [
    "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-03-05_01-00-20/",
    "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-03-04_17-31-39/",
    "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-03-04_20-58-41/",
    "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-03-06_23-41-38/",
    "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-03-07_12-02-59/",
]
methods = ['Baseline', '2D', '3D', '3D non-reg', "2D DiT"]


ssims_list, nmse_list, psnr_list, metrics_list = [], [], [], []


for path, method in zip(paths, methods):
    ssims = []
    nmses = []
    psnrs = []
    metric = Metrics(METRIC_FUNCS)

    for file in listdir(path):
        if file.endswith("preds.nii"):
            pred = nibabel.load(path + file).get_fdata()
            file_tar = file.split("preds.nii")[0]
            target = nibabel.load(path + file_tar + "targets.nii").get_fdata()

            if method in ['3D', '3D non-reg']:
                pred = pred[2:-2]
                target = target[2:-2]
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

for path, metric in zip(paths, metrics_list):
    print(f"{path} : mean is {metric.means()} and std is {metric.stddevs()}")
    with open(save_dir + name_ + "_output.txt", "a") as f:
        f.write(f"{path} : mean is {metric.means()} and std is {metric.stddevs()}\n")