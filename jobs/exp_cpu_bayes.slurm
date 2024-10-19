#!/bin/sh

####### Reserve computing resources #############

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --mem=64G
#SBATCH --partition=gpu-v100
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1

source myenv/bin/activate
EXP=experiment_calgary_campinas_mcdo_narval
EXP_GBM=experiment_calgary_campinas_mcdo_GBM_narval
BS=60
RF=16
LR=0.001
# TAGS = ["mnist","experiment_X"]
TR=False
SM_5="/home/amirmohammad.shamaei/MRI_REC_template/data/Calgary_Campinas/poisson_sampling/R5_218x170.npy" #data.metadata_test_dir=$metadata_test_dir  #ckpt_path=$CKP
SM_10="/home/amirmohammad.shamaei/MRI_REC_template/data/Calgary_Campinas/poisson_sampling/R10_218x170.npy"
CKP_5="/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-10-15_13-54-12/checkpoints/epoch_028.ckpt"
CKP_10="/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-10-15_19-59-59/checkpoints/epoch_029.ckpt"
net_path_10="/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-07-18_16-51-11/model_weights.pth"
net_path_5="/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-09-12_20-39-29/model_weights.pth"
num_of_infer=50

data_dir= "/work/souza_lab/amir/Data/GBM/h5/_norm"
metadata_test_dir= "/home/amirmohammad.shamaei/MRI_REC_template/projects/uncertainty/csvs/train_gbm_chal.csv"
num_slices= 152
# srun python /home/amirmohammad.shamaei/MRI_REC_template/src/train.py experiment=$EXP train=$TR data.batch_size=$BS +model.net.model.regularizer_num_filters=$RF \
#     model.optimizer.lr=$LR data.mask_dir=$SM_5 model.net_path=$net_path_5 model.num_of_infer=$num_of_infer ckpt_path=$CKP_5 #maybe

# srun python /home/amirmohammad.shamaei/MRI_REC_template/src/train.py experiment=$EXP train=$TR data.batch_size=$BS +model.net.model.regularizer_num_filters=$RF \
#     model.optimizer.lr=$LR data.mask_dir=$SM_10 model.net_path=$net_path_10 model.num_of_infer=$num_of_infer ckpt_path=$CKP_10

# srun python /home/amirmohammad.shamaei/MRI_REC_template/src/train.py experiment=$EXP_GBM train=$TR data.batch_size=$BS +model.net.model.regularizer_num_filters=$RF \
#     model.optimizer.lr=$LR data.mask_dir=$SM_5 model.net_path=$net_path_5 model.num_of_infer=$num_of_infer ckpt_path=$CKP_5

srun python /home/amirmohammad.shamaei/MRI_REC_template/src/train.py experiment=$EXP_GBM train=$TR data.batch_size=$BS +model.net.model.regularizer_num_filters=$RF \
    model.optimizer.lr=$LR data.mask_dir=$SM_10 model.net_path=$net_path_10 model.num_of_infer=$num_of_infer ckpt_path=$CKP_10

# source myenv/bin/activate
# EXP=experiment_calgary_campinas_bayes
# BS=64
# RF=16
# LR=0.001
# # TAGS = ["mnist","experiment_X"]
# TR=False
# SM="/home/amirmohammad.shamaei/MRI_REC_template/data/Calgary_Campinas/poisson_sampling/R5_218x170.npy" #data.metadata_test_dir=$metadata_test_dir  #ckpt_path=$CKP
# # CKP="/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-07-04_17-51-00/checkpoints/epoch_004.ckpt"
# # CKP="/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2023-11-22_22-03-34/checkpoints/epoch_199.ckpt"
# CKP="/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-09-17_22-56-24/checkpoints/epoch_016.ckpt"
# net_path_20="/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-07-03_16-57-33/model_weights.pth"
# net_path_5="/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-09-12_20-39-29/model_weights.pth"
# num_of_infer=10
# srun python /home/amirmohammad.shamaei/MRI_REC_template/src/train.py experiment=$EXP train=$TR data.batch_size=$BS +model.net.model.regularizer_num_filters=$RF \
#     model.optimizer.lr=$LR data.mask_dir=$SM model.net_path=$net_path_5 model.num_of_infer=$num_of_infer ckpt_path=$CKP

