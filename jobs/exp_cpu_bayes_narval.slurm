#!/bin/sh

####### Reserve computing resources #############

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH --account=def-rmsouza  # Specify your account here
#SBATCH --gpus-per-node=a100_4g.20gb:1  # Request an A100 GPU specifically





source /home/amirsha/myenv/bin/activate
EXP=experiment_calgary_campinas_bayes_narval
BS=32
RF=16
LR=0.001
# TAGS = ["mnist","experiment_X"]
TR=False
SM="/lustre06/project/6054398/amirsha/MRI_REC_template/data/Calgary_Campinas/poisson_sampling/R5_218x170.npy" #data.metadata_test_dir=$metadata_test_dir  #ckpt_path=$CKP
CKP_5="/lustre07/scratch/amirsha/logs/train/runs/R5_ssim_in10_2/checkpoints/epoch_009.ckpt"
CKP_10="/lustre07/scratch/amirsha/logs/train/runs/R10_ssim_in10_5/checkpoints/last.ckpt"
# CKP="/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2023-11-22_22-03-34/checkpoints/epoch_199.ckpt"
# CKP="/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-09-17_22-56-24/checkpoints/epoch_016.ckpt"
net_path_10="/lustre06/project/6054398/amirsha/weigths/R10/model_weights.pth"
net_path_5="/lustre06/project/6054398/amirsha/weigths/R5/model_weights.pth"
num_of_infer=10
srun python /lustre06/project/6054398/amirsha/MRI_REC_template/src/train.py experiment=$EXP train=$TR data.batch_size=$BS +model.net.model.regularizer_num_filters=$RF \
    model.optimizer.lr=$LR data.mask_dir=$SM model.net_path=$net_path_5 model.num_of_infer=$num_of_infer ckpt_path=$CKP_5

# source /home/amirsha/myenv/bin/activate
# EXP=experiment_calgary_campinas_bayes_narval
# BS=32
# RF=16
# LR=0.001
# # TAGS = ["mnist","experiment_X"]
# TR=False
# SM="/lustre06/project/6054398/amirsha/MRI_REC_template/data/Calgary_Campinas/poisson_sampling/R10_218x170.npy" #data.metadata_test_dir=$metadata_test_dir  #ckpt_path=$CKP
# CKP_5="/lustre07/scratch/amirsha/logs/train/runs/R5_ssim_in10_2/checkpoints/last.ckpt"
# CKP_10="/lustre07/scratch/amirsha/logs/train/runs/R10_ssim_in10_5/checkpoints/last.ckpt"
# # CKP="/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2023-11-22_22-03-34/checkpoints/epoch_199.ckpt"
# # CKP="/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-09-17_22-56-24/checkpoints/epoch_016.ckpt"
# net_path_10="/lustre06/project/6054398/amirsha/weigths/R10/model_weights.pth"
# net_path_5="/lustre06/project/6054398/amirsha/weigths/R5/model_weights.pth"
# num_of_infer=10
# srun python /lustre06/project/6054398/amirsha/MRI_REC_template/src/train.py experiment=$EXP train=$TR data.batch_size=$BS +model.net.model.regularizer_num_filters=$RF \
#     model.optimizer.lr=$LR data.mask_dir=$SM model.net_path=$net_path_10 model.num_of_infer=$num_of_infer ckpt_path=$CKP_10