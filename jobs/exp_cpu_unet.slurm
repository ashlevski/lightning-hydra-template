#!/bin/sh

####### Reserve computing resources #############

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --partition=gpu-a100
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1



source myenv/bin/activate
EXP=experiment_calgary_campinas_long_unet
metadata_test_dir="/home/amirmohammad.shamaei/MRI_REC_template/projects/longitudinal/csvs/test_mv_subv.csv"
BS=1024
num_workers=8
RF=16
LR=0.001
# TAGS = ["mnist","experiment_X"]
TR=True
SM="/home/amirmohammad.shamaei/MRI_REC_template/data/Calgary_Campinas/poisson_sampling/R10_218x170.npy" #data.metadata_test_dir=$metadata_test_dir  #ckpt_path=$CKP
# CKP="/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-01-12_11-43-04/checkpoints/epoch_046.ckpt"
# CKP="/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2023-11-22_22-03-34/checkpoints/epoch_199.ckpt"
srun python /home/amirmohammad.shamaei/MRI_REC_template/src/train.py experiment=$EXP train=$TR data.batch_size=$BS \
    model.optimizer.lr=$LR data.mask_dir=$SM data.metadata_test_dir=$metadata_test_dir data.num_workers=$num_workers
