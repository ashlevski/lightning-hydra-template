#!/bin/sh

####### Reserve computing resources #############

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:10:00
#SBATCH --mem=64G
#SBATCH --partition=gpu-a100
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1



source myenv/bin/activate
EXP=experiment_calgary_campinas_long
metadata_test_dir="/home/amirmohammad.shamaei/MRI_REC_template/data/Calgary_Campinas/test_mv1.csv"
BS=64
RF=16
LR=0.001
# TAGS = ["mnist","experiment_X"]
TR=True
SM="/home/amirmohammad.shamaei/MRI_REC_template/data/Calgary_Campinas/poisson_sampling/R15_218x170.npy" #data.metadata_test_dir=$metadata_test_dir  #ckpt_path=$CKP
# CKP= "/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2023-11-22_15-03-51/checkpoints/epoch_199.ckpt""
# CKP="/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2023-11-22_22-03-34/checkpoints/epoch_199.ckpt"
net_path="/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2023-11-30_18-25-55/model_weights.pth"
srun python /home/amirmohammad.shamaei/MRI_REC_template/src/train.py experiment=$EXP train=$TR data.batch_size=$BS +model.net.model.regularizer_num_filters=$RF \
    model.optimizer.lr=$LR data.mask_dir=$SM ckpt_path=$CKP data.metadata_test_dir=$metadata_test_dir model.net_path=$net_path
