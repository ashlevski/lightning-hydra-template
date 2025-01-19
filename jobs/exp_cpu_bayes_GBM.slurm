#!/bin/sh

####### Reserve computing resources #############

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --partition=bigmem
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1



source myenv/bin/activate
EXP=experiment_calgary_campinas_bayes_GBM
BS=64
RF=16
LR=0.001
# TAGS = ["mnist","experiment_X"]
TR=False
SM="/home/amirmohammad.shamaei/MRI_REC_template/data/Calgary_Campinas/poisson_sampling/R20_218x170.npy" #data.metadata_test_dir=$metadata_test_dir  #ckpt_path=$CKP
CKP="/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-07-04_17-51-00/checkpoints/epoch_004.ckpt"
# CKP="/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2023-11-22_22-03-34/checkpoints/epoch_199.ckpt"
net_path="/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-07-03_16-57-33/model_weights.pth"
num_of_infer=10
srun python /home/amirmohammad.shamaei/MRI_REC_template/src/train.py experiment=$EXP train=$TR data.batch_size=$BS +model.net.model.regularizer_num_filters=$RF \
    model.optimizer.lr=$LR data.mask_dir=$SM model.net_path=$net_path model.num_of_infer=$num_of_infer ckpt_path=$CKP