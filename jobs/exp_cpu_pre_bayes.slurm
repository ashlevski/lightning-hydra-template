#!/bin/sh

####### Reserve computing resources #############

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --partition=gpu-v100
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1



source myenv/bin/activate
EXP=experiment_calgary_campinas
BS=64
RF=16
LR=0.001
# TAGS = ["mnist","experiment_X"]
TR=False
TS=True
SM="/home/amirmohammad.shamaei/MRI_REC_template/data/Calgary_Campinas/poisson_sampling/R5_218x170.npy" #data.metadata_test_dir=$metadata_test_dir  #ckpt_path=$CKP
CKP_R20="/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-06-24_20-33-19/checkpoints/epoch_009.ckpt"
CKP="/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2023-11-22_22-03-34/checkpoints/epoch_199.ckpt"
CKP_R5="/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-08-20_20-54-39/checkpoints/epoch_019.ckpt"
srun python /home/amirmohammad.shamaei/MRI_REC_template/src/train.py experiment=$EXP train=$TR test=$TS data.batch_size=$BS +model.net.model.regularizer_num_filters=$RF \
    model.optimizer.lr=$LR data.mask_dir=$SM ckpt_path=$CKP_R5 #data.metadata_test_dir=$metadata_test_dir
