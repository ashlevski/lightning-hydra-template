#!/bin/sh

####### Reserve computing resources #############

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:10:00
#SBATCH --mem=128G
#SBATCH --partition=gpu-a100
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1



source myenv/bin/activate
EXP=experiment_calgary_campinas_long_mv_reg
data_dir="/work/souza_lab/amir/Data/h5"
target_dir="/work/souza_lab/amir/Data/target_rss_aligned"
metadata_train_dir="/home/amirmohammad.shamaei/MRI_REC_template/projects/longitudinal/csvs/train_mv_subv.csv"
metadata_val_dir="/home/amirmohammad.shamaei/MRI_REC_template/projects/longitudinal/csvs/test_mv_subv.csv"
metadata_test_dir="/home/amirmohammad.shamaei/MRI_REC_template/projects/longitudinal/csvs/test_mv_subv.csv"

metadata_prev_train_dir="/home/amirmohammad.shamaei/MRI_REC_template/projects/longitudinal/csvs/train_mv_fv.csv"
metadata_prev_val_dir="/home/amirmohammad.shamaei/MRI_REC_template/projects/longitudinal/csvs/test_mv_fv.csv"
metadata_prev_test_dir="/home/amirmohammad.shamaei/MRI_REC_template/projects/longitudinal/csvs/test_mv_fv.csv"

mask_dir="/home/amirmohammad.shamaei/MRI_REC_template/data/Calgary_Campinas/poisson_sampling/R10_218x170.npy"
sv_model="/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-01-17_23-50-25/model_weights.pth"
BS=256
RF=16
LR=0.001
num_workers=8
# TAGS = ["mnist","experiment_X"]
TR=True
# ckpt_path="/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2023-12-11_02-26-38/checkpoints/epoch_082.ckpt"
# ckpt_path="/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2023-12-12_09-04-07/checkpoints/epoch_083.ckpt"
srun python /home/amirmohammad.shamaei/MRI_REC_template/src/train.py experiment=$EXP train=$TR data.batch_size=$BS +model.net.single_visit_net.model.regularizer_num_filters=$RF model.optimizer.lr=$LR \
    data.metadata_train_dir=$metadata_train_dir data.metadata_val_dir=$metadata_val_dir data.metadata_test_dir=$metadata_test_dir \
    data.metadata_prev_train_dir=$metadata_prev_train_dir data.metadata_prev_val_dir=$metadata_prev_val_dir data.metadata_prev_test_dir=$metadata_prev_test_dir \
    model.net.weights_path=$sv_model data.mask_dir=$mask_dir data.data_dir=$data_dir data.target_dir=$target_dir data.num_workers=$num_workers \
    # ckpt_path=$ckpt_path
