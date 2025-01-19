#!/bin/sh

####### Reserve computing resources #############

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --partition=gpu-a100
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1



source myenv/bin/activate
EXP=experiment_calgary_campinas_selfsup
data_dir="/work/souza_lab/amir/Data/h5"
target_dir="/work/souza_lab/amir/Data/target_rss_h5"
metadata_train_dir="/home/amirmohammad.shamaei/MRI_REC_template/data/Calgary_Campinas/train_sv1.csv"
metadata_val_dir="/home/amirmohammad.shamaei/MRI_REC_template/data/Calgary_Campinas/val_sv1.csv"
metadata_test_dir="/home/amirmohammad.shamaei/MRI_REC_template/data/Calgary_Campinas/test_sv1.csv"
mask_dir="/home/amirmohammad.shamaei/MRI_REC_template/data/Calgary_Campinas/poisson_sampling/R10_218x170.npy"
BS=32
RF=16
LR=0.001
num_workers=0
# TAGS = ["mnist","experiment_X"]
TR=True
SM="/home/amirmohammad.shamaei/MRI_REC_template/data/Calgary_Campinas/poisson_sampling/R10_218x170.npy" #data.metadata_test_dir=$metadata_test_dir  #ckpt_path=$CKP
srun python /home/amirmohammad.shamaei/MRI_REC_template/src/train.py experiment=$EXP train=$TR data.batch_size=$BS +model.net.model.regularizer_num_filters=$RF \
    data.metadata_train_dir=$metadata_train_dir data.metadata_val_dir=$metadata_val_dir data.metadata_test_dir=$metadata_test_dir \
    model.optimizer.lr=$LR data.mask_dir=$mask_dir data.data_dir=$data_dir data.num_workers=$num_workers #ckpt_path=$ckpt_path
