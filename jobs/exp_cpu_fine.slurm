#!/bin/sh

####### Reserve computing resources #############

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --partition=gpu-v100
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1



source myenv/bin/activate
EXP=experiment_calgary_campinas_long_ft
max_epochs=201
BS=1
RF=16
LR=0.0001
TR=True
metadata_train_dir="/home/amirmohammad.shamaei/MRI_REC_template/data/Calgary_Campinas/csv/Test/NORM100_sv.csv"
metadata_val_dir="/home/amirmohammad.shamaei/MRI_REC_template/data/Calgary_Campinas/csv/Test/NORM100_mv.csv"
metadata_test_dir="/home/amirmohammad.shamaei/MRI_REC_template/data/Calgary_Campinas/csv/Test/NORM100_mv.csv"
#CKP= null #"/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2023-11-05_17-43-01/checkpoints/epoch_305.ckpt"ckpt_path=$CKP
srun python /home/amirmohammad.shamaei/MRI_REC_template/src/train.py experiment=$EXP train=$TR data.batch_size=$BS \
    +model.net.model.regularizer_num_filters=$RF model.optimizer.lr=$LR data.metadata_test_dir=$metadata_test_dir \
    data.metadata_train_dir=$metadata_train_dir data.metadata_val_dir=$metadata_val_dir data.metadata_test_dir=$metadata_test_dir \
    trainer.max_epochs=$max_epochs
