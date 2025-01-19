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
EXP=experiment_calgary_campinas
BS=8
RF=16

srun python /home/amirmohammad.shamaei/MRI_REC_template/src/eval.py data.batch_size=$BS +model.net.model.regularizer_num_filters=$RF 
