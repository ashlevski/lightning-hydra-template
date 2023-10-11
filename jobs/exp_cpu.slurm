#!/bin/sh

####### Reserve computing resources #############

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --partition=gpu-a100
#SBATCH --gpus-per-node=2
#SBATCH --gres=gpu:2



source myenv/bin/activate
EXP=experiment_recvarnet
srun python /home/amirmohammad.shamaei/MRI_REC_template/src/train.py experiment=$EXP