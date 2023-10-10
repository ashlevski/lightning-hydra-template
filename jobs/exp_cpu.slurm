#!/bin/sh

####### Reserve computing resources #############

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --mem=64G
#SBATCH --partition=ood-vis
#SBATCH --gpus-per-node=2



source myenv/bin/activate
EXP=experiment_linux_ddp
srun python /home/amirmohammad.shamaei/MRI_REC_template/src/train.py experiment=$EXP