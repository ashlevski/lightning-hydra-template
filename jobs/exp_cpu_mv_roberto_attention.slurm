#!/bin/sh

####### Reserve computing resources #############

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --partition=gpu-a100
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1



source myenv/bin/activate
EXP=experiment_calgary_campinas_long_mv_dit

BS=32
RF=16
LR=0.001
num_workers=4
# TAGS = ["mnist","experiment_X"]
TR=False
TS=True
data_dir="/work/souza_lab/amir/Data/roberto/Non-enhanced_e2ev/20x/"
baseline_dir="/work/souza_lab/amir/Data/roberto/easyreg_atlas/h5/"
# data_dir="/work/souza_lab/amir/Data/roberto/Non-enhanced/20x/"
ckpt_path_5="/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-04-30_11-05-35/checkpoints/epoch_137.ckpt"
ckpt_path_10="/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-04-26_09-43-42/checkpoints/epoch_071.ckpt"
ckpt_path_15="/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-04-30_11-06-35/checkpoints/epoch_030.ckpt"
ckpt_path_20="/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-04-30_01-44-07/checkpoints/epoch_020.ckpt"
# ckpt_path_3d="/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-03-04_16-08-54/checkpoints/epoch_597.ckpt"
# ckpt_path_3d_dit="/home/amirmohammad.shamaei/MRI_REC_template/logs/train/runs/2024-03-11_16-02-01/checkpoints/epoch_128.ckpt"

srun python /home/amirmohammad.shamaei/MRI_REC_template/src/train.py experiment=$EXP train=$TR test=$TS data.batch_size=$BS trainer.max_epochs=200 data.data_dir=$data_dir data.baseline_dir=$baseline_dir ckpt_path=$ckpt_path_20  
 #+model.net.single_visit_net.model.regularizer_num_filters=$RF model.optimizer.lr=$LR \
    # data.metadata_train_dir=$metadata_train_dir data.metadata_val_dir=$metadata_val_dir data.metadata_test_dir=$metadata_test_dir \
    # data.metadata_prev_train_dir=$metadata_prev_train_dir data.metadata_prev_val_dir=$metadata_prev_val_dir data.metadata_prev_test_dir=$metadata_prev_test_dir \
    # model.net.weights_path=$sv_model data.mask_dir=$mask_dir data.data_dir=$data_dir data.target_dir=$target_dir data.num_workers=$num_workers #ckpt_path=$ckpt_path

# srun python /home/amirmohammad.shamaei/MRI_REC_template/src/test.py experiment=$EXP
