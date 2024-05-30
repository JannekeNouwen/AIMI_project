#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=1

#module load 2022
#module load Python/3.10.4-GCCcore-11.3.0

DATASET_ID=501
DIMENSION="2d"
DATA="nnUNet_preprocessed"
now=$(date)
echo "Hello, this is a ULS job training"
echo "The starting time is $now"
echo "This version is trained on $DATA"
echo "Training on $DIMENSION"
timestr=$(date +"%Y-%m-%d_%H-%M-%S")
source "/home/ljulius/miniconda3/etc/profile.d/conda.sh"

source /home/${USER}/.bashrc
conda activate uls

mkdir $TMPDIR/nnUNet_raw -p
cp -r "/projects/0/nwo2021061/uls23/nnUNet_raw/Dataset501_RadboudumcBone" "$TMPDIR/nnUNet_raw/"
export nnUNet_raw="$TMPDIR/nnUNet_raw"

cp -r "/home/ljulius/algorithm/nnunet/$DATA" "$TMPDIR/"
export nnUNet_preprocessed="$TMPDIR/$DATA"

export nnUNet_results="/home/ljulius/algorithm/nnunet/nnUNet_results"


nnUNetv2_train $DATASET_ID $DIMENSION 0 -tr nnUNetTrainer_ULS_50_HalfLR

now2=$(date)
echo "Done at $now"

