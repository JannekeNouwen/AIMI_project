#!/bin/bash

#module load 2022
#module load Python/3.10.4-GCCore-11.3.0

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --mem=1G
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=1

source "/home/ljulius/miniconda3/etc/profile.d/conda.sh"
conda activate uls

now=$(date +"%Y-%m-%d_%H-%M-%S")
python /home/ljulius/scripts/test.py > "/home/ljulius/log/test-$now.log"

