#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=5G
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=1


#module load 2022
#module load Python/3.10.4-GCCcore-11.3.0

now=$(date)
echo "Hello, this is a test job. "
echo "The starting time is $now"

source "/home/ljulius/miniconda3/etc/profile.d/conda.sh"
conda activate uls

python /home/ljulius/scripts/test_gpu.py 2>&1 > /home/ljulius/log/test_gpu.log

now2=$(date)
echo "Done at $now2"
