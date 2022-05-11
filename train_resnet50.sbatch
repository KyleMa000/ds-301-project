#!/bin/bash
#SBATCH --cpus-per-task=2               
#SBATCH --time=4:30:00                  
#SBATCH --mem=16GB
#SBATCH --job-name=resnet50
#SBATCH --output=logs/resnet50.out
#SBATCH --gres=gpu:rtx8000:1

cd /scratch/dm4524/ds-301-final-project
conda activate ds301
python train.py --mode resnet50