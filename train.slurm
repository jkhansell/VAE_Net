#!/bin/bash
#SBATCH --job-name=VAE_Train
#SBATCH --partition=nukwa
#SBATCH --nodes=1

#SBATCH --time=24:00:00
#SBATCH --output=VAE_%j.out
#SBATCH --mail-user=jovillalobos@cenat.ac.cr
#SBATCH --mail-type=END,FAIL

source /home/jovillalobos/.bashrc
conda activate pytorch-cuda

python main.py