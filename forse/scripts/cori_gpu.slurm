#!/bin/bash
#SBATCH -C gpu
#SBATCH -q gpu_preempt
#SBATCH -t 600
#SBATCH -c 10
#SBATCH --gres=gpu:4
#SBATCH -A m1759
#SBATCH --requeue

srun python train_dcgan_on_cori.py 

