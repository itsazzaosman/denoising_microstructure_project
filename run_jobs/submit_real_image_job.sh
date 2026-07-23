#!/bin/bash
#SBATCH --job-name=real_image_inpaint
#SBATCH --partition=general
#SBATCH --qos=general_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/project/community/aiosman/logs/real_inpaint_%j.out
#SBATCH --error=/project/community/aiosman/logs/real_inpaint_%j.err

echo "Job started: $(date)"
echo "Running on node: $(hostname)"

export PYTHONNOUSERSITE=1
export WANDB_MODE=offline

/home/aiosman/miniconda3/envs/diffusion/bin/python /project/community/aiosman/defect_real_image.py

echo "Job finished: $(date)"
