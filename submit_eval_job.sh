#!/bin/bash
#SBATCH --job-name=diffusion_eval
#SBATCH --partition=general
#SBATCH --qos=general_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/project/community/aiosman/diffusion_project/logs/eval_%j.out
#SBATCH --error=/project/community/aiosman/diffusion_project/logs/eval_%j.err

echo "Job started: $(date)"
echo "Running on node: $(hostname)"

# Force Python to IGNORE the broken ~/.local installation
export PYTHONNOUSERSITE=1

# Tell W&B to run completely offline for evaluation
export WANDB_MODE=offline

# Run the defect script using the absolute Conda Python path
/home/aiosman/miniconda3/envs/diffusion/bin/python /project/community/aiosman/diffusion_project/defect_low_density.py

echo "Job finished: $(date)"