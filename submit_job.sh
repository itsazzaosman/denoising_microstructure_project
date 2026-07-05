#!/bin/bash
#SBATCH --job-name=diffusion_train
#SBATCH --partition=general
#SBATCH --qos=general_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/project/community/aiosman/logs/train_%j.out
#SBATCH --error=/project/community/aiosman/logs/train_%j.err

echo "Job started: $(date)"
echo "Running on node: $(hostname)"
echo "GPU info:"
nvidia-smi

# Create logs dir if it doesn't exist
mkdir -p /project/community/aiosman/logs


eval "$(/project/community/aiosman/miniconda3/bin/conda shell.bash hook)"
conda activate diffusion

python /project/community/aiosman/train_diffusion.py

echo "Job finished: $(date)"