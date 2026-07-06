#!/bin/bash
#SBATCH --job-name=diffusion_train
#SBATCH --partition=general
#SBATCH --qos=general_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/project/community/aiosman/diffusion_project/logs/train_%j.out
#SBATCH --error=/project/community/aiosman/diffusion_project/logs/train_%j.err

echo "Job started: $(date)"
echo "Running on node: $(hostname)"
echo "GPU info:"
nvidia-smi

# Create logs dir if it doesn't exist
mkdir -p /project/community/aiosman/diffusion_project/logs


# eval "$(/project/community/aiosman/miniconda3/bin/conda shell.bash hook)"
# eval "$(~/miniconda3/bin/conda shell.bash hook)"

# conda activate diffusion
export PYTHONNOUSERSITE=1

export WANDB_API_KEY="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"

# python /project/community/aiosman/diffusion_project/train_diffusion.py
# /project/community/aiosman/miniconda3/envs/diffusion/bin/python /project/community/aiosman/diffusion_project/train_diffusion.py
# /home/aiosman/miniconda3/envs/diffusion/bin/python
/home/aiosman/miniconda3/envs/diffusion/bin/python /project/community/aiosman/diffusion_project/train_diffusion.py



echo "Backing up checkpoints and samples to cloud storage..."
gcloud storage cp -r /project/community/aiosman/diffusion_project/checkpoints gs://cmu-gpucloud-aiosman/
gcloud storage cp -r /project/community/aiosman/diffusion_project/samples gs://cmu-gpucloud-aiosman/

echo "Job finished: $(date)"