#!/bin/bash
#SBATCH --job-name=qm9_diffusion
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Print job information
echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Load required modules
module load anaconda3/2022.05
module load cuda/11.8

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate conda environment (you'll need to create this on Discovery)
# If you're using the .venv instead, uncomment the next line and comment out conda activate
# source .venv/bin/activate
conda activate digress

# Set PYTHONPATH
export PYTHONPATH=.

# Run training with QM9 dataset
echo "Starting training..."
python src/main.py \
    dataset=qm9 \
    general.name=qm9_gpu_run \
    general.gpus=1 \
    train.batch_size=128 \
    train.n_epochs=1000 \
    general.wandb=online

echo "Training finished at $(date)"
