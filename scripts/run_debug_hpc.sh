#!/bin/bash
#SBATCH --job-name=qm9_debug
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Quick debug job to test environment setup
echo "Debug job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Load modules
module load anaconda3/2022.05
module load cuda/11.8

cd $SLURM_SUBMIT_DIR
mkdir -p logs

# Activate environment
conda activate digress

# Set PYTHONPATH
export PYTHONPATH=.

# Run debug experiment
echo "Running debug experiment..."
python src/main.py +experiment=debug general.gpus=1

echo "Debug finished at $(date)"
