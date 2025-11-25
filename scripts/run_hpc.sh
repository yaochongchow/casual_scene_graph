#!/bin/bash
#SBATCH --job-name=digress_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Load modules (Adjust based on Discovery's specific modules, usually these are standard)
module load python/3.10
module load cuda/11.8

# Activate Virtual Environment
source .venv/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs

# Run Training
# Replace 'config.train.batch_size=32' with your desired config overrides
echo "Starting training on $(hostname)"
python3 src/main.py general.name=digress_zinc_full dataset.name=zinc model.type=discrete

echo "Training finished"
