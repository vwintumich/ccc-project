#!/bin/bash
# Usage: sbatch scripts/train_g_triplet.sh
# Make executable first: chmod +x scripts/train_g_triplet.sh
#SBATCH --job-name=train_g
#SBATCH --account=siads696w26_class
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/train_g_%j.log
#SBATCH --error=logs/train_g_%j.err

# --- Setup ---
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"

# Create logs directory if it doesn't exist
mkdir -p logs

# Navigate to project directory
cd /home/vwinters/ccc-project/clue_misdirection

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate nlp_env

# Verify environment
echo "Python: $(which python)"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'transformers: {transformers.__version__}')"

# --- Run training script ---
python scripts/train_g_triplet.py

echo "Job finished: $(date)"
