#!/bin/bash
# Usage: sbatch scripts/embed_phrases.sh
# Make executable first: chmod +x scripts/embed_phrases.sh
#SBATCH --job-name=cale_embed
#SBATCH --account=siads696w26_class
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/embed_%j.log
#SBATCH --error=logs/embed_%j.err

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
python -c "import sentence_transformers; print(f'sentence-transformers: {sentence_transformers.__version__}')"

# --- Run embedding script ---
python scripts/embed_phrases.py --data-dir data/ --batch-size 64

echo "Job finished: $(date)"
