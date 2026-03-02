#!/bin/bash
# Usage: sbatch scripts/run_experiments.sh
# Make executable first: chmod +x scripts/run_experiments.sh
#
# Runs all 4 classification experiments (1A, 1B, 2A, 2B) with full
# hyperparameter grids on the full datasets. CPU only — no GPU needed.
#SBATCH --job-name=ccc_exper
#SBATCH --account=siads696w26_class
#SBATCH --partition=standard
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/experiments_%j.log
#SBATCH --error=logs/experiments_%j.err

# --- Setup ---
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"

# Create logs directory if it doesn't exist
mkdir -p logs

# Navigate to project directory
cd /home/vwinters/ccc-project/clue_misdirection

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate nlp_env

# Verify environment
echo "Python: $(which python)"
python -c "import sklearn; print(f'scikit-learn: {sklearn.__version__}')"
python -c "import pandas; print(f'pandas: {pandas.__version__}')"

# --- Run experiments (full data, full grids) ---
python scripts/run_experiments.py

echo "Job finished: $(date)"
