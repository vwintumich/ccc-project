#!/bin/bash
# Usage: sbatch scripts/run_experiments_test.sh
# Make executable first: chmod +x scripts/run_experiments_test.sh
#
# Test run: 20K-row subsample per dataset, reduced hyperparameter grids.
# Use this to verify the script works before submitting the full run.
#SBATCH --job-name=ccc_test
#SBATCH --account=siads696w26_class
#SBATCH --partition=standard
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/experiments_test_%j.log
#SBATCH --error=logs/experiments_test_%j.err

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

# --- Run experiments (sample mode) ---
python scripts/run_experiments.py --sample

echo "Job finished: $(date)"
