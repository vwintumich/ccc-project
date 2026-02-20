#!/bin/bash
#SBATCH --job-name=ccc_pipeline
#SBATCH --account=siads696w26_class
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --qos=class
#SBATCH --output=logs/%x-%j.out

set -e  # Stop immediately if any command fails

# Move to the notebooks directory so relative paths work
cd ~/ccc-project/indicator_clustering/notebooks

# Activate environment
eval "$(conda shell.bash hook)"
conda activate nlp_env

echo "=== Starting notebook 00: Data Extraction ==="
python 00_data_extraction_Victoria.py
echo "=== Finished notebook 00 ==="

echo "=== Starting notebook 01: Data Cleaning ==="
python 01_data_cleaning_Victoria.py
echo "=== Finished notebook 01 ==="

echo "=== Starting notebook 02: Embedding Generation ==="
python 02_embedding_generation_Victoria.py
echo "=== Finished notebook 02 ==="

echo "=== All done! ==="
