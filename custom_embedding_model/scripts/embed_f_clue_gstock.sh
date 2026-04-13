#!/bin/bash
#SBATCH --job-name=embed_fclue_gstock
#SBATCH --account=<account>
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/embed_fclue_gstock_%j.out

# ---------------------------------------------------------------------------
# Embed all f_clue phrases using g_stock (unmodified CALE).
#
# Before submitting:
#   1. Replace <account> above with your Great Lakes account name
#   2. Adjust module load / conda activate to match your environment
#   3. mkdir -p logs   (SLURM output directory must exist)
#
# Submit:
#   sbatch scripts/embed_f_clue_gstock.sh
#
# Expected runtime: 10-30 min on a V100/A40 (4-hour limit is conservative).
#
# After completion, scp outputs back to local machine:
#   scp <user>@greatlakes-xfer.arc-ts.umich.edu:<path>/data/embeddings/g_stock/f_clue.npy \
#       custom_embedding_model/data/embeddings/g_stock/
#   scp <user>@greatlakes-xfer.arc-ts.umich.edu:<path>/data/embeddings/g_stock/f_clue_index.csv \
#       custom_embedding_model/data/embeddings/g_stock/
# ---------------------------------------------------------------------------

module load python/3.10
# conda activate <your-env>   # Uncomment and set your environment name

python scripts/embed_f_clue_gstock.py \
    --input data/filtered_split/wn_synset/clue_phrases/f_clue.csv \
    --output-dir data/embeddings/g_stock \
    --batch-size 64
