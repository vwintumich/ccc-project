#!/bin/bash
#SBATCH --job-name=embed_wnex_full_gstock
#SBATCH --account=siads696w26_class
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/embed_wnex_full_gstock_%j.out

# ---------------------------------------------------------------------------
# Generate g_stock full-vocabulary wnex embeddings (8,360 words).
#
# Per Decision 7, g_stock is a fixed model whose embeddings over the broad
# usable dataset can be computed once and reused. The existing g_stock
# artifacts cover f_clue (full, 239K rows) and the val-only subsets of the
# vocabulary-indexed phrase files; this job fills in the missing
# full-vocab wnex output.
#
# Output: data/embeddings/g_stock/f_common_wnex.npy, shape (8360, 1024),
# indexed by vocabulary_wnex.csv (Decision 6 — no separate index CSV).
#
# Before submitting:
#   1. mkdir -p logs                              (SLURM output directory)
#   2. Verify the following files are present on Great Lakes under
#      /home/vwinters/ccc-project/custom_embedding_model/:
#        data/filtered_split/wn_synset/wnex/vocabulary_wnex.csv   (8,360 rows)
#        data/filtered_split/wn_synset/wnex/f_common_wnex.csv     (8,360 rows)
#
# Submit:
#   sbatch scripts/embed_wnex_full_gstock.sh
#
# Expected runtime: ~5 min on a V100/A40 (1-hour wall-time is conservative).
# Can be submitted simultaneously with embed_wnex_full_g1.sh — the jobs
# are independent.
#
# After completion, scp outputs back to the local machine:
#   scp vwinters@greatlakes-xfer.arc-ts.umich.edu:/home/vwinters/ccc-project/custom_embedding_model/data/embeddings/g_stock/f_common_wnex.npy \
#       /Users/victoria/Desktop/MADS/ccc-project/custom_embedding_model/data/embeddings/g_stock/
#   scp vwinters@greatlakes-xfer.arc-ts.umich.edu:/home/vwinters/ccc-project/custom_embedding_model/logs/embed_wnex_full_gstock_<jobid>.out \
#       /Users/victoria/Desktop/MADS/ccc-project/custom_embedding_model/logs/
# ---------------------------------------------------------------------------

# `source activate` rather than `conda activate` — `conda activate` requires
# `conda init` to have been run in the shell, which is not the case in
# non-interactive SLURM batch shells.
source activate nlp_env

# PYTHONUNBUFFERED=1 so tqdm and print() output stream to the SLURM log in
# real time rather than being buffered until the process exits.
export PYTHONUNBUFFERED=1

python scripts/embed_vocab.py \
    --model-path gabrielloiseau/CALE-MBERT-en \
    --pooling meanpool \
    --vocab-file data/filtered_split/wn_synset/wnex/vocabulary_wnex.csv \
    --phrase-file data/filtered_split/wn_synset/wnex/f_common_wnex.csv \
    --output-file data/embeddings/g_stock/f_common_wnex.npy
