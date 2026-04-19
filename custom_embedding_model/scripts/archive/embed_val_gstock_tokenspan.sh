#!/bin/bash
#SBATCH --job-name=embed_val_gstock_ts
#SBATCH --account=siads696w26_class
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/embed_val_gstock_tokenspan_%j.out

# ---------------------------------------------------------------------------
# Generate token-span-extracted validation-split embeddings for g_stock
# (unmodified CALE) across f_clue, f_common_wndef, and f_common_wnex.
#
# These embeddings are the baseline paired with g1_tokenspan for Stage 5
# ATE computation. Tokenspan extraction is NOT canonical CALE usage
# (Decision 20 establishes mean pooling as canonical); it exists here so
# g1_tokenspan — which was trained against this extraction — can be
# evaluated against its own consistent baseline.
#
# Outputs go to a separate directory (g_stock_tokenspan) so they do not
# collide with the canonical mean-pooled g_stock embeddings.
#
# The script's consistency check is automatically skipped for this run:
# tokenspan extraction is expected to differ from the saved
# g_stock/f_clue.npy (which was produced with mean pooling).
#
# Before submitting:
#   1. mkdir -p logs                              (SLURM output directory)
#   2. Verify the following files are present on Great Lakes under
#      /home/vwinters/ccc-project/custom_embedding_model/:
#        data/filtered_split/wn_synset/clue_phrases/f_clue.csv
#        data/filtered_split/wn_synset/wndef/f_common_wndef.csv
#        data/filtered_split/wn_synset/wndef/vocabulary_wndef_val.csv
#        data/filtered_split/wn_synset/wnex/f_common_wnex.csv
#        data/filtered_split/wn_synset/wnex/vocabulary_wnex_val.csv
#
# Submit:
#   sbatch scripts/embed_val_gstock_tokenspan.sh
#
# Expected runtime: ~5 min on a V100/A40 (1-hour wall-time is conservative).
# Can be submitted simultaneously with embed_val_gstock.sh and
# embed_val_g1_tokenspan.sh — the jobs are independent.
#
# After completion, scp outputs back to the local machine:
#   scp vwinters@greatlakes-xfer.arc-ts.umich.edu:/home/vwinters/ccc-project/custom_embedding_model/data/embeddings/g_stock_tokenspan/f_clue_val.npy \
#       /Users/victoria/Desktop/MADS/ccc-project/custom_embedding_model/data/embeddings/g_stock_tokenspan/
#   scp vwinters@greatlakes-xfer.arc-ts.umich.edu:/home/vwinters/ccc-project/custom_embedding_model/data/embeddings/g_stock_tokenspan/f_clue_val_index.csv \
#       /Users/victoria/Desktop/MADS/ccc-project/custom_embedding_model/data/embeddings/g_stock_tokenspan/
#   scp vwinters@greatlakes-xfer.arc-ts.umich.edu:/home/vwinters/ccc-project/custom_embedding_model/data/embeddings/g_stock_tokenspan/f_common_wndef_val.npy \
#       /Users/victoria/Desktop/MADS/ccc-project/custom_embedding_model/data/embeddings/g_stock_tokenspan/
#   scp vwinters@greatlakes-xfer.arc-ts.umich.edu:/home/vwinters/ccc-project/custom_embedding_model/data/embeddings/g_stock_tokenspan/f_common_wnex_val.npy \
#       /Users/victoria/Desktop/MADS/ccc-project/custom_embedding_model/data/embeddings/g_stock_tokenspan/
#   scp vwinters@greatlakes-xfer.arc-ts.umich.edu:/home/vwinters/ccc-project/custom_embedding_model/logs/embed_val_gstock_tokenspan_<jobid>.out \
#       /Users/victoria/Desktop/MADS/ccc-project/custom_embedding_model/logs/
# ---------------------------------------------------------------------------

# `source activate` rather than `conda activate` — `conda activate` requires
# `conda init` to have been run in the shell, which is not the case in
# non-interactive SLURM batch shells.
source activate nlp_env

# PYTHONUNBUFFERED=1 so tqdm and print() output stream to the SLURM log in
# real time rather than being buffered until the process exits.
export PYTHONUNBUFFERED=1

python scripts/embed_val.py \
    --model-path gabrielloiseau/CALE-MBERT-en \
    --output-dir data/embeddings/g_stock_tokenspan \
    --pooling tokenspan \
    --batch-size 64
