#!/bin/bash
#SBATCH --job-name=embed_val_gstock
#SBATCH --account=siads696w26_class
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/embed_val_gstock_%j.out

# ---------------------------------------------------------------------------
# Generate canonical (mean-pooled) validation-split embeddings for g_stock
# (unmodified CALE) across all three phrase types: f_clue, f_common_wndef,
# f_common_wnex. These _val embeddings are saved alongside the existing
# full-scope files in data/embeddings/g_stock/ and are the canonical baseline
# for Stage 5 ATE computation against g1 (mean pooling).
#
# Per Decision 20, mean pooling is the canonical extraction method for CALE.
# The companion wrapper scripts/embed_val_gstock_tokenspan.sh produces the
# tokenspan-pooled g_stock baseline used to evaluate g1_tokenspan fairly.
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
#        data/embeddings/g_stock/f_clue.npy          (for consistency verify)
#        data/embeddings/g_stock/f_clue_index.csv    (for consistency verify)
#
# Submit:
#   sbatch scripts/embed_val_gstock.sh
#
# Expected runtime: ~5 min on a V100/A40 (1-hour wall-time is conservative).
# Can be submitted simultaneously with the other embed_val_*.sh wrappers —
# the jobs are independent.
#
# After completion, scp outputs back to the local machine:
#   scp vwinters@greatlakes-xfer.arc-ts.umich.edu:/home/vwinters/ccc-project/custom_embedding_model/data/embeddings/g_stock/f_clue_val.npy \
#       /Users/victoria/Desktop/MADS/ccc-project/custom_embedding_model/data/embeddings/g_stock/
#   scp vwinters@greatlakes-xfer.arc-ts.umich.edu:/home/vwinters/ccc-project/custom_embedding_model/data/embeddings/g_stock/f_clue_val_index.csv \
#       /Users/victoria/Desktop/MADS/ccc-project/custom_embedding_model/data/embeddings/g_stock/
#   scp vwinters@greatlakes-xfer.arc-ts.umich.edu:/home/vwinters/ccc-project/custom_embedding_model/data/embeddings/g_stock/f_common_wndef_val.npy \
#       /Users/victoria/Desktop/MADS/ccc-project/custom_embedding_model/data/embeddings/g_stock/
#   scp vwinters@greatlakes-xfer.arc-ts.umich.edu:/home/vwinters/ccc-project/custom_embedding_model/data/embeddings/g_stock/f_common_wnex_val.npy \
#       /Users/victoria/Desktop/MADS/ccc-project/custom_embedding_model/data/embeddings/g_stock/
#   scp vwinters@greatlakes-xfer.arc-ts.umich.edu:/home/vwinters/ccc-project/custom_embedding_model/logs/embed_val_gstock_<jobid>.out \
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
    --output-dir data/embeddings/g_stock \
    --pooling meanpool \
    --batch-size 64
