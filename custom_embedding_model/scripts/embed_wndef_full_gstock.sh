#!/bin/bash
#SBATCH --job-name=embed_wndef_full_gstock
#SBATCH --account=siads696w26_class
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/embed_wndef_full_gstock_%j.out

# ---------------------------------------------------------------------------
# Generate g_stock full-vocabulary wndef embeddings (53,930 words).
#
# Per Decision 7, g_stock is a fixed model whose embeddings over the broad
# usable dataset can be computed once and reused. The existing g_stock
# wndef artifact is val-only (f_common_wndef_val.npy, 26,152 rows); this
# job fills in the full-vocab wndef output so NB 05's cross-f triplet
# accuracy comparison can evaluate the same triplet set under both wndef
# and wnex phrase types, and so wndef triplet resolution is no longer
# bottlenecked by the ~41% of distractor words absent from the val subset.
#
# The Decision 22 rationale (full-vocab wnex) applies identically here:
# individual decontextualized word embeddings carry no test-set evaluation
# signal, so including train/test-split vocabulary words does not violate
# Decision 9 (test set lockout).
#
# Output: data/embeddings/g_stock/f_common_wndef.npy, shape (53930, 1024),
# indexed by vocabulary_wndef.csv (Decision 6 — no separate index CSV).
# This supplements (does not replace) the existing val-only file.
#
# Before submitting:
#   1. mkdir -p logs                              (SLURM output directory)
#   2. Verify the following files are present on Great Lakes under
#      /home/vwinters/ccc-project/custom_embedding_model/:
#        data/filtered_split/wn_synset/wndef/vocabulary_wndef.csv   (53,930 rows)
#        data/filtered_split/wn_synset/wndef/f_common_wndef.csv     (53,930 rows)
#
# Submit:
#   sbatch scripts/embed_wndef_full_gstock.sh
#
# Expected runtime: ~2.5 min at ~380 phrases/sec (1-hour wall-time is
# conservative). Can be submitted simultaneously with embed_wndef_full_g1.sh —
# the jobs are independent.
#
# After completion, scp outputs back to the local machine:
#   scp vwinters@greatlakes-xfer.arc-ts.umich.edu:/home/vwinters/ccc-project/custom_embedding_model/data/embeddings/g_stock/f_common_wndef.npy \
#       /Users/victoria/Desktop/MADS/ccc-project/custom_embedding_model/data/embeddings/g_stock/
#   scp vwinters@greatlakes-xfer.arc-ts.umich.edu:/home/vwinters/ccc-project/custom_embedding_model/logs/embed_wndef_full_gstock_<jobid>.out \
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
    --vocab-file data/filtered_split/wn_synset/wndef/vocabulary_wndef.csv \
    --phrase-file data/filtered_split/wn_synset/wndef/f_common_wndef.csv \
    --output-file data/embeddings/g_stock/f_common_wndef.npy
