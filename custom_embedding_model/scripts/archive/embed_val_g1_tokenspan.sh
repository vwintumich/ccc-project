#!/bin/bash
#SBATCH --job-name=embed_val_g1_ts
#SBATCH --account=siads696w26_class
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/embed_val_g1_tokenspan_%j.out

# ---------------------------------------------------------------------------
# Generate validation-split embeddings for g1_tokenspan (the fine-tuned CALE
# variant trained with token span extraction, per NB 09 /
# train_g1_tokenspan.py). Covers all three phrase types: f_clue,
# f_common_wndef, f_common_wnex.
#
# Tokenspan extraction must be used at inference time because that is the
# extraction g1_tokenspan was trained against. Decision 20 establishes mean
# pooling as canonical for CALE; g1_tokenspan is retained as a historical
# comparison point but is not the corrected g1.
#
# The script's consistency check is automatically skipped for this run:
# tokenspan extraction differs from the saved g_stock/f_clue.npy (which was
# produced with mean pooling), and additionally these weights are
# fine-tuned, so the saved baseline is not a meaningful reference.
#
# Before submitting:
#   1. mkdir -p logs                              (SLURM output directory)
#   2. Verify the following files are present on Great Lakes under
#      /home/vwinters/ccc-project/custom_embedding_model/:
#        models/g1_tokenspan/model/                  (from train_g1_tokenspan.py)
#        data/filtered_split/wn_synset/clue_phrases/f_clue.csv
#        data/filtered_split/wn_synset/wndef/f_common_wndef.csv
#        data/filtered_split/wn_synset/wndef/vocabulary_wndef_val.csv
#        data/filtered_split/wn_synset/wnex/f_common_wnex.csv
#        data/filtered_split/wn_synset/wnex/vocabulary_wnex_val.csv
#
# Submit:
#   sbatch scripts/embed_val_g1_tokenspan.sh
#
# Expected runtime: ~5 min on a V100/A40 (1-hour wall-time is conservative).
# Can be submitted simultaneously with the other embed_val_*.sh wrappers —
# the jobs are independent.
#
# After completion, scp outputs back to the local machine:
#   scp vwinters@greatlakes-xfer.arc-ts.umich.edu:/home/vwinters/ccc-project/custom_embedding_model/data/embeddings/g1_tokenspan/f_clue_val.npy \
#       /Users/victoria/Desktop/MADS/ccc-project/custom_embedding_model/data/embeddings/g1_tokenspan/
#   scp vwinters@greatlakes-xfer.arc-ts.umich.edu:/home/vwinters/ccc-project/custom_embedding_model/data/embeddings/g1_tokenspan/f_clue_val_index.csv \
#       /Users/victoria/Desktop/MADS/ccc-project/custom_embedding_model/data/embeddings/g1_tokenspan/
#   scp vwinters@greatlakes-xfer.arc-ts.umich.edu:/home/vwinters/ccc-project/custom_embedding_model/data/embeddings/g1_tokenspan/f_common_wndef_val.npy \
#       /Users/victoria/Desktop/MADS/ccc-project/custom_embedding_model/data/embeddings/g1_tokenspan/
#   scp vwinters@greatlakes-xfer.arc-ts.umich.edu:/home/vwinters/ccc-project/custom_embedding_model/data/embeddings/g1_tokenspan/f_common_wnex_val.npy \
#       /Users/victoria/Desktop/MADS/ccc-project/custom_embedding_model/data/embeddings/g1_tokenspan/
#   scp vwinters@greatlakes-xfer.arc-ts.umich.edu:/home/vwinters/ccc-project/custom_embedding_model/logs/embed_val_g1_tokenspan_<jobid>.out \
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
    --model-path models/g1_tokenspan/model \
    --output-dir data/embeddings/g1_tokenspan \
    --pooling tokenspan \
    --batch-size 64
