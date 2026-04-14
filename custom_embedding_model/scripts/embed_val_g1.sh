#!/bin/bash
#SBATCH --job-name=embed_val_g1
#SBATCH --account=siads696w26_class
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/embed_val_g1_%j.out

# ---------------------------------------------------------------------------
# Generate validation-split embeddings for g_1 (fine-tuned CALE) across all
# three phrase types: f_clue, f_common_wndef, f_common_wnex. These embeddings
# are what Stage 5 compares against g_stock for ATE measurement and cross-f
# generalization analysis.
#
# Before submitting:
#   1. mkdir -p logs                              (SLURM output directory)
#   2. Verify the following files are present on Great Lakes under
#      /home/vwinters/ccc-project/custom_embedding_model/:
#        models/g1/model/                             (from train_g1.py)
#        data/filtered_split/wn_synset/clue_phrases/f_clue.csv
#        data/filtered_split/wn_synset/wndef/f_common_wndef.csv
#        data/filtered_split/wn_synset/wndef/vocabulary_wndef_val.csv
#        data/filtered_split/wn_synset/wnex/f_common_wnex.csv
#        data/filtered_split/wn_synset/wnex/vocabulary_wnex_val.csv
#        data/embeddings/g_stock/f_clue.npy          (for consistency verify)
#        data/embeddings/g_stock/f_clue_index.csv    (for consistency verify)
#
#      Note: the consistency verification re-embeds g_stock-era phrases with
#      THIS model. For g_1 that comparison is expected to DIFFER from
#      g_stock's saved embeddings — because the model has been fine-tuned.
#      If you run this job without --skip-verify it will fail the > 0.999
#      check; that is intended behavior only when running g_stock. The
#      verification is therefore skipped for the g_1 job (see below).
#
# Submit:
#   sbatch scripts/embed_val_g1.sh
#
# Expected runtime: ~5 min on a V100/A40 (1-hour wall-time is conservative).
# Can be submitted simultaneously with embed_val_gstock.sh — the jobs are
# independent.
#
# After completion, scp outputs back to the local machine:
#   scp vwinters@greatlakes-xfer.arc-ts.umich.edu:/home/vwinters/ccc-project/custom_embedding_model/data/embeddings/g1/f_clue_val.npy \
#       /Users/victoria/Desktop/MADS/ccc-project/custom_embedding_model/data/embeddings/g1/
#   scp vwinters@greatlakes-xfer.arc-ts.umich.edu:/home/vwinters/ccc-project/custom_embedding_model/data/embeddings/g1/f_clue_val_index.csv \
#       /Users/victoria/Desktop/MADS/ccc-project/custom_embedding_model/data/embeddings/g1/
#   scp vwinters@greatlakes-xfer.arc-ts.umich.edu:/home/vwinters/ccc-project/custom_embedding_model/data/embeddings/g1/f_common_wndef_val.npy \
#       /Users/victoria/Desktop/MADS/ccc-project/custom_embedding_model/data/embeddings/g1/
#   scp vwinters@greatlakes-xfer.arc-ts.umich.edu:/home/vwinters/ccc-project/custom_embedding_model/data/embeddings/g1/f_common_wnex_val.npy \
#       /Users/victoria/Desktop/MADS/ccc-project/custom_embedding_model/data/embeddings/g1/
#   scp vwinters@greatlakes-xfer.arc-ts.umich.edu:/home/vwinters/ccc-project/custom_embedding_model/logs/embed_val_g1_<jobid>.out \
#       /Users/victoria/Desktop/MADS/ccc-project/custom_embedding_model/logs/
# ---------------------------------------------------------------------------

# `source activate` rather than `conda activate` — `conda activate` requires
# `conda init` to have been run in the shell, which is not the case in
# non-interactive SLURM batch shells.
source activate nlp_env

# PYTHONUNBUFFERED=1 so tqdm and print() output stream to the SLURM log in
# real time rather than being buffered until the process exits.
export PYTHONUNBUFFERED=1

# --skip-verify: the consistency check compares AutoModel extraction against
# the saved g_stock/f_clue.npy, which is only meaningful when running g_stock.
# For g_1 the two arrays are expected to disagree (the model has been
# fine-tuned), so the verification is skipped.
python scripts/embed_val.py \
    --model-path models/g1/model \
    --output-dir data/embeddings/g1 \
    --batch-size 64 \
    --skip-verify
