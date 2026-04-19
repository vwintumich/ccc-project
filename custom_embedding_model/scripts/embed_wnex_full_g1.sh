#!/bin/bash
#SBATCH --job-name=embed_wnex_full_g1
#SBATCH --account=siads696w26_class
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/embed_wnex_full_g1_%j.out

# ---------------------------------------------------------------------------
# Generate g1 full-vocabulary wnex embeddings (8,360 words) using the
# fine-tuned CALE variant trained with mean pooling (Decision 20 canonical).
#
# This extends the existing g1 validation-only wnex embeddings to the full
# wnex vocabulary so the cross-f generalization test (Step B in
# FINDINGS.md) can be evaluated over every wnex word, not just the
# validation subset. Note: this does NOT break Decision 8 (validation-only
# during iteration) because wnex vocabulary words are decontextualized
# words, not clue rows — the test-set lockout applies to clue rows.
#
# Output: data/embeddings/g1/f_common_wnex.npy, shape (8360, 1024),
# indexed by vocabulary_wnex.csv.
#
# Before submitting:
#   1. mkdir -p logs
#   2. Verify the following files are present on Great Lakes under
#      /home/vwinters/ccc-project/custom_embedding_model/:
#        models/g1/model/                                         (fine-tuned weights)
#        data/filtered_split/wn_synset/wnex/vocabulary_wnex.csv   (8,360 rows)
#        data/filtered_split/wn_synset/wnex/f_common_wnex.csv     (8,360 rows)
#
# Submit:
#   sbatch scripts/embed_wnex_full_g1.sh
#
# Expected runtime: ~5 min on a V100/A40 (1-hour wall-time is conservative).
# Can be submitted simultaneously with embed_wnex_full_gstock.sh — the jobs
# are independent.
#
# After completion, scp outputs back to the local machine:
#   scp vwinters@greatlakes-xfer.arc-ts.umich.edu:/home/vwinters/ccc-project/custom_embedding_model/data/embeddings/g1/f_common_wnex.npy \
#       /Users/victoria/Desktop/MADS/ccc-project/custom_embedding_model/data/embeddings/g1/
#   scp vwinters@greatlakes-xfer.arc-ts.umich.edu:/home/vwinters/ccc-project/custom_embedding_model/logs/embed_wnex_full_g1_<jobid>.out \
#       /Users/victoria/Desktop/MADS/ccc-project/custom_embedding_model/logs/
# ---------------------------------------------------------------------------

source activate nlp_env

export PYTHONUNBUFFERED=1

python scripts/embed_vocab.py \
    --model-path models/g1/model \
    --pooling meanpool \
    --vocab-file data/filtered_split/wn_synset/wnex/vocabulary_wnex.csv \
    --phrase-file data/filtered_split/wn_synset/wnex/f_common_wnex.csv \
    --output-file data/embeddings/g1/f_common_wnex.npy
