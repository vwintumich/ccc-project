#!/bin/bash
#SBATCH --job-name=embed_wndef_full_g1
#SBATCH --account=siads696w26_class
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/embed_wndef_full_g1_%j.out

# ---------------------------------------------------------------------------
# Generate g1 full-vocabulary wndef embeddings (53,930 words) using the
# fine-tuned CALE variant trained with mean pooling (Decision 20 canonical).
#
# This extends the existing g1 validation-only wndef embeddings
# (f_common_wndef_val.npy, 26,152 rows) to the full wndef vocabulary. Two
# motivations:
#   1. Cross-f triplet accuracy in NB 05 requires evaluating the same
#      triplet set under both wndef and wnex; wnex ⊂ wndef means any wnex
#      word is guaranteed to be in vocabulary_wndef, but not necessarily
#      in vocabulary_wndef_val.
#   2. Roughly 41% of validation triplets are currently dropped because
#      distractor words are absent from vocabulary_wndef_val (Decision 21).
#      Full-vocab wndef embeddings should lift resolution to ~99%+.
#
# Note: this does NOT break Decision 8 (validation-only during iteration)
# because wndef vocabulary words are decontextualized words, not clue rows —
# the test-set lockout (Decision 9) applies to clue rows. The same rationale
# underlies Decision 22 (full-vocab wnex).
#
# Output: data/embeddings/g1/f_common_wndef.npy, shape (53930, 1024),
# indexed by vocabulary_wndef.csv. This supplements (does not replace) the
# existing val-only file.
#
# Before submitting:
#   1. mkdir -p logs
#   2. Verify the following files are present on Great Lakes under
#      /home/vwinters/ccc-project/custom_embedding_model/:
#        models/g1/model/                                           (fine-tuned weights)
#        data/filtered_split/wn_synset/wndef/vocabulary_wndef.csv   (53,930 rows)
#        data/filtered_split/wn_synset/wndef/f_common_wndef.csv     (53,930 rows)
#
# Submit:
#   sbatch scripts/embed_wndef_full_g1.sh
#
# Expected runtime: ~2.5 min at ~380 phrases/sec (1-hour wall-time is
# conservative). Can be submitted simultaneously with
# embed_wndef_full_gstock.sh — the jobs are independent.
#
# After completion, scp outputs back to the local machine:
#   scp vwinters@greatlakes-xfer.arc-ts.umich.edu:/home/vwinters/ccc-project/custom_embedding_model/data/embeddings/g1/f_common_wndef.npy \
#       /Users/victoria/Desktop/MADS/ccc-project/custom_embedding_model/data/embeddings/g1/
#   scp vwinters@greatlakes-xfer.arc-ts.umich.edu:/home/vwinters/ccc-project/custom_embedding_model/logs/embed_wndef_full_g1_<jobid>.out \
#       /Users/victoria/Desktop/MADS/ccc-project/custom_embedding_model/logs/
# ---------------------------------------------------------------------------

source activate nlp_env

export PYTHONUNBUFFERED=1

python scripts/embed_vocab.py \
    --model-path models/g1/model \
    --pooling meanpool \
    --vocab-file data/filtered_split/wn_synset/wndef/vocabulary_wndef.csv \
    --phrase-file data/filtered_split/wn_synset/wndef/f_common_wndef.csv \
    --output-file data/embeddings/g1/f_common_wndef.npy
