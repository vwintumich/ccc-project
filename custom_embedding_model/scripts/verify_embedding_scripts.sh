#!/bin/bash
#SBATCH --job-name=verify_embed_scripts
#SBATCH --account=siads696w26_class
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/verify_embedding_scripts_%j.out

# ---------------------------------------------------------------------------
# Run all seven verification passes for the embedding_utils.py / embed_clue.py
# / embed_vocab.py refactor. Each pass re-embeds an existing artifact with
# the new scripts and compares rowwise to the committed .npy file via
# cosine similarity (assertion threshold mean > 0.999).
#
# V1 (g_stock f_clue all) is special: the existing g_stock/f_clue.npy was
# produced by SentenceTransformer.encode() while embed_clue.py uses
# AutoModel + manual mean pooling. These are mathematically equivalent but
# numerically different at float32 precision, so V1 is expected to show
# mean cosine ~0.999+ rather than ~1.0. Decision 20 / FINDINGS.md Stage 4
# record that the two methods agree to within mean cosine > 0.999. V2–V7
# compare AutoModel to AutoModel and should show ~1.0.
#
# Outputs land in per-model _verify directories so they do not collide with
# the committed artifacts. After the job succeeds, delete those directories:
#   rm -rf data/embeddings/g_stock_verify data/embeddings/g1_verify
#
# Before submitting:
#   1. mkdir -p logs
#   2. Verify existing artifacts are present on Great Lakes under
#      /home/vwinters/ccc-project/custom_embedding_model/data/embeddings/:
#        g_stock/f_clue.npy + f_clue_index.csv
#        g_stock/f_clue_val.npy + f_clue_val_index.csv
#        g_stock/f_common_wndef_val.npy
#        g_stock/f_common_wnex_val.npy
#        g1/f_clue_val.npy + f_clue_val_index.csv
#        g1/f_common_wndef_val.npy
#        g1/f_common_wnex_val.npy
#      And that models/g1/model/ is present for the g1 runs.
#
# Submit:
#   sbatch scripts/verify_embedding_scripts.sh
#
# Expected runtime: V1 dominates (~6 min, 239K rows); V2/V3 ~1 min each
# (47K val rows); V4–V7 ~30 sec each (≤26K vocab rows). Total well under
# 30 min on a V100/A40; 2-hour wall is conservative.
#
# After completion, scp the log back for the record:
#   scp vwinters@greatlakes-xfer.arc-ts.umich.edu:/home/vwinters/ccc-project/custom_embedding_model/logs/verify_embedding_scripts_<jobid>.out \
#       /Users/victoria/Desktop/MADS/ccc-project/custom_embedding_model/logs/
# ---------------------------------------------------------------------------

source activate nlp_env

export PYTHONUNBUFFERED=1

set -e  # Abort on first failure so a FAIL in V_k doesn't silently roll into V_{k+1}.

echo "############################################################"
echo "# V1: g_stock f_clue all (meanpool)"
echo "#    Compared against SentenceTransformer-produced reference"
echo "#    Expected mean cosine ~0.999+ (not 1.0) due to different extraction"
echo "############################################################"
python scripts/embed_clue.py \
    --model-path gabrielloiseau/CALE-MBERT-en \
    --pooling meanpool --split all \
    --output-dir data/embeddings/g_stock_verify \
    --verify-against data/embeddings/g_stock/f_clue.npy

echo
echo "############################################################"
echo "# V2: g_stock f_clue validate (meanpool)"
echo "############################################################"
python scripts/embed_clue.py \
    --model-path gabrielloiseau/CALE-MBERT-en \
    --pooling meanpool --split validate \
    --output-dir data/embeddings/g_stock_verify \
    --verify-against data/embeddings/g_stock/f_clue_val.npy

echo
echo "############################################################"
echo "# V3: g1 f_clue validate (meanpool)"
echo "############################################################"
python scripts/embed_clue.py \
    --model-path models/g1/model \
    --pooling meanpool --split validate \
    --output-dir data/embeddings/g1_verify \
    --verify-against data/embeddings/g1/f_clue_val.npy

echo
echo "############################################################"
echo "# V4: g_stock wndef val (meanpool)"
echo "############################################################"
python scripts/embed_vocab.py \
    --model-path gabrielloiseau/CALE-MBERT-en \
    --pooling meanpool \
    --vocab-file data/filtered_split/wn_synset/wndef/vocabulary_wndef_val.csv \
    --phrase-file data/filtered_split/wn_synset/wndef/f_common_wndef.csv \
    --output-file data/embeddings/g_stock_verify/f_common_wndef_val.npy \
    --verify-against data/embeddings/g_stock/f_common_wndef_val.npy

echo
echo "############################################################"
echo "# V5: g_stock wnex val (meanpool)"
echo "############################################################"
python scripts/embed_vocab.py \
    --model-path gabrielloiseau/CALE-MBERT-en \
    --pooling meanpool \
    --vocab-file data/filtered_split/wn_synset/wnex/vocabulary_wnex_val.csv \
    --phrase-file data/filtered_split/wn_synset/wnex/f_common_wnex.csv \
    --output-file data/embeddings/g_stock_verify/f_common_wnex_val.npy \
    --verify-against data/embeddings/g_stock/f_common_wnex_val.npy

echo
echo "############################################################"
echo "# V6: g1 wndef val (meanpool)"
echo "############################################################"
python scripts/embed_vocab.py \
    --model-path models/g1/model \
    --pooling meanpool \
    --vocab-file data/filtered_split/wn_synset/wndef/vocabulary_wndef_val.csv \
    --phrase-file data/filtered_split/wn_synset/wndef/f_common_wndef.csv \
    --output-file data/embeddings/g1_verify/f_common_wndef_val.npy \
    --verify-against data/embeddings/g1/f_common_wndef_val.npy

echo
echo "############################################################"
echo "# V7: g1 wnex val (meanpool)"
echo "############################################################"
python scripts/embed_vocab.py \
    --model-path models/g1/model \
    --pooling meanpool \
    --vocab-file data/filtered_split/wn_synset/wnex/vocabulary_wnex_val.csv \
    --phrase-file data/filtered_split/wn_synset/wnex/f_common_wnex.csv \
    --output-file data/embeddings/g1_verify/f_common_wnex_val.npy \
    --verify-against data/embeddings/g1/f_common_wnex_val.npy

echo
echo "############################################################"
echo "# ALL SEVEN VERIFICATION RUNS COMPLETED SUCCESSFULLY"
echo "############################################################"
echo "After confirming this log shows PASS for V1–V7, the _verify"
echo "directories can be deleted:"
echo "  rm -rf data/embeddings/g_stock_verify data/embeddings/g1_verify"
