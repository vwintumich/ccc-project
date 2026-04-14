#!/bin/bash
#SBATCH --job-name=train_g1
#SBATCH --account=siads696w26_class
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/train_g1_%j.out

# ---------------------------------------------------------------------------
# Fine-tune g_stock (CALE-MBERT-en) with triplet margin loss to produce g_1,
# using CALE's canonical mean pooling (Decision 20). This is the corrected
# counterpart to train_g1_tokenspan.sh: same triplets, same hyperparameters,
# different extraction method, so the Stage 5 comparison isolates the effect
# of pooling choice alone.
#
# Before submitting:
#   1. mkdir -p logs   (SLURM output directory must exist)
#   2. Verify data/triplets/g1.csv has been uploaded from local
#      (produced by notebooks/03_train_g1.ipynb — same file as g1_tokenspan).
#
# Submit:
#   sbatch scripts/train_g1.sh
#
# Expected runtime: ~49 min on a V100/A40 (same as g1_tokenspan — same data
# size and hyperparameters; mean pooling is slightly cheaper than token span
# extraction but the difference is negligible). 4-hour wall-time is
# conservative padding.
#
# After completion:
#   1. Upload model weights from models/g1/model/ to Google Drive
#      ("Research Project - NLP CCC's" / custom_embedding_models/g1/).
#   2. scp the SLURM log and training_log.json back to the local repo
#      (weights do NOT need to come back locally — they live in Drive):
#      scp <user>@greatlakes-xfer.arc-ts.umich.edu:<path>/logs/train_g1_<jobid>.out \
#          custom_embedding_model/logs/
#      scp <user>@greatlakes-xfer.arc-ts.umich.edu:<path>/models/g1/training_log.json \
#          custom_embedding_model/models/g1/
#   3. Fill in models/g1/README.md from the SUMMARY block in the SLURM log.
#   4. Update FINDINGS.md Stage 3 section per Decision 19.
# ---------------------------------------------------------------------------

source activate nlp_env
export PYTHONUNBUFFERED=1

python scripts/train_g1.py \
    --input data/triplets/g1.csv \
    --output-dir models/g1 \
    --epochs 3 \
    --batch-size 32 \
    --lr 2e-5 \
    --margin 1.0
