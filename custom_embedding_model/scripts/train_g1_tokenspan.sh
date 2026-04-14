#!/bin/bash
#SBATCH --job-name=train_g1_tokenspan
#SBATCH --account=siads696w26_class
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/train_g1_tokenspan_%j.out

# ---------------------------------------------------------------------------
# Fine-tune g_stock (CALE-MBERT-en) with triplet margin loss to produce g1_tokenspan.
# Uses token span extraction (non-standard — see Decision 20). Historical artifact;
# superseded by train_g1.sh / train_g1.py (canonical mean pooling).
#
# Before submitting:
#   1. mkdir -p logs   (SLURM output directory must exist)
#   2. Verify data/triplets/g1.csv has been uploaded from local
#      (produced by notebooks/03_train_g1.ipynb).
#
# Submit:
#   sbatch scripts/train_g1_tokenspan.sh
#
# Expected runtime: 2-4 hours on a V100/A40 for ~70K triplet rows at
# batch_size=32, 3 epochs. 8-hour wall-time is conservative padding.
#
# After completion:
#   1. Upload model weights from models/g1_tokenspan/model/ to Google Drive
#      ("Research Project - NLP CCC's" / custom_embedding_models/g1_tokenspan/).
#   2. scp the SLURM log and training_log.json back to the local repo
#      (weights do NOT need to come back locally — they live in Drive):
#      scp <user>@greatlakes-xfer.arc-ts.umich.edu:<path>/logs/train_g1_<jobid>.out \
#          custom_embedding_model/logs/
#      scp <user>@greatlakes-xfer.arc-ts.umich.edu:<path>/models/g1_tokenspan/training_log.json \
#          custom_embedding_model/models/g1_tokenspan/
#   3. Fill in models/g1_tokenspan/README.md from the SUMMARY block in the SLURM log.
#   4. Update FINDINGS.md Stage 3 section per Decision 19.
# ---------------------------------------------------------------------------

source activate nlp_env

python scripts/train_g1_tokenspan.py \
    --input data/triplets/g1.csv \
    --output-dir models/g1_tokenspan \
    --epochs 3 \
    --batch-size 32 \
    --lr 2e-5 \
    --margin 1.0
