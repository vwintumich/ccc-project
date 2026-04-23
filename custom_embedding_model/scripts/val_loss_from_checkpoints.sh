#!/bin/bash
#SBATCH --job-name=val_loss_g1
#SBATCH --account=siads696w26_class
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/val_loss_g1_%j.out

# ---------------------------------------------------------------------------
# Post hoc validation-loss computation for g_1's per-epoch checkpoints. Runs
# a single forward pass over data/triplets/g1_val.csv with each of the three
# saved checkpoints (model_epoch1.pt / model_epoch2.pt / model_epoch3.pt) and
# reports val loss, triplet accuracy, and mean/median margin per epoch.
#
# Fills a monitoring gap in the original train_g1.py run, which logged only
# training loss. Results will tell us whether g_1 was overfitting during
# training and whether earlier epochs generalized better than the final one.
#
# Before submitting:
#   1. mkdir -p logs   (SLURM output directory must exist)
#   2. Verify models/g1/ contains model_epoch1.pt, model_epoch2.pt,
#      model_epoch3.pt, and training_log.json (from the g_1 training run).
#   3. Verify data/triplets/g1_val.csv has been uploaded from local.
#
# Submit:
#   sbatch scripts/val_loss_from_checkpoints.sh
#
# Expected runtime: ~10-15 min total (no gradient computation, single forward
# pass per checkpoint). 1-hour wall-time is conservative padding.
#
# After completion:
#   scp the SLURM log and val_loss_results.json back to the local repo:
#     scp <user>@greatlakes-xfer.arc-ts.umich.edu:<path>/logs/val_loss_g1_<jobid>.out \
#         custom_embedding_model/logs/
#     scp <user>@greatlakes-xfer.arc-ts.umich.edu:<path>/models/g1/val_loss_results.json \
#         custom_embedding_model/models/g1/
# ---------------------------------------------------------------------------

source activate nlp_env
export PYTHONUNBUFFERED=1

python scripts/val_loss_from_checkpoints.py \
    --checkpoint-dir models/g1 \
    --val-triplets data/triplets/g1_val.csv
