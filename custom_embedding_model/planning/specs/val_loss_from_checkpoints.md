# Spec: Validation Loss Computation from Epoch Checkpoints

**Stage:** 3 (post hoc — fills a monitoring gap in the original training)
**Script:** `scripts/val_loss_from_checkpoints.py`
**Date:** 2026-04-22
**Status:** Approved

## Purpose

Compute validation triplet loss and triplet accuracy at each training epoch
using the saved epoch checkpoints. This fills a gap in the original training
script, which tracked training loss per epoch but did not compute validation
loss. The results will tell us whether g1 was overfitting during training
(training loss decreasing while validation loss plateaus or increases) and
whether earlier epochs had better generalization properties than the final
epoch.

The per-epoch training losses are read from `models/g1/training_log.json`
(produced by the original training script) for side-by-side comparison.

## Inputs

All paths relative to `custom_embedding_model/` on Great Lakes
(`/home/vwinters/ccc-project/custom_embedding_model/`).

- `models/g1/model_epoch1.pt` — checkpoint after epoch 1 (state_dict +
  optimizer state)
- `models/g1/model_epoch2.pt` — checkpoint after epoch 2
- `models/g1/model_epoch3.pt` — checkpoint after epoch 3
- `data/triplets/g1_val.csv` — validation triplets (46,506 rows). Columns:
  `clue_id`, `definition`, `answer_wn`, `distractor_wn`, `anchor`,
  `positive`, `negative`.
- `models/g1/training_log.json` — training log produced by `train_g1.py`.
  Contains `per_epoch_loss` (list of per-epoch mean training losses) and
  `hyperparameters` (including `margin`, `max_length`).

## Outputs

- `models/g1/val_loss_results.json` — structured results containing:
  - Per-epoch validation loss (mean triplet margin loss, same loss function
    and margin as training: `nn.TripletMarginLoss(margin=1.0, p=2)`)
  - Per-epoch validation triplet accuracy (fraction of triplets where
    `cos(anchor, positive) > cos(anchor, negative)`)
  - Per-epoch mean margin (`cos(anchor, positive) - cos(anchor, negative)`)
  - The training losses read from `training_log.json` for comparison
  - Timestamp and environment versions

- Printed summary table to stdout (captured in SLURM log)

## Implementation Details

### §1 — Setup

Standard imports: torch, numpy, pandas, json, pathlib, time, sys,
transformers.

CLI arguments:
- `--checkpoint-dir` (Path, required): directory containing the epoch
  checkpoint files and `training_log.json` (e.g., `models/g1`)
- `--val-triplets` (Path, required): path to `g1_val.csv`
- `--batch-size` (int, default=32): batch size for forward pass

Load `training_log.json` from `--checkpoint-dir` at startup. Read the
`margin` and `max_length` from the `hyperparameters` dict — these must
match training exactly, so reading them from the log is safer than
accepting them as CLI arguments. Read `per_epoch_loss` for the summary
table. Assert that the number of epoch checkpoint files matches the
length of `per_epoch_loss`.

Print environment versions at startup (same pattern as `train_g1.py`).

### §2 — Load validation triplets

Load `g1_val.csv` with `keep_default_na=False, na_values=[""]`. Assert
46,506 rows. Assert no nulls in `anchor`, `positive`, `negative` columns.

Use the same `TripletDataset` class and `DataLoader` pattern as
`train_g1.py` (no shuffle, `drop_last=False` — we want to evaluate every
triplet).

### §3 — Define extraction function

Use the same `extract_meanpool_embedding` function from `train_g1.py`
(attention-masked mean pooling over `last_hidden_state`). Copy it directly
to keep the script self-contained — do not import from `train_g1.py`.

### §4 — Evaluate each checkpoint

For each epoch checkpoint (1, 2, 3):

1. Load the base model architecture:
   `AutoModel.from_pretrained("gabrielloiseau/CALE-MBERT-en")`
2. Load the checkpoint state dict:
   `checkpoint = torch.load(ckpt_path, map_location=device)`
   `model.load_state_dict(checkpoint["model_state_dict"])`
3. Set model to eval mode: `model.eval()`
4. Iterate over the validation DataLoader with `torch.no_grad()`:
   - Compute anchor, positive, negative embeddings using
     `extract_meanpool_embedding`
   - Compute triplet loss using `nn.TripletMarginLoss(margin=1.0, p=2)`
   - Compute cosine similarities: `cos(anchor, positive)` and
     `cos(anchor, negative)` using `F.cosine_similarity`
   - Track: total loss, number of correct triplets (cos_pos > cos_neg),
     all margins (cos_pos - cos_neg)
5. Report: mean loss, triplet accuracy, mean margin, median margin
6. Print results immediately after each checkpoint (do not wait until all
   three are done — if the job is killed, we want partial results in the
   SLURM log)

### §5 — Summary and output

Print a summary table:

```
Epoch | Train Loss | Val Loss | Val Accuracy | Val Mean Margin
------+------------+----------+--------------+----------------
    1 |      0.470 |   ?.???  |       ?.?%   |         ?.????
    2 |      0.111 |   ?.???  |       ?.?%   |         ?.????
    3 |      0.014 |   ?.???  |       ?.?%   |         ?.????
```

Save `val_loss_results.json` atomically (write to `.tmp`, rename).

## Important implementation notes

- **Use `drop_last=False`** in the DataLoader. Unlike training, we want to
  evaluate every single validation triplet, including the final partial
  batch.
- **No gradient computation.** Wrap the forward pass in `torch.no_grad()`.
  Do not call `model.train()` — use `model.eval()` to disable dropout.
- **Same loss function as training.** Use `nn.TripletMarginLoss(margin=M,
  p=2)` where M is read from `training_log.json` — the same class and
  parameters as `train_g1.py` line 284. This ensures the validation loss
  is directly comparable to the training loss.
- **Same extraction function.** Copy `extract_meanpool_embedding` from
  `train_g1.py` verbatim. The extraction method must match training.
- **Same max_length.** Read from `training_log.json` to ensure tokenization
  matches training.
- **Memory:** Each checkpoint evaluation loads the full model. After
  finishing each checkpoint, delete the model and call
  `torch.cuda.empty_cache()` before loading the next one.
- **Load CSV with `keep_default_na=False`.** The word "nan" (grandmother)
  is a valid crossword entry.

## Environment

Great Lakes (GPU). Short job — estimated ~10–15 minutes total for three
checkpoints (no gradient computation, single forward pass per checkpoint).

## SLURM script

The spec does not produce a SLURM wrapper, but the Coder should create one
at `scripts/val_loss_from_checkpoints.sh` with:
- Same SLURM headers as `train_g1.sh` but with `--time=01:00:00` (1 hour,
  conservative)
- `source activate nlp_env`
- `export PYTHONUNBUFFERED=1`
- Command: `python scripts/val_loss_from_checkpoints.py --checkpoint-dir models/g1 --val-triplets data/triplets/g1_val.csv`
