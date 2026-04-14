# Spec: Fix incomplete g1_tokenspan rename and create g1 README

**Stage:** 3 (cleanup)
**Target files:** `scripts/train_g1_tokenspan.sh`, `scripts/train_g1_tokenspan.py`, `models/g1/README.md`, `models/g1_tokenspan/README.md`
**Date:** 2026-04-14
**Status:** Draft

## Purpose

When the original g1 training scripts were renamed to g1_tokenspan (commit
f4ce874), the rename was file-level only — the file contents were not updated.
Every internal reference to `g1` (script paths, output directories, SLURM
job names, comments, docstrings) still points to the pre-rename names. This
spec fixes all stale references and also creates the missing `models/g1/README.md`.

These are historical scripts that will not be re-run, but they should be
accurate for provenance and to prevent accidental misuse.

## Changes to `scripts/train_g1_tokenspan.sh`

All line numbers refer to the current file state.

| Line | Current | Replace with |
|------|---------|-------------|
| 2 | `--job-name=train_g1` | `--job-name=train_g1_tokenspan` |
| 9 | `--output=logs/train_g1_%j.out` | `--output=logs/train_g1_tokenspan_%j.out` |
| 12 | `produce g_1.` | `produce g1_tokenspan.` |
| 13 | Remove "Step A of the design document — faithful reproduction of NB 09's training" and replace with: `Uses token span extraction (non-standard — see Decision 20). Historical artifact;` |
| 14 | `procedure on our pipeline's triplet file.` | `superseded by train_g1.sh / train_g1.py (canonical mean pooling).` |
| 22 | `sbatch scripts/train_g1.sh` | `sbatch scripts/train_g1_tokenspan.sh` |
| 28 | `models/g1/model/` | `models/g1_tokenspan/model/` |
| 29 | `custom_embedding_models/g1/` | `custom_embedding_models/g1_tokenspan/` |
| 34 | `models/g1/training_log.json` | `models/g1_tokenspan/training_log.json` |
| 35 | `custom_embedding_model/models/g1/` | `custom_embedding_model/models/g1_tokenspan/` |
| 36 | `models/g1/README.md` | `models/g1_tokenspan/README.md` |
| 40 | `conda activate nlp_env` | `source activate nlp_env` |
| 42 | `python scripts/train_g1.py \` | `python scripts/train_g1_tokenspan.py \` |
| 44 | `--output-dir models/g1 \` | `--output-dir models/g1_tokenspan \` |

## Changes to `scripts/train_g1_tokenspan.py`

| Line | Current | Replace with |
|------|---------|-------------|
| 2 | `produce g_1.` | `produce g1_tokenspan.` |
| 3-4 | Keep the NB 09 reference but add after line 8: `\nNote: This script uses token span extraction (non-standard). See Decision 20.\nSuperseded by train_g1.py which uses CALE's canonical mean pooling.` |
| 16 | `python scripts/train_g1.py \` | `python scripts/train_g1_tokenspan.py \` |
| 18 | `--output-dir models/g1 \` | `--output-dir models/g1_tokenspan \` |
| 25 | `python scripts/train_g1.py \` | `python scripts/train_g1_tokenspan.py \` |
| 89 | `description="Fine-tune CALE with triplet margin loss (g_1, Step A)"` | `description="Fine-tune CALE with triplet margin loss (g1_tokenspan, token span extraction)"` |
| 506 | `# --- Summary block for SLURM log → models/g1/README.md ---` | `# --- Summary block for SLURM log → models/g1_tokenspan/README.md ---` |
| 510 | `"SUMMARY (copy into models/g1/README.md and FINDINGS.md)"` | `"SUMMARY (copy into models/g1_tokenspan/README.md and FINDINGS.md)"` |

## Create `models/g1/README.md`

Create this file with the following content (values from the confirmed
mean pooling training run — verified via training_log.json and SLURM log
timestamps):

```markdown
# Model: g1

**Status:** trained 2026-04-14 — weights on Great Lakes

**Extraction method:** Mean pooling (canonical). This model uses CALE's
standard attention-masked mean pooling over all non-padding tokens, matching
`SentenceTransformer.encode()` behavior. See Decision 20.

## Base model

- HuggingFace ID: `gabrielloiseau/CALE-MBERT-en`

## Weights location

- Great Lakes: `/home/vwinters/ccc-project/custom_embedding_model/models/g1/model/`

Weights are **not** committed to this repo (Decision 12).

## Training details

- Triplet file: `data/triplets/g1.csv`
- Triplet design: T_1 (same triplets as g1_tokenspan — different extraction method)
- Training script: `scripts/train_g1.py`
- SLURM script: `scripts/train_g1.sh`

Hyperparameters:
  - margin: 1.0 (Decision 13)
  - learning_rate: 2e-5
  - epochs: 3
  - batch_size: 32
  - weight_decay: 0.01
  - grad_accum: 1
  - warmup_fraction: 0.1
  - max_length: 128
  - random_state: 42

Date trained: 2026-04-14
Runtime: 43.5 min (0.72 h) on Great Lakes gpu partition, Tesla V100-PCIE-16GB
Per-epoch loss: [0.470, 0.111, 0.014]

Environment versions (per Decision 19):
  - Python: 3.12.12
  - torch: 2.5.1+cu121
  - transformers: 4.57.6
  - numpy: 2.3.5
  - pandas: 3.0.0
```

## Update `models/g1_tokenspan/README.md`

Two changes:

1. Line 3: change `weights in Google Drive` to `weights on Great Lakes`
2. Line 17-18: change the Google Drive reference to:
   ```
   - Great Lakes: `/home/vwinters/ccc-project/custom_embedding_model/models/g1_tokenspan/model/`
   ```
3. Remove line 13 (`Version/commit hash` TODO) — we do not have the hash
   and it is not critical for this historical model.
4. Remove the paragraph about uploading to Google Drive (lines 20-23).

## No outputs file needed

This is a documentation/cleanup task, not an analysis notebook.
