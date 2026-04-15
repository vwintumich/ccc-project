# Spec: Rename g1.csv to g1_train.csv

**Stage:** 3
**Date:** 2026-04-14
**Status:** Draft

## Purpose

Rename the training triplet file from `g1.csv` to `g1_train.csv` so that
training and validation triplet files have parallel, explicit names:
- `data/triplets/g1_train.csv`
- `data/triplets/g1_val.csv` (to be produced by the NB 03 update)

The current name `g1.csv` relies on an implicit convention (no suffix =
full scope) that is misleading here — the file contains only training-split
rows, not the full dataset. Making the split explicit in the filename
eliminates ambiguity.

## Changes required

### 1. Rename the file

```
git mv data/triplets/g1.csv data/triplets/g1_train.csv
```

### 2. Rename the metadata file

```
git mv data/triplets/g1_meta.json data/triplets/g1_train_meta.json
```

### 3. Update all references

Every file that references `g1.csv` or `g1_meta.json` must be updated.
The following files contain references (from `rg g1\.csv` and
`rg g1_meta\.json`):

**Project documentation:**
- `CLAUDE.md` — repo structure diagram, file naming section
- `WORKFLOW.md` — Stage 3 description, summary table
- `DATA.md` — triplet files section
- `NOTEBOOKS.md` — Stage 3 notebook table
- `DECISIONS.md` — Decision 10 (triplet naming convention)
- `FINDINGS.md` — Stage 3 sections
- `planning/custom_embedding_model_design_v5.md` — §8.1, §8.3 diagram

**Scripts:**
- `scripts/train_g1.py` — reads the triplet CSV (command-line arg or default path)
- `scripts/train_g1.sh` — passes triplet path to the training script
- `scripts/train_g1_tokenspan.py` — same
- `scripts/train_g1_tokenspan.sh` — same

**Model metadata:**
- `models/g1/README.md` — "Triplet file" field
- `models/g1_tokenspan/README.md` — "Triplet file" field
- `models/g1/training_log.json` — if it references the triplet path
- `models/g1_tokenspan/training_log.json` — if it references the triplet path

**Notebooks:**
- `notebooks/03_train_g1.ipynb` — output path, metadata dict, summary cell

**Outputs:**
- `outputs/03_train_g1-results.md` — references to the triplet file

**Prior specs (update for consistency, not re-execution):**
- `planning/specs/3_train_g1.md`
- `planning/specs/train_g1.md`
- `planning/specs/fix_tokenspan_rename_and_g1_readme.md`

**SLURM logs (do not modify):**
- `logs/train_g1_tokenspan_47741274.out` — historical log, leave as-is

### 4. Update Decision 10

Decision 10 currently says: "The triplet CSV used to train g_i is named
`triplets/<g_name>.csv`." Update to reflect the new convention:
`triplets/<g_name>_train.csv` for training triplets,
`triplets/<g_name>_val.csv` for validation triplets.

## Implementation notes

- This is a pure rename + find-and-replace. No logic changes.
- Replace `g1.csv` with `g1_train.csv` and `g1_meta.json` with
  `g1_train_meta.json` in all files listed above.
- Do NOT modify SLURM log files — they are historical records.
- After the rename, run `git status` to verify no references were missed.
- The companion spec `03_val_triplets.md` already uses `g1_val.csv` and
  should reference `g1_train.csv` for the training file. Update
  `03_val_triplets.md` if any references to `g1.csv` remain.
- The companion spec `05_model_evaluation.md` already uses `g1_val.csv`
  and does not reference `g1.csv`. No update needed.

## Execution order

This rename should be done **before** the NB 03 update (03_val_triplets.md),
so that NB 03 is internally consistent when the validation triplet section
is added.
