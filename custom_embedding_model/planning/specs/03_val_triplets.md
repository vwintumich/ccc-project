# Spec: Add Validation Triplets to NB 03

**Stage:** 3
**Notebook:** `notebooks/03_train_g1.ipynb` (update existing)
**Date:** 2026-04-14
**Status:** Draft

## Purpose

Add a section to NB 03 that constructs validation-split triplets using the
same procedure as the training triplets. The output — `data/triplets/g1_val.csv`
— is consumed by NB 05 (model evaluation) for validation triplet accuracy,
the most direct test of whether g1 generalizes the training objective.

This is a small addition to an existing notebook, not a new notebook.

## Why this belongs in NB 03

Validation triplets must be constructed identically to training triplets:
same distractor source (`dataset_harder.parquet`), same phrase lookups
(`f_clue.csv`, `f_common_wndef.csv`), same join logic. Producing them
alongside the training triplets ensures they share the same code path and
makes the provenance relationship explicit. A future reader of NB 03 will
see both artifacts produced in one place.

## Current NB 03 structure

The existing notebook:
- §0: Imports and configuration
- §1: Load source data (clues_wn_filtered, f_clue, f_common_wndef, dataset_harder)
- §2: Join training rows with distractors
- §3: Look up phrases for all three triplet roles
- §4: Build and save the triplet file (g1_train.csv)
- §5: Save provenance metadata (g1_train_meta.json)
- §6: Inspection and examples
- §7: Comparison to NB 09
- §8: Summary
- §9: Write results file

## Changes required

### New §4b — Build and save validation triplet file

Insert after §4 (which saves g1_train.csv) and before §5 (metadata). The logic
mirrors §2–§4 but filters to `split == 'validate'` instead of
`split == 'train'`.

1. Filter `clues_wn` to `split == 'validate'` (should yield 47,933 rows —
   assert this).
2. Inner join with the distractor mapping (same `dist` DataFrame already
   loaded in §1) on `(clue_id, definition_wn)`.
3. Look up anchor, positive, negative phrases using the same lookups
   (`f_clue_lookup`, `f_wndef_lookup`) already built in §1.
4. Drop rows with any missing phrase.
5. Report coverage at each step (same format as §2–§3 for training rows).
6. Apply the same schema invariants as §4 (no nulls, balanced `<t></t>` in
   anchors, positives start with `<t>`, no positive==negative rows, no split
   column).
7. Save as `data/triplets/g1_val.csv` with columns:
   `clue_id, definition, answer_wn, distractor_wn, anchor, positive, negative`

### Update §5 — Add validation counts to g1_train_meta.json

Add the following fields to the existing metadata dict:
- `"n_val_rows"`: row count of g1_val.csv
- `"n_val_unique_clue_ids"`: unique clue_ids in g1_val.csv
- `"n_val_unique_pairs"`: unique (definition, answer_wn) pairs
- `"val_rows_lost_to_distractor_join"`: rows lost at the join step
- `"val_rows_lost_to_missing_phrases"`: rows lost at the phrase lookup step

### Update §8 — Summary

Add a line reporting the validation triplet file size and coverage alongside
the existing training coverage summary.

### Update §9 — Results file

Add the validation triplet statistics to the results markdown.

## Outputs

- `data/triplets/g1_val.csv` — new file, same schema as g1_train.csv
- `data/triplets/g1_train_meta.json` — updated with validation counts
- `outputs/03_train_g1-results.md` — updated with validation statistics

## Implementation notes

- The notebook already loads all four source files in §1. No additional
  file loading is needed.
- The `dist` DataFrame (distractors from dataset_harder.parquet) is already
  in memory and is not split-specific — it covers all clue_ids regardless
  of our train/validate/test split. The inner join will naturally retain
  whatever validation rows have matching distractors.
- Expected validation coverage loss should be similar to training (~2–3%
  at the distractor join, ~0.7% at phrase lookup). Report actual numbers.
- Use `keep_default_na=False` when saving/loading g1_val.csv (consistent
  with all other CSV handling in the project).

## Environment

Local (CPU). Fast — same text-manipulation-only workflow as the existing
NB 03 (a few seconds).
