# Spec: Stage 4 Embedding Verification

**Stage:** 4 (verification, pre-Stage-5/6 gate)
**Notebook:** `notebooks/04_embedding_verification.ipynb`
**Date:** 2026-04-14
**Status:** Draft

## Purpose

Verify that all four sets of validation embeddings (g_stock, g_stock_tokenspan,
g1, g1_tokenspan) are correctly named and contain what they claim to contain,
before proceeding to Stage 5 model evaluation.

### Background

During Stage 3–4 development, the original g1 model (trained with token span
extraction) was renamed to g1_tokenspan, and a new g1 was trained with mean
pooling (Decision 20). Embeddings were then generated for all four
(model, pooling) combinations. This notebook performs a systematic check that
no files were accidentally swapped, duplicated, or mislabeled during the
rename and regeneration process.

This is a lightweight local notebook (CPU only) — it loads existing `.npy`
files and computes summary statistics. No new embeddings are generated.

## Inputs

All paths relative to `custom_embedding_model/`.

**Embedding files (12 `.npy` files, 4 index CSVs):**

| Directory | Files |
|---|---|
| `data/embeddings/g_stock/` | `f_clue_val.npy`, `f_clue_val_index.csv`, `f_common_wndef_val.npy`, `f_common_wnex_val.npy` |
| `data/embeddings/g_stock_tokenspan/` | `f_clue_val.npy`, `f_clue_val_index.csv`, `f_common_wndef_val.npy`, `f_common_wnex_val.npy` |
| `data/embeddings/g1/` | `f_clue_val.npy`, `f_clue_val_index.csv`, `f_common_wndef_val.npy`, `f_common_wnex_val.npy` |
| `data/embeddings/g1_tokenspan/` | `f_clue_val.npy`, `f_clue_val_index.csv`, `f_common_wndef_val.npy`, `f_common_wnex_val.npy` |

**Vocabulary/index files (for shape validation):**

- `data/filtered_split/wn_synset/wndef/vocabulary_wndef_val.csv`
- `data/filtered_split/wn_synset/wnex/vocabulary_wnex_val.csv`

**Reference values from FINDINGS.md:**

| Phrase type | Expected shape |
|---|---|
| `f_clue_val.npy` | (47933, 1024) |
| `f_common_wndef_val.npy` | (26152, 1024) |
| `f_common_wnex_val.npy` | (3008, 1024) |

## Outputs

- `outputs/04_embedding_verification-results.md` — summary of all checks
  with PASS/FAIL verdicts

No new data artifacts are produced. This notebook is read-only with respect
to the data directory.

## Implementation details

### §1 — Setup and file loading

Standard imports: `numpy`, `pandas`, `pathlib`, `itertools.combinations`.
Define paths to all four embedding directories and the two vocabulary files.

Load all 12 `.npy` files into a nested dict keyed by `(model_name, phrase_type)`:

```python
MODEL_NAMES = ["g_stock", "g_stock_tokenspan", "g1", "g1_tokenspan"]
PHRASE_TYPES = ["f_clue_val", "f_common_wndef_val", "f_common_wnex_val"]
```

Also load all four `f_clue_val_index.csv` files (with `keep_default_na=False`).
Load the two vocabulary files for row-count validation.

### §2 — Shape and integrity checks

For each of the 12 `.npy` files:

1. **Shape check:** Assert shape matches the expected values from FINDINGS.md
   (listed above). Print a table of all 12 shapes.

2. **No NaN:** Assert `np.isnan(emb).any() == False` for each file.

3. **No all-zero rows:** Assert no rows have L2 norm exactly 0.

4. **L2 norm statistics:** For each file, compute and print min, mean, max
   L2 norm. This provides a quick fingerprint — different models should
   produce different norm distributions.

5. **Index file consistency:** Assert all four `f_clue_val_index.csv` files
   have identical `clue_id` and `definition` columns (they should — same
   validation set, same phrase file). Use `DataFrame.equals()` after
   selecting just those two columns.

Print a summary table with columns: model, phrase_type, shape, has_nan,
n_zero_rows, L2_min, L2_mean, L2_max.

### §3 — Pairwise cosine similarity: f_clue_val

Compute mean rowwise cosine similarity between every pair of the four
`f_clue_val.npy` arrays. There are C(4,2) = 6 unique pairs.

**Rowwise cosine similarity** for arrays A, B of shape (N, D):
```python
# Normalize each row to unit length
A_norm = A / np.linalg.norm(A, axis=1, keepdims=True)
B_norm = B / np.linalg.norm(B, axis=1, keepdims=True)
# Cosine similarity per row
cos_sims = (A_norm * B_norm).sum(axis=1)
# Report mean, median, min, max
```

Present results as a 4×4 symmetric matrix (diagonal = 1.000) with mean
cosine similarity in each cell. Also print the 6 unique pairs as a list
showing mean, median, min, and std for each pair.

**Interpretation guide** (print in a markdown cell before the results):
- Same weights, different pooling (g_stock ↔ g_stock_tokenspan; g1 ↔
  g1_tokenspan): expect moderate divergence. The earlier consistency check
  in `embed_val.py` found ~0.926 for g_stock f_clue (full dataset). The
  validation subset may differ slightly.
- Same pooling, different weights (g_stock ↔ g1, both meanpool;
  g_stock_tokenspan ↔ g1_tokenspan, both tokenspan): expect relatively
  high similarity — fine-tuning nudges weights but doesn't overhaul them.
- Different weights AND different pooling: expect the lowest similarity of
  any pair.
- **Red flag:** Any off-diagonal cell ≥ 0.999 would suggest the two files
  contain effectively the same embeddings despite different names.

### §4 — Pairwise cosine similarity: f_common_wndef_val

Same 4×4 matrix as §3, but for the `f_common_wndef_val.npy` files
(26,152 rows each).

This is especially important because the triplet loss trained on
f_common_wndef phrases — so the effect of fine-tuning may be most visible
here.

### §5 — Pairwise cosine similarity: f_common_wnex_val

Same 4×4 matrix for the `f_common_wnex_val.npy` files (3,008 rows each).

This is the cross-f generalization question: the model was trained on wndef
phrases, not wnex phrases. If g1 changes wnex embeddings substantially (low
cosine vs g_stock), that hints at semantic generalization. If wnex embeddings
barely move (high cosine vs g_stock), that hints at format-specific learning.
This is a preview of the Step B analysis in Stages 5/6 — the notebook should
note the observation but defer the full interpretation to Stages 5 and 6.

### §6 — Overall verdict

Print a clear PASS or FAIL verdict based on these criteria:

1. **PASS** if all 12 files have the expected shapes — **FAIL** otherwise
2. **PASS** if no NaN or all-zero rows found — **FAIL** otherwise
3. **PASS** if all off-diagonal cells in all three similarity matrices are
   < 0.999 (no accidental duplicates) — **FAIL** otherwise
4. **PASS** if the similarity pattern is internally consistent: for each
   phrase type, pairs differing only in pooling should show similar
   divergence to each other, and pairs differing only in weights should
   show similar divergence to each other. Flag (as a WARNING, not FAIL) if
   the pattern is unexpected or hard to interpret.

Write the overall verdict and the full set of tables to the results file.

### Results file

Write `outputs/04_embedding_verification-results.md` containing:
- Date and environment info
- The shape/integrity summary table from §2
- The three 4×4 similarity matrices from §3–§5
- The 6-pair detail tables (mean, median, min, std) for each phrase type
- The overall verdict from §6
- Any observations or warnings

## Environment

Local (CPU). No GPU or Great Lakes needed. This notebook loads `.npy` files
and computes cosine similarities — numpy only.

## Notebook structure

- Use §-numbered markdown sections matching the sections above
- Include environment auto-detection for local/Great Lakes/Colab
- Standard notebook header with authorship and purpose
- Summary cell at end with verdict and any notable observations
- Write results to `outputs/04_embedding_verification-results.md`
