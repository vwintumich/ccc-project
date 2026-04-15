# Spec: Model Evaluation (Pre-Hypothesis Testing)

**Stage:** 5 (precedes hypothesis testing)
**Notebook:** `notebooks/05_model_evaluation.ipynb`
**Date:** 2026-04-14
**Status:** Draft

## Purpose

Evaluate the basic health and generalization properties of g1 (mean pooling)
before interpreting its ATE results. This notebook answers three questions
that must be resolved before hypothesis testing can be meaningfully
interpreted:

1. **Does g1 generalize the training objective?** Validation triplet accuracy
   measures whether the model satisfies the triplet constraint on data it
   was never trained on — the most direct test of overfitting.
2. **Is the embedding space healthy, or has it collapsed?** NB 09 showed
   evidence that the original g1 compressed f_common_wndef phrases together.
   We check whether our g1 exhibits the same pathology.
3. **What do the raw similarity components look like?** Before computing the
   ATE (a difference), we examine the T=0 and T=1 similarity distributions
   individually to understand what's driving any change.

All analyses use pre-computed validation-split embeddings from Stage 4.
No new GPU work is required. Scope is limited to the canonical mean-pooling
models (g_stock and g1); tokenspan models are out of scope.

## Inputs

All paths relative to `custom_embedding_model/`.

**Embeddings (all .npy, float32, 1024-dim):**
- `data/embeddings/g_stock/f_clue_val.npy` — (47933, 1024)
- `data/embeddings/g_stock/f_clue_val_index.csv`
- `data/embeddings/g_stock/f_common_wndef_val.npy` — (26152, 1024)
- `data/embeddings/g_stock/f_common_wnex_val.npy` — (3008, 1024)
- `data/embeddings/g1/f_clue_val.npy` — (47933, 1024)
- `data/embeddings/g1/f_clue_val_index.csv`
- `data/embeddings/g1/f_common_wndef_val.npy` — (26152, 1024)
- `data/embeddings/g1/f_common_wnex_val.npy` — (3008, 1024)

**Vocabulary and clue files:**
- `data/filtered_split/wn_synset/clues_val.csv`
- `data/filtered_split/wn_synset/wndef/vocabulary_wndef_val.csv`
- `data/filtered_split/wn_synset/wnex/vocabulary_wnex_val.csv`

**Validation triplets (produced by updated NB 03 — see prerequisite below):**
- `data/triplets/g1_val.csv`

## Outputs

- `outputs/05_model_evaluation-results.md` — all numerical results
- `outputs/figures/05_val_triplet_accuracy.png`
- `outputs/figures/05_collapse_pairwise_cosine.png`
- `outputs/figures/05_collapse_singular_values.png`
- `outputs/figures/05_t0_t1_distributions.png`

No new data artifacts (.npy, .csv) are produced.

## Implementation Details

### §0 — Imports and configuration

Standard environment auto-detection (Local / Great Lakes / Colab). Define
`COMPONENT_ROOT`, `EMBEDDINGS_DIR`, `WN_DIR`, `OUTPUT_DIR`. Version
reporting cell (Decision 18): pandas, numpy, scipy, matplotlib, seaborn.

### §1 — Load embeddings and index files

Load all 8 .npy files (4 per model: f_clue_val, f_common_wndef_val,
f_common_wnex_val for g_stock and g1). Load f_clue_val_index.csv for both
models, vocabulary_wndef_val.csv, vocabulary_wnex_val.csv, and clues_val.csv.
All CSVs loaded with `keep_default_na=False, na_values=[""]`.

Assert all shapes match FINDINGS.md expectations:
- f_clue_val: (47933, 1024)
- f_common_wndef_val: (26152, 1024)
- f_common_wnex_val: (3008, 1024)

Build lookup dicts:
- `wndef_word_to_row`: word -> row index in f_common_wndef_val.npy
  (from vocabulary_wndef_val.csv)
- `wnex_word_to_row`: word -> row index in f_common_wnex_val.npy
  (from vocabulary_wnex_val.csv)
- `clue_key_to_row`: (clue_id, definition) -> row index in f_clue_val.npy
  (from f_clue_val_index.csv — use g_stock's index; NB 04 confirmed both
  models' indexes are identical)

### §2 — Validation triplet accuracy

**Purpose:** Test whether g1 generalizes the training objective to held-out
data. This is the most direct overfitting diagnostic.

**Prerequisite:** NB 03 must be updated to also produce `data/triplets/g1_val.csv`
using the same construction procedure on validation-split rows. See the
companion spec `03_train_g1_val_triplets.md`. The validation triplet file
has the same schema as `g1_train.csv` (columns: clue_id, definition, answer_wn,
distractor_wn, anchor, positive, negative).

**Step 2a — Load validation triplets and resolve embeddings.**

1. Load `data/triplets/g1_val.csv` with `keep_default_na=False, na_values=[""]`.
   Print row count.
2. For each row, resolve embedding row indices:
   - anchor: look up (clue_id, definition) in clue_key_to_row
   - positive: look up answer_wn in wndef_word_to_row
   - negative: look up distractor_wn in wndef_word_to_row
3. Drop rows where any embedding lookup fails (should be zero if
   the triplet file and embeddings were built from the same upstream
   artifacts, but assert defensively).
4. Report: N triplets loaded, N with successful embedding lookups.

**Step 2b — Compute triplet accuracy.**

For each surviving validation triplet, using pre-computed embeddings:
```
cos_pos = rowwise_cosine(anchor_embs, positive_embs)
cos_neg = rowwise_cosine(anchor_embs, negative_embs)
correct = cos_pos > cos_neg
accuracy = correct.mean()
margin = (cos_pos - cos_neg).mean()
```

Compute for both g_stock and g1. Report:

| Metric | g_stock | g1 |
|--------|---------|-----|
| Triplet accuracy (% correct) | — | — |
| Mean margin (cos_pos - cos_neg) | — | — |
| Median margin | — | — |
| % triplets with margin > 0.1 | — | — |
| % triplets with margin > 0.5 | — | — |
| N validation triplets evaluated | — | — |

The g_stock accuracy is the baseline: how well does the untrained model
already separate positives from negatives? If g1's accuracy is only
marginally higher (or lower), that's evidence of poor generalization.
If g1's accuracy is much higher, the model learned a generalizable signal.

**Step 2c — Margin distribution visualization.**

Plot overlaid histograms of (cos_pos - cos_neg) for g_stock and g1.
Save as `outputs/figures/05_val_triplet_accuracy.png` (300 dpi).

### §3 — Collapse detection

**Purpose:** Determine whether g1's embedding space has collapsed — i.e.,
whether all embeddings have moved closer together globally, reducing the
space's expressiveness.

**Step 3a — Mean pairwise cosine similarity among random word pairs.**

For each (model, phrase_type) combination in
{g_stock, g1} x {f_common_wndef_val, f_common_wnex_val}:

1. Sample 50,000 random pairs of distinct row indices
   (random_state=42). For wnex (3,008 words), the maximum number of
   distinct pairs is C(3008,2) = ~4.5M, so 50,000 is fine. Use the
   same sampled pairs for both models within a phrase type.
2. Compute cosine similarity for each pair.
3. Report mean, median, std, 5th percentile, 95th percentile.

If g1's mean pairwise cosine is substantially higher than g_stock's, that
is evidence of collapse. The magnitude of the difference matters: a small
increase might reflect tighter clustering of semantically related words
(good), while a large increase (especially combined with reduced variance)
indicates indiscriminate compression (bad).

Report as a table:

| Model | Phrase type | Mean | Median | Std | P5 | P95 |
|-------|-------------|------|--------|-----|----|-----|

**Step 3b — Embedding variance and effective dimensionality.**

For each (model, phrase_type) in
{g_stock, g1} x {f_common_wndef_val, f_common_wnex_val}:

1. Center the embedding matrix (subtract the mean vector).
2. Compute singular values via `np.linalg.svd(centered, full_matrices=False)`.
   The singular values are `s`; squared singular values are proportional to
   variance explained per component.
3. Compute:
   - **Total variance:** `sum(s**2)` (a single number summarizing spread)
   - **Effective dimensionality (participation ratio):**
     `(sum(s**2))**2 / sum(s**4)`. This ranges from 1 (all variance in one
     dimension = maximally collapsed) to D (variance uniformly spread across
     all D dimensions = maximally isotropic). Higher is healthier.
   - **Fraction of variance in top 10, 50, 100 components:** helps visualize
     whether the space is "spiky" (a few dominant dimensions) or spread out.

Report as a table:

| Model | Phrase type | Total var | Eff. dim | Top-10 var % | Top-50 var % | Top-100 var % |
|-------|-------------|-----------|----------|-------------|-------------|--------------|

**Step 3c — Visualization.**

Two figures:

1. `05_collapse_pairwise_cosine.png`: Overlaid histograms of random pairwise
   cosine similarities, one panel per phrase type, g_stock vs g1 as separate
   colors. (2 panels: wndef, wnex.)

2. `05_collapse_singular_values.png`: Cumulative explained variance curves
   (x = number of components, y = fraction of total variance). One panel per
   phrase type, g_stock vs g1 as separate lines. (2 panels: wndef, wnex.)

### §4 — T=0 and T=1 similarity distributions

**Purpose:** Examine the two components of the ATE separately before taking
their difference. This reveals whether changes in ATE are driven by T=0
shifting, T=1 shifting, or both.

**Step 4a — Assemble (clue, definition, answer) evaluation pairs.**

Starting from clues_val.csv:
1. Look up each row's f_clue_val embedding via clue_key_to_row.
2. Look up definition_wn in wndef_word_to_row -> definition's
   f_common_wndef_val embedding.
3. Look up answer_wn in wndef_word_to_row -> answer's
   f_common_wndef_val embedding.
4. Keep only rows where all three lookups succeed.
5. Report coverage: N rows retained, N dropped, and the reason
   (definition not in wndef_val vocab, answer not in wndef_val vocab,
   or clue not in index).

**Step 4b — Compute T=0 and T=1.**

For each surviving evaluation pair, under each model (g_stock, g1):
```
T0 = cos_sim(g(f_common_wndef(definition)), g(f_common_wndef(answer)))
T1 = cos_sim(g(f_clue(definition)),         g(f_common_wndef(answer)))
```

This produces 4 arrays: g_stock_T0, g_stock_T1, g1_T0, g1_T1.

Report:

| Metric | g_stock T=0 | g_stock T=1 | g1 T=0 | g1 T=1 |
|--------|-------------|-------------|--------|--------|
| Mean | | | | |
| Median | | | | |
| Std | | | | |
| P5 | | | | |
| P95 | | | | |

**Step 4c — Visualization.**

`05_t0_t1_distributions.png`: Four overlaid density plots (or a 2x2 grid
of histograms). Layout:
- Top row: T=0 distributions (g_stock blue, g1 orange)
- Bottom row: T=1 distributions (g_stock blue, g1 orange)

Alternatively, a single 1x2 layout:
- Left panel: g_stock (T=0 and T=1 overlaid)
- Right panel: g1 (T=0 and T=1 overlaid)

Use whichever layout the Coder judges clearest for comparison. Include
vertical lines at means. The key visual question: did T=0 shift up (as in
NB 09) while T=1 stayed put?

### §5 — Representational Similarity Analysis (RSA)

**Purpose:** A single summary statistic for how much the overall similarity
structure changed between g_stock and g1. Complements the collapse detection
(which measures marginal properties) with a relational measure.

For each phrase type in {f_common_wndef_val, f_common_wnex_val}:

1. Sample 1,000 words from the vocabulary (random_state=42). For wnex
   (3,008 words), sample 1,000. For wndef (26,152 words), sample 1,000.
2. Extract the corresponding rows from both g_stock and g1 embedding arrays.
3. Compute the 1000x1000 pairwise cosine similarity matrix under g_stock
   and under g1.
4. Extract the upper triangle (excluding diagonal) of each matrix, flatten
   to a vector. This gives ~499,500 values per model.
5. Compute Spearman correlation between the two vectors.
   Use `scipy.stats.spearmanr`.

Report:

| Phrase type | N words sampled | Spearman rho | p-value |
|-------------|-----------------|-------------|---------|
| f_common_wndef_val | 1,000 | — | — |
| f_common_wnex_val | 1,000 | — | — |

Interpretation guidance in the markdown cell:
- rho near 1.0: g1 preserved the relative similarity structure (words that
  were similar under g_stock are still similar under g1, and vice versa)
- rho near 0: the similarity structure was fundamentally reorganized
- Low rho + high collapse = the space compressed indiscriminately
- Low rho + healthy dimensionality = the space reorganized structurally

### §6 — Summary

Markdown summary cell recapping all key numbers. Highlight:
- Whether g1 generalizes the training objective (triplet accuracy)
- Whether the embedding space collapsed (pairwise cosine, effective dim)
- Whether T=0 and T=1 shifted as expected or pathologically
- Whether the similarity structure was preserved or reorganized (RSA)
- Explicit statement of what this implies for interpreting the ATE in the
  subsequent hypothesis testing notebook

## Environment

Local (CPU). No GPU needed — all embeddings are pre-computed.

## Notebook structure

- §-numbered markdown sections before each logical block
- Environment auto-detection for Local / Great Lakes / Colab
- Version reporting cell after imports (Decision 18)
- `rowwise_cosine()` helper defined once in §1 and reused throughout
  (same implementation as DATA.md)
- Write results file to `outputs/05_model_evaluation-results.md`
- All figures saved to `outputs/figures/` at 300 dpi
