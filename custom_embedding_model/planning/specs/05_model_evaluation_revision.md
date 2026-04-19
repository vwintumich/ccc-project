# Spec: NB 05 revision — cross-f triplet accuracy and full-vocab analyses

**Stage:** 5
**Notebook:** `notebooks/05_model_evaluation.ipynb`
**Date:** 2026-04-19
**Status:** Approved

**Note to Coder:** This is a revision of the existing committed notebook
`notebooks/05_model_evaluation.ipynb`. Read the current notebook first,
then rewrite it following this spec. The current notebook's code patterns
(helper functions, loading idioms, figure style) should be preserved where
the spec doesn't specify a change. §4, §5, and §7 reuse val-only data
identical to the current notebook — their numerical outputs should match
the previous run exactly.

## Purpose

Revise NB 05 to add cross-f triplet accuracy as the central new analysis,
use full-vocabulary wndef and wnex embeddings where appropriate, and add
wnex-based T=0/T=1 distributions. This implements the Step B cross-f
generalization test from the design document: does g1's learned
discriminative structure transfer to a phrase type (wnex) it was never
trained on?

Full-vocabulary embeddings (53,930 wndef / 8,360 wnex) are used for
triplet accuracy (§2, §3) because they resolve distractor lookups that
failed with val-only vocabularies (~41% dropout, Decision 21).
Validation-only embeddings (26,152 wndef / 3,008 wnex) are used for
collapse detection (§4) and RSA (§7) because these are model diagnostics
that inform model selection and must not include test-split words
(Decision 9). T=0/T=1 sections (§5, §6) evaluate validation clues only,
so the choice doesn't matter — val-only is used for consistency.

This is an exploratory pass — be thorough, compute more than strictly
needed. A future revision will tighten the narrative.

## Inputs

### Embedding files — full-vocabulary (for §2, §3 triplet accuracy)

| File | Shape | Index | Notes |
|------|-------|-------|-------|
| `g_stock/f_common_wndef.npy` | (53930, 1024) | `vocabulary_wndef.csv` | **NEW** |
| `g1/f_common_wndef.npy` | (53930, 1024) | `vocabulary_wndef.csv` | **NEW** |
| `g_stock/f_common_wnex.npy` | (8360, 1024) | `vocabulary_wnex.csv` | Recent |
| `g1/f_common_wnex.npy` | (8360, 1024) | `vocabulary_wnex.csv` | Recent |

### Embedding files — val-only (for §4 collapse, §5/§6 T=0/T=1, §7 RSA)

| File | Shape | Index | Notes |
|------|-------|-------|-------|
| `g_stock/f_clue_val.npy` | (47933, 1024) | `g_stock/f_clue_val_index.csv` | Existing |
| `g1/f_clue_val.npy` | (47933, 1024) | `g1/f_clue_val_index.csv` | Existing |
| `g_stock/f_common_wndef_val.npy` | (26152, 1024) | `vocabulary_wndef_val.csv` | Existing |
| `g1/f_common_wndef_val.npy` | (26152, 1024) | `vocabulary_wndef_val.csv` | Existing |
| `g_stock/f_common_wnex_val.npy` | (3008, 1024) | `vocabulary_wnex_val.csv` | Existing |
| `g1/f_common_wnex_val.npy` | (3008, 1024) | `vocabulary_wnex_val.csv` | Existing |

### Data files

| File | Path (relative to `custom_embedding_model/`) |
|------|----------------------------------------------|
| `clues_val.csv` | `data/filtered_split/wn_synset/clues_val.csv` |
| `vocabulary_wndef.csv` | `data/filtered_split/wn_synset/wndef/vocabulary_wndef.csv` |
| `vocabulary_wnex.csv` | `data/filtered_split/wn_synset/wnex/vocabulary_wnex.csv` |
| `vocabulary_wndef_val.csv` | `data/filtered_split/wn_synset/wndef/vocabulary_wndef_val.csv` |
| `vocabulary_wnex_val.csv` | `data/filtered_split/wn_synset/wnex/vocabulary_wnex_val.csv` |
| `g1_val.csv` | `data/triplets/g1_val.csv` |

## Outputs

- `outputs/05_model_evaluation-results.md` (overwritten)
- `outputs/figures/05_val_triplet_accuracy.png` (overwritten — now ~46K triplets)
- `outputs/figures/05_crossf_triplet_accuracy.png` (**new**)
- `outputs/figures/05_collapse_pairwise_cosine.png` (overwritten — val-only, same as before)
- `outputs/figures/05_collapse_singular_values.png` (overwritten — val-only, same as before)
- `outputs/figures/05_t0_t1_wndef_distributions.png` (renamed from `05_t0_t1_distributions.png`)
- `outputs/figures/05_t0_t1_wnex_distributions.png` (**new**)

## Notebook structure

### §0 — Imports and configuration

Same as current. No changes beyond the date stamp.

### §1 — Load embeddings and index files

**Replace** the current loading section entirely. Load two sets of
embeddings: full-vocabulary (for triplet accuracy) and val-only (for
collapse detection, T=0/T=1, RSA).

**Full-vocabulary embeddings** (for §2, §3):
- **f_common_wndef (full)** for g_stock and g1 (53,930 rows each — **new**)
- **f_common_wnex (full)** for g_stock and g1 (8,360 rows each)
- `vocabulary_wndef.csv` (full, 53,930 rows)
- `vocabulary_wnex.csv` (full, 8,360 rows)

**Val-only embeddings** (for §4, §5, §6, §7):
- **f_clue_val** for g_stock and g1 (47,933 rows each)
- **f_common_wndef_val** for g_stock and g1 (26,152 rows each)
- **f_common_wnex_val** for g_stock and g1 (3,008 rows each)
- `f_clue_val_index.csv` (g_stock copy as canonical)
- `vocabulary_wndef_val.csv` (26,152 rows)
- `vocabulary_wnex_val.csv` (3,008 rows)

**Other data files:**
- `clues_val.csv`
- `g1_val.csv`

Build lookup dicts — **separate dicts for full-vocab and val-only** to
prevent accidental cross-use:
- `wndef_word_to_row`: word → row in vocabulary_wndef.csv / f_common_wndef.npy (full)
- `wnex_word_to_row`: word → row in vocabulary_wnex.csv / f_common_wnex.npy (full)
- `wndef_val_word_to_row`: word → row in vocabulary_wndef_val.csv / f_common_wndef_val.npy
- `wnex_val_word_to_row`: word → row in vocabulary_wnex_val.csv / f_common_wnex_val.npy
- `clue_key_to_row`: (clue_id, definition) → row in f_clue_val.npy

**Expected shapes for assertion:**

```python
EXPECTED_SHAPES = {
    "f_clue_val":          (47933, 1024),
    "f_common_wndef":      (53930, 1024),
    "f_common_wnex":       (8360, 1024),
    "f_common_wndef_val":  (26152, 1024),
    "f_common_wnex_val":   (3008, 1024),
}
```

**Registry approach:** Store embeddings in a dict keyed by `(model, phrase)`.
The phrase keys are `"f_clue_val"`, `"f_common_wndef"`, `"f_common_wnex"`,
`"f_common_wndef_val"`, `"f_common_wnex_val"`.

**Markdown cell** before loading should explain the full-vocab vs val-only
split: full-vocab is used for triplet accuracy (§2, §3) where we need
distractor resolution across the full vocabulary; val-only is used for
model diagnostics (§4, §7) and ATE (§5, §6) that inform model selection
and must not include test-split words (Decision 9).

Keep the `rowwise_cosine` helper from the current notebook (unchanged).

### §2 — Validation triplet accuracy (wndef, full-vocabulary)

**Markdown cell** explains: this is the same analysis as the original §2
but with dramatically better resolution. Previously ~41% of triplets were
dropped because distractors weren't in the val-only wndef vocabulary
(Decision 21). Full-vocab wndef embeddings (53,930 words) resolve nearly
all distractors.

**§2a — Resolve triplet embedding rows.**
Load `g1_val.csv` (46,506 rows). For each triplet:
- anchor: look up `(clue_id, definition)` in `clue_key_to_row`
- positive: look up `answer_wn` in `wndef_word_to_row`
- negative: look up `distractor_wn` in `wndef_word_to_row`

Report per-role resolution counts. Assert anchors and positives resolve
100%. Report how many negatives now resolve (expect ~46,200+ out of
46,506 — only the ~222 distractors completely absent from our vocabulary
should fail, per g1_train_meta.json).

**§2b — Triplet accuracy table.**
Compute for both g_stock and g1 on the resolved subset:
- Triplet accuracy (% where cos(A,P) > cos(A,N))
- Mean margin, median margin
- % triplets with margin > 0.1, > 0.5
- N triplets evaluated

Use the same `triplet_stats` function pattern as the current notebook but
parameterized to accept the phrase-type embedding array (so it can be
reused in §3 for wnex).

**§2c — Margin distribution figure.**
Same overlaid histogram as current (`05_val_triplet_accuracy.png`), now
with ~46K triplets.

### §3 — Cross-f triplet accuracy (matched comparison)

This is the central new analysis. It answers: does g1's discriminative
structure generalize from wndef (the phrase type it was trained on) to
wnex (a phrase type it has never seen)?

**Markdown cell** explains the matched-comparison methodology:
- For a fair comparison, we must evaluate the *exact same set of triplets*
  under both wndef and wnex embeddings
- The binding constraint is the wnex vocabulary (8,360 words), since
  wnex ⊂ wndef
- A triplet enters the matched set if `answer_wn` ∈ vocabulary_wnex AND
  `distractor_wn` ∈ vocabulary_wnex (anchors always resolve via f_clue_val)
- The same triplet subset is then evaluated using wndef embeddings and
  wnex embeddings for the positive and negative

**§3a — Identify the matched subset.**
From the 46,506 validation triplets, find those where both `answer_wn`
and `distractor_wn` are in `wnex_word_to_row`. Report:
- N matched triplets (and % of total)
- N triplets where answer resolves in wnex but distractor doesn't
- N triplets where distractor resolves in wnex but answer doesn't
- N triplets where neither resolves in wnex

**§3b — Matched triplet accuracy table.**
On the matched subset, compute triplet accuracy for all four combinations:
- g_stock with wndef
- g1 with wndef
- g_stock with wnex
- g1 with wnex

Present as a single table with columns: Metric, g_stock (wndef), g1 (wndef),
g_stock (wnex), g1 (wnex). Rows: accuracy, mean margin, median margin,
% margin > 0.1, N triplets.

The key comparisons:
- **g1 wndef vs g1 wnex**: does the triplet accuracy transfer?
- **g1 wnex vs g_stock wnex**: does g1 improve over the baseline on wnex?

**§3c — Margin distribution figure.**
A 2×1 figure with two panels:
- Left panel: g_stock vs g1 margin distributions (wndef, matched subset)
- Right panel: g_stock vs g1 margin distributions (wnex, matched subset)

Same overlaid-histogram style as §2c. Save as
`05_crossf_triplet_accuracy.png`.

### §4 — Collapse detection (val-only)

Same two-part structure as the current §3, using **val-only** embeddings.
These are model diagnostics that inform model selection — using full-vocab
embeddings here would include test-split words, violating the spirit of
Decision 9.

**§4a — Mean pairwise cosine among random word pairs.**
Sample 50,000 random distinct-row pairs (random_state=42) per
(model, phrase_type). Report for all four combinations:
- (g_stock, f_common_wndef_val), (g1, f_common_wndef_val)
- (g_stock, f_common_wnex_val), (g1, f_common_wnex_val)

Same table format as current: Mean, Median, Std, P5, P95.

**§4b — Embedding variance and effective dimensionality.**
Compute participation ratio and cumulative variance curves for all four.
Same table format as current.

**§4c — Figures.**
Same two-figure layout as current (pairwise cosine histograms, cumulative
variance curves), with two panels each for wndef_val and wnex_val. Save as
`05_collapse_pairwise_cosine.png` and `05_collapse_singular_values.png`.

### §5 — T=0 and T=1 similarity distributions (wndef)

Same as the current §4 but explicitly labeled as wndef-based. Uses
**val-only** wndef embeddings for the decontextualized lookups. Since
evaluation pairs come from `clues_val.csv`, all definition_wn and
answer_wn words are validation-split words and resolve in the val-only
vocabulary. Val-only is used for consistency with §4 and §7 — these
sections collectively form the validation-scoped model diagnostics.

**§5a — Assemble evaluation pairs.**
From `clues_val.csv` (47,933 rows), look up:
- `(clue_id, definition)` → f_clue_val row (via `clue_key_to_row`)
- `definition_wn` → f_common_wndef_val row (via `wndef_val_word_to_row`)
- `answer_wn` → f_common_wndef_val row (via `wndef_val_word_to_row`)

Report per-role resolution counts (expect 100%). Compute T=0 and T=1
for g_stock and g1.

**§5b — Table and ATE preview.**
Same format as current. Report mean, median, std, P5, P95 for each of
g_stock T=0, g_stock T=1, g1 T=0, g1 T=1. Print ATE = mean(T=1 − T=0).

**§5c — Figure.**
Same 1×2 panel layout. Save as `05_t0_t1_wndef_distributions.png`.

### §6 — T=0 and T=1 similarity distributions (wnex)

**New section.** Same analysis as §5 but using wnex for the
decontextualized terms:

- T=0: cos(g(f_common_wnex(def)), g(f_common_wnex(ans)))
- T=1: cos(g(f_clue(def)), g(f_common_wnex(ans)))

**§6a — Assemble wnex evaluation pairs.**
From `clues_val.csv`, keep rows where BOTH `definition_wn` AND
`answer_wn` are in `wnex_val_word_to_row`. Report:
- Total val clue rows
- Rows where definition_wn is in wnex vocab
- Rows where answer_wn is in wnex vocab
- Rows where both are (the evaluation set for this section)

This will be a substantially smaller set than §5 — wnex_val covers only
3,008 of 26,152 val vocabulary words (11.5%). Report the exact count
prominently.

**§6b — Table and ATE preview.**
Same format as §5b but on the wnex evaluation pairs. Include a note
comparing the wnex ATE to the wndef ATE from §5, and flagging that
these are computed on different (overlapping but not identical) subsets
of clues_val.

**§6c — Figure.**
Same 1×2 panel layout. Save as `05_t0_t1_wnex_distributions.png`.

**§6d — Matched ATE comparison (optional but recommended).**
For a cleaner wndef-vs-wnex ATE comparison, find the intersection of
evaluation pairs that resolve under BOTH wndef_val and wnex_val (the
binding constraint is wnex_val, since wndef_val covers all val vocabulary
words). On this matched subset, compute and report both the wndef ATE
and the wnex ATE. This eliminates any concern that the difference in
ATE is driven by different clue subsets rather than different phrase types.

### §7 — RSA (val-only)

Same as the current §5, using **val-only** embeddings for the same
model-selection discipline as §4.

Sample 1,000 words per phrase type (random_state=42). For wnex_val
(3,008 words), 1,000 is ~33% of the vocabulary. For wndef_val (26,152
words), 1,000 is ~3.8%.

Compute Spearman ρ between g_stock and g1 pairwise cosine upper triangles.
Same table format as current.

### §8 — Write results file

Rebuild `outputs/05_model_evaluation-results.md` with all sections.
Include all tables, resolution counts, ATE previews, and figure paths.
Add a new "## §3 — Cross-f triplet accuracy" section. Update all section
references to match the new numbering.

### Summary cell

Update to reflect the expanded scope. List all outputs (results file +
all figures). Note which analyses are new vs. updated from the previous
version.

## Implementation details

### Reusable functions

The `triplet_stats` function from the current §2b should be generalized
to accept a phrase-type embedding array as a parameter, so it can be
called for both wndef and wnex:

```python
def triplet_stats(model, anchor_rows, pos_rows, neg_rows, pos_neg_phrase):
    """Compute triplet accuracy for one (model, phrase_type) combination.

    Parameters:
        model: model name key in embeddings dict
        anchor_rows: int array of f_clue_val row indices
        pos_rows: int array of vocabulary row indices
        neg_rows: int array of vocabulary row indices
        pos_neg_phrase: phrase key in embeddings dict (e.g. "f_common_wndef")
    """
```

### Data loading

- All CSVs: `keep_default_na=False, na_values=[""]`
- All `.npy` loads: assert shape matches EXPECTED_SHAPES
- Cross-check `.npy` row counts against vocabulary file lengths
- Use `pathlib` for all paths

### Random seeds

- `np.random.seed(42)` at the top of §0 (same as current)
- Pair sampling in §4a: `np.random.default_rng(42)`
- RSA word sampling in §7: `np.random.default_rng(42)`

### Figures

- 300 dpi PNG, saved to `outputs/figures/`
- Common bin edges within each figure for direct visual comparison
- Consistent color scheme: g_stock = tab:blue, g1 = tab:orange
  (matching the current notebook)

## Environment

Local (CPU). No GPU work — all embeddings are pre-computed.

## Key interpretive questions the results should address

1. **Does g1's triplet accuracy transfer cross-f?** Compare g1's accuracy
   on the matched wnex subset against (a) g_stock's wnex accuracy and
   (b) g1's own wndef accuracy on the same subset.

2. **Does the collapse pattern hold?** The val-only analysis showed ~35%
   variance drop on both wndef and wnex. The revised §4 uses the same
   val-only data, so the numbers should match the previous run exactly
   (serving as a consistency check for the refactored notebook).

3. **How does the wnex ATE compare to the wndef ATE?** Is the misdirection
   signal (negative ATE) present under wnex too? Does g1 make it better
   or worse? The matched comparison in §6d is the cleanest version of
   this question.

4. **Resolution improvement:** How much did full-vocab embeddings improve
   triplet resolution (§2) compared to the val-only version (Decision 21)?
