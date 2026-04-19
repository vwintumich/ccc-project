# Questions — 05_model_evaluation

## Q1 — Distractor words not in `vocabulary_wndef_val.csv`

**Spec reference:** §2a, step 3.

> Drop rows where any embedding lookup fails (should be zero if the triplet
> file and embeddings were built from the same upstream artifacts, but assert
> defensively).

**Observed:** 19,158 of 46,506 validation triplets (41.2%) have a
`distractor_wn` word that is not present in `vocabulary_wndef_val.csv`.
All 46,506 anchors and all 46,506 positives resolve. Only the negative
column is affected.

**Why:** The distractor words come from `dataset_harder.parquet` and are
drawn from the full WordNet vocabulary. `vocabulary_wndef.csv` covers all
words appearing as definition or answer in `clues_wn_filtered.csv`
(53,930 words total), but `vocabulary_wndef_val.csv` is the subset of those
words that appear specifically in the validation split (26,152 words).
A distractor that is a valid WordNet word but happens to appear only in
the training or test split rows of `clues_wn_filtered.csv` will be in
`vocabulary_wndef.csv` (and so will have a `g_stock` full-dataset
embedding) but **not** in `vocabulary_wndef_val.csv` (so it has no
`g_stock/f_common_wndef_val.npy` or `g1/f_common_wndef_val.npy` row).

Per Decision 8, `g1` embeddings cover the validation split only during
iteration; full-dataset `g1` embeddings are deferred to Stage 6. So for
~41% of validation triplets, the negative cannot be resolved to a `g1`
embedding at all.

**Options I considered:**

1. **Drop unresolved triplets and report the count explicitly.** Fastest,
   keeps the notebook moving, and the sample size for triplet accuracy
   (~27,348) is still large. Risk: the surviving triplets may be
   biased — e.g. if distractors that also appear as validation
   definitions/answers are systematically "easier" negatives than
   distractors that don't.
2. **Re-generate g1 / g_stock embeddings over the union of
   `vocabulary_wndef_val.csv` and the distractors used in `g1_val.csv`.**
   This means another Great Lakes run. Keeps the full 46,506-triplet
   sample, but violates the Stage 4 artifact freeze and adds a new
   vocabulary file.
3. **Use `g_stock` full-dataset `f_common_wndef.npy` for the negative
   lookups on both models, and accept that the "g1 negative" embedding
   is actually the `g_stock` negative.** This breaks the clean
   comparison and is not methodologically sound.

**Decision I need from the Architect:** which option to take. For now,
the notebook takes **Option 1**: drops unresolved triplets, reports the
exact count and fraction, and continues. A `NOTE:` banner is printed at
the dropout step so the behavior is visible in the notebook output. If
the Architect wants Option 2, we can re-run Stage 4 for the enlarged
vocabulary and then rerun this notebook unchanged (it will see zero
failures and the assertion will pass trivially).

**Scope impact:** only §2 (validation triplet accuracy) is affected.
§3 (collapse detection), §4 (T=0 / T=1), and §5 (RSA) all use
`f_common_wndef_val.npy` row lookups by validation-split words, which
are fully covered.
