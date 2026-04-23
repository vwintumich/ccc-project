# Spec: POS WordNet Census — Unambiguous Reliability Patch

**Stage:** Exploration (not part of the numbered pipeline)
**Notebook:** `planning/exploration/pos_wordnet_census.ipynb`
**Date:** 2026-04-21
**Status:** Approved

## Purpose

Patch the existing `pos_wordnet_census.ipynb` to replace the binary
reliability classification (`reliable` / `arbitrary`) with a three-way
classification that correctly handles single-synset words. Currently,
words with exactly one synset but zero lemma counts are labelled
"arbitrary" — but there is no sense selection to get wrong for these
words. They are trivially correct for both DI-2 (no POS choice) and
DI-3 (no ordering choice).

## The change

Replace the binary `reliability` column with a three-way classification:

| Category | Condition | Interpretation |
|----------|-----------|----------------|
| **unambiguous** | `n_synsets == 1` | Only one synset exists; sense selection is trivially correct regardless of lemma counts |
| **frequency-confirmed** | `n_synsets > 1` AND `has_nonzero_count` AND `sense0_is_max_within_pos` | Multiple senses exist, and frequency evidence supports sense[0] |
| **arbitrary** | `n_synsets > 1` AND (NOT `has_nonzero_count` OR NOT `sense0_is_max_within_pos`) | Multiple senses exist, and we lack evidence (or have contrary evidence) that sense[0] is the most common |

## Cells to modify

### §2d — Reliability column and heatmap

Replace:
```python
vocab_census['reliability'] = np.where(
    vocab_census['has_nonzero_count'] & vocab_census['sense0_is_max_within_pos'],
    'reliable',
    'arbitrary',
)
```

With logic implementing the three-way classification above. Use the
column name `reliability` (same as before). The three values should be
`'unambiguous'`, `'frequency-confirmed'`, `'arbitrary'`.

Update the heatmap in the same cell: three columns instead of two,
reindexed as `['unambiguous', 'frequency-confirmed', 'arbitrary']`.
Update the heatmap title/subtitle to reflect the new categories.

### §2c — Lemma count reliability reporting

After the existing reporting, add a summary block that reports the
three-way breakdown:
- Total unambiguous (n_synsets == 1): count and percentage
- Total frequency-confirmed: count and percentage
- Total arbitrary: count and percentage

This contextualises the existing `has_nonzero_count` statistics by
showing how many of the "zero count" words are actually unambiguous.

### §4d — Sense reliability in training

The existing statistics use `has_nonzero_count` to define "arbitrary."
Add a supplementary block that reports the three-way breakdown at the
triplet level:

- Triplets where ALL three wndef roles (anchor_wn, positive, negative)
  are either unambiguous or frequency-confirmed (i.e., none are
  arbitrary): count and percentage
- Triplets where at least one role is arbitrary: count and percentage

This replaces the existing `trip_any_arb` / `trip_both_hnz` statistics
as the primary reliability summary. Keep the existing statistics in
place (they are still informative) but add the new ones after them
with a markdown note explaining the distinction.

To classify each role: look up `n_synsets` from `vocab_census` for
`definition_wn`, `answer_wn`, and `distractor_wn`, and apply the
three-way logic. The simplest approach is to add `n_synsets` to the
`VC_COLS` list pulled onto the triplet dataframe in the §4 assembly
cell, then compute per-role reliability in §4d.

### §5d — Sense reliability in evaluation

Same pattern as §4d. Add a supplementary block reporting:

- Pairs where BOTH definition_wn and answer_wn are either unambiguous
  or frequency-confirmed: count and percentage
- Pairs where at least one is arbitrary: count and percentage

Again, keep existing statistics and add the new ones after them.

To classify each role: look up `n_synsets` from `vocab_census` for
`definition_wn` and `answer_wn`. Same approach — add `n_synsets` to
the `VC_COLS` list pulled onto the validation dataframe in the §5
assembly cell.

### §7 — Results file

Update the results file to include the three-way breakdown in the
vocabulary census section, the training triplet section, and the
validation pair section. The new numbers should appear alongside (not
replacing) the existing `has_nonzero_count` statistics.

### §6 — Summary cell

Update the summary cell text to reflect the three-way classification.
In particular, the current phrasing about "arbitrary" fractions should
be revised to distinguish unambiguous single-synset words from
genuinely arbitrary multi-synset words. The key narrative point: the
scale of DI-2 and DI-3 is smaller than the binary classification
suggested, because nearly half the vocabulary has only one synset.

## What NOT to change

- Do not modify §3 (contextual POS tagging) — unaffected.
- Do not modify §4a, §4b, §4c, §5a, §5b, §5c — unaffected.
- Do not change any figure filenames or output paths.
- Do not remove existing statistics — add the three-way breakdown
  alongside them.
- The `VC_COLS` list needs `n_synsets` added, but do not remove any
  existing columns from it.
