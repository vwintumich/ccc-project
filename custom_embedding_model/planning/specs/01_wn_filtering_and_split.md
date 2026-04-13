# Spec: WordNet Filtering and Split Assignment

**Stage:** 1
**Notebook:** `notebooks/01_wn_filtering_and_split.ipynb`
**Date:** 2026-04-13
**Status:** Draft

## Purpose

Filter `clues_filtered.csv` to rows where both the definition and the answer
have at least one WordNet synset. Assign the 30/20/50 train/validate/test
split at the (definition, answer) pair level. Construct the full unified
vocabulary. This notebook produces the foundational dataset and vocabulary
files that all downstream phrase construction, triplet building, and
evaluation depend on.

## Inputs

- `../../data/clues_filtered.csv` — 457,342 rows, columns: `row_id`,
  `clue_id`, `surface`, `definition`, `answer`

## Outputs

All outputs go to `data/filtered_split/wn_synset/`:

| File | Description |
|------|-------------|
| `clues_wn_filtered.csv` | Filtered rows with `definition_wn`, `answer_wn`, and `split` columns added |
| `clues_val.csv` | Convenience subset: rows where `split == 'validate'` |
| `vocabulary.csv` | All unique words (definition or answer) in the filtered dataset; columns: `word`, `row` |
| `vocabulary_val.csv` | Validation-split subset of vocabulary; columns: `word`, `row` (own 0-indexed ordering) |

Must also produce: `outputs/01_wn_filtering_and_split-results.md`

## Implementation Details

### §1 — Imports and configuration

Standard imports: `pandas`, `numpy`, `pathlib`, `time`, `nltk` (WordNet).

```python
from nltk.corpus import wordnet as wn
```

Download WordNet data if not present. Define `DATA_DIR`, `OUTPUT_DIR`, etc.
Include environment auto-detection (Local / Great Lakes / Colab).

Load `clues_filtered.csv` with `keep_default_na=False, na_values=[""]`.
Use only the columns needed: `row_id`, `clue_id`, `surface`, `definition`,
`answer`.

Assert 457,342 rows loaded.

### §2 — WordNet lookup function

Implement `wordnet_lookup(text)` that returns a tuple
`(wn_form, synsets, strip_applied)`:

- `wn_form`: the string form that succeeded (lowercased, underscored,
  possibly article-stripped), or `None` if no synsets found
- `synsets`: the list of synsets (may be empty)
- `strip_applied`: which prefix was stripped (`None`, `"a"`, `"an"`,
  `"the"`, or `"to"`), for diagnostic reporting

**Lookup order:**

1. Lowercase the input, replace spaces with underscores. Try
   `wn.synsets(lookup)`. If synsets found, return immediately with
   `strip_applied=None`.
2. If no synsets, check whether the lowercased input starts with any of
   the prefixes `"a "`, `"an "`, `"the "`, `"to "` (checked in this
   order). For the first matching prefix, strip it, replace spaces with
   underscores in the remainder, and retry `wn.synsets()`. If synsets
   found, return with the appropriate `strip_applied` value.
3. If all attempts fail, return `(None, [], None)`.

**Important:** Only one prefix is tried — the first one that matches the
input. We do not attempt multiple prefixes per word. The order
`"a "`, `"an "`, `"the "`, `"to "` is chosen so that the most common
crossword-definition articles are tried first.

**Sanity checks (assert in the notebook):**

- `wordnet_lookup("shade")` → finds synsets, no strip
- `wordnet_lookup("a shade")` → finds synsets via `"a"` strip
- `wordnet_lookup("an animal")` → finds synsets via `"an"` strip
- `wordnet_lookup("the law")` → finds synsets via `"the"` strip
- `wordnet_lookup("to flee")` → finds synsets via `"to"` strip
- `wordnet_lookup("PLOT")` → finds synsets (uppercase handled)
- `wordnet_lookup("ice cream")` → finds synsets (multi-word)
- `wordnet_lookup("xyzzy")` → no synsets found

### §3 — Apply WordNet lookup to definitions and answers

Apply `wordnet_lookup` to every unique `definition` value and every unique
`answer` value. Work on unique values first, then map results back to the
full dataframe — this avoids redundant WordNet lookups (many rows share
the same definition or answer).

Create columns:
- `definition_wn`: the WordNet-ready form that succeeded, or `None`
- `answer_wn`: same for answers
- (Temporary) `def_strip` and `ans_strip`: which prefix was stripped, for
  the diagnostic

### §4 — Article-stripping diagnostic

Before filtering, report how many unique definitions and unique answers
were recovered by each prefix strip. Print a table like:

```
Article-stripping recovery (unique definitions):
  No strip needed:  N (x.x%)
  Recovered by "a":   N (x.x%)
  Recovered by "an":  N (x.x%)
  Recovered by "the": N (x.x%)
  Recovered by "to":  N (x.x%)
  No synsets found:   N (x.x%)
```

And the same for answers. This diagnostic informs whether the expanded
stripping was worthwhile. Include these numbers in the results file.

### §5 — Filter to rows with WordNet coverage

Drop rows where `definition_wn` is `None` OR `answer_wn` is `None`.
Report:
- Rows before filter
- Rows dropped (definition failed): count
- Rows dropped (answer failed): count
- Rows dropped (both failed): count
- Rows remaining after filter

Drop the temporary `def_strip` and `ans_strip` columns.

### §6 — Split assignment

Assign the train/validate/test split at the level of unique
(definition, answer) pairs:

1. Build a dataframe of unique `(definition, answer)` pairs from the
   filtered data.
2. Use `sklearn.model_selection.train_test_split` twice with
   `random_state=42`:
   - First split: 50% test, 50% remainder
   - Second split: 60% train, 40% validate (of the remainder = 30%/20%
     of total)
3. Map split assignments back to the full dataframe via merge on
   `(definition, answer)`.
4. Store as a `split` column with values `'train'`, `'validate'`,
   `'test'`.

**Validation checks:**
- Assert every row has a split value
- Assert no (definition, answer) pair appears in more than one split
- Print actual split fractions (rows and pairs) — they should be
  approximately 30/20/50 but won't be exact due to pair-level assignment

### §7 — Save clues_wn_filtered.csv

Output columns (in this order): `row_id`, `clue_id`, `surface`,
`definition`, `answer`, `definition_wn`, `answer_wn`, `split`.

Save to `data/filtered_split/wn_synset/clues_wn_filtered.csv` with
`index=False`. Create parent directories if needed.

Assert the output has the expected columns and no null values in
`definition_wn`, `answer_wn`, or `split`.

### §8 — Save clues_val.csv

Filter `clues_wn_filtered` to `split == 'validate'` and save as
`data/filtered_split/wn_synset/clues_val.csv` with `index=False`.

Report the number of validation rows.

### §9 — Vocabulary construction

Build the unified vocabulary from all unique words appearing as either
`definition_wn` or `answer_wn` in `clues_wn_filtered.csv`:

1. Collect all unique `definition_wn` values and all unique `answer_wn`
   values. Take their union.
2. Sort alphabetically to establish a deterministic canonical ordering.
3. Assign a `row` column: 0-indexed integers matching the sort order.
4. Save as `data/filtered_split/wn_synset/vocabulary.csv` with columns
   `word`, `row`. Use `index=False`.

**Validation checks:**
- Assert no duplicate words
- Assert `row` values are contiguous from 0 to len-1
- Assert every `definition_wn` and `answer_wn` value in the clue file
  appears in the vocabulary

Report vocabulary size.

### §10 — Validation vocabulary construction

Build `vocabulary_val.csv` from the subset of vocabulary words that
appear as `definition_wn` or `answer_wn` in validation-split rows:

1. Filter `clues_wn_filtered` to `split == 'validate'`.
2. Collect all unique `definition_wn` and `answer_wn` values from
   those rows. Take their union.
3. Sort alphabetically.
4. Assign its own 0-indexed `row` column (independent of the full
   vocabulary ordering).
5. Save as `data/filtered_split/wn_synset/vocabulary_val.csv`.

**Validation checks:**
- Assert every word in `vocabulary_val.csv` also exists in
  `vocabulary.csv`
- Assert no duplicates

Report validation vocabulary size and fraction of full vocabulary.

### §11 — Summary statistics and results file

Print and write to `outputs/01_wn_filtering_and_split-results.md`:

**Coverage measurements:**
- `clues_filtered.csv` rows (input)
- `clues_wn_filtered.csv` rows (output)
- Fraction retained after WordNet filter
- Rows recovered by each article strip (from §4 diagnostic)

**Split statistics:**
- Unique (definition, answer) pairs: total, train, validate, test
  (counts and fractions)
- Rows per split (counts and fractions)

**Vocabulary statistics:**
- `vocabulary.csv` word count
- `vocabulary_val.csv` word count
- Validation vocabulary as fraction of full vocabulary

**Overlap statistics:**
- Words appearing as both a definition and an answer
- Words appearing only as a definition
- Words appearing only as an answer

## Environment

Local (CPU). No GPU needed — this is pure pandas + WordNet lookups.

## Notebook Structure

- Use §-numbered markdown sections matching the sections above
- Include environment auto-detection for Local / Great Lakes / Colab
- Notebook header per CLAUDE.md conventions:
  - Primary author: Victoria
  - Builds on: `clue_misdirection/notebooks/01_data_cleaning.ipynb`
    (Victoria — WordNet lookup logic, article stripping, underscore
    conversion)
  - Environment: Local
- Notebook summary cell with all coverage statistics, output file
  locations, and any findings worth recording in FINDINGS.md
- Write results to `outputs/01_wn_filtering_and_split-results.md`
