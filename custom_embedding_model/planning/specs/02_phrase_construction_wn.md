# Spec: WordNet Phrase Construction

**Stage:** 2
**Notebook:** `notebooks/02_phrase_construction_wn.ipynb`
**Date:** 2026-04-13
**Status:** Draft

## Purpose

Construct phrase files for all three WordNet-based f's: f_clue,
f_common_wndef, and f_common_wnex. For each vocabulary-based f, also
produce the filtered clue file, vocabulary files, and validation vocabulary
files that define that f's data scope. Report coverage statistics at every
stage.

This is the WordNet-specific phrase construction notebook. Future resource
families (dictionary APIs, LLM-generated phrases) will have their own
`02_phrase_construction_<resource>.ipynb` notebooks.

## Inputs

- `data/filtered_split/wn_synset/clues_wn_filtered.csv` — 239,406 rows;
  columns: `row_id`, `clue_id`, `surface`, `definition`, `answer`,
  `definition_wn`, `answer_wn`, `split`
- `data/filtered_split/wn_synset/vocabulary.csv` — 53,930 words;
  columns: `word`, `row`
- `data/filtered_split/wn_synset/vocabulary_val.csv` — 26,152 words;
  columns: `word`, `row`
- `../../notebooks/clue_utils.py` — shared definition-finding and
  delimiter-placement logic
- WordNet corpus via NLTK

## Outputs

All outputs go under `data/filtered_split/wn_synset/`:

**f_clue** (in `clue_phrases/`):

| File | Description |
|------|-------------|
| `f_clue.csv` | One row per (clue_id, definition) pair with valid tagging; columns: `clue_id`, `definition`, `split`, `phrase` |

**f_common_wndef** (in `wndef/`):

| File | Description |
|------|-------------|
| `f_common_wndef.csv` | One row per word with valid phrase; columns: `word`, `row`, `synset_name`, `phrase`, `self_ref` |
| `clues_wndef_filtered.csv` | Subset of `clues_wn_filtered.csv` where both `definition_wn` and `answer_wn` appear in `vocabulary_wndef.csv`; inherits all columns including `split` |
| `vocabulary_wndef.csv` | Words with valid f_common_wndef phrase; columns: `word`, `row` (own 0-indexed ordering) |
| `vocabulary_wndef_val.csv` | Validation-split subset; columns: `word`, `row` (own 0-indexed ordering) |

**f_common_wnex** (in `wnex/`):

| File | Description |
|------|-------------|
| `f_common_wnex.csv` | One row per word with valid phrase; columns: `word`, `row`, `synset_name`, `phrase` |
| `clues_wnex_filtered.csv` | Subset of `clues_wn_filtered.csv` where both `definition_wn` and `answer_wn` appear in `vocabulary_wnex.csv`; inherits all columns including `split` |
| `vocabulary_wnex.csv` | Words with valid f_common_wnex phrase; columns: `word`, `row` (own 0-indexed ordering) |
| `vocabulary_wnex_val.csv` | Validation-split subset; columns: `word`, `row` (own 0-indexed ordering) |

Must also produce: `outputs/02_phrase_construction_wn-results.md`

## Implementation Details

### §1 — Imports and configuration

Standard imports: `pandas`, `numpy`, `pathlib`, `time`, `sys`, `nltk`
(WordNet).

Add `../../notebooks` to `sys.path` and import `clue_utils`:
```python
sys.path.insert(0, str(PROJECT_ROOT / "notebooks"))
from clue_utils import tag_definition_in_surface
```

Include environment auto-detection (Local / Great Lakes / Colab).
Include version-reporting cell per Decision 18 (pandas, numpy, nltk,
WordNet corpus version).

Load `clues_wn_filtered.csv` and `vocabulary.csv` with
`keep_default_na=False, na_values=[""]`.

Assert expected row counts: 239,406 clue rows, 53,930 vocabulary words.

### §2 — f_clue construction

For every row in `clues_wn_filtered.csv`, attempt to construct an f_clue
phrase by calling `tag_definition_in_surface(definition, surface)`.

This function returns `None` when the definition cannot be unambiguously
located in the surface (e.g., the definition appears more than once, or
matching fails). Rows where it returns `None` are dropped from `f_clue.csv`
but remain in `clues_wn_filtered.csv` — f_clue does not further filter the
scope.

**Output columns:** `clue_id`, `definition`, `split`, `phrase`

Save to `data/filtered_split/wn_synset/clue_phrases/f_clue.csv` with
`index=False`. Create the `clue_phrases/` directory if needed.

**Validation checks:**
- Assert no duplicate (clue_id, definition) pairs
- Assert every `phrase` value contains exactly one `<t>` and one `</t>`
- Assert the `split` column matches the source data

**Coverage report:**
- Total rows in `clues_wn_filtered.csv`
- Rows with valid f_clue phrase (count and fraction)
- Rows dropped and why (print a few examples of failures for inspection)

### §3 — f_common_wndef construction

For each word in `vocabulary.csv`, look up the most frequent WordNet synset
(index 0 from `wn.synsets(word)`) and construct:

```
"<t>word</t>: <synset.definition()>"
```

Where `word` is the vocabulary word as stored (the `_wn` form — lowercased,
underscored). For display in the phrase, replace underscores with spaces in
the target word:

```python
display_word = word.replace("_", " ")
phrase = f"<t>{display_word}</t>: {synset.definition()}"
```

Every word in `vocabulary.csv` has at least one synset (that was the
NB 01 filter), so f_common_wndef should produce a phrase for every word.
If any word unexpectedly has no synsets, print a warning and exclude it.

**Self-referential detection:** After constructing each phrase, check whether
the target word also appears *untagged* in the phrase. Strip the one tagged
occurrence (`<t>display_word</t>`), then search for a case-insensitive
word-boundary match of `display_word` in the remainder. Record the result as
a boolean `self_ref` column.

```python
import re
untagged = phrase.replace(f"<t>{display_word}</t>", "", 1)
self_ref = bool(re.search(r"\b" + re.escape(display_word) + r"\b", untagged, re.IGNORECASE))
```

These words are kept in the vocabulary (not filtered out) — the flag exists
so that downstream evaluation can subset and check whether self-referential
definitions behave differently.

**Output columns:** `word`, `row`, `synset_name`, `phrase`, `self_ref`

The `row` column here is a new 0-indexed ordering for vocabulary_wndef,
independent of the full vocabulary's row numbering.

Save `f_common_wndef.csv` to `data/filtered_split/wn_synset/wndef/`.

**Validation checks:**
- Assert the number of rows equals (or very nearly equals) the vocabulary
  size — f_common_wndef should cover all or nearly all words
- Assert every phrase contains exactly one `<t>` and one `</t>`
- Assert no duplicate words
- Assert `self_ref` column is boolean dtype

### §4 — f_common_wndef vocabulary and filtered clues

**Vocabulary files:**

Build `vocabulary_wndef.csv` from the words that have valid f_common_wndef
phrases. Sort alphabetically, assign 0-indexed `row` column. Save to
`wndef/vocabulary_wndef.csv`.

**Filtered clue file:**

Build `clues_wndef_filtered.csv` by keeping only rows from
`clues_wn_filtered.csv` where BOTH `definition_wn` AND `answer_wn` appear
in `vocabulary_wndef.csv`. This ensures every row in the filtered file has
valid decontextualized phrases for both the definition and the answer.
Inherit all columns including `split`. Save to `wndef/clues_wndef_filtered.csv`.

**Validation vocabulary:**

Build `vocabulary_wndef_val.csv` from the subset of `vocabulary_wndef.csv`
words that appear as `definition_wn` or `answer_wn` in validation-split
rows of **`clues_wndef_filtered.csv`** (not `clues_wn_filtered.csv`). This
ensures the validation vocabulary contains only words that will actually be
needed for ATE evaluation on wndef-filtered validation clues — no wasted
embeddings. Sort alphabetically, assign own 0-indexed `row` column. Save
to `wndef/vocabulary_wndef_val.csv`.

**Validation checks:**
- Assert every word in `vocabulary_wndef_val.csv` exists in
  `vocabulary_wndef.csv`
- Assert every `definition_wn` and `answer_wn` in `clues_wndef_filtered.csv`
  exists in `vocabulary_wndef.csv`
- Assert no (definition, answer) pair spans multiple splits

**Coverage report:**
- vocabulary_wndef size vs. full vocabulary (count and fraction)
- vocabulary_wndef_val size vs. vocabulary_wndef (count and fraction)
- clues_wndef_filtered rows vs. clues_wn_filtered rows (count and fraction)
- Actual train/validate/test split fractions in clues_wndef_filtered

### §5 — f_common_wnex construction

For each word in `vocabulary.csv`, look up the most frequent WordNet synset
(index 0) and check whether it has usage examples
(`synset.examples()`). If examples exist, find the first example where the
target word appears exactly once, and wrap it in `<t></t>` delimiters.

**Matching the word in the example:** The word in the vocabulary is in
`_wn` form (lowercased, underscored). The usage example is natural text.
Search for the word (with underscores replaced by spaces) using
case-insensitive word-boundary matching. The word must appear exactly once
for unambiguous tagging.

```python
display_word = word.replace("_", " ")
pattern = re.compile(r"\b" + re.escape(display_word) + r"\b", re.IGNORECASE)
matches = list(pattern.finditer(example))
if len(matches) == 1:
    m = matches[0]
    phrase = example[:m.start()] + "<t>" + example[m.start():m.end()] + "</t>" + example[m.end():]
```

A word is excluded from f_common_wnex if:
- Its most frequent synset has no usage examples, OR
- None of the examples contain the word exactly once

**Output columns:** `word`, `row`, `synset_name`, `phrase`

The `row` column is a new 0-indexed ordering for vocabulary_wnex.

Save `f_common_wnex.csv` to `data/filtered_split/wn_synset/wnex/`.

**Validation checks:**
- Assert every phrase contains exactly one `<t>` and one `</t>`
- Assert no duplicate words

**Coverage report:**
- Words with valid f_common_wnex phrase vs. full vocabulary (count and
  fraction) — this is expected to be substantially lower than f_common_wndef

### §6 — f_common_wnex vocabulary and filtered clues

Same structure as §4 but for wnex. **Important:** the order of operations
is the same — build the vocabulary first, then the filtered clue file, then
the validation vocabulary from the filtered clue file's validation rows:

1. `vocabulary_wnex.csv` — words with valid f_common_wnex phrases, sorted
   alphabetically, 0-indexed `row`
2. `clues_wnex_filtered.csv` — rows from `clues_wn_filtered.csv` where both
   `definition_wn` and `answer_wn` appear in `vocabulary_wnex.csv`
3. `vocabulary_wnex_val.csv` — words from `vocabulary_wnex.csv` that appear
   as `definition_wn` or `answer_wn` in validation rows of
   **`clues_wnex_filtered.csv`**; own 0-indexed `row`

Same validation checks as §4.

**Coverage report:**
- vocabulary_wnex size vs. full vocabulary (count and fraction)
- vocabulary_wnex_val size vs. vocabulary_wnex (count and fraction)
- clues_wnex_filtered rows vs. clues_wn_filtered rows (count and fraction)
- Actual train/validate/test split fractions in clues_wnex_filtered

### §7 — Summary statistics and results file

Print and write to `outputs/02_phrase_construction_wn-results.md`:

**Versions section** (per Decision 18).

**f_clue coverage:**
- clues_wn_filtered rows (input)
- Rows with valid f_clue phrase (count and fraction)
- Rows dropped (count and reasons)

**f_common_wndef coverage:**
- Full vocabulary size (input)
- Words with valid phrase (count and fraction)
- Self-referential words (count and fraction of wndef vocabulary)
- clues_wndef_filtered rows (count and fraction of clues_wn_filtered)
- vocabulary_wndef_val size
- Actual train/validate/test fractions in clues_wndef_filtered

**f_common_wnex coverage:**
- Full vocabulary size (input)
- Words with valid phrase (count and fraction)
- clues_wnex_filtered rows (count and fraction of clues_wn_filtered)
- vocabulary_wnex_val size
- Actual train/validate/test fractions in clues_wnex_filtered

**Cross-f comparison:**
- Words in vocabulary_wndef but not vocabulary_wnex (count)
- Words in both (count)
- Rows in clues_wndef_filtered but not clues_wnex_filtered (count)

## Environment

Local (CPU). No GPU needed — this is pandas + WordNet lookups +
string manipulation.

## Notebook Structure

- Use §-numbered markdown sections matching the sections above
- Include environment auto-detection for Local / Great Lakes / Colab
- Version-reporting cell per Decision 18
- Notebook header per CLAUDE.md conventions:
  - Primary author: Victoria
  - Builds on: `clue_misdirection/notebooks/02_embedding_generation.ipynb`
    (Victoria — phrase construction logic, CALE delimiter insertion,
    WordNet synset lookup)
  - Environment: Local
- Notebook summary cell with all coverage statistics, output file
  locations, and findings worth recording in FINDINGS.md
- Write results to `outputs/02_phrase_construction_wn-results.md`
