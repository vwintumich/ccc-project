# Workflow — ccc-project (Shared Pipeline)

This document describes the two shared notebooks that sit upstream of all
project components. Their outputs — `puzzle_metadata.csv` and
`clues_filtered.csv` — are shared artifacts in `ccc-project/data/` and should
not be regenerated without a documented reason.

Component-specific workflows (WordNet filtering, phrase construction, model
training) are documented in their respective `WORKFLOW.md` files.

---

## Repo Structure

```
ccc-project/
├── data/                              # SHARED — do not modify
│   ├── data.sqlite3                   # George Ho source DB (660,613 clues)
│   ├── clues_raw.csv                  # Extracted from sqlite; shared input
│   ├── puzzle_metadata.csv            # Produced by puzzle_metadata.ipynb
│   ├── clues_filtered.csv             # Produced by structural_filtering.ipynb
│   └── publisher_lookup.csv           # Lookup table for metadata extraction
├── notebooks/
│   ├── puzzle_metadata.ipynb          # Extracts publisher, series, setter, puzzle_no, clue_direction
│   └── structural_filtering.ipynb     # Filters clues_raw.csv to valid CCC clues
├── DATA_RAW.md                        # Raw data schema and metadata extraction logic
├── custom_embedding_model/            # Active component
├── clue_misdirection/                 # Complete — do not modify
└── indicator_clustering/              # Complete — do not modify
```

---

## `puzzle_metadata.ipynb`

**Notebook:** `notebooks/puzzle_metadata.ipynb`
**Environment:** Local (CPU)
**Inputs:** `data/clues_raw.csv`, `data/publisher_lookup.csv`
**Output:** `data/puzzle_metadata.csv`

Extracts structured metadata for every puzzle in the dataset by parsing
`puzzle_name`, `source_url`, and `clue_number`. This notebook is independent
of structural filtering and can be run at any time. Its output is a standalone
file keyed on `clue_id` — never joined into `clues_filtered.csv`
automatically; downstream notebooks join it in when needed.

Derived columns:

| Column | Description |
|--------|-------------|
| `clue_id` | Join key — unique in `clues_raw.csv`; not unique in `clues_filtered.csv` |
| `publisher` | Canonical publisher name |
| `series` | Puzzle series within the publication (e.g., "Toughie", "QC") |
| `setter` | Setter pseudonym or surname; pipe-separated for collaborations |
| `puzzle_no` | Puzzle number (int); NaN where not extractable |
| `puzzle_date` | Publication date; extracted from `puzzle_name` or `source_url` where missing |
| `clue_no` | Parsed grid number (int); NaN if unparseable |
| `clue_direction` | `"across"` or `"down"`; parsed from `clue_number`; NaN if unparseable |

See `DATA_RAW.md` §4 for the full extraction logic, source-by-source format
descriptions, and edge cases.

**Critical:** `clue_id` is unique in `clues_raw.csv` but is **not** unique in
`clues_filtered.csv` after multi-definition expansion. When joining
`puzzle_metadata.csv` onto filtered or downstream files, join on `clue_id` and
expect multiple matching rows.

---

## `structural_filtering.ipynb`

**Notebook:** `notebooks/structural_filtering.ipynb`
**Environment:** Local (CPU)
**Inputs:** `data/clues_raw.csv`
**Output:** `data/clues_filtered.csv`

Filters `clues_raw.csv` to clues satisfying CCC structural constraints,
independent of any external linguistic resource. This is the canonical shared
upstream artifact for all active and future project components.

Columns loaded from `clues_raw.csv`: `clue_id`, `clue`, `answer`, `definition`.

Filtering steps:

1. Remove rows where `clue`, `definition`, or `answer` is null
2. Clean the clue surface: strip `/` unconditionally; exclude rows where `*`
   is the first character followed by an uppercase letter (strip `*`
   elsewhere); exclude rows where brackets surround or partially surround an
   all-caps sequence, strip brackets from remaining rows
3. Validate that `answer` matches the length/format code in `clue`
4. Parse double-definition clues: split `definition` on `/` into a list of
   candidate definitions; retain only candidates that appear as intact whole
   words in the surface text (accepting `<word>'s` as a valid match for
   `<word>`); keep the clue row only if at least one valid definition appears
   at the start or end of the surface; expand to one row per valid definition
5. **Bracket, `*`, and `/` diagnostics** (not filters): display surviving
   rows containing each character for reference; decisions documented in
   `DECISIONS.md`

Definition-in-surface matching uses `clue_utils.py` (see below), which
handles word-boundary matching and apostrophe-s. The same utility is used in
`02_phrase_construction.ipynb` when placing `<t></t>` delimiters for f_clue,
ensuring consistency between filtering and phrase construction.

Output schema (fixed column order):

| Column | Description |
|--------|-------------|
| `clue_id` | From George Ho sqlite; not unique after multi-definition expansion |
| `surface` | Clue text with answer format stripped; used for f_clue phrase construction |
| `definition` | Single valid definition substring (one row per definition) |
| `answer` | Answer string, uppercase |

**Critical:** Do not apply WordNet constraints, assign train/validate/test
splits, or join puzzle metadata here. Do not output `clue`, `surface_normalized`,
`answer_format`, `clue_no`, `clue_direction`, `source`, or `num_definitions` —
these are either derivable on demand, belong in `puzzle_metadata.csv`, or are
intermediate artifacts not needed downstream.

## `clue_utils.py`

**File:** `notebooks/clue_utils.py` (importable from shared notebooks and
component notebooks alike)

Contains the shared definition-finding and delimiter-placement logic used by
both `structural_filtering.ipynb` and `custom_embedding_model/notebooks/02_phrase_construction.ipynb`.

Key functions:
- **`find_definition_in_surface(definition, surface)`** — returns the span
  (start, end) of the definition in the surface, using word-boundary matching.
  Accepts `<word>'s` as a valid match for `<word>`. Returns `None` if not
  found.
- **`tag_definition_in_surface(definition, surface)`** — wraps the definition
  span in `<t></t>` delimiters and returns the tagged surface string for use
  as an f_clue phrase. Uses the original `surface` (preserving capitalization
  and punctuation) for the output phrase. Returns `None` if the definition
  cannot be unambiguously located.

Both functions operate on the original `surface` string, not on a normalized
form — CALE was trained on naturalistic text and produces better embeddings
when the input preserves capitalization and punctuation. Normalization is
applied internally only for the matching step, not carried through to the
output phrase.

This utility is a committed, versioned artifact. If the logic changes, any
previously generated `f_clue.csv` is potentially stale and must be regenerated.
Document any changes in `DECISIONS.md`.

---

## What Gets Computed Here vs. in Components

| Artifact | Notebook | Location |
|----------|----------|----------|
| `puzzle_metadata.csv` | `puzzle_metadata.ipynb` | `ccc-project/data/` |
| `clues_filtered.csv` | `structural_filtering.ipynb` | `ccc-project/data/` |
| WordNet filtering, split assignment | Component notebooks | `<component>/data/` |
| Phrase construction, embeddings, model training | Component notebooks | `<component>/data/` |

---

## Critical Rules

- **Do not modify** `clues_raw.csv`, `data.sqlite3`, or `publisher_lookup.csv`.
- **Do not regenerate** `puzzle_metadata.csv` or `clues_filtered.csv` without
  a documented reason agreed on by the team.
- **Do not join** puzzle metadata into `clues_filtered.csv` at this stage.
  Downstream components join it in when and if needed.
- **Do not modify** `clue_misdirection/` or `indicator_clustering/` — both
  are complete.
