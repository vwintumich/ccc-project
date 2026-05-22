# Claude Code Configuration — ccc-project

## Project Summary

This repository contains an ongoing NLP research project investigating
**semantic misdirection in cryptic crossword clues** — the phenomenon by which
the surface reading of a cryptic clue misleads embedding-based models trying
to connect the definition word to the correct answer.

The project is structured as a collection of components, each in its own
subdirectory. Two components are complete and must not be modified. One
component is actively under development. The shared `data/` directory and
shared `notebooks/` directory at the root level serve all components.

**Raw data source:** George Ho's cryptic crossword clue dataset (660,613
clues), available at https://cryptics.georgeho.org/data.db, previously
extracted to `clues_raw.csv`.

## Team

- **Victoria Winters**: Research questions, project management, phrase
  construction, embeddings, AI-assisted notebook development and repo management
- **Nathan Cantwell**: Triplet margin loss implementation, model training,
  Great Lakes compute

Faculty Advisor: Dr. Kevyn Collins-Thompson (University of Michigan)

## Repo Structure

```
ccc-project/
├── data/                              # SHARED — do not modify
│   ├── data.sqlite3                   # George Ho source DB (660,613 clues)
│   ├── clues_raw.csv                  # Extracted from sqlite; do not re-extract
│   ├── publisher_lookup.csv           # Lookup table for metadata extraction
│   ├── puzzle_metadata.csv            # Produced by puzzle_metadata.ipynb
│   └── clues_filtered.csv             # Produced by structural_filtering.ipynb
├── notebooks/                         # Shared upstream notebooks (order-independent)
│   ├── puzzle_metadata.ipynb          # Extracts publisher, series, setter, puzzle_no, clue_direction
│   ├── structural_filtering.ipynb     # Filters clues_raw.csv to valid CCC clues
│   └── clue_utils.py                  # Shared definition-finding and delimiter-placement logic
├── CLAUDE.md                          # This file
├── WORKFLOW.md                        # Shared pipeline documentation
├── DATA_RAW.md                        # Raw data schema and metadata extraction logic
├── custom_embedding_model/            # Set aside — see Decision 28
│   └── CLAUDE.md                      # Component-specific configuration
├── clue_misdirection/                 # Complete — do not modify
└── indicator_clustering/              # Complete — do not modify
```

## The Shared Pipeline

Two notebooks and one utility module at the project root produce shared
artifacts consumed by all components. The notebooks are independent of each
other and can be run in any order.

### `puzzle_metadata.ipynb`

**Reads:** `data/clues_raw.csv`, `data/publisher_lookup.csv`
**Writes:** `data/puzzle_metadata.csv`

Extracts structured metadata for every puzzle in the dataset by parsing
`puzzle_name`, `source_url`, and `clue_number`. Produces one row per
`clue_id` with columns: `publisher`, `series`, `setter`, `puzzle_no`,
`puzzle_date`, `clue_no`, and `clue_direction`. See `DATA_RAW.md` §4 for
the full source-by-source extraction logic, edge cases, and lookup table
format.

This file is a standalone artifact — never joined into `clues_filtered.csv`
automatically. Downstream notebooks join it in when needed, using `clue_id`
as the key. Note that `clue_id` is unique in `clues_raw.csv` but is **not**
unique in `clues_filtered.csv` after multi-definition expansion.

### `structural_filtering.ipynb`

**Reads:** `data/clues_raw.csv`
**Writes:** `data/clues_filtered.csv`

Filters `clues_raw.csv` to clues satisfying CCC structural constraints.
Loads only `clue_id`, `clue`, `answer`, and `definition` from the raw file.

Filtering steps:
1. Remove rows where `definition` or `answer` is null
2. Validate that `answer` matches the length/format code in `clue`
3. Parse double-definition clues: split on `/`, retain only candidate
   definitions that appear as intact whole words in the surface (accepting
   `<word>'s` as a valid match for `<word>`), keep the clue only if at
   least one valid definition appears at the start or end of the surface,
   expand to one row per valid definition
4. **Bracket diagnostic** (not a filter): count and display surviving rows
   containing `[`; view them in a cell output; document the decision in
   `DECISIONS.md`

Output columns: `clue_id`, `surface`, `definition`, `answer`.

Definition-in-surface matching uses `clue_utils.py` — the same utility used
in `02_phrase_construction.ipynb` when placing `<t></t>` delimiters, ensuring
the two steps are always consistent.

### `clue_utils.py`

A shared Python module containing the definition-finding and delimiter-placement
logic used by both `structural_filtering.ipynb` and component phrase
construction notebooks. Key functions:

- **`find_definition_in_surface(definition, surface)`** — locates the
  definition in the surface using word-boundary matching; accepts `<word>'s`
  as a valid match for `<word>`; returns the span or `None`
- **`tag_definition_in_surface(definition, surface)`** — wraps the located
  definition in `<t></t>` delimiters, preserving the original surface's
  capitalization and punctuation; returns the tagged string or `None` if
  the definition cannot be unambiguously located

This module is a committed, versioned artifact. If its logic changes, any
previously generated `f_clue.csv` is potentially stale. Document changes
in `DECISIONS.md`.

## Shared Data Directory Rules

- **Do not modify** `data/clues_raw.csv`, `data/data.sqlite3`, or
  `data/publisher_lookup.csv`. These are fixed inputs.
- **Do not re-extract** from `data.sqlite3`. `clues_raw.csv` is the
  authoritative extracted form.
- **Do not regenerate** `puzzle_metadata.csv` or `clues_filtered.csv` without
  agreement from the team and a documented reason.
- **Do not join** puzzle metadata into `clues_filtered.csv` at this level.
  Components join it in when needed.
- **Do not modify** anything in `clue_misdirection/` or
  `indicator_clustering/` — both are complete.

## Notebook Style

Write markdown cells as if the reader is new to the project — each cell should
fully explain the purpose and reasoning of the Python code that follows. At the
same time, eliminate redundancy: do not restate what the code makes obvious, do
not repeat content already established in a prior cell, and do not restate in
an inline comment what the preceding markdown cell just said. Justify length by
conceptual necessity, not by comprehensiveness for its own sake. Code cells
open with a `# ===` banner; inline comments are appropriate where logic is
non-obvious. See `custom_embedding_model/notebooks/archive/09_learned_g_misdirection.ipynb`
for an example of a well-executed notebook in this project.

## Notebook Header

Every notebook must start with a markdown cell containing:

```
# [Title]

**Primary author:** [who wrote this notebook]

**Builds on:**
- *[Notebook or document name]* ([Author] — brief description of what was drawn from it)

**Prompt engineering:** Victoria
**AI assistance:** Claude / Claude Code (Anthropic)
**Environment:** Local

[Brief purpose statement: what this notebook does, what it reads, what it
produces, and how it fits into the overall pipeline.]
```

## Notebook Summary Cell

Every notebook must end with a markdown cell that:
- Summarizes what was done
- Reports row counts at each filter step (count and fraction of previous stage)
- States the size and location of all output files produced
- Highlights any findings or edge cases worth noting
- Records wall-clock runtime for any computationally significant steps

## Coding Standards

- **Use `pathlib`** for all file paths.
- **Pin random seeds** (`random_state=42`) for reproducibility everywhere.
- **No hardcoded absolute paths.** Use relative paths from the notebook
  directory, or named variables (`DATA_DIR`, `OUTPUT_DIR`) defined at the top.
- **`keep_default_na=False`:** Always use this flag when loading any CSV that
  contains word, definition, or answer columns. The word "nan" (meaning
  grandmother) is a valid crossword entry; without this flag, pandas silently
  converts it to `NaN`. Use `na_values=[""]` alongside it.
- **Record runtimes:** Use `time.time()` or equivalent for any step taking
  more than a few seconds. Print and include in the summary cell.

## Working in a Component

Before writing any code or notebooks inside a component subdirectory, read
that component's `CLAUDE.md`. Each component has its own workflow, data
conventions, and critical rules that extend and refine what is documented here.

- **`custom_embedding_model/`** — read `custom_embedding_model/CLAUDE.md`
- `clue_misdirection/` and `indicator_clustering/` are complete; do not modify

## Key Reference Files

- `WORKFLOW.md` — shared pipeline documentation: what the two shared notebooks
  do, what they produce, and how components consume their outputs
- `DATA_RAW.md` — raw data schema (`clues_raw.csv` columns, source breakdown)
  and full puzzle metadata extraction logic (source-by-source `puzzle_name`
  formats, lookup table structure, edge cases)
