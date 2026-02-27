# Claude Code Configuration — clue_misdirection

## Project Summary

This is the **supervised learning component (Part A)** of a MADS capstone
project (SIADS 696, Winter 2026) exploring semantic misdirection in cryptic
crossword clues. The research question: *How much does the surface reading of
a cryptic clue mislead embedding-based models trying to connect the definition
to the answer?*

The project has two analyses:
1. **Retrieval analysis** — rank all known answer words by cosine similarity
   to the definition under various embedding conditions; measure how clue
   context degrades retrieval of the true answer. This is the primary evidence
   for misdirection.
2. **Binary classifier** — distinguish real definition–answer pairs from
   distractor pairs across easy and harder datasets, with and without
   clue-context features. This satisfies the course requirement for supervised
   learning (3 model families, CV, ablation, etc.).

## Team

- **Victoria Winters** (CCC Domain Expert): Research questions, project management, embeddings, AI-assisted notebook refactoring and repo management
- **Hans Li** (Supervised Learning): Clue misdirection, AI prompting support
- **Nathan Cantwell** (Unsupervised Learning): Indicator clustering, definition clustering comparison
- **Sahana Sundar** (Evaluation): Evaluation and visualization, supervised/unsupervised support as needed
- **All:** Data cleaning, report writing, intermittent reporting to faculty advisor

Faculty Advisor: Dr. Kevyn Collins-Thompson (University of Michigan)

## Repo Structure

```
ccc-project/
├── data/                       # Shared data directory (both components read from here)
│   ├── data.sqlite3            # George Ho source DB (660,613 clues)
│   ├── clues_raw.csv           # Extracted from sqlite by NB00 (shared input)
│   ├── indicators_raw.csv      # Other *_raw.csv tables (used by indicator_clustering)
│   └── ...
├── clue_misdirection/          # ← YOU ARE HERE (supervised learning, Part A)
│   ├── CLAUDE.md               # This file
│   ├── PLAN.md                 # 12-step pipeline plan (from design doc v3)
│   ├── NOTEBOOKS.md            # Inventory of existing and planned notebooks
│   ├── DATA.md                 # Data dictionary and schema
│   ├── DECISIONS.md            # Locked-in team decisions
│   ├── FINDINGS.md             # Running log of findings and progress
│   ├── requirements.txt        # Python dependencies for this component
│   ├── notebooks/              # Pipeline notebooks (run in order)
│   │   └── archive/            # Prior exploratory notebooks (reference only)
│   ├── data/                   # Intermediate outputs (clues_filtered.csv, embeddings/, etc.)
│   └── outputs/                # Figures, results tables
├── indicator_clustering/       # Unsupervised component (Part B, separate, mostly complete)
│   └── ...
├── README.md                   # Top-level project overview
└── requirements.txt            # (TODO) Consolidated dependencies for both components
```

**Note on shared data:** The `*_raw.csv` files currently live in
`indicator_clustering/data/`. The target state is a shared `ccc-project/data/`
directory that both components read from. Migration of indicator_clustering
paths is pending (a teammate is actively working on that component). For now,
clue_misdirection notebooks should reference `../data/clues_raw.csv` as their
raw input, which will resolve correctly under either the current or target
directory layout.

## Tech Stack

- **Python 3.10+** (Conda base environment)
- **Embedding model:** `gabrielloiseau/CALE-MBERT-en` (CALE, 1024-dim) via
  `sentence-transformers`. Uses `<t></t>` delimiters around target words.
  (backup: `BAAI/bge-base-en-v1.5`, 768-dim)
- **Key libraries:** scikit-learn, pandas, numpy, nltk (WordNet), matplotlib,
  seaborn, sentence-transformers, torch
- **Compute:** GPU steps run on UM Great Lakes cluster or Google Colab (T4 GPU).
  All other steps run locally on CPU. Embedding the full filtered dataset
  (~100K rows upper bound) takes under 20 minutes on a T4 GPU and produces
  files under 1 GB — no sampling before embedding is needed.
- **Data formats:** CSVs for tabular data, `.npy` for dense embedding arrays,
  `.parquet` for the feature table (mixed types, named columns)

## Coding Standards

### Notebook Header

Every notebook must start with a markdown cell containing:

```
# [Title]

**Primary author:** [who wrote this notebook]

**Builds on:**
- *[Notebook Name]* ([Author] — brief description of what was drawn from it)
- ...

**Prompt engineering:** Victoria
**AI assistance:** Claude / Claude Code (Anthropic)
**Environment:** [Local / Great Lakes / Colab]

[Brief purpose statement, inputs, outputs, and which PLAN.md step(s)
this notebook implements.]
```

This intellectual lineage block credits the authors of prior work that
each notebook draws on. Most existing work comes from Hans, with data
cleaning contributions from Victoria and Sahana.

### Notebook Summary Cell

Every notebook must end with a markdown cell that:
- Summarizes the data cleaning, analysis, or modeling performed
- States the quality and size of the output data
- Highlights key findings or observations
- Uses visualizations where appropriate
- Notes anything worth recording in FINDINGS.md

Any information that would go into a cleaning log or results log should
be explained in the notebook itself, not in a separate file.

### Comments and Markdown Cells

This is a graded academic project. Comments and markdown cells should
explain **why**, not just **what**. Write for a graduate student who is
new to NLP and may not know why a particular step matters. For example:

- Bad: `# Filter to WordNet entries`
- Good: `# Filter to rows where both definition and answer have at least
  one WordNet synset. We need WordNet coverage to construct sense-specific
  embeddings (common vs. obscure) in Step 2, and to compute the 22
  relationship features in Step 3.`

### General Standards

- **Notebook naming:** `NN_short_description.ipynb` (e.g., `01_data_cleaning.ipynb`)
- **Use `pathlib`** for all file paths.
- **Pin random seeds** (`random_state=42`) for reproducibility everywhere.
- **No hardcoded absolute paths.** Use relative paths from the notebook directory
  or a `DATA_DIR` / `OUTPUT_DIR` variable defined at the top of each notebook.
- **Validate data** at boundaries: assert no NaNs before modeling
  (`assert not df.isnull().any().any()`).
- **Feature scaling:** Use `StandardScaler` fitted only on training folds for
  KNN and Logistic Regression. Random Forest is scale-invariant.
- **Cross-validation:** Use `GroupKFold` with 5 folds, grouped by
  definition–answer pair (or by definition word for stricter leakage prevention).
  Same fold assignments across all experiments.
- **Figures:** Save all figures to `outputs/figures/` as PNG (300 dpi).
- **`keep_default_na=False`:** Always use `keep_default_na=False` when
  loading any CSV that contains `word`, `definition_wn`, or `answer_wn`
  columns. The word "nan" (meaning grandmother) is a valid crossword
  definition and answer; without this flag, pandas silently converts it
  to `NaN`.
- **Clue-context embedding lookups:** When looking up clue-context
  embeddings, use `clue_context_phrases.csv` with a composite key
  (`clue_id`, `definition_wn`) rather than `clue_context_index.csv`
  alone, because `clue_id` is non-unique for double-definition clues.

### Notebook Version Control

- **`nbstripout` is installed as a git filter.** All committed notebooks have outputs
  and execution counts stripped automatically. Do not manually clear outputs before
  committing — the filter handles it.
- **`nbdime` is configured for diffs and merges.** Notebook diffs are rendered in a
  human-readable format rather than raw JSON.
- **Rendered HTML snapshots live in `docs/`.** After a clean `Restart & Run All`,
  render to HTML and commit. Naming convention: `NN_name_YYYY-MM-DD_HHMM.html`.
  See `docs/README.md` for the notebook index.
- **Coordinate edits in Slack.** Post before editing a notebook you don't own.

**Setup (one-time, every team member):**
```bash
pip install nbstripout nbdime
nbstripout --install
nbdime config-git --enable
```


## Source Data

George Ho's cryptic crossword clue dataset (660,613 clues).
- Download: https://cryptics.georgeho.org/data.db → saves as `data.sqlite3`
- The sqlite extraction to `clues_raw.csv` (and other `*_raw.csv` files) was
  done in the indicator_clustering NB00. We start from `clues_raw.csv`.
- **Do not use** Hans's `clues_single_word.csv` — it applied an overly
  restrictive single-word filter. We are re-deriving the cleaned dataset
  from `clues_raw.csv` to preserve multi-word definitions and answers that
  appear in WordNet, and to accommodate double-definition clues.

## Important Terminology

- **`clue`** — the raw clue text including the answer format in parentheses,
  e.g., "Plant in a garden party (5)"
- **`surface`** — the clue text with the answer format stripped, e.g.,
  "Plant in a garden party". This is what we embed. Victoria's cleaning
  notebook (Cell 14) creates this column using a regex that strips trailing
  `(N)` or `(N,M)` patterns.
- **`definition`** — the definition substring within the surface text.
  May be multi-word. For double-definition clues (where the `definition`
  field contains `/`-separated alternatives), we keep any definition that
  appears in WordNet.
- **`answer`** — the answer word or phrase. May be multi-word as long as
  it has a WordNet synset.

## What NOT to Change

- Do not modify anything in `indicator_clustering/` — that component is
  mostly complete and a teammate is actively working on it.
- Do not include wordplay type as a model feature (team decision; see DECISIONS.md).
- Do not use `WidthType.PERCENTAGE` in any generated reports (breaks in Google Docs).

## Key Reference Files

- `PLAN.md` — The 12-step pipeline plan. Start here to understand what to build.
- `NOTEBOOKS.md` — What already exists and what's missing.
- `DATA.md` — Schema and data flow.
- `DECISIONS.md` — Locked-in choices; do not revisit these.
- `FINDINGS.md` — Running log of findings as the pipeline is built.
- `supervised_learning_plan_v3.docx` — The authoritative design document.
  All `.md` files are derived from it.
- `CONTEXT.md` — Hans's writeup of his prior exploratory work. Useful
  background, but plan v3 supersedes his experimental design.
