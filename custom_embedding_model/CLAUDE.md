# Claude Code Configuration — custom_embedding_model

## Project Summary

This is the **custom embedding model component** of an ongoing NLP research
project investigating semantic misdirection in cryptic crossword clues. It
builds on a completed Milestone II project (SIADS 696, Winter 2026) in which
we found that embedding a definition word in the context of its cryptic clue
roughly doubles its median retrieval rank among candidate answer words —
evidence that the clue's surface text misleads the embedding.

The current phase investigates whether we can fine-tune our embedding model
(CALE) using triplet margin loss so that it becomes more adept at embedding
a clue-contextualized definition as a sense more aligned with the clue's
answer. The workflow is experimental rather than a linear pipeline: we try
different phrase construction strategies (**f functions**) and different
triplet designs, training multiple candidate models (**g models**), then
evaluate them on a held-out validation set before finally evaluating the
chosen model on a locked test set.

The longer-term goal — building a model that measures misdirection directly
using embeddings from the fine-tuned CALE — will live in a separate directory.
This component focuses entirely on the fine-tuning step.

**The authoritative design document** for this component is
`custom_embedding_model_design_v5.md` (in the project Google Drive). Read it
before making any significant decisions about data, phrase construction,
triplet design, or file organization.

## Team

- **Victoria Winters**: Research questions, project management, phrase
  construction, embeddings, AI-assisted notebook development and repo management
- **Nathan Cantwell**: Triplet margin loss implementation (NB 09), model
  training, Great Lakes compute

Faculty Advisor: Dr. Kevyn Collins-Thompson (University of Michigan)

## Repo Structure

```
ccc-project/
├── data/                              # SHARED — do not modify
│   ├── data.sqlite3                   # George Ho source DB (660,613 clues)
│   ├── clues_raw.csv                  # Extracted from sqlite; shared input
│   ├── puzzle_metadata.csv            # Produced by puzzle_metadata.ipynb
│   ├── clues_filtered.csv             # Produced by structural_filtering.ipynb
│   └── publisher_lookup.csv           # Lookup table for metadata extraction
├── notebooks/                         # Shared upstream notebooks (order-independent)
│   ├── puzzle_metadata.ipynb          # Extracts publisher, series, setter, puzzle_no, clue_direction
│   ├── structural_filtering.ipynb     # Filters clues_raw.csv to valid CCC clues
│   └── clue_utils.py                  # Shared definition-finding and delimiter-placement logic
├── DATA_RAW.md                        # Raw data schema and metadata extraction logic
├── custom_embedding_model/            # ← YOU ARE HERE
│   ├── CLAUDE.md                      # This file
│   ├── WORKFLOW.md                    # Branching workflow and stage dependencies
│   ├── DATA.md                        # Data dictionary, file registry, schema
│   ├── NOTEBOOKS.md                   # Notebook inventory and status
│   ├── DECISIONS.md                   # Locked-in decisions; do not revisit
│   ├── FINDINGS.md                    # Running log of coverage measurements and results
│   ├── data/
│   │   ├── filtered_split/
│   │   │   └── wn_synset/             # Rows where definition + answer have ≥1 WN synset
│   │   │       ├── clues_wn_filtered.csv  # Full scope, with split column
│   │   │       ├── clues_val.csv          # Validation rows (convenience subset)
│   │   │       ├── vocabulary.csv         # All wn_synset words; canonical ordering = index
│   │   │       ├── vocabulary_val.csv     # Validation-split subset; IS the index
│   │   │       ├── clue_phrases/          # f_clue only; does not further filter rows
│   │   │       │   └── f_clue.csv         # Indexed by (clue_id, definition)
│   │   │       ├── wndef/                 # Further filtered: words with WN definition
│   │   │       │   ├── clues_wndef_filtered.csv
│   │   │       │   ├── vocabulary_wndef.csv
│   │   │       │   ├── vocabulary_wndef_val.csv
│   │   │       │   └── f_common_wndef.csv
│   │   │       └── wnex/                  # Further filtered: words with WN usage example
│   │   │           ├── clues_wnex_filtered.csv
│   │   │           ├── vocabulary_wnex.csv
│   │   │           ├── vocabulary_wnex_val.csv
│   │   │           └── f_common_wnex.csv
│   │   ├── triplets/                  # One file per g; spans subset conditions and f's
│   │   │   ├── g1_train.csv
│   │   │   └── g1_train_meta.json     # Provenance: which f for each triplet role, row counts
│   │   └── embeddings/                # One subfolder per g model
│   │       ├── g_stock/               # Stock CALE embeddings
│   │       │   ├── f_clue.npy         # Full wn_synset clues
│   │       │   ├── f_clue_index.csv   # (clue_id, definition, row)
│   │       │   ├── f_common_wndef.npy # Indexed by vocabulary_wndef.csv
│   │       │   └── f_common_wnex.npy  # Indexed by vocabulary_wnex.csv
│   │       └── g1/                    # Fine-tuned g_1 embeddings (validation only)
│   │           ├── f_clue_val.npy
│   │           ├── f_clue_val_index.csv
│   │           ├── f_common_wndef_val.npy  # Indexed by vocabulary_wndef_val.csv
│   │           └── f_common_wnex_val.npy   # Indexed by vocabulary_wnex_val.csv
│   ├── models/
│   │   ├── g_stock/
│   │   │   └── README.md              # HuggingFace ID + version; no weights stored
│   │   └── g1/
│   │       └── README.md              # Google Drive path, hyperparams, date trained
│   ├── notebooks/
│   │   └── archive/
│   ├── scripts/                       # SLURM scripts for GPU work on Great Lakes
│   ├── outputs/                       # Figures, ATE tables, evaluation summaries
│   └── requirements.txt
├── clue_misdirection/                 # Complete — do not modify
└── indicator_clustering/              # Complete — do not modify
```
│   ├── CLAUDE.md                      # This file
│   ├── WORKFLOW.md                    # Branching workflow and stage dependencies
│   ├── DATA.md                        # Data dictionary, file registry, schema
│   ├── NOTEBOOKS.md                   # Notebook inventory and status
│   ├── DECISIONS.md                   # Locked-in decisions; do not revisit
│   ├── FINDINGS.md                    # Running log of coverage measurements and results
│   ├── data/
│   │   ├── filtered_split/
│   │   │   └── wn_synset/             # Rows where definition + answer have ≥1 WN synset
│   │   │       ├── clues_wn_filtered.csv  # Full scope, with split column
│   │   │       ├── clues_val.csv          # Validation rows (convenience subset)
│   │   │       ├── vocabulary.csv         # All wn_synset words; canonical ordering = index
│   │   │       ├── vocabulary_val.csv     # Validation-split subset; IS the index
│   │   │       ├── clue_phrases/          # f_clue only; does not further filter rows
│   │   │       │   └── f_clue.csv         # Indexed by (clue_id, definition)
│   │   │       ├── wndef/                 # Further filtered: words with WN definition
│   │   │       │   ├── clues_wndef_filtered.csv
│   │   │       │   ├── vocabulary_wndef.csv
│   │   │       │   ├── vocabulary_wndef_val.csv
│   │   │       │   └── f_common_wndef.csv
│   │   │       └── wnex/                  # Further filtered: words with WN usage example
│   │   │           ├── clues_wnex_filtered.csv
│   │   │           ├── vocabulary_wnex.csv
│   │   │           ├── vocabulary_wnex_val.csv
│   │   │           └── f_common_wnex.csv
│   │   ├── triplets/                  # One file per g; spans subset conditions and f's
│   │   │   ├── g1_train.csv
│   │   │   └── g1_train_meta.json     # Provenance: which f for each triplet role, row counts
│   │   └── embeddings/                # One subfolder per g model
│   │       ├── g_stock/               # Stock CALE embeddings
│   │       │   ├── f_clue.npy         # Full wn_synset clues
│   │       │   ├── f_clue_index.csv   # (clue_id, definition, row)
│   │       │   ├── f_common_wndef.npy # Indexed by vocabulary_wndef.csv
│   │       │   └── f_common_wnex.npy  # Indexed by vocabulary_wnex.csv
│   │       └── g1/                    # Fine-tuned g_1 embeddings (validation only)
│   │           ├── f_clue_val.npy
│   │           ├── f_clue_val_index.csv
│   │           ├── f_common_wndef_val.npy  # Indexed by vocabulary_wndef_val.csv
│   │           └── f_common_wnex_val.npy   # Indexed by vocabulary_wnex_val.csv
│   ├── models/
│   │   ├── g_stock/
│   │   │   └── README.md              # HuggingFace ID + version; no weights stored
│   │   └── g1/
│   │       └── README.md              # Google Drive path, hyperparams, date trained
│   ├── notebooks/
│   │   └── archive/
│   ├── scripts/                       # SLURM scripts for GPU work on Great Lakes
│   ├── outputs/                       # Figures, ATE tables, evaluation summaries
│   └── requirements.txt
├── clue_misdirection/                 # Complete — do not modify
└── indicator_clustering/              # Complete — do not modify
```

## Tech Stack

- **Python 3.10+** (Conda base environment)
- **Embedding model:** `gabrielloiseau/CALE-MBERT-en` (CALE, 1024-dim) via
  `sentence-transformers`. Uses `<t></t>` delimiters to focus embeddings on
  a target word within a passage of text.
- **Key libraries:** pandas, numpy, nltk (WordNet), matplotlib, seaborn,
  sentence-transformers, torch
- **Compute:** GPU steps (model training, embedding generation) run on UM
  Great Lakes cluster. All other steps run locally on CPU.
- **Data formats:** CSVs for tabular data and phrase files, `.npy` for dense
  embedding arrays
- **Model weights:** Fine-tuned model weights are stored in the shared Google
  Drive folder **"Research Project - NLP CCC's"** (owned by Nathan). The
  `models/<g_name>/` directories in the repo are placeholders only, each
  containing a `README.md` with the Google Drive path, HuggingFace base model
  identifier and version hash, training script and hyperparameters used, and
  date trained.

## Important Terminology

Understanding the two-step chain from word to embedding is essential for all
work in this component.

- **f (phrase construction strategy)** — a procedure that takes a word and
  returns a tagged passage of text ready for CALE embedding. Each strategy is
  named descriptively: `f_<sense>_<construction>`. The passage is *not* an
  embedding — it is the raw text input to a g model. Different f's encode
  different assumptions about which sense of a word to capture and how
  naturally to express it.

- **g (embedding model)** — takes a tagged passage produced by f and returns
  a 1024-dimensional embedding vector. `g_stock` is the unmodified CALE model.
  Fine-tuned variants are `g_1`, `g_2`, etc. The full chain is:
  word → f(word) → g(f(word)) → 1024-dim embedding.

- **f_clue** — the special phrase construction strategy for clue-contextualized
  definitions. Takes a clue's surface text and wraps the definition word(s) in
  `<t></t>` delimiters. Indexed by (clue_id, definition) rather than by word.
  g(f_clue(definition)) is the "treatment" embedding — what the model sees when
  the clue's misleading surface reading is present.

- **T (triplet)** — a training example of the form (Anchor, Positive,
  Negative), where the Anchor is always g_stock(f_clue(definition)). `T_1`
  denotes the initial triplet design from NB 09.

- **ATE (Average Treatment Effect)** — a diagnostic measure decomposable into
  T=0 (decontextualized similarity) and T=1 (clue-contextualized similarity):
  ATE = mean(T=1 − T=0) = mean(cos_sim(g(f_clue(def)), g(f(ans))) −
  cos_sim(g(f(def)), g(f(ans)))). A negative ATE indicates misdirection. ATE
  changing under a fine-tuned g confirms the model learned something; the T=0
  and T=1 components reveal *what* it learned. Always interpret ATE through
  its components, not as a standalone optimization target.

- **`clue`** — the raw clue text including the answer format in parentheses,
  e.g., "Plant in a garden party (5)"
- **`surface`** — the clue text with the answer format stripped, e.g.,
  "Plant in a garden party". This is what f_clue operates on.
- **`definition`** — the definition substring within the surface text. May be
  multi-word.
- **`answer`** — the answer word or phrase. May be multi-word as long as it
  has a WordNet synset.
- **`vocabulary`** — the unified set of all unique words appearing as
  definitions or answers in the filtered dataset. Definitions and answers share
  one vocabulary and one set of phrase/embedding files.

## File and Naming Conventions

### Data directory organization

Filtered clue files, vocabulary files, and phrase files are organized under
`data/filtered_split/<scope>/`. Each scope directory represents a specific
filtering constraint applied on top of `clues_filtered.csv`. Within a scope,
subset directories (e.g. `wndef/`, `wnex/`) each correspond to one phrase
construction strategy and contain the clue subset, vocabulary files, and
phrase file for that strategy. `clue_phrases/` is a special subdirectory
holding `f_clue.csv`, which belongs to the full scope and does not further
filter rows.

Triplet files live in `data/triplets/`, separate from both `filtered_split/`
and `embeddings/`, because a triplet spans multiple subset conditions and f's.
Each triplet file `<g_name>.csv` has a companion `<g_name>_meta.json`
documenting which f was used for each triplet role (anchor, positive, negative),
the source paths of the phrase files used, and the resulting row counts.

Embedding files live in `data/embeddings/<g_name>/`, organized by g model
rather than by subset. This reflects how embeddings are used: hypothesis
testing loads all embeddings for a given g at once. The phrase filename
(e.g. `f_common_wnex_val.npy`) identifies the subset and f; the vocabulary
file it is indexed by is documented in `DATA.md`.

### Notebook naming

Notebooks are named starting with two-digit stage numbers and no prefix:
`01_wn_filtering_and_split.ipynb`, `02_phrase_construction.ipynb`, etc. Do
not use an `nb_` prefix.



- **No suffix** = full dataset scope (e.g., `f_clue.npy`, `f_common_wndef.npy`)
- **`_val` suffix** = validation split only (e.g., `f_clue_val.npy`,
  `f_common_wndef_val.npy`)

This applies to both phrase files and embedding files. Phrase files are always
generated for the full dataset; the `_val` suffix is used only for embeddings.

### Vocabulary files as indexes

Vocabulary files (`vocabulary.csv`, `vocabulary_val.csv`, etc.) use a fixed
canonical row ordering established at creation time and never changed. This
ordering is the index for all corresponding embedding arrays — the row number
of a word in `vocabulary.csv` is its row in any `.npy` embedding array for
that vocabulary. Do not reorder vocabulary files.

For f_clue embeddings, an explicit `_index.csv` companion file is always
stored alongside the `.npy` file, mapping (clue_id, definition) composite
keys to row positions.

### Strict f definitions — no fallbacks

Each f is defined only for words where the required phrase can be constructed
without any fallback. If a word lacks the required resource (e.g., no WordNet
usage example for f_common_wnex), that word is simply absent from that f's
vocabulary file and phrase file. Do not silently substitute a different
construction method. The identity of each f must remain unambiguous.

### All phrase, vocabulary, embedding, and triplet files are committed artifacts

Generate once, save, reuse. Nothing is recomputed on the fly. Do not
regenerate existing files unless there is a documented bug fix, in which
case update DECISIONS.md and FINDINGS.md accordingly.

## Coding Standards

### Notebook Header

Every notebook must start with a markdown cell containing:

```
# [Title]

**Primary author:** [who wrote this notebook]

**Builds on:**
- *[Notebook or document name]* ([Author] — brief description of what was drawn from it)
- ...

**Prompt engineering:** Victoria
**AI assistance:** Claude / Claude Code (Anthropic)
**Environment:** [Local / Great Lakes / Colab]

[Brief purpose statement: what this notebook does, what it reads, what it
produces, and how it fits into the overall workflow.]
```

### Notebook Summary Cell

Every notebook must end with a markdown cell that:
- Summarizes what was done
- Reports coverage statistics where applicable (how many words / rows remain
  after each constraint, as a count and fraction of the previous stage)
- States the size and location of all output files produced
- Highlights any findings worth recording in FINDINGS.md
- Records wall-clock runtime for any computationally significant steps

### Notebook Style

Write markdown cells as if the reader is new to the project — each cell should
fully explain the purpose and reasoning of the Python code that follows. At the
same time, eliminate redundancy: do not restate what the code makes obvious, do
not repeat content already established in a prior cell, and do not restate in
an inline comment what the preceding markdown cell just said. Justify length by
conceptual necessity, not by comprehensiveness for its own sake. Code cells
open with a `# ===` banner; inline comments are appropriate where logic is
non-obvious. See `custom_embedding_model/notebooks/archive/09_learned_g_misdirection.ipynb`
for an example of a well-executed notebook in this project.

### General Standards

- **Use `pathlib`** for all file paths.
- **Pin random seeds** (`random_state=42`) for reproducibility everywhere.
- **No hardcoded absolute paths.** Use relative paths from the notebook
  directory, or `DATA_DIR` / `OUTPUT_DIR` / `MODELS_DIR` variables defined
  at the top of each notebook.
- **`keep_default_na=False`:** Always use this flag when loading any CSV that
  contains word, definition, or answer columns. The word "nan" (meaning
  grandmother) is a valid crossword entry; without this flag, pandas silently
  converts it to `NaN`.
- **Validate array shapes** after loading `.npy` files: assert that the number
  of rows matches the length of the corresponding vocabulary or index file.
- **Record runtimes:** Use `time.time()` or equivalent to record wall-clock
  time for any step taking more than a few seconds. Print and include in the
  summary cell.
- **Figures:** Save all figures to `outputs/figures/` as PNG (300 dpi).
- **Test set is locked.** Never load, inspect, or embed test-set data until a
  final model has been chosen and that decision is documented in DECISIONS.md.

### GPU Scripts

Computationally intensive steps (model training, embedding generation) run as
standalone `.py` scripts submitted via SLURM on Great Lakes, not inside
notebooks. Each script should:
- Accept command-line arguments for key parameters (model path, input file,
  output directory, batch size, sample flag)
- Print progress and intermediate results to stdout (captured in SLURM logs)
- Record and print total wall-clock runtime at completion
- Save outputs atomically (write to a temp path, then rename) to avoid
  partial files if a job is killed

**Training scripts** must additionally (per Decision 24):
- Accept a `--val-input` argument for the validation triplet file
- Compute validation loss at the end of each epoch (same loss function,
  `model.eval()`, `torch.no_grad()`)
- Log both training and validation loss per epoch to stdout and to the
  structured training log
- Save per-epoch model checkpoints

After a GPU job completes, scp the output files back locally before proceeding
with analysis notebooks.

### Notebook Version Control

- **`nbstripout` is installed as a git filter.** All committed notebooks have
  outputs and execution counts stripped automatically. Do not manually clear
  outputs before committing — the filter handles it.
- **`nbdime` is configured for diffs and merges.** Notebook diffs are rendered
  in a human-readable format rather than raw JSON.
- **Rendered HTML snapshots** should be saved after any clean `Restart & Run
  All` run. Naming convention: `NB_name_YYYY-MM-DD.html`. Commit alongside
  the notebook.
- **Coordinate edits in Slack** before editing a notebook you don't own.

**Setup (one-time, every team member):**
```bash
pip install nbstripout nbdime
nbstripout --install
nbdime config-git --enable
```

## Source Data

George Ho's cryptic crossword clue dataset (660,613 clues).
- Download: https://cryptics.georgeho.org/data.db → saves as `data.sqlite3`
- `clue_id` originates in the George Ho sqlite database and is unique per row
  in `clues_raw.csv`. After multi-definition expansion in
  `structural_filtering.ipynb`, it is no longer unique in `clues_filtered.csv`.
- We start from `clues_raw.csv`, extracted from the sqlite DB by
  `indicator_clustering/NB00`, living in `ccc-project/data/`.
- The shared upstream pipeline produces two artifacts this component reads from:
  - `ccc-project/data/clues_filtered.csv` — columns: `clue_id`, `surface`,
    `definition`, `answer`
  - `ccc-project/data/puzzle_metadata.csv` — joined in when needed
- **Do not use** any filtered datasets from `clue_misdirection/` as inputs.

## What NOT to Change

- **Do not modify** anything in `clue_misdirection/` or
  `indicator_clustering/` — both are complete.
- **Do not modify** the shared `ccc-project/data/` directory.
- **Do not regenerate** `clues_filtered.csv` or `puzzle_metadata.csv` from
  within this component.
- **Do not touch the test set** until a final model is chosen and the decision
  is documented in DECISIONS.md.
- **Do not add fallbacks inside an f function.** If a word lacks the required
  resource, it is absent from that f — period.
- **Do not reorder vocabulary files.** The canonical ordering is the index for
  all corresponding embedding arrays.
- **Do not revisit decisions** recorded in DECISIONS.md without first
  discussing with the team.
- **Do not regenerate triplet files** without updating the companion
  `_meta.json` and documenting the reason in DECISIONS.md.

## Key Reference Files

- `WORKFLOW.md` — the branching workflow: what gets computed once, what gets
  computed per-f, and what gets computed per-g. Start here to understand the
  overall shape of the project.
- `DATA.md` — data dictionary, vocabulary file registry, embedding file
  registry, and schema for each data artifact.
- `NOTEBOOKS.md` — inventory of notebooks and scripts, their roles, and
  current status.
- `DECISIONS.md` — locked-in choices. Do not revisit these.
- `FINDINGS.md` — running log of coverage measurements, ATE results, and
  experimental findings as the project progresses.
- `ccc-project/DATA_RAW.md` — raw data schema and puzzle metadata extraction
  logic. Shared across all components.
- `custom_embedding_model_design_v5.md` (in planning/)— the authoritative
  design document. All `.md` files in this directory are derived from it.
