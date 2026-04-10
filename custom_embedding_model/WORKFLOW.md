# Workflow — custom_embedding_model

This document describes the stage-by-stage workflow for the custom embedding
model project. Unlike the linear pipeline in `clue_misdirection`, this project
has a branching structure: one upstream dataset feeds multiple phrase
construction strategies (f's), each of which feeds its own triplet and
embedding pipeline. Reading this document before writing any code is essential.

The authoritative design document is `custom_embedding_model_design_v4.md`
(Google Drive, "Research Project - NLP CCC's"). This file summarizes the
workflow decisions derived from it.

---

## The Branching Structure

```
clues_raw.csv
    │
    ▼
Stage 0: Structural filtering
    │
    ▼
clues_filtered.csv
    │
    ▼
Stage 1: WordNet filtering + split assignment
    │
    ▼
clues_wn_filtered.csv (with split column)
    │                    │
    │                    ▼
    │           Stage 1b: g_stock f_clue embeddings (GPU)
    │           embeddings/g_stock/f_clue.npy + index
    │
    ▼
Stage 2: Per-f phrase construction + coverage measurement
    │
    ├── f_common_wndef branch
    │   ├── clues_wndef_filtered.csv (inherits split column)
    │   ├── vocabulary_wndef.csv / vocabulary_wndef_val.csv
    │   └── phrases/f_common_wndef.csv
    │
    └── f_common_wnex branch
        ├── clues_wnex_filtered.csv (inherits split column)
        ├── vocabulary_wnex.csv / vocabulary_wnex_val.csv
        └── phrases/f_common_wnex.csv
              │
              ▼
    Stage 3: Triplet construction + model training (per g_i)
              │
              ├── triplets/g1.csv (training split only)
              └── models/g1/ (weights → Google Drive)
                    │
                    ▼
    Stage 4: Embedding generation (per g_i, validation split only, GPU)
              └── embeddings/g1/f_<name>_val.npy
                    │
                    ▼
    Stage 5: Hypothesis testing (shared comparison notebook)
              └── FINDINGS.md entries
                    │
                    ▼
    Stage 6: Final evaluation (locked — test split only)
              └── FINDINGS.md entries
```

---

## Stage 0: Structural Filtering

**Notebook:** `notebooks/nb_00_structural_filtering.ipynb`
**Environment:** Local (CPU)
**Inputs:** `../data/clues_raw.csv`
**Outputs:** `data/clues_filtered.csv`

Filters `clues_raw.csv` to clues satisfying CCC structural constraints,
independent of any external linguistic resource:

1. Remove rows with missing clue, answer, or definition
2. Remove bracketed clues (mis-parsed entries)
3. Validate answer adheres to the length/format code in the clue
4. Parse double-definition clues (split on `/`, expand to multiple rows)
5. Verify definition appears as intact whole-word(s) in the surface text
6. Verify definition appears at the edge of the surface text

Retain all available columns including linguistic annotation (wordplay_type,
indicator) and provenance metadata (author, series, publisher). The precise
column list should be confirmed against `clues_raw.csv` during development.
See DATA.md for the target schema.

**Critical:** Do not apply any WordNet constraints here. Do not assign the
train/validate/test split here. This file is the shared upstream artifact
that all downstream branches read from.

---

## Stage 1: WordNet Filtering and Split Assignment

**Notebook:** `notebooks/nb_01_wn_filtering_and_split.ipynb`
**Environment:** Local (CPU)
**Inputs:** `data/clues_filtered.csv`
**Outputs:**
- `data/clues_wn_filtered.csv` (with `split` column)
- `data/clues_val.csv` (convenience subset)
- `data/vocabulary.csv`
- `data/vocabulary_val.csv`

### 1a: WordNet filtering

Filter `clues_filtered.csv` to rows where both the definition and answer have
at least one WordNet synset. This is the broadest WordNet constraint and serves
as the superset for all f-specific datasets. See DATA.md for the WordNet lookup
procedure (including article stripping and underscore conversion for multi-word
entries).

**Output:** `data/clues_wn_filtered.csv`

### 1b: Split assignment

Assign the 30/20/50 train/validate/test split at the level of unique
(definition, answer) pairs — not individual clue rows. Multiple clues can share
the same pair; all rows sharing a pair must land in the same split.

The split is assigned once and never changed. It is stored as a `split` column
in `clues_wn_filtered.csv` with values `'train'`, `'validate'`, `'test'`.
Use `random_state=42`.

All f-specific filtered datasets are subsets of `clues_wn_filtered.csv` and
inherit their split assignments from this column. They do not receive
independent splits. The actual resulting split fractions for each f-specific
dataset should be reported in FINDINGS.md as part of coverage measurement.

Save `data/clues_val.csv` as a convenience subset (rows where split ==
'validate').

### 1c: Full vocabulary construction

Build the unified vocabulary from all unique words appearing as either a
definition or answer in `clues_wn_filtered.csv`. Save as `data/vocabulary.csv`
with a `row` column giving the canonical position (0-indexed). This ordering
is fixed permanently — do not reorder.

Save `data/vocabulary_val.csv` as the subset of vocabulary words appearing in
validation-split clues, with its own canonical ordering.

### 1d: g_stock f_clue embedding generation (GPU)

**Script:** `scripts/embed_f_clue_gstock.py` + `scripts/embed_f_clue_gstock.sh`
**Environment:** Great Lakes (GPU)
**Inputs:** `data/clues_wn_filtered.csv`, `data/phrases/f_clue.csv`

Wait — phrase construction (Stage 2) must happen before this embedding step.
Therefore Stage 1d executes *after* Stage 2 has produced `phrases/f_clue.csv`.
It is listed under Stage 1 conceptually (because it uses the wn-filtered
dataset scope) but depends on Stage 2 output.

Encode all f_clue phrases for the full `clues_wn_filtered.csv` dataset using
g_stock. Save:
- `data/embeddings/g_stock/f_clue.npy`
- `data/embeddings/g_stock/f_clue_index.csv` (columns: clue_id, definition, row)

This is computed once for g_stock and reused by all f-specific analyses. When
a stricter f-specific dataset looks up f_clue embeddings, it filters this index
by clue_id and definition to find the relevant rows.

**If inclusion criteria are later expanded** (more rows added to
`clues_wn_filtered.csv`), generate embeddings only for the new rows and append
them to the existing `.npy` and index files. Do not regenerate existing rows.

Record runtime in FINDINGS.md.

---

## Stage 2: Per-f Phrase Construction and Coverage Measurement

**Notebook:** `notebooks/nb_02_phrase_construction.ipynb`
**Environment:** Local (CPU)
**Inputs:** `data/clues_wn_filtered.csv`, WordNet (via NLTK)
**Outputs:** Per-f filtered clue files, vocabulary files, phrase files

This notebook handles all WordNet-based f's (currently f_common_wndef and
f_common_wnex) in sequence. A separate notebook will be created for any
dictionary- or LLM-based f's if needed.

**Critical — strict f definitions:** Each f is defined only for words where
the required phrase can be constructed without any fallback. If a word lacks
the required resource, it is absent from that f's vocabulary and phrase file.
Do not add silent fallbacks. The identity of each f must remain unambiguous.

### f_clue construction

Build f_clue phrases for all rows in `clues_wn_filtered.csv` by wrapping the
definition word(s) in `<t></t>` delimiters within the surface text. Use
word-boundary matching to locate the definition. Drop any rows where the
definition appears more than once in the surface (ambiguous delimiter
placement).

**Output:** `data/phrases/f_clue.csv`
**Index:** (clue_id, definition) — one row per (clue, definition) pair

### f_common_wndef construction

For each word in `vocabulary.csv`, look up its most common WordNet synset
(sense index 0) and construct the phrase as `"<t>word</t>: <synset definition
text>"`. Every word with at least one synset has a valid f_common_wndef phrase,
so this f covers the full vocabulary.

Filter `clues_wn_filtered.csv` to rows where both definition and answer have
valid f_common_wndef phrases (this should be nearly all rows, but verify).
Save:
- `data/clues_wndef_filtered.csv` (inherits split column from clues_wn_filtered)
- `data/vocabulary_wndef.csv` (full; canonical ordering = index)
- `data/vocabulary_wndef_val.csv` (validation-split subset)
- `data/phrases/f_common_wndef.csv`

### f_common_wnex construction

For each word in `vocabulary.csv`, look up its most common WordNet synset
(sense index 0) and attempt to use its usage example as the phrase, wrapping
the target word in `<t></t>`. A phrase is valid only if a usage example exists
*and* the target word appears exactly once in it. Words without a valid usage
example have no f_common_wnex phrase and are absent from the wnex vocabulary.

Filter `clues_wn_filtered.csv` to rows where both definition and answer have
valid f_common_wnex phrases. Save:
- `data/clues_wnex_filtered.csv` (inherits split column)
- `data/vocabulary_wnex.csv` (full; canonical ordering = index)
- `data/vocabulary_wnex_val.csv` (validation-split subset)
- `data/phrases/f_common_wnex.csv`

### Coverage measurement (summary cell)

At each stage, report and record in the notebook summary cell:
- Rows remaining and fraction of previous stage
- Unique words with valid phrases and fraction of full vocabulary
- Resulting split fractions (train/validate/test) for each filtered dataset
- Number of validation-split clues and vocabulary words for each f

These numbers should also be recorded in FINDINGS.md under "Coverage
Measurements."

---

## Stage 3: Triplet Construction and Model Training

**Notebook:** `notebooks/nb_03_train_<g_name>.ipynb` (one per g_i)
**Script:** `scripts/train_<g_name>.py` + `scripts/train_<g_name>.sh`
**Environment:** Notebook locally for inspection; script on Great Lakes (GPU)
**Inputs:** Phrase files, `clues_<f>_filtered.csv` (training split only)
**Outputs:** `data/triplets/<g_name>.csv`, model weights (Google Drive),
  `models/<g_name>/README.md`

### Triplet construction

Assemble (anchor, positive, negative) text triplets from the training split
only. The validation and test splits must never appear in a triplet file.

Each triplet row contains three text strings:
- **anchor:** f_clue phrase for the definition in its clue context
- **positive:** f phrase for the positive target word
- **negative:** f phrase for the negative word (distractor)

The triplet is constructed from one clue row plus phrase lookups — not from
joins across multiple clue rows. Save the triplet dataset as a committed
artifact: `data/triplets/<g_name>.csv`. This file can be reused if retraining
with the same design.

Naming: the triplet file shares the name of the g model it was used to train
(e.g., `triplets/g1.csv` for g_1).

### Model training

Fine-tune g_stock using triplet margin loss (Schroff et al., 2015) on the
triplet dataset. Key parameters:
- Base model: `gabrielloiseau/CALE-MBERT-en`
- Margin: α = 1.0
- Use `random_state=42` for any stochastic elements
- Log training loss at regular intervals

Monitor training loss as a sanity check only — it does not tell you what the
model learned. Scientific evaluation happens in Stage 5.

Save model weights to Google Drive ("Research Project - NLP CCC's", owned by
Nathan). Commit `models/<g_name>/README.md` to the repo containing:
- Google Drive path to weights
- HuggingFace base model identifier and version hash
- Training script name and all hyperparameters
- Triplet file used
- Date trained
- Wall-clock runtime

Record runtime in FINDINGS.md.

---

## Stage 4: Embedding Generation

**Script:** `scripts/embed_<g_name>.py` + `scripts/embed_<g_name>.sh`
**Environment:** Great Lakes (GPU)
**Inputs:** Model weights (from Google Drive), phrase files, vocabulary files
**Outputs:** Validation-split embedding arrays under `data/embeddings/<g_name>/`

For each g_i being evaluated, generate embeddings for the validation split
only. Full-dataset embeddings are generated only for the final chosen g (Stage
6).

For each f used in the current experiments:
- Encode phrases for `vocabulary_<f>_val.csv` words using g_i
- Save as `data/embeddings/<g_i>/f_<name>_val.npy`
- The index is `vocabulary_<f>_val.csv` itself — no separate index file needed

For f_clue:
- Encode f_clue phrases for validation-split clues using g_i
- Save as `data/embeddings/<g_i>/f_clue_val.npy`
- Save `data/embeddings/<g_i>/f_clue_val_index.csv` (clue_id, definition, row)

After the job completes, scp output files back to local machine before
proceeding to Stage 5.

Record runtime in FINDINGS.md alongside the vocabulary size, clue count, and
Great Lakes partition used.

---

## Stage 5: Hypothesis Testing

**Primary notebook:** `notebooks/nb_05_hypothesis_testing.ipynb`
**Per-g exploration notebooks (optional):** `notebooks/nb_05_explore_<g_name>.ipynb`
**Environment:** Local (CPU)
**Inputs:** Embedding arrays from `data/embeddings/`
**Outputs:** FINDINGS.md entries, figures in `outputs/figures/`

### Shared comparison notebook

This notebook accumulates results across all trained g's and is added to as
new models are trained. It should not be rerun from scratch each time — add
new sections for each new g_i.

For each g_i, compute and record:

**ATE on validation set:**
For each (clue, definition, answer) pair in the validation split:
- delta = cos_sim(g(f_clue(def)), g(f(ans))) − cos_sim(g(f(def)), g(f(ans)))
- Report: mean delta, median delta, % pairs with negative delta, 95% CI

Compare g_i against g_stock. A less negative ATE under g_i indicates the
model is partially counteracting misdirection.

**Cross-f generalization test (for g_1 specifically):**
Compute cos_sim(g_1(f_common_wnex(word)), g_stock(f_common_wnex(word))) for
validation-split wnex vocabulary words. Compare against g_stock baseline.
If g_1 compresses f_common_wnex phrases (even though it was trained on
f_common_wndef), that is evidence of semantic generalization. If not, that is
evidence of format-specific overfitting.

Report the size of the wnex subset alongside results.

Record all numerical results in FINDINGS.md.

### Per-g exploration notebooks

For deeper investigation of a specific g_i (neighborhood structure, which
words moved most, failure cases), create a separate notebook named
`nb_05_explore_<g_name>.ipynb`. Move to `notebooks/archive/` when done with
that model.

---

## Stage 6: Final Evaluation (Locked)

**Trigger:** A final g has been chosen and the decision documented in
DECISIONS.md.

**Script:** `scripts/embed_final_<g_name>.py` + corresponding `.sh`
**Notebook:** `notebooks/nb_06_final_evaluation.ipynb`
**Environment:** Embedding generation on Great Lakes (GPU); evaluation locally

**Do not begin this stage until the final g is documented in DECISIONS.md.**
Do not load, inspect, or embed test-split data at any earlier stage.

Generate full-dataset embeddings for the chosen g:
- `data/embeddings/<g_name>/f_<name>.npy` (full vocabulary, no _val suffix)
- `data/embeddings/<g_name>/f_clue.npy` + index (full clues_wn_filtered scope)

Run ATE and any other evaluation metrics on the test split. Record all results
in FINDINGS.md.

---

## What Gets Computed When: Summary Table

| Artifact | Scope | Stage | Environment |
|----------|-------|-------|-------------|
| `clues_filtered.csv` | All structurally valid clues | 0 | Local |
| `clues_wn_filtered.csv` + split | All clues with ≥1 WN synset | 1 | Local |
| `clues_val.csv` | Validation subset of above | 1 | Local |
| `vocabulary.csv` / `_val` | Full WN vocab / val subset | 1 | Local |
| `phrases/f_clue.csv` | All wn_filtered clues | 2 | Local |
| `embeddings/g_stock/f_clue.npy` | Full wn_filtered clues | 1d (after 2) | Great Lakes |
| `clues_<f>_filtered.csv` + vocab + phrases | Per-f subset | 2 | Local |
| `triplets/<g_name>.csv` | Training split only | 3 | Local |
| Model weights | — | 3 | Great Lakes → Google Drive |
| `embeddings/<g_i>/f_<name>_val.npy` | Validation split | 4 | Great Lakes |
| ATE results, figures | — | 5 | Local |
| Full-dataset embeddings for final g | All splits | 6 | Great Lakes |
| Test-set evaluation | Test split | 6 | Local |

---

## Critical Rules

- **The test set is locked.** Do not load, inspect, or embed test-split data
  until Stage 6, after the final g is chosen and documented in DECISIONS.md.
- **One split.** The 30/20/50 split is assigned once in Stage 1 on
  `clues_wn_filtered.csv`. All downstream datasets inherit it. Do not reassign.
- **No fallbacks in f functions.** A word either has a valid phrase for a
  given f or it does not. Absent words are absent — they are not substituted
  with a different construction method.
- **Vocabulary ordering is permanent.** Never reorder a vocabulary file after
  creation. The row order is the index for all corresponding embedding arrays.
- **All phrase, vocabulary, and embedding files are committed artifacts.**
  Generate once, save, reuse. Do not regenerate without a documented reason
  in DECISIONS.md.
- **Do not modify** `clue_misdirection/`, `indicator_clustering/`, or
  `ccc-project/data/`.
