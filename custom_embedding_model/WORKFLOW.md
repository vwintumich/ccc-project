# Workflow — custom_embedding_model

This document describes the stage-by-stage workflow for the custom embedding
model project. Unlike the linear pipeline in `clue_misdirection`, this project
has a branching structure: one upstream dataset feeds multiple phrase
construction strategies (f's), each of which feeds its own triplet and
embedding pipeline. Reading this document before writing any code is essential.

The authoritative design document is `custom_embedding_model_design_v5.md`
(in `planning/`). This file summarizes the workflow decisions derived from it.

**Upstream dependency:** This component reads from two shared artifacts
produced by `ccc-project/notebooks/`:
- `ccc-project/data/clues_filtered.csv` — columns: `clue_id`, `surface`,
  `definition`, `answer`
- `ccc-project/data/puzzle_metadata.csv` — joined in when needed

It also imports `ccc-project/notebooks/clue_utils.py` for definition-finding
and delimiter-placement logic used in f_clue phrase construction.

Do not regenerate these files from within this component.

---

## The Branching Structure

```
ccc-project/data/clues_filtered.csv        (shared upstream artifact)
    │
    ▼
Stage 1: WordNet filtering + split assignment
    │
    ▼
filtered_split/wn_synset/
    ├── clues_wn_filtered.csv (with split column)
    ├── vocabulary.csv / vocabulary_val.csv
    └── clue_phrases/
        └── f_clue.csv
    │
    ▼
Stage 2: Per-f phrase construction + coverage measurement
    │
    ├── wndef/
    │   ├── clues_wndef_filtered.csv
    │   ├── vocabulary_wndef.csv / vocabulary_wndef_val.csv
    │   └── f_common_wndef.csv
    │
    └── wnex/
        ├── clues_wnex_filtered.csv
        ├── vocabulary_wnex.csv / vocabulary_wnex_val.csv
        └── f_common_wnex.csv
              │
              ▼
    Stage 3: Triplet construction + model training (per g_i)
              │
              ├── triplets/g1_train.csv + g1_train_meta.json
              └── models/g1/ (weights → Google Drive)
                    │
                    ▼
    Stage 4: Embedding generation (per g_i, validation split only, GPU)
              └── embeddings/g1/f_<n>_val.npy
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

## Stage 1: WordNet Filtering and Split Assignment

**Notebook:** `notebooks/01_wn_filtering_and_split.ipynb`
**Environment:** Local (CPU)
**Inputs:** `../../data/clues_filtered.csv`
**Outputs:**
- `data/filtered_split/wn_synset/clues_wn_filtered.csv` (with `split` column)
- `data/filtered_split/wn_synset/clues_val.csv` (convenience subset)
- `data/filtered_split/wn_synset/vocabulary.csv`
- `data/filtered_split/wn_synset/vocabulary_val.csv`

### 1a: WordNet filtering

Filter `clues_filtered.csv` to rows where both the definition and answer have
at least one WordNet synset. This is the broadest WordNet constraint and serves
as the superset for all f-specific datasets. See DATA.md for the WordNet lookup
procedure (including article stripping and underscore conversion for multi-word
entries).

**Output:** `data/filtered_split/wn_synset/clues_wn_filtered.csv`

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

Save `data/filtered_split/wn_synset/clues_val.csv` as a convenience subset
(rows where split == 'validate').

### 1c: Full vocabulary construction

Build the unified vocabulary from all unique words appearing as either a
definition or answer in `clues_wn_filtered.csv`. Save as
`data/filtered_split/wn_synset/vocabulary.csv` with a `row` column giving the
canonical position (0-indexed). This ordering is fixed permanently — do not
reorder.

Save `data/filtered_split/wn_synset/vocabulary_val.csv` as the subset of
vocabulary words appearing in validation-split clues, with its own canonical
ordering.

### 1d: g_stock f_clue embedding generation (GPU)

**Script:** `scripts/embed_f_clue_gstock.py` + `scripts/embed_f_clue_gstock.sh`
**Environment:** Great Lakes (GPU)
**Inputs:** `data/filtered_split/wn_synset/clues_wn_filtered.csv`,
  `data/filtered_split/wn_synset/clue_phrases/f_clue.csv`

Wait — phrase construction (Stage 2) must happen before this embedding step.
Therefore Stage 1d executes *after* Stage 2 has produced `f_clue.csv`.
It is listed under Stage 1 conceptually (because it uses the wn_synset
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

**Notebook naming convention:** `02_phrase_construction_<resource>.ipynb` —
one notebook per resource family. Each resource family has its own filtering
notebook (Stage 1) and phrase construction notebook (Stage 2). If a future
resource does not require WordNet, it branches from `clues_filtered.csv`
with its own filtering step and produces a sibling scope directory under
`data/filtered_split/`.

**WordNet notebook:** `notebooks/02_phrase_construction_wn.ipynb`
**Environment:** Local (CPU)
**Inputs:** `data/filtered_split/wn_synset/clues_wn_filtered.csv`, WordNet (via NLTK)
**Outputs:** Per-f subset directories under `data/filtered_split/wn_synset/`,
  plus `data/filtered_split/wn_synset/clue_phrases/f_clue.csv`

This notebook handles all WordNet-based f's (currently f_common_wndef and
f_common_wnex) in sequence.

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

**Output:** `data/filtered_split/wn_synset/clue_phrases/f_clue.csv`
**Index:** (clue_id, definition) — one row per (clue, definition) pair

Note: `clue_phrases/` does not further filter the `wn_synset` row scope.
All rows in `clues_wn_filtered.csv` that have unambiguous definition placement
are represented here.

### f_common_wndef construction

For each word in the full vocabulary, look up its most frequent WordNet sense
(index 0) and construct `"<t>word</t>: <WordNet definition text>"`. Save rows
with valid phrases only under `data/filtered_split/wn_synset/wndef/`:
- `clues_wndef_filtered.csv` (inherits split column)
- `vocabulary_wndef.csv` (full; canonical ordering = index)
- `vocabulary_wndef_val.csv` (validation-split subset)
- `f_common_wndef.csv`

### f_common_wnex construction

For each word in the full vocabulary, look up its most frequent WordNet sense
and use the WordNet usage example sentence, with the target word wrapped in
`<t></t>`. Defined only for words with a valid usage example where the target
word appears exactly once. Save under `data/filtered_split/wn_synset/wnex/`:
- `clues_wnex_filtered.csv` (inherits split column)
- `vocabulary_wnex.csv` (full; canonical ordering = index)
- `vocabulary_wnex_val.csv` (validation-split subset)
- `f_common_wnex.csv`

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

**Notebook:** `notebooks/03_train_<g_name>.ipynb` (one per g_i)
**Script:** `scripts/train_<g_name>.py` + `scripts/train_<g_name>.sh`
**Environment:** Notebook locally for inspection; script on Great Lakes (GPU)
**Inputs:** Phrase files from `data/filtered_split/wn_synset/`,
  `clues_<f>_filtered.csv` (training split only)
**Outputs:** `data/triplets/<g_name>.csv`, `data/triplets/<g_name>_meta.json`,
  model weights (Google Drive), `models/<g_name>/README.md`

### Triplet construction

Assemble (anchor, positive, negative) text triplets from the training split
only. The validation and test splits must never appear in a triplet file.

Each triplet row contains three fully constructed phrase strings:
- **anchor:** f_clue phrase for the definition in its clue context
- **positive:** f phrase for the positive target word
- **negative:** f phrase for the negative word (distractor)

A triplet file spans multiple subset directories and f's — the anchor comes
from `clue_phrases/`, while the positive and negative come from `wndef/` or
`wnex/` or another subset. The training rows are the intersection of clues
that have valid phrases under all three f's used. Save as a committed
artifact: `data/triplets/<g_name>.csv`.

Save a companion `data/triplets/<g_name>_meta.json` documenting:
- Which f was used for each triplet role (anchor, positive, negative)
- Source paths of the phrase files used
- Number of rows in the triplet file
- Split used (always `'train'`)
- `random_state`

This metadata file is the authoritative provenance record for the triplet
dataset. `models/<g_name>/README.md` references it rather than duplicating it.

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
- Save as `data/embeddings/<g_i>/f_<n>_val.npy`
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

**Primary notebook:** `notebooks/05_hypothesis_testing.ipynb`
**Per-g exploration notebooks (optional):** `notebooks/05_explore_<g_name>.ipynb`
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
`05_explore_<g_name>.ipynb`. Move to `notebooks/archive/` when done with
that model.

---

## Stage 6: Final Evaluation (Locked)

**Trigger:** A final g has been chosen and the decision documented in
DECISIONS.md.

**Script:** `scripts/embed_final_<g_name>.py` + corresponding `.sh`
**Notebook:** `notebooks/06_final_evaluation.ipynb`
**Environment:** Embedding generation on Great Lakes (GPU); evaluation locally

**Do not begin this stage until the final g is documented in DECISIONS.md.**
Do not load, inspect, or embed test-split data at any earlier stage.

Generate full-dataset embeddings for the chosen g:
- `data/embeddings/<g_name>/f_<n>.npy` (full vocabulary, no _val suffix)
- `data/embeddings/<g_name>/f_clue.npy` + index (full clues_wn_filtered scope)

Run ATE and any other evaluation metrics on the test split. Record all results
in FINDINGS.md.

---

## What Gets Computed When: Summary Table

| Artifact | Location | Stage | Environment |
|----------|----------|-------|-------------|
| `puzzle_metadata.csv` | `ccc-project/data/` | Shared | Local |
| `clues_filtered.csv` | `ccc-project/data/` | Shared | Local |
| `clues_wn_filtered.csv` + split | `filtered_split/wn_synset/` | 1 | Local |
| `clues_val.csv` | `filtered_split/wn_synset/` | 1 | Local |
| `vocabulary.csv` / `_val` | `filtered_split/wn_synset/` | 1 | Local |
| `f_clue.csv` | `filtered_split/wn_synset/clue_phrases/` | 2 (WN) | Local |
| `embeddings/g_stock/f_clue.npy` + index | `embeddings/g_stock/` | 1d (after 2) | Great Lakes |
| `clues_<f>_filtered.csv` + vocab + phrase | `filtered_split/wn_synset/<f>/` | 2 (WN) | Local |
| `triplets/<g_name>.csv` + `_meta.json` | `triplets/` | 3 | Local |
| Model weights | Google Drive | 3 | Great Lakes |
| `embeddings/<g_i>/f_<n>_val.npy` | `embeddings/<g_i>/` | 4 | Great Lakes |
| ATE results, figures | `outputs/` | 5 | Local |
| Full-dataset embeddings for final g | `embeddings/<g_name>/` | 6 | Great Lakes |
| Test-set evaluation | — | 6 | Local |

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
