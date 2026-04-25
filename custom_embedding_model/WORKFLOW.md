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
    Stage 5: Model evaluation (shared comparison notebook)
              └── FINDINGS.md entries
                    │
                    ▼
    Stage 6: Hypothesis testing (per-g investigation notebooks)
              └── FINDINGS.md entries
                    │
                    ▼
    Stage 7: Final evaluation (locked — test split only)
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
model learned. Scientific evaluation happens in Stages 5 and 6.

### Learning curves (Decision 27)

After the full model is trained, train additional models on subsets of the
training data (e.g., 10%, 25%, 50%, 75%) using the same hyperparameters.
Evaluate each on the full validation set (same metrics as Decision 24:
val loss, val accuracy, val margin). Save results to
`models/<g_name>/learning_curve_results.json`.

This answers "how much training data is necessary?" and reveals whether
problems are driven by data quantity vs. data quality. Since learning curves
are specific to a triplet design and phrase construction, each g_i needs its
own.

### Runtime tracking (Decision 27)

Before submitting GPU jobs, estimate wall-clock time for each component in
FINDINGS.md:
- Base model training (full triplet set, all epochs)
- Each learning curve subset run
- Validation loss computation per epoch

After each job completes, record actual runtime alongside the estimate. These
records build a reference for planning future jobs.

### Outputs and documentation

Save model weights to Google Drive ("Research Project - NLP CCC's", owned by
Nathan). Commit `models/<g_name>/README.md` to the repo containing:
- Google Drive path to weights
- HuggingFace base model identifier and version hash
- Training script name and all hyperparameters
- Triplet file used
- Date trained
- Wall-clock runtime (estimated and actual)

Record runtime in FINDINGS.md.

---

## Stage 4: Embedding Generation

**Script:** `scripts/embed_<g_name>.py` + `scripts/embed_<g_name>.sh`
**Environment:** Great Lakes (GPU)
**Inputs:** Model weights (from Google Drive), phrase files, vocabulary files
**Outputs:** Embedding arrays under `data/embeddings/<g_name>/`

For each g_i being evaluated, generate the following embeddings:

**Vocabulary-indexed embeddings (full vocabulary, per Decision 23):**
- Encode phrases for `vocabulary_<f>.csv` words using g_i
- Save as `data/embeddings/<g_i>/f_<sense>_<construction>.npy`
- The index is `vocabulary_<f>.csv` itself — no separate index file needed

**f_clue embeddings (training and validation splits, per Decision 25):**
- Encode f_clue phrases for training-split clues using g_i
- Save as `data/embeddings/<g_i>/f_clue_train.npy`
- Save `data/embeddings/<g_i>/f_clue_train_index.csv` (clue_id, definition, row)
- Encode f_clue phrases for validation-split clues using g_i
- Save as `data/embeddings/<g_i>/f_clue_val.npy`
- Save `data/embeddings/<g_i>/f_clue_val_index.csv` (clue_id, definition, row)
- Test-split f_clue embeddings are NOT generated until Stage 7 (Decision 9)

Full-dataset f_clue embeddings (covering all splits) are generated only for
the final chosen g (Stage 7).

After the job completes, scp output files back to local machine before
proceeding to Stage 5 and Stage 6.

Record runtime in FINDINGS.md alongside the vocabulary size, clue count, and
Great Lakes partition used.

---

## Stage 5: Model Evaluation

**Primary notebook:** `notebooks/05_model_evaluation.ipynb`
**Environment:** Local (CPU)
**Inputs:** Embedding arrays from `data/embeddings/`, training log and
  validation loss results from `models/<g_name>/`
**Outputs:** FINDINGS.md entries, figures in `outputs/figures/`

For each g_i, evaluate in the following order. The progression moves from
basic model health to intrinsic task performance to comparative analysis
to the research question. Earlier sections should be satisfactory before
interpreting later sections. Evaluate on the training phrase type (e.g.
wndef for g1) — cross-format generalization (e.g. wnex) is a hypothesis
testing question for Stage 6.

### §1: Training dynamics and overfitting

Using `training_log.json` and `val_loss_results.json` from the model
directory, report training loss and validation loss per epoch, the
train/val loss ratio, validation accuracy, and validation mean margin
across epochs. Identify the best-generalizing checkpoint (lowest
validation loss). Note whether the deployed model is the best checkpoint
or a later, more overfit epoch.

### §2: Task performance in training and research metrics

**Triplet accuracy in both metrics (Decision 26):** Compute triplet
accuracy on both training and validation triplets using L2 distance (the
training metric) and cosine similarity (the research metric). This
requires f_clue_train embeddings (Decision 25) for training triplet
accuracy and f_clue_val embeddings for validation triplet accuracy, plus
full-vocabulary embeddings for the positive/negative lookups. Report:
- Training vs validation accuracy gap (overfitting diagnostic)
- L2 vs cosine accuracy gap (metric mismatch diagnostic)

If training accuracy is not directly computable (e.g. f_clue_train
embeddings were not generated for a legacy model), note this gap and
infer training accuracy from the training loss where possible.

### §3: Embedding space geometry

Characterize how g_i uses the embedding space, for both vocabulary-
indexed (e.g. wndef) and f_clue embedding populations:
- **Norm distributions:** L2 norm histograms, means, standard deviations
  for g_i and g_stock (overlay). Tests whether the model exploited the
  magnitude degree of freedom in L2 training.
- **Pairwise cosine among random pairs:** Mean, median, and distribution.
  High values indicate loss of discriminability.
- **Effective dimensionality:** Participation ratio of singular values.
  Low values indicate dimensional collapse.
- **Compression relative to g_stock:** The delta in pairwise cosine,
  total variance, and effective dimensionality between g_stock and g_i.

### §4: Structural comparison to g_stock

**RSA (Representational Similarity Analysis):** Spearman correlation of
pairwise cosine matrices between g_stock and g_i to measure how much
fine-tuning reorganized the similarity structure (as opposed to merely
compressing it).

### §5: Misdirection (ATE decomposition)

For each (clue, definition, answer) pair in the validation split, compute
T=0 = cos_sim(g(f(def)), g(f(ans))) and T=1 = cos_sim(g(f_clue(def)),
g(f(ans))). Report means, medians, and distributions. Decompose ATE into
its T=0 and T=1 components — ATE changing confirms the model learned
something; the components reveal *what* it learned. Always interpret ATE
through its components, not as a standalone optimization target.

Record all numerical results in FINDINGS.md.

---

## Stage 6: Hypothesis Testing

**Per-g investigation notebooks:** `notebooks/06_<g_name>_hypothesis_testing.ipynb`
**Per-g exploration notebooks (optional):** `notebooks/06_explore_<g_name>.ipynb`
**Environment:** Local (CPU)
**Inputs:** Embedding arrays from `data/embeddings/`, design issue
  documentation from `planning/`
**Outputs:** FINDINGS.md entries, figures in `outputs/figures/`

Stage 5 characterizes *what* a fine-tuned model did to the embedding space.
Stage 6 investigates *why* — systematically testing whether identified design
issues (in phrase construction, sense selection, triplet design, etc.)
account for the observed empirical findings. Each investigation notebook
is guided by a design document in `planning/` that maps specific design
issues to specific empirical findings and proposes tests to connect them.

For deeper exploration of a specific g_i (neighborhood structure, which
words moved most, failure cases), create a separate notebook named
`06_explore_<g_name>.ipynb`. Move to `notebooks/archive/` when done with
that model.

---

## Stage 7: Final Evaluation (Locked)

**Trigger:** A final g has been chosen and the decision documented in
DECISIONS.md.

**Script:** `scripts/embed_final_<g_name>.py` + corresponding `.sh`
**Notebook:** `notebooks/07_final_evaluation.ipynb`
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
| `embeddings/<g_i>/f_clue_train.npy` + index | `embeddings/<g_i>/` | 4 | Great Lakes |
| `embeddings/<g_i>/f_<n>_val.npy` | `embeddings/<g_i>/` | 4 | Great Lakes |
| Model evaluation results, figures | `outputs/` | 5 | Local |
| Hypothesis testing results, figures | `outputs/` | 6 | Local |
| Full-dataset embeddings for final g | `embeddings/<g_name>/` | 7 | Great Lakes |
| Test-set evaluation | — | 7 | Local |

---

## Critical Rules

- **The test set is locked.** Do not load, inspect, or embed test-split data
  until Stage 7, after the final g is chosen and documented in DECISIONS.md.
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
