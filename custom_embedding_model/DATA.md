# Data Dictionary — custom_embedding_model

This file describes every data artifact in the project: what it contains,
how it is indexed, its schema, and any gotchas. Read this before writing any
code that reads or writes data files.

For the workflow that produces these files, see WORKFLOW.md. For locked-in
decisions about data design, see DECISIONS.md.

---

## General Conventions

### keep_default_na=False

Always use `keep_default_na=False` when loading any CSV that contains word,
definition, or answer columns. The word "nan" (meaning grandmother in some
languages) is a valid crossword entry; without this flag, pandas silently
converts it to `NaN`.

```python
df = pd.read_csv('data/vocabulary.csv', keep_default_na=False)
```

### pathlib for all paths

Use `pathlib.Path` for all file paths. Define `DATA_DIR`, `PHRASES_DIR`,
`EMBEDDINGS_DIR`, and `MODELS_DIR` at the top of each notebook or script.

### Vocabulary files as indexes

Vocabulary files use a fixed canonical row ordering established at creation
and never changed. The `row` column value of a word is its row index in any
corresponding `.npy` embedding array. Do not reorder vocabulary files.

### Embedding array lookup pattern

```python
import numpy as np
import pandas as pd

# For vocabulary-based embeddings
vocab = pd.read_csv(DATA_DIR / 'vocabulary_wndef.csv', keep_default_na=False)
embeddings = np.load(EMBEDDINGS_DIR / 'g_stock/f_common_wndef.npy')
word_row = vocab.loc[vocab['word'] == 'plant', 'row'].iloc[0]
word_embedding = embeddings[word_row]  # shape: (1024,)

# For f_clue embeddings
index = pd.read_csv(EMBEDDINGS_DIR / 'g_stock/f_clue_index.csv',
                    keep_default_na=False)
embeddings = np.load(EMBEDDINGS_DIR / 'g_stock/f_clue.npy')
row = index.loc[
    (index['clue_id'] == 1042) & (index['definition'] == 'plant'), 'row'
].iloc[0]
clue_embedding = embeddings[row]  # shape: (1024,)
```

### Validating array shapes

After loading any `.npy` file, assert that the number of rows matches the
length of the corresponding vocabulary or index file:

```python
assert len(vocab) == embeddings.shape[0], \
    f"Vocab length {len(vocab)} != embedding rows {embeddings.shape[0]}"
assert embeddings.shape[1] == 1024, \
    f"Expected 1024-dim embeddings, got {embeddings.shape[1]}"
```

---

## Clue Data Files

### `../../data/clues_filtered.csv`

**Produced by:** Shared upstream pipeline (`data_preparation/structural_filtering.ipynb`)
**Contains:** All clues passing CCC structural constraints
**Does not contain:** Split column, WordNet constraints

| Column | Type | Description |
|--------|------|-------------|
| clue_id | int | Unique row identifier from source DB |
| surface | str | Clue text with answer format stripped, e.g. "Plant in a garden party" |
| definition | str | Definition substring within surface text. May be multi-word. |
| answer | str | Answer word or phrase (may be multi-word) |

Note: Additional columns from `clues_raw.csv` may be retained. See
`ccc-project/WORKFLOW.md` for the full upstream pipeline.

### `data/filtered_split/wn_synset/clues_wn_filtered.csv`

**Produced by:** Stage 1 (`01_wn_filtering_and_split.ipynb`)
**Contains:** Subset of clues_filtered.csv where both definition and answer
have at least one WordNet synset, plus split assignment
**Inherits:** All columns from clues_filtered.csv

Additional columns:

| Column | Type | Description |
|--------|------|-------------|
| definition_wn | str | WordNet-ready form of definition: lowercased, spaces→underscores, article-stripped if needed |
| answer_wn | str | WordNet-ready form of answer (same transformations) |
| split | str | 'train', 'validate', or 'test'. Assigned at (definition, answer) pair level with random_state=42. |

### `data/clues_val.csv`

**Produced by:** Stage 1
**Contains:** Rows from clues_wn_filtered.csv where split == 'validate'
**Purpose:** Convenience subset to avoid repeated filtering in downstream code

### `data/clues_<f>_filtered.csv`

Examples: `clues_wndef_filtered.csv`, `clues_wnex_filtered.csv`

**Produced by:** Stage 2 (nb_02_phrase_construction.ipynb)
**Contains:** Subset of clues_wn_filtered.csv where both definition and answer
have a valid phrase under f
**Inherits:** All columns from clues_wn_filtered.csv including split column

The split fractions may differ slightly from 30/20/50 due to subsetting.
Report actual fractions in FINDINGS.md.

---

## Vocabulary Files

All vocabulary files have the same structure. The `row` column is the
authoritative index into any corresponding `.npy` embedding array. Row
numbering starts at 0. The ordering is fixed at creation and never changed.

| Column | Type | Description |
|--------|------|-------------|
| word | str | The vocabulary word (lowercased, underscored for multi-word entries) |
| row | int | 0-indexed row position in corresponding embedding arrays |

### File Registry

| File | Contents | Index for |
|------|----------|-----------|
| `vocabulary.csv` | All unique words in clues_wn_filtered.csv (def or ans) | g_stock full vocab embeddings |
| `vocabulary_val.csv` | Validation-split subset of vocabulary.csv | g_i val vocab embeddings |
| `vocabulary_wndef.csv` | Words with valid f_common_wndef phrase | g_stock/g_i wndef embeddings |
| `vocabulary_wndef_val.csv` | Validation-split subset | g_i wndef_val embeddings |
| `vocabulary_wnex.csv` | Words with valid f_common_wnex phrase | g_stock/g_i wnex embeddings |
| `vocabulary_wnex_val.csv` | Validation-split subset | g_i wnex_val embeddings |
| `vocabulary_multisyn.csv` | Words with ≥2 WordNet synsets (future use) | TBD |

---

## Phrase Files

Phrase files live inside their respective scope/subset directories under
`data/filtered_split/`. They are generated once for the full usable dataset
and reused. No `_val` suffix — phrases are always full-dataset.

### `data/filtered_split/wn_synset/clue_phrases/f_clue.csv`

**Produced by:** Stage 2 (`02_phrase_construction_wn.ipynb`)
**Indexed by:** (clue_id, definition) composite key
**Contains:** One row per (clue, definition) pair in clues_wn_filtered.csv
where a valid f_clue phrase could be constructed (i.e., definition appears
unambiguously once in the surface text)

| Column | Type | Description |
|--------|------|-------------|
| clue_id | int | Matches clue_id in clues_wn_filtered.csv |
| definition | str | Original definition string (not definition_wn) |
| split | str | Inherited split assignment |
| phrase | str | Tagged passage: surface text with definition wrapped in `<t></t>` |

### `data/filtered_split/wn_synset/wndef/f_common_wndef.csv`

**Produced by:** Stage 2 (`02_phrase_construction_wn.ipynb`)
**Indexed by:** word (matches vocabulary_wndef.csv ordering)
**Contains:** One row per word in vocabulary_wndef.csv

| Column | Type | Description |
|--------|------|-------------|
| word | str | Vocabulary word |
| row | int | Row position (matches vocabulary_wndef.csv) |
| synset_name | str | WordNet synset identifier, e.g. "plant.n.01" |
| phrase | str | `"<t>word</t>: <synset definition text>"` |
| self_ref | bool | True if the target word also appears untagged in the phrase (i.e., the word appears in its own WordNet definition). These words are not filtered out — the flag enables downstream subsetting for evaluation. |

### `data/filtered_split/wn_synset/wnex/f_common_wnex.csv`

**Produced by:** Stage 2 (`02_phrase_construction_wn.ipynb`)
**Indexed by:** word (matches vocabulary_wnex.csv ordering)
**Contains:** One row per word in vocabulary_wnex.csv (only words with a valid
usage example where the target word appears exactly once)

| Column | Type | Description |
|--------|------|-------------|
| word | str | Vocabulary word |
| row | int | Row position (matches vocabulary_wnex.csv) |
| synset_name | str | WordNet synset identifier |
| phrase | str | Usage example with target word wrapped in `<t></t>` |

---

## Triplet Files

### `data/triplets/<g_name>_train.csv`

Examples: `triplets/g1_train.csv`

**Contains:** Training-split triplets used to fine-tune the named g model.
Multiple models may share a triplet file if they differ only in extraction
method (e.g., g1_tokenspan and g1 both train on `triplets/g1_train.csv`).
**Critical:** Contains training-split rows only. Validation and test rows
must never appear in a triplet file.

| Column | Type | Description |
|--------|------|-------------|
| clue_id | int | Source clue identifier |
| definition | str | Definition string |
| answer_wn | str | Answer word (WordNet form) |
| anchor | str | f_clue phrase for this (clue, definition) pair |
| positive | str | f phrase for the positive target word |
| negative | str | f phrase for the negative word |
| f_name | str | Name of the f strategy used for positive/negative |

---

## Embedding Files

All embedding files live in `data/embeddings/<g_name>/`. Files without `_val`
suffix cover the full vocabulary or full clue dataset. Files with `_val` suffix
cover the validation split only. Embedding dimension is always 1024.

### Naming Convention

```
embeddings/<g_name>/f_<sense>_<construction>.npy        # full vocabulary
embeddings/<g_name>/f_<sense>_<construction>_val.npy    # validation vocab only (legacy)
embeddings/<g_name>/f_clue.npy                          # full clues_wn scope (final model only)
embeddings/<g_name>/f_clue_train.npy                    # training-split clues (Decision 25)
embeddings/<g_name>/f_clue_val.npy                      # validation-split clues
```

### Vocabulary-based embedding files

Indexed by the corresponding vocabulary file. No separate index file needed.

| File | Index file | Shape | Produced by |
|------|-----------|-------|-------------|
| `g_stock/f_common_wndef.npy` | `vocabulary_wndef.csv` | (53930, 1024) | `embed_vocab.py` via `embed_wndef_full_gstock.sh` |
| `g_stock/f_common_wnex.npy` | `vocabulary_wnex.csv` | (8360, 1024) | `embed_vocab.py` via `embed_wnex_full_gstock.sh` |
| `g1/f_common_wndef.npy` | `vocabulary_wndef.csv` | (53930, 1024) | `embed_vocab.py` via `embed_wndef_full_g1.sh` |
| `g1/f_common_wnex.npy` | `vocabulary_wnex.csv` | (8360, 1024) | `embed_vocab.py` via `embed_wnex_full_g1.sh` |
| `g_stock/f_common_wndef_val.npy` | `vocabulary_wndef_val.csv` | (26152, 1024) | archived `embed_val.py`; superseded by full-vocab (Decision 23), retained for backward compat |
| `g_stock/f_common_wnex_val.npy` | `vocabulary_wnex_val.csv` | (3008, 1024) | archived `embed_val.py`; superseded by full-vocab (Decision 23), retained for backward compat |
| `g1/f_common_wndef_val.npy` | `vocabulary_wndef_val.csv` | (26152, 1024) | archived `embed_val.py`; superseded by full-vocab (Decision 23), retained for backward compat |
| `g1/f_common_wnex_val.npy` | `vocabulary_wnex_val.csv` | (3008, 1024) | archived `embed_val.py`; superseded by full-vocab (Decision 23), retained for backward compat |
| `g_stock_tokenspan/f_common_wndef_val.npy` | `vocabulary_wndef_val.csv` | (26152, 1024) | archived `embed_val.py` |
| `g_stock_tokenspan/f_common_wnex_val.npy` | `vocabulary_wnex_val.csv` | (3008, 1024) | archived `embed_val.py` |
| `g1_tokenspan/f_common_wndef_val.npy` | `vocabulary_wndef_val.csv` | (26152, 1024) | archived `embed_val.py` |
| `g1_tokenspan/f_common_wnex_val.npy` | `vocabulary_wnex_val.csv` | (3008, 1024) | archived `embed_val.py` |

### f_clue embedding files

Always accompanied by an explicit `_index.csv` file.

**Index schema** (`f_clue_index.csv` or `f_clue_val_index.csv`):

| Column | Type | Description |
|--------|------|-------------|
| clue_id | int | Clue identifier |
| definition | str | Definition string (use keep_default_na=False) |
| row | int | Row position in the corresponding .npy array |

| File | Scope | Index file | Produced by |
|------|-------|-----------|-------------|
| `g_stock/f_clue.npy` | Full clues_wn_filtered | `g_stock/f_clue_index.csv` | archived `embed_f_clue_gstock.py` (reproducible via `embed_clue.py --split all`) |
| `g_stock/f_clue_val.npy` | Validation clues only | `g_stock/f_clue_val_index.csv` | archived `embed_val.py` (reproducible via `embed_clue.py --split validate`) |
| `g_stock_tokenspan/f_clue_val.npy` | Validation clues only | `g_stock_tokenspan/f_clue_val_index.csv` | archived `embed_val.py` |
| `g1_tokenspan/f_clue_val.npy` | Validation clues only | `g1_tokenspan/f_clue_val_index.csv` | archived `embed_val.py` |
| `g1/f_clue_train.npy` | Training clues only | `g1/f_clue_train_index.csv` | `embed_clue.py --split train` (Decision 25; not yet generated for g1) |
| `g1/f_clue_val.npy` | Validation clues only | `g1/f_clue_val_index.csv` | archived `embed_val.py` (reproducible via `embed_clue.py --split validate`) |

---

## Model Files

### `models/<g_name>/README.md`

The only file committed to the repo for each model. Contains:

```
# Model: <g_name>

## Base model
HuggingFace ID: gabrielloiseau/CALE-MBERT-en
Version/commit hash: <hash>

## Weights location
Great Lakes: /home/vwinters/ccc-project/custom_embedding_model/models/<g_name>/model/

## Training details
Triplet file: data/triplets/<triplet_name>.csv
Training script: scripts/train_<g_name>.py
Hyperparameters:
  - margin: 1.0
  - learning_rate: <value>
  - epochs: <value>
  - batch_size: <value>
  - grad_accum: <value>
  - random_state: 42
Date trained: YYYY-MM-DD
Runtime: <hours> on Great Lakes (<partition>, <GPU type>)
```

### `models/g_stock/README.md`

```
# Model: g_stock

## Description
The unmodified stock CALE model. No weights are stored locally or in
Google Drive — load directly from HuggingFace at runtime.

HuggingFace ID: gabrielloiseau/CALE-MBERT-en
Version/commit hash: <pin this when first used>
```

---

## WordNet Lookup Conventions

When looking up WordNet synsets for a word, apply transformations in this
order to maximize coverage:

1. Try the word as-is (lowercased)
2. Try with spaces replaced by underscores (e.g., "ice cream" → "ice_cream")
3. Try with leading article stripped (e.g., "a shade" → "shade"), but only
   after step 2 fails

Store the WordNet-ready form in `definition_wn` and `answer_wn` columns.
Preserve the original `definition` and `answer` columns for surface text
matching (finding the definition within the surface string).

When constructing f_clue phrases, always use the original `definition` string
(not `definition_wn`) to locate the definition within the surface text.

---

## ATE Computation

The primary evaluation measure is the Average Treatment Effect (ATE):

For each (clue, definition, answer) pair in the evaluation split:
```
delta_i = cos_sim(g(f_clue(def_i)), g(f(ans_i)))
        − cos_sim(g(f(def_i)),      g(f(ans_i)))
```

```
ATE = mean(delta_i)
```

A **negative ATE** indicates misdirection: clue context pulls the definition
embedding away from the answer. A **less negative ATE** under a fine-tuned g_i
compared to g_stock indicates the model is partially counteracting misdirection.

Use rowwise cosine similarity (normalize rows, then dot product):
```python
def rowwise_cosine(A, B):
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-10)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-10)
    return np.sum(A_norm * B_norm, axis=1)
```

Report: mean delta, median delta, standard error, 95% CI, % pairs with
negative delta.
