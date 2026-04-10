# Notebook and Script Inventory — custom_embedding_model

This file describes every notebook and script in the project: what it does,
what it reads, what it produces, and its current status. Because this project
is experimental rather than a linear pipeline, notebooks are not strictly
numbered in sequence — the stage number in the name reflects the workflow
stage (see WORKFLOW.md), not a required run order within a stage.

Runtimes for GPU-intensive steps are tracked in FINDINGS.md, not here.

**Upstream notebooks** (shared, in `ccc-project/notebooks/`) are not listed
here. See `ccc-project/WORKFLOW.md` for `puzzle_metadata.ipynb`,
`structural_filtering.ipynb`, and `clue_utils.py`.

---

## Status Key

- ✅ **Complete** — runs cleanly, outputs committed
- 🔄 **In progress** — actively being developed
- ❌ **Not yet created** — needs to be built
- 📋 **Reference only** — do not run or modify; mine for code and context

---

## Pipeline Notebooks (`notebooks/`)

### Stage 1

| Notebook | Status | Environment | Reads | Writes |
|----------|--------|-------------|-------|--------|
| `01_wn_filtering_and_split.ipynb` | ❌ | Local | `../../data/clues_filtered.csv` | `filtered_split/wn_synset/clues_wn_filtered.csv`, `filtered_split/wn_synset/clues_val.csv`, `filtered_split/wn_synset/vocabulary.csv`, `filtered_split/wn_synset/vocabulary_val.csv` |

Applies the WordNet synset filter (both definition and answer must have at
least one synset). Assigns the 30/20/50 train/validate/test split at the
(definition, answer) pair level using `random_state=42`. Constructs the full
unified vocabulary. Reports actual split fractions and vocabulary size.

The `definition_wn` and `answer_wn` columns created here use the WordNet
lookup convention in DATA.md (lowercase, underscore conversion, article
stripping).

Builds on: `clue_misdirection/notebooks/01_data_cleaning.ipynb` (Victoria —
WordNet lookup logic, article stripping, underscore conversion)

---

### Stage 1d (GPU — runs after Stage 2)

| Script | Status | Environment | Reads | Writes |
|--------|--------|-------------|-------|--------|
| `scripts/embed_f_clue_gstock.py` | ❌ | Great Lakes (GPU) | `filtered_split/wn_synset/clues_wn_filtered.csv`, `filtered_split/wn_synset/clue_phrases/f_clue.csv` | `data/embeddings/g_stock/f_clue.npy`, `data/embeddings/g_stock/f_clue_index.csv` |
| `scripts/embed_f_clue_gstock.sh` | ❌ | Great Lakes (SLURM) | — | — |

Encodes all f_clue phrases for the full `clues_wn_filtered.csv` dataset using
g_stock. Computed once and reused by all f-specific analyses. Must be run
after Stage 2 has produced `phrases/f_clue.csv`. If inclusion criteria are
later expanded, append new rows rather than regenerating. Record runtime in
FINDINGS.md.

---

### Stage 2

| Notebook | Status | Environment | Reads | Writes |
|----------|--------|-------------|-------|--------|
| `02_phrase_construction.ipynb` | ❌ | Local | `filtered_split/wn_synset/clues_wn_filtered.csv`, WordNet | See below |

Handles all WordNet-based f's in sequence. Also constructs f_clue phrases
for the full wn_synset scope using `clue_utils.py` (imported from
`ccc-project/notebooks/`) for definition-finding and delimiter-placement —
the same logic used in `structural_filtering.ipynb`, ensuring consistency.
For each f, produces a subset directory under `filtered_split/wn_synset/`
containing the filtered clue file, vocabulary files, and phrase file.
Specifically:
- `filtered_split/wn_synset/clue_phrases/f_clue.csv`
- `filtered_split/wn_synset/wndef/clues_wndef_filtered.csv`
- `filtered_split/wn_synset/wndef/vocabulary_wndef.csv` + `_val`
- `filtered_split/wn_synset/wndef/f_common_wndef.csv`
- `filtered_split/wn_synset/wnex/clues_wnex_filtered.csv`
- `filtered_split/wn_synset/wnex/vocabulary_wnex.csv` + `_val`
- `filtered_split/wn_synset/wnex/f_common_wnex.csv`

Reports coverage statistics at each stage (rows remaining, vocabulary words
with valid phrases, resulting split fractions). These numbers go in
FINDINGS.md under "Coverage Measurements."

**Critical:** No fallbacks. Words without a valid phrase for a given f are
absent from that f's vocabulary and phrase file. See WORKFLOW.md Stage 2.

Builds on: `clue_misdirection/notebooks/02_embedding_generation.ipynb`
(Victoria — phrase construction logic, CALE delimiter insertion, WordNet
synset lookup)

---

### Stage 3

| Notebook | Status | Environment | Reads | Writes |
|----------|--------|-------------|-------|--------|
| `03_train_g1.ipynb` | ❌ | Local (inspection) / Great Lakes (GPU) | Phrase files from `filtered_split/wn_synset/`, training-split rows | `data/triplets/g1.csv`, `data/triplets/g1_meta.json`, `models/g1/README.md` |
| `scripts/train_g1.py` | ❌ | Great Lakes (GPU) | Same | Model weights → Google Drive |
| `scripts/train_g1.sh` | ❌ | Great Lakes (SLURM) | — | — |

Constructs triplets from the training split, drawing anchor phrases from
`clue_phrases/`, and positive/negative phrases from the relevant subset
directory (e.g. `wndef/`). The triplet file represents the intersection of
rows with valid phrases under all three f's used. Saves `g1_meta.json`
documenting which f was used for each role, source paths, and row counts.
Fine-tunes g_stock using triplet margin loss (α = 1.0). Saves model weights
to Google Drive and commits README to repo. Record runtime in FINDINGS.md.

**Critical:** Triplet file contains training-split rows only. Validation and
test rows must never appear.

Builds on: `custom_embedding_model/notebooks/archive/09_learned_g_misdirection.ipynb`
(Nathan — triplet construction, training loop, CALE concept-aligned extraction)

---

### Stage 4

| Script | Status | Environment | Reads | Writes |
|--------|--------|-------------|-------|--------|
| `scripts/embed_g1.py` | ❌ | Great Lakes (GPU) | Model weights, phrase files, vocabulary_*_val.csv files | `data/embeddings/g1/*_val.npy` files |
| `scripts/embed_g1.sh` | ❌ | Great Lakes (SLURM) | — | — |

Generates validation-split embeddings for g_1:
- `embeddings/g1/f_common_wndef_val.npy` (indexed by `wndef/vocabulary_wndef_val.csv`)
- `embeddings/g1/f_common_wnex_val.npy` (indexed by `wnex/vocabulary_wnex_val.csv`)
- `embeddings/g1/f_clue_val.npy` + `f_clue_val_index.csv`

After job completes, scp output files back locally. Record runtime in
FINDINGS.md alongside vocabulary size, clue count, and cluster partition.

---

### Stage 5

| Notebook | Status | Environment | Reads | Writes |
|----------|--------|-------------|-------|--------|
| `05_hypothesis_testing.ipynb` | ❌ | Local | Embedding arrays, vocabulary files, clue data | FINDINGS.md entries, `outputs/figures/` |

**Primary shared comparison notebook.** Accumulates ATE results and cross-f
generalization tests across all trained g's. Add new sections for each new
g_i — do not rerun from scratch.

For each g_i computes:
- ATE on validation set (mean delta, median, SE, 95% CI, % negative)
- Cross-f generalization test (for g_1: do f_common_wnex similarities change?)
- g_stock baseline for format sensitivity comparison

See DATA.md for the ATE computation formula and rowwise cosine implementation.

---

### Stage 6 (Locked)

| Notebook | Status | Environment | Reads | Writes |
|----------|--------|-------------|-------|--------|
| `06_final_evaluation.ipynb` | ❌ | Local | Full-dataset embeddings for final g | FINDINGS.md entries |
| `scripts/embed_final_<g_name>.py` | ❌ | Great Lakes (GPU) | Model weights, all phrase files | Full-dataset embedding arrays |

**Do not create or run until final g is chosen and documented in DECISIONS.md.**

---

## Archive Notebooks (`notebooks/archive/`)

These notebooks are reference only. Do not run or modify them.

| Notebook | Author | Status | Description |
|----------|--------|--------|-------------|
| `09_learned_g_misdirection.ipynb` | Nathan | 📋 Reference | Initial g_1 training and evaluation. Constructs triplets from `dataset_harder.parquet`, fine-tunes g_stock with triplet margin loss (α=1.0, 80/20 train/test split), compares g_stock vs. g_1 on ATE. Key finding: g_1 ATE = −0.282 vs. g_stock ATE = −0.072 (more negative, not less). T=0 similarity jumped from 0.548 to 0.758 while T=1 stayed flat at 0.476 — evidence g_1 compressed f_common_wndef phrases rather than learning to counteract misdirection. Ran on Great Lakes (GPU). See FINDINGS.md for full results. |

Per-g exploration notebooks will be moved here when work on that model is
complete. Name them `05_explore_<g_name>.ipynb` before archiving.

---

## Scripts (`scripts/`)

| Script | Purpose | Environment |
|--------|---------|-------------|
| `embed_f_clue_gstock.py` | Encode f_clue phrases with g_stock over full wn_filtered dataset | Great Lakes (GPU) |
| `embed_f_clue_gstock.sh` | SLURM submission for above | Great Lakes |
| `train_g1.py` | Triplet construction + g_1 fine-tuning | Great Lakes (GPU) |
| `train_g1.sh` | SLURM submission for above | Great Lakes |
| `embed_g1.py` | Validation-split embedding generation for g_1 | Great Lakes (GPU) |
| `embed_g1.sh` | SLURM submission for above | Great Lakes |

All GPU scripts should:
- Accept command-line arguments for key parameters
- Print progress and results to stdout (captured in SLURM logs)
- Record and print total wall-clock runtime at completion
- Save outputs atomically (write to temp path, then rename)
