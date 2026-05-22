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
| `01_wn_filtering_and_split.ipynb` | ✅ | Local | `../../data/clues_filtered.csv` | `filtered_split/wn_synset/clues_wn_filtered.csv`, `filtered_split/wn_synset/clues_val.csv`, `filtered_split/wn_synset/vocabulary.csv`, `filtered_split/wn_synset/vocabulary_val.csv` |

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
| `scripts/embed_clue.py` | ✅ | Great Lakes (GPU) | `filtered_split/wn_synset/clue_phrases/f_clue.csv` | `data/embeddings/<g_name>/f_clue[_train\|_val].npy`, `..._index.csv` |
| `scripts/embed_f_clue_gstock.py` + `.sh` | 📋 Archived | Great Lakes (GPU) | — | — (superseded by `embed_clue.py`, kept in `scripts/archive/` for reference) |

Encodes f_clue phrases for a given `(model, split, pooling)` combination.
Driven by `--split` flag: `all` reproduces the full-dataset g_stock output
(Decision 7), `validate` reproduces `embed_val.py`'s f_clue portion,
`train` is available for completeness. Must be run after Stage 2 has
produced `f_clue.csv`. Record runtime in FINDINGS.md.

---

### Stage 2

**Naming convention:** `02_phrase_construction_<resource>.ipynb` — one
notebook per resource family. Future non-WordNet resources (dictionary APIs,
LLM-generated phrases) would each get their own notebook and may branch
from a different scope directory. See WORKFLOW.md Stage 2.

| Notebook | Status | Environment | Reads | Writes |
|----------|--------|-------------|-------|--------|
| `02_phrase_construction_wn.ipynb` | ✅ | Local | `filtered_split/wn_synset/clues_wn_filtered.csv`, WordNet | See below |

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

#### g1_tokenspan (token span extraction — see Decision 20)

| Notebook | Status | Environment | Reads | Writes |
|----------|--------|-------------|-------|--------|
| `03_train_g1.ipynb` | ✅ | Local | Phrase files from `filtered_split/wn_synset/`, `dataset_harder.parquet`, training- and validation-split rows | `data/triplets/g1_train.csv`, `data/triplets/g1_val.csv`, `data/triplets/g1_train_meta.json` |
| `scripts/train_g1_tokenspan.py` | ✅ | Great Lakes (GPU) | `data/triplets/g1_train.csv` | Model weights (Great Lakes) |
| `scripts/train_g1_tokenspan.sh` | ✅ | Great Lakes (SLURM) | — | — |

Constructs triplets from the training and validation splits, drawing anchor
phrases from `clue_phrases/`, and positive/negative phrases from the relevant
subset directory (e.g. `wndef/`). Each triplet file represents the
intersection of rows with valid phrases under all three f's used. Saves
`g1_train_meta.json` documenting which f was used for each role, source
paths, and row counts for both splits.

The training triplet file (`g1_train.csv`) is used by the training scripts.
The validation triplet file (`g1_val.csv`) is used by NB 05 (model
evaluation) for validation triplet accuracy. Both are constructed with the
same procedure; only the split differs. Test-split rows must never appear
in either file.

**Note:** The training triplet file is shared between g1_tokenspan and g1 —
both models train on identical text triplets. Only the extraction method
differs (token span vs. mean pooling). g1_tokenspan uses token span
extraction (non-standard for CALE). See Decision 20. The training script
and SLURM wrapper are preserved as historical artifacts.

Builds on: `custom_embedding_model/notebooks/archive/09_learned_g_misdirection.ipynb`
(Nathan — triplet construction, training loop, CALE concept-aligned extraction)

#### g1 (mean pooling — canonical)

| Script | Status | Environment | Reads | Writes |
|--------|--------|-------------|-------|--------|
| `scripts/train_g1.py` | ✅ | Great Lakes (GPU) | `data/triplets/g1_train.csv` (same triplets as g1_tokenspan, different extraction) | Model weights (Great Lakes) |
| `scripts/train_g1.sh` | ✅ | Great Lakes (SLURM) | — | — |

Same triplets as g1_tokenspan, but trained using mean pooling (canonical
CALE extraction per Decision 20). Uses `SentenceTransformer` training APIs
or equivalent attention-masked mean pooling. Training completed 2026-04-14.

---

### Stage 4

| Script | Status | Environment | Reads | Writes |
|--------|--------|-------------|-------|--------|
| `scripts/embedding_utils.py` | ✅ | Great Lakes (GPU) | — (library) | — (library; shared by `embed_clue.py` and `embed_vocab.py`) |
| `scripts/embed_clue.py` | ✅ | Great Lakes (GPU) | `filtered_split/wn_synset/clue_phrases/f_clue.csv` | `data/embeddings/<g_name>/f_clue[_train\|_val].npy` + `_index.csv` |
| `scripts/embed_vocab.py` | ✅ | Great Lakes (GPU) | Any `(vocab, phrase)` pair | `data/embeddings/<g_name>/<phrase_name>.npy` |
| `scripts/embed_wnex_full_gstock.sh` | ✅ | Great Lakes (SLURM) | — | `g_stock/f_common_wnex.npy` (8360 rows) |
| `scripts/embed_wnex_full_g1.sh` | ✅ | Great Lakes (SLURM) | — | `g1/f_common_wnex.npy` (8360 rows) |
| `scripts/verify_embedding_scripts.sh` | ✅ | Great Lakes (SLURM) | — | Seven verification passes comparing new scripts against committed artifacts |
| `scripts/embed_val.py` + wrappers | 📋 Archived | Great Lakes (GPU) | — | — (superseded by `embed_clue.py` + `embed_vocab.py`, kept in `scripts/archive/` for reference) |

Stage 4 uses a two-script architecture with shared machinery in
`embedding_utils.py`: `embed_clue.py` handles clue-contextualized (f_clue)
phrases with a `--split` filter, and `embed_vocab.py` handles
decontextualized vocabulary-indexed phrases with `--vocab-file` and
`--phrase-file` passed as arguments. Both support `--pooling
meanpool|tokenspan` (Decision 20 canonical is `meanpool`) and an optional
`--verify-against` flag for rowwise consistency checks against existing
`.npy` artifacts.

Existing embedding runs (committed artifacts):
1. g_stock_tokenspan (token span extraction, stock CALE weights) — ✅ generated via archived `embed_val.py`
2. g1_tokenspan (token span extraction, fine-tuned weights) — ✅ generated via archived `embed_val.py`
3. g_stock (mean pooling, stock CALE weights) — ✅ generated via archived scripts; full-vocab wnex added by `embed_wnex_full_gstock.sh`
4. g1 (mean pooling, fine-tuned weights) — ✅ generated via archived `embed_val.py`; full-vocab wnex added by `embed_wnex_full_g1.sh`

After each job completes, transfer output files back locally. Record
runtime in FINDINGS.md alongside vocabulary size, clue count, and cluster
partition.

| Notebook | Status | Environment | Reads | Writes |
|----------|--------|-------------|-------|--------|
| `04_embedding_verification.ipynb` | ✅ | Local | All `data/embeddings/*/` `.npy` and index files, vocabulary files | `outputs/04_embedding_verification-results.md` |

Pre-Stage-5 verification that all four embedding sets are correctly named,
distinct, and contain valid data. Checks shapes against FINDINGS.md, confirms
no NaN/zero rows, verifies index file consistency, and computes pairwise
mean cosine similarity matrices across all (model, phrase_type) combinations
to confirm no accidental duplicates. Does not produce new data artifacts.

---

### Stage 5: Model Evaluation

| Notebook | Status | Environment | Reads | Writes |
|----------|--------|-------------|-------|--------|
| `05_model_evaluation.ipynb` | ✅ | Local | Full-vocab and val-only embedding arrays, vocabulary files, clue data, validation triplets | `outputs/05_model_evaluation-results.md`, `outputs/figures/` |

**Primary model evaluation notebook.** Completed 2026-04-14, revised
2026-04-19 to add cross-f triplet accuracy (§3), wnex T=0/T=1 (§6), and
full-vocab wndef triplet resolution (§2). Uses full-vocab embeddings for
triplet accuracy, val-only for model diagnostics (Decision 23 / Decision 9).

For g_stock and g1, computes:
- Validation triplet accuracy (wndef, full-vocab — 46,506 triplets, 100% resolved)
- Cross-f triplet accuracy (matched wndef vs wnex on 2,985 triplets)
- Collapse detection (pairwise cosine, effective dimensionality — val-only)
- T=0 and T=1 distributions with ATE preview (wndef and wnex — val-only)
- RSA (val-only)

See DATA.md for the ATE computation formula and rowwise cosine implementation.

---

### Exploration (not part of numbered pipeline)

| Notebook | Status | Environment | Reads | Writes |
|----------|--------|-------------|-------|--------|
| `planning/exploration/pos_wordnet_census.ipynb` | ✅ | Local | `vocabulary.csv`, `clues_wn_filtered.csv`, `clues_val.csv`, `g1_train.csv`, WordNet, spaCy | `outputs/pos_wordnet_census-results.md`, `outputs/pos_mismatch_examples.md`, 6 figures |
| `planning/exploration/wordplay_ate_breakdown.ipynb` | ✅ | Local | `clues_val.csv`, `wordplay_metadata.csv`, val-only embeddings | `outputs/wordplay_ate_breakdown-results.md`, 7 figures |
| `planning/exploration/cale_fclue_norm_bimodality.ipynb` | ✅ | Local | g_stock and g1 f_clue and vocabulary embeddings, `clues_val.csv`, `wordplay_metadata.csv`, `puzzle_metadata.csv` | `outputs/cale_fclue_norm_bimodality-results.md`, 12 figures |
| `planning/exploration/cale_norm_bimodality.ipynb` | ✅ | Local | All g_stock embedding arrays, vocabulary files, `f_clue.csv`, `wordplay_metadata.csv` | `outputs/cale_norm_bimodality-results.md`, 8 figures |

POS census characterizes sense selection reliability and noun dominance across
vocabulary, training, and validation data. Wordplay ATE breakdown decomposes
misdirection patterns by clue type (structural and letterplay).
`cale_fclue_norm_bimodality` is the initial characterization of the L2 norm
bimodality in g_stock f_clue embeddings: definition position effects, wordplay
stratification, dimension-level analysis, propagation to cosine/ATE, and g1
comparison. `cale_norm_bimodality` extends the investigation with formal
bimodality testing (ΔBIC + Ashman's D) across all g_stock embedding
populations, ICC analysis of word-level vs context-level variance, cross-format
norm correlations, and surface-feature regression.

---

### Stage 6: Hypothesis Testing

*Not pursued. See Decision 28.*

---

### Stage 7: Final Evaluation

*Not pursued. See Decision 28.*

---

## Archive Notebooks (`notebooks/archive/`)

These notebooks are reference only. Do not run or modify them.

| Notebook | Author | Status | Description |
|----------|--------|--------|-------------|
| `09_learned_g_misdirection.ipynb` | Nathan | 📋 Reference | Initial g_1 training and evaluation. Constructs triplets from `dataset_harder.parquet`, fine-tunes g_stock with triplet margin loss (α=1.0, 80/20 train/test split), compares g_stock vs. g_1 on ATE. Key finding: g_1 ATE = −0.282 vs. g_stock ATE = −0.072 (more negative, not less). T=0 similarity jumped from 0.548 to 0.758 while T=1 stayed flat at 0.476 — evidence g_1 compressed f_common_wndef phrases rather than learning to counteract misdirection. Ran on Great Lakes (GPU). See FINDINGS.md for full results. |

This component was set aside after the g1 evaluation (Decision 28).

---

## Scripts (`scripts/`)

| Script | Purpose | Environment |
|--------|---------|-------------|
| `embedding_utils.py` | Shared embedding machinery (model loading, extraction, save helpers) imported by `embed_clue.py` and `embed_vocab.py` | Great Lakes (library) |
| `embed_clue.py` | Embed f_clue phrases for a given model, split (`train`/`validate`/`all`), and pooling method | Great Lakes (GPU) |
| `embed_vocab.py` | Embed vocabulary-indexed phrases for a given model, `(vocab, phrase)` pair, and pooling method | Great Lakes (GPU) |
| `embed_wnex_full_gstock.sh` | SLURM submission: g_stock full-vocab wnex (8,360 words) | Great Lakes |
| `embed_wnex_full_g1.sh` | SLURM submission: g1 full-vocab wnex (8,360 words) | Great Lakes |
| `verify_embedding_scripts.sh` | SLURM submission: seven verification passes reproducing every existing embedding artifact with the new scripts | Great Lakes |
| `train_g1_tokenspan.py` | g1_tokenspan fine-tuning (token span extraction — historical, see Decision 20) | Great Lakes (GPU) |
| `train_g1_tokenspan.sh` | SLURM submission for above | Great Lakes |
| `train_g1.py` | g1 fine-tuning (mean pooling — canonical) | Great Lakes (GPU) |
| `train_g1.sh` | SLURM submission for above | Great Lakes |
| `val_loss_from_checkpoints.py` | ✅ Compute validation loss from saved g1 epoch checkpoints (Decision 24) | Great Lakes (GPU) |
| `val_loss_from_checkpoints.sh` | ✅ SLURM submission for above | Great Lakes |
| `archive/embed_f_clue_gstock.py` + `.sh` | 📋 Archived — superseded by `embed_clue.py` | Great Lakes |
| `archive/embed_val.py` + four `.sh` wrappers | 📋 Archived — superseded by `embed_clue.py` + `embed_vocab.py` | Great Lakes |

All GPU scripts should:
- Accept command-line arguments for key parameters
- Print progress and results to stdout (captured in SLURM logs)
- Record and print total wall-clock runtime at completion
- Save outputs atomically (write to temp path, then rename)
