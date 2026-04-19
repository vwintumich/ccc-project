# Findings — custom_embedding_model

Running log of coverage measurements, compute runtimes, ATE results, and
experimental findings as the project progresses. Add entries as work
completes — do not wait until the end to record results.

For locked-in decisions, see DECISIONS.md. For the workflow, see WORKFLOW.md.

---

## Prior Work: Initial g_1 Training and Evaluation (NB 09)

These findings come from `notebooks/archive/09_learned_g_misdirection.ipynb`
(Nathan Cantwell), which represents the first attempt at fine-tuning a custom
embedding model. This work was done prior to the `custom_embedding_model`
directory and used a different dataset and split than what we are building
here. It is recorded as prior work, not as a result of the current pipeline.

### Data and setup

- Input dataset: `clue_misdirection/data/dataset_harder.parquet` (harder
  binary classification dataset from Milestone II)
- Split: 80/20 train/test at the (definition, answer) pair level
- Training set: ~190,943 triplet rows (sample mode: 37,836 rows, 20,000
  unique pairs)
- Test set: ~48,172 rows (sample mode: 20,000 rows)
- Environment: Great Lakes (GPU, CUDA)

### Triplet design (T_1)

| Component | Description | Construction |
|-----------|-------------|--------------|
| Anchor | f_clue(definition) | Definition tagged in clue surface with `<t></t>` |
| Positive | f_common_wndef(answer) | `"<t>answer</t>: <WordNet definition>"` |
| Negative | f_common_wndef(distractor) | Same format for distractor word |

Base model: `gabrielloiseau/CALE-MBERT-en` (g_stock)
Margin: α = 1.0. Mixed precision, gradient accumulation.

### ATE results (sample mode, 20K test rows)

The ATE is defined as:
mean(cos_sim(g(f_clue(def)), g(f(ans))) − cos_sim(g(f(def)), g(f(ans))))

A negative ATE indicates misdirection. The hypothesis was that g_1 would show
a *less negative* ATE than g_stock. The opposite occurred.

| Metric | g_stock | g_1 |
|--------|---------|-----|
| Decontextualized similarity T=0 (mean) | 0.548 | 0.758 |
| Contextualized similarity T=1 (mean) | 0.476 | 0.476 |
| Delta = T=1 − T=0 (ATE, mean) | −0.072 | −0.282 |
| ATE 95% CI | [−0.074, −0.071] | [−0.283, −0.281] |
| Delta median | −0.065 | −0.281 |
| % pairs with negative delta | 76.9% | 100.0% |

**Key observation:** The T=1 (clue-context) similarity stayed almost identical
between g_stock and g_1 (0.476 vs. 0.476). The T=0 (decontextualized)
similarity jumped dramatically (0.548 → 0.758). g_1 pulled decontextualized
f_common_wndef phrases much closer together in embedding space, making the
ATE more negative rather than less.

### Retrieval-based ATE (sample mode)

| Metric | g_stock | g_1 |
|--------|---------|-----|
| Median rank T=0 (decontextualized) | 618 | 821 |
| Median rank T=1 (clue-context) | 936 | 1240 |
| Rank delta (T=1 − T=0) | +318 | +419 |
| % pairs worsened by context | 55.6% | 53.6% |

### Interpretation

g_1 did not learn to counteract misdirection. The evidence suggests it learned
to compress f_common_wndef phrases ("word: WordNet definition" format) closer
together in embedding space — a form of format-specific overfitting. Two
contributing factors identified:

1. **Unnatural phrase format:** f_common_wndef uses a dictionary-entry format
   (`"<t>word</t>: definition"`) that CALE was not trained on. The model may
   have keyed on this format signature rather than semantic content.

2. **Positive target:** The positive was f_common_wndef(answer), not
   f_common_wndef(definition). This pulls the clue-contextualized definition
   embedding toward the answer's decontextualized embedding — a conceptually
   indirect objective that may not be well-suited to counteracting misdirection.

### Planned investigation

Step A: Reproduce g_1 exactly using the new pipeline.
Step B: Test the formatting hypothesis — compute g_1(f_common_wnex(word)) for
validation-set words and compare to g_stock(f_common_wnex(word)). If g_1
also compresses wnex phrases (even though it was not trained on them), that
suggests semantic generalization. If not, that confirms format-specific
overfitting.

---

## Stage 0: Structural Filtering

*Not yet run.*

---

## Stage 1: WordNet Filtering and Split Assignment

**Notebook:** `01_wn_filtering_and_split.ipynb` — completed 2026-04-13
**Environment:** Local, `crossword` kernel (NLTK 3.9.2, scikit-learn 1.8.0)

### Coverage measurements

| Metric | Value |
|--------|-------|
| clues_filtered.csv rows | 457,262 |
| clues_wn_filtered.csv rows | 239,406 |
| Fraction retained after WN filter | 52.4% |
| Unique (definition, answer) pairs | 150,805 |
| Train pairs (target ~30%) | 45,241 (30.0%) |
| Validate pairs (target ~20%) | 30,161 (20.0%) |
| Test pairs (target ~50%) | 75,403 (50.0%) |
| Train rows | 72,107 (30.1%) |
| Validate rows | 47,933 (20.0%) |
| Test rows | 119,366 (49.9%) |
| vocabulary.csv words | 53,930 |
| vocabulary_val.csv words | 26,152 |
| Val vocab as fraction of full vocab | 48.5% |

### Article-stripping recovery

Expanded from Milestone II's `"a "` only to `"a "`, `"an "`, `"the "`,
`"to "` (Decision 16). Recovery counts for unique definitions:

| Prefix | Unique defs recovered | Unique answers recovered |
|--------|----------------------|-------------------------|
| `"a"` | 1,579 | 43 |
| `"an"` | 257 | 1 |
| `"the"` | 615 | 231 |
| `"to"` | 737 | 13 |

The three new prefixes recovered 1,609 additional unique definitions beyond
what `"a "` alone would have captured.

### Vocabulary overlap

| Category | Count |
|----------|-------|
| Words appearing as both definition and answer | 18,200 |
| Words appearing only as definition | 8,965 |
| Words appearing only as answer | 26,765 |

### Version discrepancy finding

Running NB 01 under the default `python3` kernel (NLTK 3.8.1) vs. the
`crossword` kernel (NLTK 3.9.2) produced a 49-row difference (239,455 vs.
239,406). Root cause: different NLTK versions resolve slightly different
sets of words to WordNet synsets. Led to Decision 17 (pin NLTK and
scikit-learn) and Decision 18 (version provenance in notebooks).

---

## Stage 2: Phrase Construction

**Notebook:** `02_phrase_construction_wn.ipynb` — completed 2026-04-13
**Environment:** Local, `crossword` kernel (NLTK 3.9.2, pandas 3.0.0, numpy 2.3.5)

### f_clue coverage

| Metric | Value |
|--------|-------|
| clues_wn_filtered.csv rows (input) | 239,406 |
| Rows with valid f_clue phrase | 239,406 (100.0%) |
| Rows dropped | 0 (0.0%) |

`tag_definition_in_surface` succeeded on every row — no ambiguous definition
placements in the wn_synset-filtered dataset.

### f_common_wndef coverage

| Metric | Value |
|--------|-------|
| vocabulary.csv words (input) | 53,930 |
| Words with valid f_common_wndef phrase | 53,930 (100.0%) |
| Self-referential phrases | 1,139 (2.1%) |
| clues_wndef_filtered.csv rows | 239,406 (100.0% of clues_wn_filtered) |
| vocabulary_wndef_val.csv words | 26,152 |
| Train fraction | 72,107 (30.1%) |
| Validate fraction | 47,933 (20.0%) |
| Test fraction | 119,366 (49.9%) |

100% coverage is expected: every word in vocabulary.csv has at least one synset
(the NB 01 filter), and synset[0] always has a definition. wndef vocabulary
equals the full vocabulary; wndef-filtered clues equal the full clue set.

1,139 words (2.1%) have self-referential definitions — the word appears
untagged in its own WordNet definition text (e.g., "admit: declare to be true
or admit the existence…"). These are flagged with `self_ref=True` in
`f_common_wndef.csv` for downstream subsetting but are not filtered out.

### f_common_wnex coverage

| Metric | Value |
|--------|-------|
| vocabulary.csv words (input) | 53,930 |
| Words with valid f_common_wnex phrase | 8,360 (15.5%) |
| clues_wnex_filtered.csv rows | 24,327 (10.2% of clues_wn_filtered) |
| vocabulary_wnex_val.csv words | 3,008 |
| Train fraction | 7,075 (29.1%) |
| Validate fraction | 4,825 (19.8%) |
| Test fraction | 12,427 (51.1%) |

Low coverage is expected: many WordNet synsets lack usage examples, and among
those with examples, the target word must appear exactly once for unambiguous
tagging. wnex is a strict subset of wndef (8,360 words in both, 0 in wnex
only, 45,570 in wndef only).

### Cross-f comparison

| Metric | Value |
|--------|-------|
| Words in wndef but not wnex | 45,570 |
| Words in wnex but not wndef | 0 |
| Words in both | 8,360 |
| Rows in clues_wndef_filtered but not clues_wnex_filtered | 215,079 |

---

## Stage 1d: g_stock f_clue Embedding Generation

**Script:** `scripts/embed_f_clue_gstock.py` — completed 2026-04-13
**Environment:** Great Lakes, `nlp_env` conda environment

| Metric | Value |
|--------|-------|
| Total rows embedded | 239,406 |
| Embedding array shape | (239406, 1024) |
| Embedding dtype | float32 |
| L2 norm range | [24.4009, 34.1953] |
| NaN values | 0 |
| All-zero rows | 0 |
| Encoding rate | 675 phrases/sec (batch_size=64) |
| Encoding time | 354.7s (5 min 51 sec) |
| Great Lakes partition | gpu (gl1002) |
| Date run | 2026-04-13 |

### Output files

| File | Size | Location |
|------|------|----------|
| `f_clue.npy` | 936 MB | `data/embeddings/g_stock/` |
| `f_clue_index.csv` | 4.9 MB | `data/embeddings/g_stock/` |

### Environment versions

| Package | Version |
|---------|---------|
| Python | 3.12.12 |
| torch | 2.5.1+cu121 |
| sentence-transformers | 5.2.2 |
| transformers | 4.52.3 |
| numpy | 2.3.5 |
| conda environment | nlp_env |

### Notes

- Used `SentenceTransformer.encode()` (not manual AutoModel extraction).
- `normalize_embeddings=False` — raw magnitudes preserved; downstream ATE
  computation normalizes as needed.
- Encountered `np.save()` temp file naming bug: numpy auto-appends `.npy`,
  causing atomic rename to fail. Files were saved successfully but required
  manual rename on Great Lakes. Bug fixed in commit c3653b9.

---

## Stage 3: g1_tokenspan Training

**Script:** `scripts/train_g1_tokenspan.py` (submitted via `scripts/train_g1_tokenspan.sh`) — completed 2026-04-13
**Environment:** Great Lakes, `nlp_env` conda environment

**Note on extraction method:** This model was trained using token span
extraction (averaging hidden states only within the `<t></t>` span), which
is NOT CALE's canonical embedding method. See Decision 20. The model is
named `g1_tokenspan` to distinguish it from the corrected `g1` (to be
trained with mean pooling). Results from this model should be compared
against `g_stock_tokenspan` (g_stock with token span extraction) for a fair
baseline, not against the canonical `g_stock`.

| Metric | Value |
|--------|-------|
| Triplet file | data/triplets/g1_train.csv |
| Training rows | 69,921 |
| Base model | gabrielloiseau/CALE-MBERT-en |
| Margin α | 1.0 |
| Learning rate | 2e-5 |
| Epochs | 3 |
| Batch size | 32 |
| Per-epoch loss | [0.488, 0.104, 0.013] |
| Great Lakes partition | gpu (Tesla V100-PCIE-16GB) |
| Wall-clock runtime | 49.0 min (0.82 h) |
| Date trained | 2026-04-13 |
| Great Lakes path | models/g1_tokenspan/model/ |

### Environment versions

| Package | Version |
|---------|---------|
| Python | 3.12.12 |
| torch | 2.5.1+cu121 |
| transformers | 4.57.6 |
| numpy | 2.3.5 |
| pandas | 3.0.0 |
| conda environment | nlp_env |

### Notes

- The SLURM script was missing `PYTHONUNBUFFERED=1`, so stdout was buffered
  and SLURM log progress appeared in chunks rather than streaming. Cosmetic
  only — no impact on training or saved artifacts.
- A harmless `conda activate` error was printed at job start but the correct
  environment was in use throughout. Cosmetic only — no impact on results.

---

## Stage 3: g1 Training (mean pooling)

**Script:** `scripts/train_g1.py` (submitted via `scripts/train_g1.sh`) — completed 2026-04-14
**Environment:** Great Lakes, `nlp_env` conda environment

This model uses CALE's canonical mean pooling (Decision 20). Compare against
g_stock (mean pooling) for a fair baseline.

| Metric | Value |
|--------|-------|
| Triplet file | data/triplets/g1_train.csv |
| Training rows | 69,921 |
| Base model | gabrielloiseau/CALE-MBERT-en |
| Margin α | 1.0 |
| Learning rate | 2e-5 |
| Epochs | 3 |
| Batch size | 32 |
| Per-epoch loss | [0.470, 0.111, 0.014] |
| Great Lakes partition | gpu (Tesla V100-PCIE-16GB) |
| Wall-clock runtime | 43.5 min (0.72 h) |
| Date trained | 2026-04-14 |
| Great Lakes path | models/g1/model/ |

### Environment versions

| Package | Version |
|---------|---------|
| Python | 3.12.12 |
| torch | 2.5.1+cu121 |
| transformers | 4.57.6 |
| numpy | 2.3.5 |
| pandas | 3.0.0 |
| conda environment | nlp_env |

---

## Stage 4: Embedding Generation (Validation Split)

### Extraction method finding

During implementation of `scripts/embed_val.py`, a consistency check comparing
AutoModel + token span extraction against `SentenceTransformer.encode()` on
g_stock f_clue phrases found mean cosine similarity of 0.926 — confirming
the two methods produce substantively different embeddings. This led to
Decision 20 (mean pooling is canonical for CALE) and the renaming of g1 to
g1_tokenspan. See Decision 20 for full evidence.

### g_stock_tokenspan — completed 2026-04-14

**Script:** `scripts/embed_val.py` (submitted via `scripts/embed_val_gstock_tokenspan.sh`)
**Environment:** Great Lakes, `nlp_env` conda environment

| Embedding | Shape | Indexed by |
|-----------|-------|------------|
| `f_clue_val.npy` | (47933, 1024) | `f_clue_val_index.csv` (47,933 rows) |
| `f_common_wndef_val.npy` | (26152, 1024) | `vocabulary_wndef_val.csv` (26,152 rows) |
| `f_common_wnex_val.npy` | (3008, 1024) | `vocabulary_wnex_val.csv` (3,008 rows) |

Transferred to local machine 2026-04-14. All shapes verified against
vocabulary/index files.

### g1_tokenspan — completed 2026-04-14

**Script:** `scripts/embed_val.py` (submitted via `scripts/embed_val_g1_tokenspan.sh`)
**Environment:** Great Lakes, `nlp_env` conda environment

| Embedding | Shape | Indexed by |
|-----------|-------|------------|
| `f_clue_val.npy` | (47933, 1024) | `f_clue_val_index.csv` (47,933 rows) |
| `f_common_wndef_val.npy` | (26152, 1024) | `vocabulary_wndef_val.csv` (26,152 rows) |
| `f_common_wnex_val.npy` | (3008, 1024) | `vocabulary_wnex_val.csv` (3,008 rows) |

Transferred to local machine 2026-04-14. All shapes verified against
vocabulary/index files.

### g_stock (mean pooling) — completed 2026-04-14

**Script:** `scripts/embed_val.py` (submitted via `scripts/embed_val_gstock.sh`)
**Environment:** Great Lakes, `nlp_env` conda environment

| Embedding | Shape | Indexed by |
|-----------|-------|------------|
| `f_clue_val.npy` | (47933, 1024) | `f_clue_val_index.csv` (47,933 rows) |
| `f_common_wndef_val.npy` | (26152, 1024) | `vocabulary_wndef_val.csv` (26,152 rows) |
| `f_common_wnex_val.npy` | (3008, 1024) | `vocabulary_wnex_val.csv` (3,008 rows) |

Total runtime 2.9 min. Transferred to local machine 2026-04-14. All shapes
verified against vocabulary/index files.

### g1 (mean pooling) — completed 2026-04-14

**Script:** `scripts/embed_val.py` (submitted via `scripts/embed_val_g1.sh`)
**Environment:** Great Lakes, `nlp_env` conda environment

| Embedding | Shape | Indexed by |
|-----------|-------|------------|
| `f_clue_val.npy` | (47933, 1024) | `f_clue_val_index.csv` (47,933 rows) |
| `f_common_wndef_val.npy` | (26152, 1024) | `vocabulary_wndef_val.csv` (26,152 rows) |
| `f_common_wnex_val.npy` | (3008, 1024) | `vocabulary_wnex_val.csv` (3,008 rows) |

Total runtime 2.9 min. Transferred to local machine 2026-04-14. All shapes
verified against vocabulary/index files.

### Embedding scripts refactor — 2026-04-19

**Scripts:** `scripts/embedding_utils.py`, `scripts/embed_clue.py`,
`scripts/embed_vocab.py` (replacing `embed_f_clue_gstock.py` and
`embed_val.py`, archived to `scripts/archive/` after verification passes).

**Motivation:** Produce full-vocabulary wnex embeddings for g_stock and g1
(8,360 words each) and provide a cleaner two-script architecture for
future models and phrase types.

**Verification:** Seven runs (`verify_embedding_scripts.sh`) completed
2026-04-19 on Tesla V100-PCIE-16GB. All seven PASSED with mean cosine
1.000000 — including V1 (AutoModel meanpool vs. SentenceTransformer.encode()
reference), which matched perfectly rather than the ~0.999 expected. Total
verification runtime ~13 min. Old scripts archived to `scripts/archive/`.

### g_stock full-vocab wnex — completed 2026-04-19

**Script:** `scripts/embed_vocab.py` via `scripts/embed_wnex_full_gstock.sh`
**Environment:** Great Lakes, `nlp_env` conda environment

| Metric | Value |
|--------|-------|
| Vocabulary size | 8,360 |
| Embedding array shape | (8360, 1024) |
| Embedding dtype | float32 |
| L2 norm range | [26.3180, 33.4561] |
| Encoding rate | 380 phrases/sec (batch_size=64) |
| Encoding time | 22.0s |
| Total runtime | 26.0s (0.4 min) |
| Great Lakes partition | gpu (Tesla V100-PCIE-16GB) |
| Date run | 2026-04-19 |

Output: `data/embeddings/g_stock/f_common_wnex.npy` (32.7 MB), indexed by
`vocabulary_wnex.csv`.

### g1 full-vocab wnex — completed 2026-04-19

**Script:** `scripts/embed_vocab.py` via `scripts/embed_wnex_full_g1.sh`
**Environment:** Great Lakes, `nlp_env` conda environment

| Metric | Value |
|--------|-------|
| Vocabulary size | 8,360 |
| Embedding array shape | (8360, 1024) |
| Embedding dtype | float32 |
| L2 norm range | [25.5698, 31.4078] |
| Encoding rate | 379 phrases/sec (batch_size=64) |
| Encoding time | 22.0s |
| Total runtime | 25.8s (0.4 min) |
| Great Lakes partition | gpu (Tesla V100-PCIE-16GB) |
| Date run | 2026-04-19 |

Output: `data/embeddings/g1/f_common_wnex.npy` (32.7 MB), indexed by
`vocabulary_wnex.csv`.

### g_stock full-vocab wndef — completed 2026-04-19

**Script:** `scripts/embed_vocab.py` via `scripts/embed_wndef_full_gstock.sh`
**Environment:** Great Lakes, `nlp_env` conda environment

| Metric | Value |
|--------|-------|
| Vocabulary size | 53,930 |
| Embedding array shape | (53930, 1024) |
| Embedding dtype | float32 |
| L2 norm range | [23.5365, 33.6164] |
| Encoding rate | 317 phrases/sec (batch_size=64) |
| Encoding time | 170.1s |
| Total runtime | 179.4s (3.0 min) |
| Great Lakes partition | gpu (Tesla V100-PCIE-16GB) |
| Date run | 2026-04-19 |

Output: `data/embeddings/g_stock/f_common_wndef.npy` (210.7 MB), indexed by
`vocabulary_wndef.csv`.

### g1 full-vocab wndef — completed 2026-04-19

**Script:** `scripts/embed_vocab.py` via `scripts/embed_wndef_full_g1.sh`
**Environment:** Great Lakes, `nlp_env` conda environment

| Metric | Value |
|--------|-------|
| Vocabulary size | 53,930 |
| Embedding array shape | (53930, 1024) |
| Embedding dtype | float32 |
| L2 norm range | [24.0275, 30.6785] |
| Encoding rate | 321 phrases/sec (batch_size=64) |
| Encoding time | 167.8s |
| Total runtime | 175.1s (2.9 min) |
| Great Lakes partition | gpu (Tesla V100-PCIE-16GB) |
| Date run | 2026-04-19 |

Output: `data/embeddings/g1/f_common_wndef.npy` (210.7 MB), indexed by
`vocabulary_wndef.csv`.

### Environment versions (all five embedding jobs on 2026-04-19)

| Package | Version |
|---------|---------|
| Python | 3.12.12 |
| torch | 2.5.1+cu121 |
| transformers | 4.57.6 |
| numpy | 2.3.5 |
| pandas | 3.0.0 |
| conda environment | nlp_env |

### Stage 4 Verification (NB 04) — completed 2026-04-14

**Notebook:** `04_embedding_verification.ipynb`
**Environment:** Local (CPU)

All four embedding sets (g_stock, g_stock_tokenspan, g1, g1_tokenspan)
verified across all three phrase types (f_clue_val, f_common_wndef_val,
f_common_wnex_val). Five hard checks passed:

| Criterion | Result |
|---|---|
| All 12 shapes match expected | PASS |
| No NaN anywhere | PASS |
| No all-zero rows | PASS |
| f_clue_val index files consistent | PASS |
| No off-diagonal mean cosine ≥ 0.999 | PASS |

Max off-diagonal mean cosine across all three matrices: 0.9212
(g_stock ↔ g_stock_tokenspan on f_clue_val).

#### Cross-model divergence (g_stock ↔ g1, both meanpool)

| Phrase type | Mean cosine |
|---|---|
| f_clue_val | 0.296 |
| f_common_wndef_val | 0.391 |
| f_common_wnex_val | 0.326 |

Fine-tuning moved embeddings dramatically across all three phrase types, not
just the wndef phrases the model was trained on. Whether this reflects
compression (format-specific overfitting) or semantic reorganization is
deferred to Stage 5.

Full pairwise matrices and per-pair detail tables in
`outputs/04_embedding_verification-results.md`.

---

## Stage 5: Model Evaluation (NB 05)

**Notebook:** `05_model_evaluation.ipynb` — completed 2026-04-14,
revised 2026-04-19
**Environment:** Local (CPU)
**Scope:** Canonical mean-pooling models only (g_stock and g1). Tokenspan
variants are out of scope per Decision 20.

### Validation triplet accuracy (wndef, full-vocabulary)

Validation triplet file: `data/triplets/g1_val.csv` (46,506 rows).
Full-vocabulary wndef embeddings (Decision 23) resolved 100% of triplets,
eliminating the ~41% dropout from the initial val-only run (Decision 21).

| Metric | g_stock | g1 |
|--------|---------|-----|
| Triplet accuracy (% correct) | 38.8% | 90.0% |
| Mean margin (cos_pos − cos_neg) | −0.054 | +0.125 |
| Median margin | −0.044 | +0.122 |
| N triplets evaluated | 46,506 | 46,506 |

### Cross-f triplet accuracy (matched comparison)

Matched subset: 2,985 triplets (6.4%) where both answer and distractor
are in `vocabulary_wnex.csv`. The same triplets evaluated under wndef
and wnex embeddings:

| Metric | g_stock (wndef) | g1 (wndef) | g_stock (wnex) | g1 (wnex) |
|--------|----------------|------------|----------------|-----------|
| Triplet accuracy | 45.6% | 88.3% | 40.3% | 67.2% |
| Mean margin | −0.023 | +0.109 | −0.052 | +0.041 |
| Median margin | −0.017 | +0.110 | −0.042 | +0.041 |

**Key finding:** g1's wnex triplet accuracy (67.2%) is well above g_stock's
wnex baseline (40.3%), demonstrating that g1 learned discriminative
structure that partially transfers to a phrase type it was never trained on.
However, accuracy drops from 88.3% (wndef) to 67.2% (wnex), and mean
margin drops from +0.109 to +0.041, indicating the transfer is partial —
g1 learned more about wndef-specific structure than about general semantic
discrimination.

### Collapse detection (val-only)

Random pairwise cosine among 50,000 random word pairs (random_state=42):

| Model | f_common_wndef_val mean | f_common_wnex_val mean |
|-------|------------------------|------------------------|
| g_stock | 0.398 | 0.299 |
| g1 | 0.571 | 0.506 |

Embedding variance and effective dimensionality:

| Model | Phrase type | Total variance | Eff. dim (participation ratio) |
|-------|-------------|---------------|-------------------------------|
| g_stock | f_common_wndef_val | 13,454,020 | 43.6 |
| g1 | f_common_wndef_val | 8,720,421 | 48.7 |
| g_stock | f_common_wnex_val | 1,943,473 | 47.7 |
| g1 | f_common_wnex_val | 1,228,944 | 77.5 |

Compression occurred on both wndef (trained on) and wnex (never trained on).
Total variance dropped ~35% on both phrase types. Effective dimensionality
increased slightly, indicating uniform contraction rather than dimensional
collapse.

### T=0 and T=1 similarity distributions (wndef, val-only)

47,933 evaluation pairs from clues_val.csv (100% resolved):

| Metric | g_stock | g1 |
|--------|---------|-----|
| T=0 mean (def vs ans) | 0.576 | 0.715 |
| T=1 mean (clue-def vs ans) | 0.513 | 0.591 |
| ATE (mean of T=1 − T=0) | −0.063 | −0.124 |

T=0 rose by +0.139 while T=1 rose by only +0.078. The ATE became more
negative — the same pattern observed in the NB 09 prior work.

### T=0 and T=1 similarity distributions (wnex, val-only)

4,825 evaluation pairs where both definition_wn and answer_wn are in the
wnex validation vocabulary (10.1% of clues_val):

| Metric | g_stock | g1 |
|--------|---------|-----|
| T=0 mean (def vs ans) | 0.495 | 0.590 |
| T=1 mean (clue-def vs ans) | 0.486 | 0.547 |
| ATE (mean of T=1 − T=0) | −0.009 | −0.043 |

### Matched ATE comparison (wndef vs wnex on identical pairs)

4,825 pairs resolving under both wndef_val and wnex_val:

| Phrase format | g_stock ATE | g1 ATE |
|---------------|-------------|--------|
| wndef | −0.071 | −0.134 |
| wnex | −0.009 | −0.043 |

**Key finding:** The wnex ATE is much less negative than wndef under both
models. g1 roughly doubles the ATE magnitude on both phrase types (wndef:
−0.071 → −0.134; wnex: −0.009 → −0.043). The wndef format itself appears
to carry format-specific signal that amplifies the measured misdirection
effect — or the wnex subset of clues has inherently less misdirection.
Either way, g1's core problem (making the ATE more negative) persists
across phrase types, pointing to the triplet design (T_1) rather than
the phrase format as the root cause.

### RSA (Spearman correlation of pairwise cosines, val-only)

1,000 words sampled per phrase type (random_state=42):

| Phrase type | Spearman ρ | p-value |
|-------------|-----------|---------|
| f_common_wndef_val | 0.112 | <0.001 |
| f_common_wnex_val | 0.075 | <0.001 |

The near-zero ρ values indicate the pairwise similarity structure was
fundamentally reorganized by fine-tuning, not merely shifted.

### Figures

- `outputs/figures/05_val_triplet_accuracy.png`
- `outputs/figures/05_crossf_triplet_accuracy.png`
- `outputs/figures/05_collapse_pairwise_cosine.png`
- `outputs/figures/05_collapse_singular_values.png`
- `outputs/figures/05_t0_t1_wndef_distributions.png`
- `outputs/figures/05_t0_t1_wnex_distributions.png`

Full numerical results in `outputs/05_model_evaluation-results.md`.

---

## Stage 6: Final Evaluation

*Locked — do not populate until final g is chosen and documented in DECISIONS.md.*
