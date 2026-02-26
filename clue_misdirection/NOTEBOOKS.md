# Notebook Inventory — Clue Misdirection

## Directory Layout

```
clue_misdirection/notebooks/
├── 01_data_cleaning.ipynb          # Pipeline notebooks (run in order)
├── 02_embedding_generation.ipynb
├── 03_feature_engineering.ipynb
├── ...
└── archive/                        # Prior exploratory notebooks (reference only)
    ├── Data_Cleaning_for_Clues_-_Pairs_in_WordNet.ipynb
    ├── Data_Cleaning_for_Clues__1_.ipynb
    ├── Hans_Supervised_Learning_EDA.ipynb
    ├── Hans_Supervised_Learning_EDA_WITH_OUTPUTS.ipynb
    ├── Hans_Supervised_Learning.ipynb
    ├── Hans_Supervised_Learning_Models.ipynb
    ├── Hans_Control_Experiment_Normal_English.ipynb
    └── Hans_Negative_Strategies_Experiment.ipynb
```

Pipeline notebooks live directly in `notebooks/`. Prior exploratory notebooks
from Victoria, Sahana, and Hans live in `notebooks/archive/` for reference.

## Status Key

- ✅ **Usable** — can be incorporated directly or with minor edits
- 🔄 **Needs rework** — contains useful code/insights but needs significant changes
- 📋 **Reference only** — superseded or exploratory; mine for insights, don't run
- ❌ **Not yet created** — needs to be built

---

## Indicator Clustering Notebooks (reference for patterns)

These notebooks live in the `indicator_clustering/` component. Do not modify
them (a teammate is actively working on that component), but use them as
references for notebook structure, coding conventions, and the embedding
deduplication pattern.

| Notebook | Author | Status | Description | Relevance |
|----------|--------|--------|-------------|-----------|
| `01_data_cleaning.ipynb` | Victoria | ✅ Usable (as reference) | Stage 1 cleaning for indicator clustering. 131 cells. Reads from `clues_raw.csv` and other `*_raw.csv` files. Demonstrates the notebook header format (intellectual lineage block), path setup with environment auto-detection (Colab/local), and summary reporting conventions. | Reference for notebook structure, header format, path conventions |
| `02_embedding_generation.ipynb` | Victoria | ✅ Usable (as reference) | Stage 2 embedding generation for indicator clustering. 16 cells. Loads deduplicated unique indicator strings, embeds each once with BAAI/bge-m3 via `sentence-transformers`, saves `.npy` + index CSV, and includes a verification cell. | **Key reference** for our Step 2: the deduplicated embedding pattern, index file contract, and verification approach |

---

## Victoria's Data Cleaning Notebooks (`notebooks/archive/`)

These contain the most relevant prior work for our Step 1 (data cleaning).

| Notebook | Size | Cells | Status | Description |
|----------|------|-------|--------|-------------|
| `Data_Cleaning_for_Clues_-_Pairs_in_WordNet.ipynb` | 76 KB | 85 (64 code, 21 markdown) | 🔄 Needs rework | **Primary base for Step 1.** EDA and data cleaning focused on definition–answer pairs. Key contributions: creates `surface` column by stripping answer format from `clue` text (Cell 14); creates `surface_normalized` for validation (Cell 17); parses double-definition clues by splitting on `/` into `definition_list` (Cells 26–33); validates definitions appear as intact words in surface text; computes `answer_format` from the answer string (Cell 23). WordNet filtering was started (Cells 59+) but not completed — the notebook drifts into exploratory WordNet analysis and falls back to single-word filtering at the end (Cell 77+). Has saved cell outputs. |
| `Data_Cleaning_for_Clues__1_.ipynb` | 58 KB | 20 (17 code, 3 markdown) | 📋 Reference | Earlier, simpler version of clue data cleaning (Victoria/Sahana). Reads from `clues_raw.csv`. Filters to single-word definitions and answers, drops NaNs (notes that ~36% of single-word definitions are NaN), and joins with indicator tables. Minimal markdown documentation. Has saved cell outputs. Check for any filter steps not covered by the Pairs in WordNet notebook. |

---

## Hans's Notebooks (`notebooks/archive/`)

Hans's exploratory work using `clues_single_word.csv` and `all-mpnet-base-v2`.
These notebooks are complete and contain valuable code and findings, but the
dataset (single-word only), embedding model, and feature set are all being
replaced in our pipeline per plan v3. Use as reference for code patterns and
to verify our new results against his preliminary findings.

| Notebook | Size | Cells | Status | Description | Plan Steps |
|----------|------|-------|--------|-------------|------------|
| `Hans_Supervised_Learning_EDA.ipynb` | 21 KB | 42 (33 code, 9 markdown) | 📋 Reference | EDA for supervised learning data preparation. Filters to single-word definitions and answers, explores data quality, investigates WordNet synonym coverage (~10.5% of answers are direct synonyms), saves `clues_single_word.csv`. Includes whole-word boundary verification using `\b` regex (ported from Victoria's indicator cleaning). Has saved cell outputs. | 1 |
| `Hans_Supervised_Learning_EDA_WITH_OUTPUTS.ipynb` | 29 KB | 31 (22 code, 9 markdown) | 📋 Reference | Condensed version of the EDA notebook with different cell organization but same purpose. Has saved cell outputs. Can be ignored if reading the main EDA notebook. | 1 |
| `Hans_Supervised_Learning.ipynb` | 465 KB | 34 (21 code, 13 markdown) | 🔄 Needs rework | **Main retrieval-based misdirection analysis.** Loads `clues_single_word.csv`, deduplicates to unique (definition, answer) pairs, samples 10,000 pairs, embeds definitions and answers with `all-mpnet-base-v2`, and runs retrieval evaluation (rank true answer among all 8,598 candidates by cosine similarity). Compares context-free vs. context-informed embeddings. Key finding: +512 mean rank degradation with context. Also contains a 3-model comparison (MiniLM, MPNet, BGE-M3) and misdirection-by-wordplay-type analysis. Has saved cell outputs. Code for embedding generation, retrieval evaluation, and cosine similarity computation is reusable with model updates. | 2, 4 |
| `Hans_Supervised_Learning_Models.ipynb` | 255 KB | 27 (15 code, 12 markdown) | 🔄 Needs rework | **Classification-based misdirection analysis.** Builds a balanced binary dataset (5,000 real + 5,000 random distractor pairs), engineers 10 features (3 embedding, 3 WordNet, 4 surface), trains KNN/LogReg/RF with 5-fold stratified CV, runs ablation (remove one feature at a time), sensitivity analysis (training set size), and failure analysis (3 categories). Key finding: <0.5pp context gap; `wn_path_sim` is the dominant feature. Includes whole-word verification filter. Has saved cell outputs. The CV scaffolding, feature engineering functions, ablation approach, and evaluation reporting can all be adapted for our expanded 47-feature pipeline. | 3, 6, 8, 9, 10, 11, 12 |
| `Hans_Control_Experiment_Normal_English.ipynb` | 158 KB | 22 (13 code, 9 markdown) | 📋 Reference | Control experiment comparing cryptic crossword misdirection against normal English. Builds (sentence, word, related_word) triples from WordNet hypernym/hyponym pairs found in Brown + Reuters corpora. Uses same 10 features and 3 models. Key finding: in normal English, removing context features *hurts* accuracy (context is useful), while in cryptic crosswords, removing context *helps* (the "sign flip"). Excludes `wn_path_sim` to prevent it from masking context effects. Has saved cell outputs. Not part of plan v3 pipeline, but the sign-flip finding is important background for interpreting our results. | — |
| `Hans_Negative_Strategies_Experiment.ipynb` | 189 KB | 21 (10 code, 11 markdown) | 📋 Reference | Tests three distractor generation strategies (known answers, clue vocabulary, generic English vocabulary) across both cryptic and normal English datasets. Key finding: misdirection signal (<1pp context gap for cryptic) is robust across all strategies. Has saved cell outputs. Informed the harder dataset design in plan v3 (Decision 6) but uses a different approach — our pipeline uses cosine-similarity-based distractors instead. | 5, 7 |

---

## Pipeline Notebooks (`notebooks/`)

These are the target notebooks that implement the plan, living directly in
the `notebooks/` directory. Each should follow the coding standards in
CLAUDE.md (intellectual lineage header, explanatory markdown cells written
for NLP newcomers, summary cell at the end).

| Notebook | Plan Steps | Status | Sources to Draw From |
|----------|------------|--------|----------------------|
| `01_data_cleaning.ipynb` | 1 | ❌ Not yet created | Victoria's *Pairs in WordNet* notebook (primary base: surface extraction, double-definition parsing, answer format validation) + Victoria/Sahana's *Data Cleaning for Clues* (secondary check for additional filters) + Hans's *EDA* notebook (whole-word verification, data quality checks) + indicator_clustering *01_data_cleaning* (notebook structure reference) |
| `02_embedding_generation.ipynb` | 2 | ✅ Complete | Step 2 embedding generation. 27 cells. CPU portion: loads clues_filtered.csv, derives WordNet-ready strings, constructs CALE context phrases with `<t></t>` delimiters for all 7 embedding types, saves phrase CSVs. GPU portion: `scripts/embed_phrases.py` encodes all phrases with CALE-MBERT-en on Great Lakes V100, produces 6 output files (~1.8 GB). Verification cells validate shapes, consistency, and semantic sense of embeddings. |
| `03_feature_engineering.ipynb` | 3 | ✅ Complete | 47 features (15 context-free cosine + 6 context-informed cosine + 22 WordNet relationship + 4 surface) computed for 240,211 rows. Output: `data/features_all.parquet`. Includes merge fix for double-definition clues (composite key on `clue_id` + `definition`) and standalone feature functions designed for later extraction to `scripts/feature_utils.py` (Decision 18). |
| `04_retrieval_analysis.ipynb` | 4 | ✅ Complete | Step 4 retrieval analysis. 24 cells. Loads CALE embeddings (definition, answer, clue-context) + `clues_filtered.csv` + `clue_context_phrases.csv`. Runs 4×3 retrieval matrix (4 definition conditions × 3 answer conditions) over 127,608 unique pairs against 45,254 candidate answers. Clue Context uses median-rank aggregation across clue rows per pair (Decision 5). Outputs: `outputs/retrieval_results_unique_pairs.csv`, `outputs/retrieval_results_all_rows.csv`, `outputs/figures/retrieval_bar_chart.png`, `outputs/figures/retrieval_heatmap.png`. Key finding: misdirection confirmed — Clue Context roughly doubles median rank vs. Allsense context-free (2,160 vs. 1,015). |
| `05_dataset_construction.ipynb` | 5, 7 | ❌ Not yet created | Hans's *Models* notebook (random distractor generation for easy dataset) + Hans's *Negative Strategies* notebook (reference for distractor strategies — our harder dataset uses cosine-similarity-based selection per Decision 6) |
| `06_experiments_easy.ipynb` | 6 | ❌ Not yet created | Hans's *Models* notebook (KNN/LogReg/RF training, CV setup, evaluation reporting — adapt to GroupKFold and 47 features) |
| `07_experiments_harder.ipynb` | 8 | ❌ Not yet created | Hans's *Models* notebook (same modeling scaffolding as Step 6) |
| `08_results_and_evaluation.ipynb` | 9, 10, 11, 12 | ❌ Not yet created | Hans's *Models* notebook (feature importance, ablation, sensitivity, failure analysis — expand to 47-feature groups and harder dataset) |

---

## Scripts (`scripts/`)

Standalone scripts for GPU or batch workloads that are too heavy or
environment-specific for notebook cells.

| Script | Description |
|--------|-------------|
| `scripts/embed_phrases.py` | GPU embedding script for Step 2. Loads phrase CSVs, encodes with CALE, saves `.npy` + index CSVs. Supports `--sample N` for testing and `--batch-size`. |
| `scripts/embed_phrases.sh` | SLURM submission script for Great Lakes (1 V100, 32 GB, 1 hr). |
| `scripts/embed_phrases_test.sh` | SLURM test script (`--sample 100`, 10 min). |
| `scripts/feature_utils.py` | ❌ Not yet created. Shared feature computation functions for NB 05 and NB 07 (Decision 18). Extracted from NB 03 logic. Computes cosine similarity, WordNet relationship, and surface features for arbitrary (definition, answer) pairs. Created when NB 05 is built. |

---

## Other Reference Documents

| File | Description |
|------|-------------|
| `CONTEXT.md` | Hans's comprehensive writeup of his prior work (18 KB). Covers all four of his work streams (retrieval, classification, control experiment, negative strategies), feature reference, key findings, and open questions. **Read for background context**, but plan v3 supersedes his experimental design. |
| `supervised_learning_plan_v3.docx` | The authoritative design document. All `.md` files and pipeline notebooks are derived from it. |

---

## Recommended Reading Order

When starting work, read these in order to build context:

1. `PLAN.md` — understand the 12-step pipeline plan
2. `CLAUDE.md` — coding standards, terminology, and project structure
3. `CONTEXT.md` — Hans's framing and prior findings
4. Victoria's *Data Cleaning for Clues - Pairs in WordNet.ipynb* — the primary
   base for Step 1
5. Indicator_clustering `02_embedding_generation.ipynb` — the deduplication
   pattern for Step 2
6. Hans's *Hans_Supervised_Learning.ipynb* — the main existing retrieval and
   embedding codebase
7. Hans's *Hans_Supervised_Learning_Models.ipynb* — the classification
   scaffolding for Steps 6–8
