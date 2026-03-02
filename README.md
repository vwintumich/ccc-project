# Exploring Wordplay and Misdirection in Cryptic Crossword Clues

University of Michigan MADS Capstone Project — SIADS 696, Winter 2026
Team: Victoria Winters, Sahana Sundar, Nathan Cantwell, Hans Li
Faculty Advisor: Dr. Kevyn Collins-Thompson

## Project Overview

This project applies natural language processing to cryptic crossword clues
(CCCs), a domain that poses unique challenges for language models due to
deliberate semantic misdirection and strict hidden grammatical structure.

The project has two independent components that investigate complementary
aspects of CCC language:

### 1. Indicator Clustering (`indicator_clustering/`)

Unsupervised clustering of 12,622 unique CCC indicator words and phrases to
explore whether their semantic embeddings naturally reflect the structure of
CCC wordplay types. Uses BGE-M3 embeddings (1024-dim) with UMAP
dimensionality reduction, HDBSCAN, and agglomerative clustering.

**Key findings:**

- No natural k=8 grouping corresponding to the eight standard wordplay types
  emerges from unconstrained clustering. Agglomerative silhouette improves
  monotonically with k (0.246 at k=4 to 0.431 at k=250) with no plateau.
- Wordplay types with distinct metaphorical bases separate cleanly:
  Homophone vs. Reversal achieves ARI=0.611 (k=2). Types sharing
  placement/containment metaphors (Hidden, Container, Insertion) are
  inseparable (ARI=0.045 at k=3) — a 13.5x contrast.
- Seed-word constraints provide only marginal purity improvement over
  unconstrained clustering (0.598 vs. 0.563).
- HDBSCAN reveals meaningful sub-structure within the dominant anagram type
  (6,610 indicators), identifying 149 conceptual metaphor sub-clusters
  (repair, disorder, movement, transformation, etc.).

### 2. Clue Misdirection (`clue_misdirection/`)

Supervised learning experiments quantifying how much the surface text of a
cryptic clue misleads embedding-based models attempting to connect a definition
to its answer. Uses CALE-MBERT-en embeddings with `<t></t>` target-word
delimiters, retrieval analysis, and binary classification with 47 engineered
features.

**Key findings:**

- Retrieval analysis confirms misdirection: embedding a definition in clue
  context roughly doubles the median retrieval rank from 1,015 to 2,160
  (out of 45,254 candidates), a 43% relative decrease in top-10 hit rate.
- On the harder classification task (cosine-similarity distractors), Random
  Forest achieves 0.757 accuracy / 0.827 ROC AUC with 32 features.
  Context-informed features improve accuracy by +5.5 to +9.4pp across all
  models, indicating that context features, while harmful for retrieval,
  are informative for discriminating real from distractor pairs.
- Group-level ablation: removing context-informed features causes the largest
  drop (-10.8pp), followed by relationship features (-8.7pp); surface
  features contribute minimally (-0.6pp).

## Data

Both components use George Ho's CCC dataset (660,613 clues), available under
the Open Database License (ODbL v1.0):

- Download: https://cryptics.georgeho.org/data.db
- The file will download as `data.sqlite3`
- Place it in `indicator_clustering/data/` and/or `clue_misdirection/data/`
  before running notebooks

See `DATA_LICENSE` for full attribution and license details.

## Repository Structure

```
ccc-project/
  indicator_clustering/          # Unsupervised clustering component
    notebooks/                   # Pipeline notebooks 00–07 (run in order)
      archive/                   # Superseded and exploratory notebooks
    data/                        # Data files (see setup above)
    outputs/                     # Metrics CSVs and generated figures
      figures/report/            # Publication-quality figures
    docs/                        # Rendered HTML notebooks (GitHub Pages)
      README.md                  # Index of rendered notebooks
    CLAUDE.md                    # Claude Code project configuration
    PROJECT_OVERVIEW.md          # Research context and task definitions
    DOMAIN_KNOWLEDGE.md          # CCC wordplay taxonomy
    FINDINGS_AND_DECISIONS.md    # Empirical results and advisor guidance
    OPEN_QUESTIONS.md            # Unresolved decisions

  clue_misdirection/             # Supervised learning component
    notebooks/                   # Pipeline notebooks 00–08 (run in order)
      archive/                   # Superseded and exploratory notebooks
    scripts/                     # Python and shell scripts for GPU jobs
    data/                        # Data files and embeddings (~1.8 GB)
    outputs/                     # Results CSVs and generated figures
      figures/                   # Retrieval, importance, and evaluation plots
    docs/                        # Rendered HTML notebooks (GitHub Pages)
      README.md                  # Index of rendered notebooks
    CLAUDE.md                    # Claude Code project configuration
    PLAN.md                      # 12-step pipeline plan
    FINDINGS.md                  # Research findings log
    DECISIONS.md                 # Key decisions and rationale
    DATA.md                      # Data dictionary and schemas
    NOTEBOOKS.md                 # Notebook descriptions and purposes
    requirements.txt             # Component-specific dependencies

  README.md                      # This file
  requirements.txt               # Python dependencies (CPU/analysis)
  LICENSE                        # MIT License
  DATA_LICENSE                   # ODbL v1.0 for derived datasets
```

## Rendered Notebooks

Rendered notebooks with full outputs (figures, tables, metrics) are available
via GitHub Pages:

**https://vwintumich.github.io/ccc-project/**

- [Indicator Clustering notebooks](indicator_clustering/docs/README.md) —
  8 notebooks covering data extraction through evaluation and control
  experiments
- [Clue Misdirection notebooks](clue_misdirection/docs/README.md) —
  9 notebooks covering model comparison through results and failure analysis

## Running the Notebooks

Each component's notebooks form a sequential pipeline — run them in numerical
order. Later notebooks depend on outputs from earlier stages.

### Indicator Clustering

| Stage | Notebook | Environment |
|-------|----------|-------------|
| 0 | 00_data_extraction | Local |
| 1 | 01_data_cleaning | Local |
| 2 | 02_embedding_generation | GPU (Great Lakes / Colab) |
| 3 | 03_dimensionality_reduction | GPU (Great Lakes / Colab) |
| 4 | 04_clustering | Local |
| 5 | 05_constrained_and_targeted | Local |
| 6 | 06_evaluation_and_figures | Local |
| 7 | 07_definitions_control | GPU for Section 2; Local otherwise |

### Clue Misdirection

| Stage | Notebook | Environment |
|-------|----------|-------------|
| 0 | 00_model_comparison | Local |
| 1 | 01_data_cleaning | Local |
| 2 | 02_embedding_generation | GPU (Great Lakes / Colab) |
| 3 | 03_feature_engineering | Local |
| 4 | 04_retrieval_analysis | Local |
| 5 | 05_dataset_construction | Local |
| 6 | 06_experiments_easy | Local (or Great Lakes for full data) |
| 7 | 07_experiments_harder | Local (or Great Lakes for full data) |
| 8 | 08_results_and_evaluation | Local |

### Note on GPU steps

Embedding generation notebooks require a GPU and `sentence-transformers` /
`torch`. They are designed to run on the University of Michigan Great Lakes
cluster or Google Colab with a GPU runtime (Runtime > Change runtime type >
T4 GPU). Generated embedding files are not included in this repository due to
size and must be produced by running the relevant notebooks.

## Environment

**CPU/analysis dependencies** (both components):

```bash
pip install -r requirements.txt
```

**GPU/embedding dependencies** (clue_misdirection):

```bash
pip install -r clue_misdirection/requirements.txt
```

This installs `sentence-transformers`, `torch`, and `pyarrow` in addition to
the base dependencies. On Great Lakes and Colab, `torch` with CUDA support is
pre-installed.

Key libraries: scikit-learn, hdbscan, umap-learn, pandas, numpy, nltk,
matplotlib, seaborn. The indicator clustering component uses BGE-M3
(`BAAI/bge-m3`) and the clue misdirection component uses CALE-MBERT-en
(`oskar-h/cale-modernbert-base`); both are downloaded automatically by
`sentence-transformers` on first use.

## References

- Ho, G. (2022). A Dataset of Cryptic Crossword Clues.
  https://cryptics.georgeho.org/
- Liétard, B., & Loiseau, G. (2025). CALE: Concept-aligned embeddings for both within-lemma and inter-lemma sense differentiation (arXiv:2508.04494). arXiv. __https://doi.org/10.48550/arXiv.2508.04494__
- Tiernan, A., & Runnalls, L. (2025). Minute Cryptic. St. Martin's Griffin.
