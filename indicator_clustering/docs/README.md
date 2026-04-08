# Rendered Notebooks — CCC Indicator Clustering

These HTML files are rendered versions of the project notebooks with full outputs
(figures, tables, metrics). Each file can be viewed directly in a browser via
GitHub Pages.

The notebooks form a sequential pipeline. Each stage builds on the outputs of the
previous one.

## Current Renders

| Stage | Notebook | HTML Render | Description |
|-------|----------|-------------|-------------|
| 0 | 00_data_extraction | [00_data_extraction_2026-02-26_1502.html](https://vwintumich.github.io/ccc-project/indicator_clustering/docs/00_data_extraction_2026-02-26_1502.html) | Extract 6 tables from SQLite database to CSV |
| 1 | 01_data_cleaning | [01_data_cleaning.html](https://vwintumich.github.io/ccc-project/indicator_clustering/docs/01_data_cleaning.html) | Verify indicators, compute ground-truth labels, export clean datasets |
| 2 | 02_embedding_generation | [02_embedding_generation_2026-02-26_1524.html](https://vwintumich.github.io/ccc-project/indicator_clustering/docs/02_embedding_generation_2026-02-26_1524.html) | Generate 1024-dim BGE-M3 embeddings for 12,622 unique indicators |
| 3 | 03_dimensionality_reduction | [03_dimensionality_reduction_2026-02-26_1503.html](https://vwintumich.github.io/ccc-project/indicator_clustering/docs/03_dimensionality_reduction_2026-02-26_1503.html) | PCA and UMAP reduction to 10D (clustering) and 2D (visualization) |
| 4 | 04_clustering | [04_clustering_2026-02-26_1516.html](https://vwintumich.github.io/ccc-project/indicator_clustering/docs/04_clustering_2026-02-26_1516.html) | Unconstrained HDBSCAN and agglomerative clustering exploration |
| 5 | 05_constrained_and_targeted | [05_constrained_and_targeted_2026-02-26_1518.html](https://vwintumich.github.io/ccc-project/indicator_clustering/docs/05_constrained_and_targeted_2026-02-26_1518.html) | Seed-word constraints, subset experiments, anagram sub-clustering |
| 6 | 06_evaluation_and_figures | [06_evaluation_and_figures.html](https://vwintumich.github.io/ccc-project/indicator_clustering/docs/06_evaluation_and_figures.html) | Publication-quality figures and systematic evaluation for the report (**Note:** Report figures saved to `outputs/figures/report/`.) |
| 7 | 07_definitions_control | [07_definitions_control.html](https://vwintumich.github.io/ccc-project/indicator_clustering/docs/07_definitions_control.html) | Definitions-as-control experiment (Section 6 interpretation still template) |

### Older Renders (superseded)

These files are earlier renders that have been replaced by newer versions above.

| File | Superseded by |
|------|---------------|
| [00_data_extraction_2026-02-26_1137.html](https://vwintumich.github.io/ccc-project/indicator_clustering/docs/00_data_extraction_2026-02-26_1137.html) | 00_data_extraction_2026-02-26_1502.html |
| [01_data_cleaning_2026-02-26_1137.html](https://vwintumich.github.io/ccc-project/indicator_clustering/docs/01_data_cleaning_2026-02-26_1137.html) | 01_data_cleaning.html |

## Environment Note

Notebooks 02 and 03 require GPU access (UMich Great Lakes or Google Colab).
Notebook 07 Section 2 (embedding generation) also requires GPU. All other
notebooks and sections run locally.
