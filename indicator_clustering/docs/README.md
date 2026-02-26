# Rendered Notebooks — CCC Indicator Clustering

These HTML files are rendered versions of the project notebooks with full outputs
(figures, tables, metrics). Open any file in a browser to view.

The notebooks form a sequential pipeline. Each stage builds on the outputs of the
previous one.

| Notebook | Description |
|----------|-------------|
| 00_data_extraction | Extract 6 tables from SQLite database to CSV |
| 01_data_cleaning | Verify indicators, compute ground-truth labels, export clean datasets |
| 02_embedding_generation | Generate 1024-dim BGE-M3 embeddings for 12,622 unique indicators |
| 03_dimensionality_reduction | PCA and UMAP reduction to 10D (clustering) and 2D (visualization) |
| 04_clustering | Unconstrained HDBSCAN and agglomerative clustering exploration |
| 05_constrained_and_targeted | Seed-word constraints, subset experiments, anagram sub-clustering |
| 06_evaluation_and_figures | Publication-quality figures and systematic evaluation for the report |

**Environment note:** Notebooks 02 and 03 require GPU access (UMich Great Lakes or
Google Colab). All other notebooks run locally.
