# Rendered Notebooks — CCC Clue Misdirection

These HTML files are rendered versions of the project notebooks with full outputs
(figures, tables, metrics). Click any link to view the rendered notebook on GitHub Pages.

| Notebook | HTML Snapshot | Description |
|----------|---------------|-------------|
| 00 — Model Comparison | [00_model_comparison.html](https://vwintumich.github.io/ccc-project/clue_misdirection/docs/00_model_comparison.html) | Compare CALE, BGE, and MPNet embedding models; validate CALE delimiter mechanism |
| 01 — Data Cleaning | [01_data_cleaning_2026-03-02.html](https://vwintumich.github.io/ccc-project/clue_misdirection/docs/01_data_cleaning_2026-03-02.html) | Filter clues, validate WordNet coverage, export clean dataset (241,397 rows) |
| 02 — Embedding Generation | [02_embedding_generation.html](https://vwintumich.github.io/ccc-project/clue_misdirection/docs/02_embedding_generation.html) | Construct CALE context phrases and generate embeddings (~1.8 GB) |
| 03 — Feature Engineering | [03_feature_engineering.html](https://vwintumich.github.io/ccc-project/clue_misdirection/docs/03_feature_engineering.html) | Compute 47 features (cosine, WordNet, surface) for 240,211 rows |
| 04 — Retrieval Analysis | [04_retrieval_analysis_2026-02-28_1858.html](https://vwintumich.github.io/ccc-project/clue_misdirection/docs/04_retrieval_analysis_2026-02-28_1858.html) | 4×3 retrieval matrix, WordNet reachability analysis, misdirection confirmation |
| 04 — Retrieval Analysis (earlier) | [04_retrieval_analysis_2026-02-26_1543.html](https://vwintumich.github.io/ccc-project/clue_misdirection/docs/04_retrieval_analysis_2026-02-26_1543.html) | Earlier snapshot before WordNet reachability additions |
| 05 — Dataset Construction | [05_dataset_construction_2026-02-26_1813.html](https://vwintumich.github.io/ccc-project/clue_misdirection/docs/05_dataset_construction_2026-02-26_1813.html) | Construct easy (random) and harder (cosine-similarity) distractor datasets |
| 06 — Experiments Easy | [06_experiments_easy_sample20K.html](https://vwintumich.github.io/ccc-project/clue_misdirection/docs/06_experiments_easy_sample20K.html) | Exp 1A/1B on easy dataset (sample 20K; full-data accuracy in results_summary.csv) |
| 07 — Experiments Harder | [07_experiments_harder_sample20K.html](https://vwintumich.github.io/ccc-project/clue_misdirection/docs/07_experiments_harder_sample20K.html) | Exp 2A/2B on harder dataset (sample 20K; full-data accuracy in results_summary.csv) |
| 08 — Results & Evaluation | [08_results_and_evaluation_2026-02-27.html](https://vwintumich.github.io/ccc-project/clue_misdirection/docs/08_results_and_evaluation_2026-02-27.html) | Table 8, feature importance, ablation, sensitivity, failure analysis (full-data results) |
| 08 — Results & Evaluation (sample) | [08_results_and_evaluation_sample20K_2026-02-27.html](https://vwintumich.github.io/ccc-project/clue_misdirection/docs/08_results_and_evaluation_sample20K_2026-02-27.html) | Same analyses on 20K sample for development testing |

See `PLAN.md` for the full 12-step pipeline plan.
