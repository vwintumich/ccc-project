# Notebook Inventory

Generated: February 19, 2026

---

## Summary

| # | Notebook | Author | Pipeline Stage | State | Environment |
|---|----------|--------|----------------|-------|-------------|
| 1 | Get CSV from SQLITE3.ipynb | Victoria | Stage 0: Data extraction | Working | Local |
| 2 | Cryptic Crossword Data Exploration.ipynb | Victoria | Stage 1: EDA | Working | Local |
| 3 | Data Cleaning for Indicator Clustering.ipynb | Victoria | Stage 1: Data cleaning (early) | Broken | Local (wrong DB path) |
| 4 | Data Cleaning for Indicator Clustering - Single Word Indicators.ipynb | Victoria/Sahana | Stage 1: Data cleaning (single-word focus) | Broken (import errors) | Colab |
| 5 | Data Cleaning for Indicator Clustering copy.ipynb | Victoria | Stage 1: Data cleaning (canonical) | Working | Local/Colab |
| 6 | Sentence Transformers Exploration.ipynb | Sahana | Exploration: POS-weighted embeddings | Working | Colab |
| 7 | Data Cleaning for Clues - Pairs in WordNet.ipynb | Victoria/Sahana | Stage 1 (alt task): Definition-answer cleaning | Partially working | Colab |
| 8 | NC_Indicators_w_Context_Clustering_Models.ipynb | Nathan (NC) | Stages 2-4: Context embeddings + KMeans | Working (poor results) | Colab |
| 9 | NC_Indicators_wo_Context_Clustered_Embeddings_Compared.ipynb | Nathan (NC) | Stages 2-4: Multi-model comparison class | Working | Colab |
| 10 | NC_Comprehensive_Embeddings.ipynb | Nathan (NC) | Stage 2: Embedding generation (7 models) | Working | Colab |
| 11 | NC_PCA_Analysis.ipynb | Nathan (NC) | Stage 3: PCA + UMAP visualization | Broken (missing parquet) | Colab |
| 12 | Sahana Copy of NC_Indicators_w_Context_Clustering_Models_WIP.ipynb | Sahana | Stages 2-4: Copy of NC's context notebook | Working (poor results) | Colab |
| 13 | Hierarchical_Clustering_Indicators_with_BGE_M3_Embeddings (3).ipynb | Victoria/Sahana | Stages 2-4: BGE-M3 + HDBSCAN + Agglomerative | Broken locally (missing sentence-transformers); has saved outputs | Local/Colab |

---

## Detailed Notebook Descriptions

### 1. Get CSV from SQLITE3.ipynb
- **Author:** Victoria
- **Purpose:** Extract 6 tables from the raw `cryptic.sqlite3` database and save them as CSV files (`indicators_raw.csv`, `clues_raw.csv`, etc.). Renames columns for consistency.
- **State:** Working. All cells have outputs. Produces the CSV files that every other notebook depends on.
- **Outputs:** `indicators_raw.csv`, `indicators_by_clue_raw.csv`, `indicators_consolidated_raw.csv`, `clues_raw.csv`, `charades_raw.csv`, `charades_by_clue_raw.csv`
- **Notes:** Uses a different DB filename (`cryptic.sqlite3`) than notebook #3 (`cryptic_data.sqlite3`). This is the canonical extraction notebook.

### 2. Cryptic Crossword Data Exploration.ipynb
- **Author:** Victoria
- **Purpose:** Broad EDA on the full dataset: table sizes, indicator counts by wordplay type, word counts for clues/answers/indicators, duplicate analysis, histograms.
- **State:** Working. All cells executed. Good overview of data characteristics.
- **Outputs:** No saved files (exploratory only).
- **Notes:** Uses the sqlite3 file directly (not the CSVs). Contains the initial idea for the clustering task in a markdown cell.

### 3. Data Cleaning for Indicator Clustering.ipynb
- **Author:** Victoria
- **Purpose:** Early attempt at data cleaning. Loads from sqlite3 directly, explores indicator tables, discusses data requirements and dilemmas.
- **State:** Broken. Cell 4 fails with `DatabaseError: no such table: clues` because it connects to `cryptic_data.sqlite3` which appears to have a different schema or be missing. Only a few cells executed successfully.
- **Outputs:** None.
- **Notes:** Superseded by notebook #5 ("copy" version). Contains the same markdown discussion text as #5 but less developed code. The dilemma discussions are duplicated in #5.

### 4. Data Cleaning for Indicator Clustering - Single Word Indicators.ipynb
- **Author:** Victoria/Sahana
- **Purpose:** Clean and filter indicators to single-word only. Uses `wordfreq`, `pyenchant`, and WordNet to validate words. Investigates letter lengths, suspicious indicators.
- **State:** Broken. First cell fails with `ModuleNotFoundError: No module named 'wordfreq'` (pip install fails due to no network). Many cells reference columns (`zipf_score`, `enchant_check`, `in_wordnet`) that were never created due to the import failure. Later cells are uncommented but depend on earlier failures.
- **Outputs:** Attempts to save `df_ind_one_word.csv` but it's unclear if this succeeded. The last cell tries to mount Google Drive (Colab-specific).
- **Notes:** This notebook was the precursor to the verification approach in notebook #5. The single-word-only approach was later broadened (per KCT guidance) to include multi-word indicators. Some of the validation logic (Zipf, enchant, WordNet) was abandoned in favor of Victoria's "checksum" verification method.

### 5. Data Cleaning for Indicator Clustering copy.ipynb (CANONICAL)
- **Author:** Victoria
- **Purpose:** The canonical data cleaning notebook. Loads from CSVs (produced by notebook #1). Implements Victoria's "checksum" verification: an indicator is verified if it appears as intact words in the clue surface text. Produces the verified indicator dataset (14,196 indicators). Also explores hidden word detection, alternation detection, and anagram detection in clue surfaces.
- **State:** Working. All cells executed with outputs. This is the most complete and important data cleaning notebook.
- **Outputs:** `verified_indicators.csv` (14,196 indicators), `verified_indicators_one_word.csv` (single-word subset)
- **Critical issue:** The exported CSVs do NOT include wordplay labels. They are just lists of indicator strings. This is flagged as OPEN in FINDINGS_AND_DECISIONS.md (Q9). Downstream notebooks that need labels must re-derive them from `df_indicators`.
- **Notes:** Contains environment auto-detection (Colab vs. local). Includes extensive markdown explanations. The alternation/hidden/anagram verification sections at the bottom are exploratory and not part of the main pipeline output.

### 6. Sentence Transformers Exploration.ipynb
- **Author:** Sahana
- **Purpose:** Explore whether POS-weighted embeddings (using spaCy POS tags to up-weight nouns/verbs and down-weight function words) improve sentence similarity. Uses `all-MiniLM-L6-v2`.
- **State:** Working. All cells executed. Demonstrates that POS weighting changes similarity scores but doesn't clearly improve them for this task.
- **Outputs:** None (exploratory).
- **Notes:** This approach was discussed in a meeting and deemed not worth pursuing (per FINDINGS_AND_DECISIONS.md). The notebook is a useful reference but does not feed into the pipeline.

### 7. Data Cleaning for Clues - Pairs in WordNet.ipynb
- **Author:** Victoria/Sahana
- **Purpose:** Clean the full clues dataset for a DIFFERENT task: analyzing definition-answer pairs using WordNet. Verifies definitions appear at start/end of clue, checks answer format matches required format, explores WordNet similarity between definitions and answers.
- **State:** Partially working. Most cells execute. The last few cells fail with `NameError: name 'df_ind_one_word' is not defined` (a variable from a different notebook context). The WordNet exploration at the bottom is incomplete.
- **Outputs:** No saved files.
- **Notes:** This is for the supervised/definition-answer task, NOT the indicator clustering task. It's the only notebook addressing the second research question. Useful but incomplete.

### 8. NC_Indicators_w_Context_Clustering_Models.ipynb
- **Author:** Nathan (NC)
- **Purpose:** Embed indicators WITH their clue context (using `[INDICATOR]` highlighting in the full clue text), then cluster with KMeans (k=8) across three models (MiniLM, MPNet, DistilRoBERTa). Includes prediction function.
- **State:** Working but with very poor results (ARI ~0.003-0.005, silhouette ~0.015-0.017). The prediction cell fails with `NameError: name 'model' is not defined` because the model objects from the pipeline are stored in a dict but the prediction code references a bare `model` variable.
- **Outputs:** No saved files.
- **Key finding:** KMeans on contextualized embeddings produces near-random clustering. This approach has been ruled out per FINDINGS_AND_DECISIONS.md.

### 9. NC_Indicators_wo_Context_Clustered_Embeddings_Compared.ipynb
- **Author:** Nathan (NC)
- **Purpose:** Comprehensive `EmbeddingComparison` class that compares embedding models (MiniLM, MPNet) with multiple clustering methods (KMeans, DBSCAN, HDBSCAN, hierarchical). Includes auto-tuning for DBSCAN epsilon, dendrogram generation, cluster member export. Embeds indicators WITHOUT context (just the indicator phrase).
- **State:** Working. Ran HDBSCAN on all-mpnet-base-v2 embeddings of 14,195 verified indicators. Results: silhouette -0.2345 (poor; likely because noise reassignment distorts the score).
- **Outputs:** Cluster CSVs and summary text files saved to the outputs/ directory on Google Drive.
- **Notes:** This is the most engineering-heavy notebook. The `EmbeddingComparison` class is ~700 lines. It uses PCA (not UMAP) for dimensionality reduction before clustering, which may explain poorer results compared to notebook #13's UMAP approach. The noise-reassignment strategy (assigning all DBSCAN/HDBSCAN noise points to nearest centroid) is questionable and inflates cluster sizes.

### 10. NC_Comprehensive_Embeddings.ipynb
- **Author:** Nathan (NC)
- **Purpose:** Generate embeddings for all 91,448 indicator instances (NOT unique indicators) across 7 models (MiniLM, MiniLM-L12, MPNet, E5-base, E5-large, BGE-M3, Multilingual-MPNet). Saves both "with context" (highlighted clue) and "without context" (indicator alone) embeddings as parquet files.
- **State:** Working. Shows the last model (Multilingual-MPNet) being embedded; earlier models were presumably run in previous sessions.
- **Outputs:** Parquet files per model in Google Drive data/ directory (e.g., `Multilingual-MPNet_indicator_embeddings.parquet`).
- **Critical issue:** Embeds at the INSTANCE level (91,448 rows = one row per clue-indicator pair), NOT the unique indicator level (14,196). This contradicts the settled decision in FINDINGS_AND_DECISIONS.md that embeddings should be computed on unique indicators only. These embeddings should NOT be used for the final pipeline.

### 11. NC_PCA_Analysis.ipynb
- **Author:** Nathan (NC)
- **Purpose:** Load pre-computed embeddings from parquet, apply PCA (90% variance), then UMAP for visualization. Compare "with context" vs "without context" embeddings via UMAP scatter plots colored by wordplay type.
- **State:** Broken. Fails with `FileNotFoundError` when loading the Multilingual-MPNet parquet (the file doesn't exist at the expected path). No visualization outputs.
- **Outputs:** None (intended to save UMAP plots and a parquet with UMAP coordinates).
- **Notes:** Depends on notebook #10's parquet outputs. Would need the parquet files to be in the right location to run.

### 12. Sahana Copy of NC_Indicators_w_Context_Clustering_Models_WIP.ipynb
- **Author:** Sahana (copy of Nathan's work)
- **Purpose:** Exact duplicate of notebook #8 (NC_Indicators_w_Context_Clustering_Models.ipynb). Same code, same outputs, same errors.
- **State:** Working with same poor results as #8. Same prediction error.
- **Outputs:** None.
- **Notes:** This is a redundant copy. Sahana's Google Drive path differs from Nathan's but the code and results are identical.

### 13. Hierarchical_Clustering_Indicators_with_BGE_M3_Embeddings (3).ipynb
- **Author:** Victoria (HDBSCAN), Sahana (agglomerative)
- **Purpose:** The most successful clustering notebook. Embeds all 14,196 verified indicators using BGE-M3 (1024-dim), reduces with UMAP (10-dim), clusters with HDBSCAN. Also runs agglomerative clustering with Ward's method at k=6, 8, 12, 26 (matching seed set sizes).
- **State:** Broken on local (missing `sentence_transformers` package in local conda env). Has full outputs from a previous successful run. The agglomerative section runs on the same `embeddings_reduced` from the HDBSCAN section.
- **Outputs (from prior run):**
  - HDBSCAN: 353 clusters, silhouette 0.29, 4,076 noise points (29%)
  - Agglomerative (Ward's): silhouette scores 0.38-0.46 depending on k
  - Cluster cohesion analysis (tightest: "filling for" at 0.004, "buried in" at 0.005)
- **Notes:** This is the notebook that produced the findings reported in FINDINGS_AND_DECISIONS.md. It does NOT save embeddings to .npy files as required by the pipeline architecture in CLAUDE.md. The embeddings are generated inline and only exist in memory during the notebook session.

---

## Redundancy Analysis

### Full duplicates
- **#8 and #12 are identical.** `Sahana Copy of NC_Indicators_w_Context_Clustering_Models_WIP.ipynb` is a verbatim copy of `NC_Indicators_w_Context_Clustering_Models.ipynb`. Keep #8, delete #12.

### Superseded notebooks
- **#3 is superseded by #5.** `Data Cleaning for Indicator Clustering.ipynb` is an earlier, broken version of `Data Cleaning for Indicator Clustering copy.ipynb`. The "copy" version has all the same markdown discussions plus working code and the verification breakthrough. Keep #5, archive #3.
- **#4 is partially superseded by #5.** The single-word notebook's validation approach (wordfreq/enchant/WordNet) was abandoned in favor of the checksum verification in #5. However, #4 contains unique analysis of suspicious indicators by letter length that could be referenced. The notebook is broken and would need significant repair to run again.

### Overlapping scope
- **#8, #9, and #10 overlap significantly.** All three are Nathan's work on embedding + clustering. #8 does context-based KMeans (ruled out). #9 is the comprehensive comparison class. #10 generates embeddings at the instance level (which contradicts project decisions). These could be consolidated into one or two notebooks.
- **#9 and #13 both attempt the same core task** (embed indicators, reduce dimensions, cluster) but with different approaches. #13 (BGE-M3 + UMAP + HDBSCAN) produces far better results than #9 (various models + PCA + various methods). #13 is closer to the canonical pipeline.

---

## Pipeline Coverage Gaps

| Pipeline Stage | Status |
|----------------|--------|
| Stage 0: Data extraction (SQLite to CSV) | Covered by #1 |
| Stage 1: Data cleaning + verification | Covered by #5 (but missing labeled output format) |
| Stage 2: Embedding generation (BGE-M3, unique indicators) | Partially covered by #13 (inline, not saved to .npy) |
| Stage 3: Dimensionality reduction (UMAP) | Partially covered by #13 (inline, not saved to .npy) |
| Stage 4: Clustering | Partially covered by #13 (HDBSCAN + agglomerative) |
| Stage 5: Evaluation and visualization | Not covered by any notebook |

### Key gaps:
1. **No notebook saves embeddings to .npy files** as specified in the pipeline architecture. Notebook #13 computes them inline. A dedicated Stage 2 notebook is needed.
2. **No notebook outputs verified indicators WITH labels** (the Q9 open question). The exported CSVs are label-free.
3. **No dedicated evaluation/visualization notebook** (Stage 5). Evaluation metrics are scattered across #9 and #13 but there is no systematic comparison, sensitivity analysis, or publication-quality visualization notebook.
4. **No epsilon sensitivity analysis** has been performed (required per KCT, Feb 15).
5. **No notebook follows the naming convention** `NN_descriptive_name_Author.ipynb` specified in CLAUDE.md.
6. **No notebook uses the standard environment auto-detection block** consistently (notebook #5 is closest).

---

## Recommended Action Plan (not yet implemented)

1. Rename and number notebooks to match the pipeline convention.
2. Consolidate Nathan's notebooks (#8, #9, #10, #11) into one or two, discarding the instance-level and context-based approaches that have been ruled out.
3. Delete the redundant Sahana copy (#12).
4. Archive the broken/superseded notebooks (#3, #4).
5. Create a dedicated Stage 2 notebook that saves embeddings to .npy and indicator indices to .csv.
6. Create a dedicated Stage 5 evaluation notebook.
7. Update notebook #5 to also export indicators with wordplay labels (resolving Q9).
