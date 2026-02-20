# CLAUDE.md — Indicator Clustering Project Instructions

This file is read automatically by Claude Code at the start of every session. Read it in full before doing anything else. Then read PROJECT_OVERVIEW.md, DOMAIN_KNOWLEDGE.md, FINDINGS_AND_DECISIONS.md, and OPEN_QUESTIONS.md before writing any code.

---

## Project Identity

This is a MADS (University of Michigan) graduate student project on clustering cryptic crossword clue (CCC) indicator words/phrases to see if they group into interpretable wordplay categories. This file covers conventions that apply to all notebooks.

**Team:** Sahana, Nathan (NC), Victoria, Hans
**Faculty Advisor:** Dr. Kevyn Collins-Thompson (KCT)
**Prompt Engineering Credit:** Victoria
**Report Due:** Tuesday, March 3, 2026

---

## Before Writing Any Code: Read All Five Project Docs

1. CLAUDE.md (this file)
2. PROJECT_OVERVIEW.md — research context, dataset facts, both tasks
3. DOMAIN_KNOWLEDGE.md — CCC terminology, wordplay taxonomy, cluster hierarchy
4. FINDINGS_AND_DECISIONS.md — empirical results to date, advisor guidance, settled decisions
5. OPEN_QUESTIONS.md — unresolved decisions; consult before making choices in these areas

---

## Notebook Authorship Convention

Every notebook must include a markdown cell at the top crediting:
- The primary author(s) by name
- Any notebooks it builds on (by filename and author)
- Victoria for prompt engineering if AI assistance was used in its creation
- Claude (Anthropic) as AI assistant if applicable

Example header cell:
```
# Notebook Title
**Primary author:** Nathan (NC)
**Builds on:** 01_data_cleaning_Victoria.ipynb
**Prompt engineering:** Victoria
**AI assistance:** Claude (Anthropic)
**Environment:** Great Lakes (GPU required)
```

---

## Notebook Style: Learner-Oriented Writing

This is a learning-oriented project for graduate students who are developing their NLP and data science skills. All notebooks must be written as instructional documents, not just executable code. This is a firm requirement that affects both markdown and code style.

**Markdown cells must:**
- Explain *why* each step is being taken, not just what it does
- Define NLP/ML terminology when first introduced (e.g., silhouette score, UMAP, epsilon)
- Describe expected output before each major code block
- Explain how to interpret any metric, plot, or result produced
- Note which computational environment is appropriate for the cell or section

**Code comments must:**
- Explain non-obvious parameter choices (e.g., why min_cluster_size=10 was chosen)
- Use descriptive intermediate variable names rather than chaining
- Prefer readable over clever

It is acceptable — and encouraged — to implement a simpler approach first and iterate. The notebooks should reflect the learning process, not just the final result.

---

## Computational Environment Rules

Each notebook must include an environment setup section near the top with clearly labeled, runnable cells for each supported environment.

**The three environments:**
- **Local / personal laptop** — data cleaning, light EDA, loading saved embeddings, visualization
- **Google Colab** — moderate computation; include a note when GPU must be enabled in Runtime > Change runtime type
- **UMich Great Lakes cluster** — required for embedding generation (BGE-M3 on 14,196 indicators) and large-scale clustering parameter sweeps

**Great Lakes session settings** (include in a markdown cell in any Great Lakes notebook):
- Partition: gpu
- GPUs: 1 (V100 or A40)
- CPUs: 4
- Memory: 32GB
- Wall time: 2 hours for embeddings; 1 hour for clustering (adjust as needed)
- Note: Great Lakes uses individual user accounts; data must be in each user's /scratch directory or a shared project directory

**Standard data path configuration** — use this pattern at the top of every notebook. It auto-detects the environment rather than requiring manual configuration:

```python
import os
from pathlib import Path

# --- Environment Auto-Detection ---
IS_COLAB = 'google.colab' in str(get_ipython())
IS_GREATLAKES = 'SLURM_JOB_ID' in os.environ  # Great Lakes sets this automatically

if IS_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    PROJECT_ROOT = Path('/content/drive/MyDrive/SIADS 692 Milestone II/Milestone II - NLP Cryptic Crossword Clues')
elif IS_GREATLAKES:
    # Update YOUR_UNIQNAME to your actual UMich uniqname
    PROJECT_ROOT = Path('/scratch/YOUR_UNIQNAME/ccc_project')
else:
    # Local: move up from notebooks/ to project root
    # Adjust the number of .parent calls based on where this notebook sits
    PROJECT_ROOT = Path.cwd().parent

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Project root: {PROJECT_ROOT}")
print(f"Data directory: {DATA_DIR}")
print(f"Output directory: {OUTPUT_DIR}")
```

---

## Modular Pipeline Architecture

The project is divided into sequential stages. Each stage saves its outputs as files. Downstream notebooks must load from those files — never recompute embeddings or earlier outputs inside a downstream notebook.

```
Stage 0: Data Extraction
  Notebook: 00_data_extraction_Victoria.ipynb
  Inputs:  data.sqlite3 (SQLite database from cryptics.georgeho.org)
  Outputs: clues_raw.csv, indicators_raw.csv, indicators_by_clue_raw.csv,
           indicators_consolidated_raw.csv, charades_raw.csv, charades_by_clue_raw.csv
  Runs on: Local or Colab

Stage 1: Data Cleaning & Verification
  Notebook: 01_data_cleaning_Victoria.ipynb
  Inputs:  *_raw.csv files from Stage 0
  Outputs: verified_indicators_unique.csv   (12,622 unique indicator strings)
           verified_clues_labeled.csv       (76,015 clue-indicator pairs with labels)
  Runs on: Local or Colab

Stage 2: Embedding Generation                [GREAT LAKES or COLAB GPU]
  Notebook: 02_embedding_generation_Victoria.ipynb
  Inputs:  verified_indicators_unique.csv
  Outputs: embeddings_bge_m3_all.npy        (shape: 12622 x 1024)
           indicator_index_all.csv          (row number -> indicator string)
  Model:   BAAI/bge-m3 via SentenceTransformer

Stage 3: Dimensionality Reduction
  Inputs:  embeddings_bge_m3_*.npy
  Outputs: embeddings_umap_2d.npy           (for visualization)
           embeddings_umap_10d.npy          (for clustering input)
           embeddings_pca_Nd.npy            (as needed)
  Runs on: Great Lakes or Colab (GPU helps)

Stage 4: Clustering
  Inputs:  Dimensionality-reduced embeddings
  Outputs: cluster_labels_hdbscan.csv
           cluster_labels_agglomerative.csv
  Runs on: Great Lakes (parameter sweeps); Colab (single runs)

Stage 5: Evaluation and Visualization
  Inputs:  Cluster label files + indicator index
  Outputs: figures/ directory, metrics_summary.csv
  Runs on: Local or Colab
```

Each notebook must begin with a cell that checks for required input files and prints a clear error message if they are missing, rather than failing silently.

---

## Coding Conventions

- **Language:** Python 3, Jupyter notebooks (.ipynb)
- **Key libraries:** pandas, numpy, scikit-learn, sentence-transformers, umap-learn, hdbscan, matplotlib, seaborn
- **Reproducibility:** Always set random_state=42 (or np.random.seed(42)) at the top of every notebook
- **File naming:** NN_descriptive_name_Author.ipynb (e.g., 02_embeddings_NC.ipynb)
- **Save outputs** with a version suffix or timestamp so reruns don't silently overwrite previous results
- **Do not hardcode paths** — always use the ENV config block above

---

## Hard Rules (Do Not Violate)

- **No domain-knowledge-based data cleaning.** Use only objective, algorithmic criteria to filter or clean data. Manually removing indicators because they "seem wrong" introduces bias and constitutes data leakage.
- **No stemming or lemmatization before embedding.** The BGE-M3 model handles morphological variation. Do not reduce indicators to stems before passing them to SentenceTransformer.
- **Never recompute embeddings in a downstream notebook.** Load from the .npy file saved in Stage 2.
- **Do not use KMeans as a primary method.** It performed very poorly on this data (silhouette score 0.033 vs DBSCAN's 0.754). It may appear as a baseline comparison only.
- **Do not tune HDBSCAN/DBSCAN epsilon arbitrarily.** Always examine the pairwise distance distribution of the embeddings first and select epsilon based on that distribution. Document the reasoning.
- **Do not rely on silhouette score alone.** Always accompany quantitative metrics with cluster visualizations and qualitative inspection of example indicators per cluster.

---

## GitHub Requirements

- Repository must be public OR the project coach must be invited as a collaborator
- Final submission must include: all notebooks, report PDF, data (or URLs/sample if >25MB), README
- README must describe how to run notebooks in order and what environment each requires
- Do not edit or add files to the repository after the project deadline — this triggers a late penalty
