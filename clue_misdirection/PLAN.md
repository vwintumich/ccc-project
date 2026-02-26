# Execution Plan — Clue Misdirection (Supervised Learning)

Derived from `supervised_learning_plan_v3.docx` (February 2026).
Steps map to Section 11 of the design document.

Every notebook should follow the coding standards in CLAUDE.md, including
the intellectual lineage header and the summary cell at the end.

---

## Step 1: Filter and Clean the Raw Dataset

**Design doc ref:** Section 3.2

**Input:**
- `../data/clues_raw.csv` (extracted from the 660,613-clue sqlite DB by the
  indicator_clustering NB00)

**Output:**
- `data/clues_filtered.csv` — rows passing all filters, with derived columns
  (`surface`, `surface_normalized`, `definition_list`, `answer_format`, etc.)
- `data/cleaning_log.md` — record of how many rows removed at each filter
  step (this information should also be explained in the notebook's summary
  cell)

**Requirements:**
- Filter to rows where `definition`, `answer`, and `clue` are all non-null
- Create a `surface` column by stripping the trailing answer format (e.g.,
  "(7)" or "(3,4)") from the `clue` text. This is the text we embed — never
  embed the raw `clue` with format information included.
- Create `surface_normalized` (lowercase, no punctuation/accents) for
  validation checks
- Parse double-definition clues: where the `definition` field contains `/`,
  split into a `definition_list`. For each definition in the list, verify it
  appears as intact word(s) in the `surface` text. Keep any definition that
  has ≥1 synset in WordNet. A single clue may produce multiple rows in the
  output if it has multiple valid definitions.
- For single-definition clues: definition must appear intact as standalone
  word(s) within the `surface` text
- Answer must adhere to the length/format specified in the clue
- Both definition and answer must have ≥1 synset in WordNet (definitions
  and answers may be multi-word — WordNet handles some multi-word entries)
- Record row counts at each filter step in the notebook
- Track unique (definition, answer) pairs vs. total (clue, definition, answer)
  rows — this distinction matters throughout the pipeline
- Assign a `def_answer_pair_id` to each unique (definition, answer) pair for
  use in GroupKFold cross-validation

**Existing work to draw from:**
- Victoria's *Data Cleaning for Clues - Pairs in WordNet.ipynb* — primary
  base. Implements `surface` extraction (Cell 14), `surface_normalized`,
  double-definition parsing via `definition_list` (Cells 26–33), and
  answer format validation. WordNet filtering was started but not completed.
- Victoria/Sahana's *Data Cleaning for Clues.ipynb* — earlier version with
  some of the same cleaning logic. Check for any additional filters.
- Hans's *Hans_Supervised_Learning_EDA.ipynb* — review for any cleaning
  steps not covered by Victoria's notebooks (e.g., whole-word boundary
  verification using `\b` regex, non-alpha removal). Note: Hans started
  from `clues_single_word.csv` which we are not using.
- The indicator_clustering *01_data_cleaning.ipynb* — reference for the
  notebook structure, header format, and path setup conventions.

**Notebook:** `01_data_cleaning.ipynb`

---

## Step 2: Generate Embeddings

**Design doc ref:** Sections 5.1–5.4

**Input:**
- `data/clues_filtered.csv`
- WordNet (via NLTK)

**Output:**
- `data/embeddings/` directory containing:
  - `definition_embeddings.npy` — shape (N_unique_defs, 3, 1024): for each
    unique definition string, three embeddings (allsense_avg, common, obscure)
  - `definition_index.csv` — maps row position in the .npy to the definition
    string
  - `answer_embeddings.npy` — shape (N_unique_answers, 3, 1024): for each
    unique answer string, three embeddings (allsense_avg, common, obscure)
  - `answer_index.csv` — maps row position in the .npy to the answer string
  - `clue_context_embeddings.npy` — shape (N_rows, 1024): per-row embedding
    for word1_clue_context
  - `clue_context_index.csv` — maps row position to clue_id (aligned with
    `clues_filtered.csv` row order)

**Embedding types (7 per clue row):**

| # | Name | Source | Deduplication | How constructed |
|---|------|--------|---------------|-----------------|
| 1 | word1_allsense | Definition | Per unique definition | Embed definition in each WordNet synset context with `<t></t>`, average all |
| 2 | word1_clue_context | Definition in clue | Per row (unique clue text) | Definition embedded within the `surface` sentence using `<t></t>` delimiters |
| 3 | word1_common | Definition | Per unique definition | Definition in most-common WordNet synset context with `<t></t>` |
| 4 | word1_obscure | Definition | Per unique definition | Definition in least-common WordNet synset context with `<t></t>` |
| 5 | word2_allsense | Answer | Per unique answer | Embed answer in each WordNet synset context with `<t></t>`, average all |
| 6 | word2_common | Answer | Per unique answer | Answer in most-common WordNet synset context with `<t></t>` |
| 7 | word2_obscure | Answer | Per unique answer | Answer in least-common WordNet synset context with `<t></t>` |

**Efficiency:** Embeddings 1, 3, 4, 5, 6, 7 depend only on the definition
or answer string, not the clue, so they are computed once per unique string
and looked up via the index files. Only embedding 2 (which depends on the
specific clue sentence) must be computed per row.

**Requirements:**
- Model: `gabrielloiseau/CALE-MBERT-en` (CALE, 1024-dim)
  (backup: `BAAI/bge-base-en-v1.5`, 768-dim)
- All embeddings use CALE's `<t></t>` delimiter mechanism. For synset-based
  embeddings, construct a context sentence from the synset's definition text
  and usage example, with `<t></t>` around the target word. Fall back to
  `<t>word</t>` if the word does not appear in the context text.
- **Runs on GPU** (Great Lakes or Colab T4). Full dataset embedding takes
  under 20 minutes on T4 and produces files under 1 GB total.
- Embed the entire filtered dataset — no sampling. Subsets can be taken
  downstream as needed.
- Store as `.npy` files with corresponding index CSVs. All downstream work
  runs on CPU.
- Verification cell at the end: reload saved files, assert shapes match
  indexes, spot-check a known embedding.

**Existing work to draw from:**
- *02_embedding_generation.ipynb* (indicator_clustering, Victoria) — reference
  for the deduplicated embedding pattern, index file contract, verification
  approach, and notebook structure. Uses BAAI/bge-m3; we use CALE but the
  workflow is the same.
- Hans's *Hans_Supervised_Learning.ipynb* — contains embedding generation
  code using all-mpnet-base-v2. The retrieval evaluation logic is useful
  reference but the model and embedding scheme need full replacement.

**Notebook:** `02_embedding_generation.ipynb` (+ optional `.py` script for
Great Lakes batch submission)

---

## Step 3: Compute All 47 Features

**Design doc ref:** Section 6

**Input:**
- `data/embeddings/definition_embeddings.npy` + `definition_index.csv`
- `data/embeddings/answer_embeddings.npy` + `answer_index.csv`
- `data/embeddings/clue_context_embeddings.npy` + `clue_context_index.csv`
- `data/clues_filtered.csv`
- WordNet (via NLTK)

**Output:**
- `data/features_all.parquet` — one row per (clue, definition, answer) with
  all 47 features plus metadata columns

**Requirements:**

Use the index files to look up the correct definition and answer embeddings
for each (clue, definition, answer) row before computing cosine similarities.

- **Context-Free Meaning (15):** C(6,2) = 15 cosine similarities among the
  6 embeddings not involving clue context (word1_allsense, word1_common,
  word1_obscure, word2_allsense, word2_common, word2_obscure)
- **Context-Informed Meaning (6):** 6 cosine similarities between
  word1_clue_context and each of the 6 context-free embeddings above
- **Relationship (22):** 20 boolean two-hop WordNet relationship types +
  max path similarity + shared synset count
- **Surface (4):** edit distance, length ratio, shared first letter (bool),
  character overlap ratio
- Verify: 15 + 6 + 22 + 4 = 47 total
- `assert not df.isnull().any().any()` — every feature must be a valid number
- For relationship features: pairs with no 2-hop connection get False for all
  20 booleans, 0.0 for path similarity, 0 for shared synset count

**Existing work to draw from:**
- Hans's *Hans_Supervised_Learning.ipynb* and
  *Hans_Supervised_Learning_Models.ipynb* — contain feature computation for
  a smaller 10-feature set. The cosine similarity and WordNet relationship
  code can be adapted but needs substantial expansion to the full 47-feature
  spec.

**Note:** The feature computation logic in this notebook is self-contained for
grading readability. The same logic is later extracted into
`scripts/feature_utils.py` for reuse when computing features for distractor
pairs in Steps 5 and 7 (see Decision 18).

**Notebook:** `03_feature_engineering.ipynb`

---

## Step 4: Retrieval Analysis (Descriptive, Pre-Modeling)

**Design doc ref:** Sections 4.1, 4.1.1, 9

**Input:**
- `data/embeddings/definition_embeddings.npy` + `definition_index.csv`
- `data/embeddings/answer_embeddings.npy` + `answer_index.csv`
- `data/embeddings/clue_context_embeddings.npy` + `clue_context_index.csv`
- `data/clues_filtered.csv`

**Output:**
- `outputs/retrieval_results_unique_pairs.csv` — primary retrieval stats
  over unique (definition, answer) pairs (5 def × 3 answer = 15 cells)
- `outputs/retrieval_results_all_rows.csv` — supplementary retrieval stats
  over all (clue, definition, answer) rows using Average condition only
- `outputs/figures/retrieval_bar_chart.png` — grouped bar chart, median rank
  (log scale) by definition condition, grouped by answer condition
- `outputs/figures/retrieval_heatmap.png` — heatmap of mean cosine similarity

**Requirements:**

*Primary analysis (unique pairs):*
- Report over **unique (definition, answer) pairs**, not all clue rows
- For context-informed conditions with multiple clues per pair: take median
  rank across clues for that pair, then compute summary stats over unique pairs
- Definition-side conditions: Average, Common, Obscure, Clue Context, Full Clue
- Answer-side conditions: Average, Common, Obscure
- Metrics per cell: Top-1, Top-5, Top-10, Top-50, Top-100 hit rates;
  mean rank; median rank; mean cosine similarity
- This is the **primary evidence for misdirection** — present prominently

*Supplementary analysis (all rows):*
- Compute the same metrics over all (clue, definition, answer) rows, using
  only the Average (decontextualized) definition and answer embeddings for
  the context-free condition
- Compare with the unique-pairs results to discuss what is representative
  of CCC puzzles in general. If creators tend to reuse certain (definition,
  answer) pairs and vary the clue context, this comparison can reveal whether
  the misdirection signal differs for frequently-reused pairs vs. rare ones.
- Note in the discussion that the all-rows view may inflate the misdirection
  measure (see Decision 5 in DECISIONS.md)

**Existing work to draw from:**
- Hans's *Hans_Supervised_Learning.ipynb* — contains preliminary retrieval
  results (Table 1 in design doc) using all-mpnet-base-v2 with 3 conditions.
  Needs re-running with CALE embeddings, the full 5×3 matrix, and the
  unique-pairs reporting unit.

**Known pitfalls (from NB 03 experience):**
- Use `keep_default_na=False` on ALL index CSV loads (the word "nan" is a
  valid crossword entry).
- Use `clue_context_phrases.csv` with a composite key (`clue_id`,
  `definition_wn`) for clue-context embedding lookups — NOT
  `clue_context_index.csv` alone, because `clue_id` is non-unique for
  double-definition clues.
- ~35% of definitions are single-synset. The Common vs Obscure retrieval
  comparison is uninformative for these pairs. Consider reporting what
  percentage of unique pairs this affects.

**Notebook:** `04_retrieval_analysis.ipynb`

---

## Step 5: Construct the Easy Dataset

**Design doc ref:** Section 7.1

**Input:**
- `data/features_all.parquet`

**Output:**
- `data/dataset_easy.parquet` — balanced 1:1 (real + distractor), all 47 features

**Requirements:**
- For each real (clue, definition, answer) row, generate one distractor by
  keeping the same (clue, definition) and substituting a randomly sampled
  answer word (excluding the true answer)
- Compute all 47 features for distractor rows (this requires generating or
  looking up embeddings for the new answer — use the existing answer embedding
  index from Step 2, since distractor answers are drawn from the pool of
  known answers). Import feature computation functions from
  `scripts/feature_utils.py` (extracted from NB 03 logic) to compute the
  same 47 features for distractor pairs.
- Label column: 1 = real, 0 = distractor
- Random seed for reproducibility

**Notebook:** `05_dataset_construction.ipynb` (both easy and harder in one notebook)

---

## Step 6: Run Experiments 1A and 1B (Easy Dataset)

**Design doc ref:** Section 8.3

**Input:**
- `data/dataset_easy.parquet`

**Output:**
- `outputs/results_easy.csv` — accuracy, F1, precision, recall (± SD) for
  all 3 models × 2 conditions. Also ROC AUC for Logistic Regression.
- Saved best hyperparameters per model

**Requirements:**
- **Exp 1A:** All 47 features. Three models: KNN, Logistic Regression,
  Random Forest.
- **Exp 1B:** Remove 6 context-informed features → 41 features.
- 5-fold GroupKFold CV (grouped by def–answer pair). Same folds across conditions.
- StandardScaler on train folds only for KNN and LogReg.
- GridSearchCV (or RandomizedSearchCV for RF) within each fold.
- Report mean ± SD of accuracy, F1, precision, recall. ROC AUC for LogReg.
- Expect high accuracy in both — this is the baseline.

**Existing work to draw from:**
- Hans's *Hans_Supervised_Learning_Models.ipynb* — contains KNN, LogReg, RF
  code with 5-fold stratified CV and a 10-feature set. Adapt the CV and
  modeling scaffolding; expand to 47 features and GroupKFold.

**Notebook:** `06_experiments_easy.ipynb`

---

## Step 7: Construct the Harder Dataset

**Design doc ref:** Section 7.2

**Input:**
- `data/features_all.parquet`
- `data/embeddings/definition_embeddings.npy` + `definition_index.csv`
  (for word1_average — the first of the three definition embeddings)
- `data/embeddings/answer_embeddings.npy` + `answer_index.csv`
  (for word2_average — the first of the three answer embeddings)

**Output:**
- `data/dataset_harder.parquet` — balanced 1:1, **without** the 15 context-free
  meaning features (32 features for Exp 2A, 26 for Exp 2B)

**Requirements:**
- For each real definition, rank all candidate answer words by cosine
  similarity between `word1_average` (definition, no context) and
  `word2_average` (answer, no context)
- Sample distractors from the top-k most similar answers (excluding the
  true answer)
- Real pairs average ~0.4 cosine similarity; distractors should be in a
  similar range
- Import feature computation functions from `scripts/feature_utils.py`
  (extracted from NB 03 logic) to compute the same features for distractor
  pairs.
- Remove the 15 context-free meaning features (they are artifacts of
  the cosine-similarity-based construction)
- Remaining: 6 context-informed + 22 relationship + 4 surface = 32 features

**Notebook:** `05_dataset_construction.ipynb` (same notebook as Step 5)

---

## Step 8: Run Experiments 2A and 2B (Harder Dataset)

**Design doc ref:** Section 8.3

**Input:**
- `data/dataset_harder.parquet`

**Output:**
- `outputs/results_harder.csv`
- Saved best hyperparameters per model

**Requirements:**
- **Exp 2A:** 32 features (context-informed meaning + relationship + surface).
- **Exp 2B:** 26 features (relationship + surface only — remove context-informed).
- Same 3 models, same CV scheme, same scaling approach. ROC AUC for LogReg.
- **This is where the misdirection hypothesis is tested via classification.**
  If 2A < 2B, context hurts → supports misdirection. If 2A > 2B, context helps.

**Existing work to draw from:**
- Hans's *Hans_Supervised_Learning_Models.ipynb* — same modeling scaffolding
  as Step 6.
- Hans's *Hans_Negative_Strategies_Experiment.ipynb* — contains experiments
  with different distractor strategies that informed the harder dataset
  design. Reference only; the approach differs from plan v3.

**Notebook:** `07_experiments_harder.ipynb`

---

## Step 9: Compile Summary Results Table

**Design doc ref:** Sections 10.2, 10.5

**Input:**
- `outputs/results_easy.csv`, `outputs/results_harder.csv`

**Output:**
- `outputs/results_summary.csv` — Table 8 from design doc
- `outputs/figures/results_summary_table.png` (formatted for report)

**Requirements:**
- Rows: KNN, Logistic Regression, Random Forest
- Columns: Exp 1A, 1B, Δ Easy, Exp 2A, 2B, Δ Hard
- All values: mean accuracy ± SD from 5-fold CV
- Δ Easy is sanity check; Δ Hard is the classifier's take on misdirection
- Include tradeoff discussion (see rubric): easy vs. harder dataset
  (accuracy vs. informativeness), interpretability vs. performance
  (LogReg coefficients vs. RF accuracy), feature richness vs. misdirection
  detection

**Notebook:** `08_results_and_evaluation.ipynb` (Steps 9–12 can share a notebook)

---

## Step 10: Feature Importance + Ablation

**Design doc ref:** Section 10.3

**Input:**
- Best-performing model from Step 8

**Output:**
- `outputs/figures/feature_importance.png`
- `outputs/ablation_results.csv`

**Requirements:**
- On best model: Gini importance (RF) or permutation importance (model-agnostic);
  standardized coefficients for LogReg
- Group-level ablation: remove one feature group at a time, report accuracy change
- The experimental design (2A vs 2B) already constitutes group-level ablation
  of context features

**Notebook:** `08_results_and_evaluation.ipynb`

---

## Step 11: Sensitivity Analysis

**Design doc ref:** Section 10.4

**Input:**
- Best-performing model from Step 8

**Output:**
- `outputs/figures/sensitivity_*.png` (at least one)

**Requirements:**
- At least one of:
  - Hyperparameter sensitivity (vary key param, plot accuracy)
  - Distractor similarity threshold (vary top-k, plot accuracy)
  - Learning curve (accuracy vs. training set size)

**Notebook:** `08_results_and_evaluation.ipynb`

---

## Step 12: Failure Analysis

**Design doc ref:** Section 10.6

**Input:**
- Predictions from best model on harder dataset

**Output:**
- `outputs/failure_analysis.md` — standalone record of misclassified examples
  and failure categories
- Section in notebook summarizing the same analysis with discussion

**Requirements:**
- Examine ≥3 specific misclassified examples
- For each: definition, answer, real vs. distractor, predicted probability
- Group into ≥3 categories: polysemy confusion, semantic near-miss,
  surface feature artifact (or others as observed)
- Suggest future improvements (no need to implement)

**Notebook:** `08_results_and_evaluation.ipynb`

---

## Rubric Coverage Checklist

| Rubric Item                    | Pts | Plan Steps       |
|--------------------------------|-----|------------------|
| Introduction                   | 5   | 4, 8 (narrative) |
| Related Work                   | 5   | (from proposal)  |
| Data Source                    | 5   | 1                |
| Feature Engineering            | 8   | 2, 3             |
| Methods Description            | 8   | 6, 8             |
| Overall Results                | 8   | 9                |
| Feature Importance & Ablation  | 6   | 10               |
| Sensitivity Analysis           | 4   | 11               |
| Tradeoffs                      | 4   | 9 (discussion)   |
| Failure Analysis               | 5   | 12               |
| Discussion                     | 4   | All              |
| Ethical Considerations         | 4*  | Shared w/ Part B |
