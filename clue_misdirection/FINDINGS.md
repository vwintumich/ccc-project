# Findings — Clue Misdirection (Supervised Learning)

Running log of findings, observations, and progress as the pipeline is built.
For locked-in decisions, see DECISIONS.md. For the execution plan, see PLAN.md.

---

## Prior Work (Hans's Preliminary Results)

These findings are from Hans's exploratory work using `clues_single_word.csv`
and `all-mpnet-base-v2` (768-dim). They will be updated as we re-run the
pipeline with the CALE model, broader dataset, and full experimental design.

- **Retrieval analysis:** Context-free definition retrieves the true answer
  at top-1 in 3.5% of cases (median rank 177.5 out of 8,598 candidates).
  Adding clue context drops top-1 to 1.0% and median rank to 684 — a +512
  rank degradation. Context hurts in 70.4% of cases.
- **Classification:** KNN/LogReg/RF on 10 features with random distractors
  achieve ~83% accuracy. Adding context features provides <0.5pp improvement.
  `wn_path_sim` is the dominant feature.
- **Control experiment (normal English):** In normal English, removing context
  features *hurts* accuracy. In cryptic crosswords, removing context *helps*.
  This "sign flip" suggests misdirection is specific to cryptic clues, not
  a general property of context.
- **Robustness:** Misdirection signal holds across 3 embedding models, 3
  classifiers, and 3 distractor strategies.

See `CONTEXT.md` for Hans's full writeup.

---

## Pipeline Progress

### Step 1: Data Cleaning
*Completed.* Notebook: `notebooks/01_data_cleaning.ipynb`

- **Final dataset:** 241,397 rows covering 129,429 unique (definition, answer)
  pairs (average 1.9 clues per pair). This is significantly broader than the
  ~10,000 rows estimated in the design doc, which assumed single-word only.
- **Filtering pipeline:** 660,613 raw → 241,397 filtered through 7 filter
  steps: null removal, bracketed-clue removal, answer format validation,
  double-definition parsing, definition-in-surface verification (with `\b`
  word boundaries), definition-at-edge check, and WordNet coverage for both
  definition and answer.
- **Article stripping:** During WordNet lookup, definitions and answers with a
  leading "a " (e.g., "a shade") are retried with the article stripped. This
  simple heuristic recovered additional matches. More aggressive lemmatization
  or stemming could improve WordNet coverage further and is a potential future
  improvement.
- **`wordplay_type` not available:** The column is not present in
  `clues_raw.csv` and was dropped from the output schema (consistent with
  Decision 4 — wordplay type is excluded from the model).
- **Multi-definition expansion:** ~5% of clues had `/`-separated definitions.
  After splitting and validating, each valid definition produces its own row,
  contributing to the total of 241,397 rows.
```

### Step 2: Embedding Generation
*Not yet started.*

### Step 3: Feature Engineering
*Not yet started.*

### Step 4: Retrieval Analysis
*Not yet started.*

### Steps 5–8: Dataset Construction and Experiments
*Not yet started.*

### Steps 9–12: Results, Ablation, Sensitivity, Failure Analysis
*Not yet started.*
