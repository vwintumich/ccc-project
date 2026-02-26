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

### Step 2: Embedding Generation — 
Model Investigation Phase *Completed.* Notebook: `notebooks/00_model_comparison.ipynb`
Embedding Generation *Not yet started.*

- **Model identifier corrected:** The design doc specified
  `oskar-h/cale-modernbert-base`, which does not exist on HuggingFace. The
  actual CALE models are published by the paper's authors under
  `gabrielloiseau/`. Three variants exist: CALE-XLM-R (multilingual),
  CALE-XLLEX (multilingual), and CALE-MBERT-en (English, ModernBERT-based).
  We use `gabrielloiseau/CALE-MBERT-en`.
- **Embedding dimension is 1024, not 768:** The CALE-MBERT-en model produces
  1024-dimensional embeddings, contrary to the original plan's assumption of
  768. All downstream shapes and storage estimates are updated accordingly.
- **CALE delimiter mechanism validated:** CALE uses `<t></t>` tags to focus
  the embedding on a target word within a sentence. This produces genuinely
  distinct embeddings for the target word vs. the full sentence
  (cos ≈ 0.66 for "plant" in clue context). Standard sentence-transformer
  models (bge-base-en-v1.5, all-mpnet-base-v2) produce nearly identical
  embeddings for a token-extracted target word vs. the full sentence
  (cos ≈ 0.90–0.93), making them poor at distinguishing Word1_clue_context
  from Sentence1.
- **Bare-word embeddings unreliable with CALE:** Without context, CALE
  produces embeddings that do not discriminate well between related and
  unrelated words (cos(plant, aster) ≈ 0.77, cos(plant, banana) ≈ 0.79).
  This is resolved by the allsense-average approach (below).
- **Allsense-average replaces bare-word "average" embeddings:** Instead of
  embedding a standalone word, we embed the word in each of its WordNet
  synset contexts (using `<t></t>` delimiters) and average the resulting
  embeddings. Each input is a short, focused context sentence — exactly
  matching CALE's training distribution. This produces rich, sense-grounded
  "average" representations.
- **Sentence1 (full clue embedding) dropped:** The full clue sentence
  embedding served no distinct purpose beyond the contextualized definition
  embedding (Word1_clue_context). For CALE, the sentence embedding without
  delimiters behaves like an ungrounded representation that correlates more
  with bare-word embeddings than with sense-specific ones. The contextualized
  definition embedding directly captures what we need for the misdirection
  analysis. Removing Sentence1 reduces the embedding types from 8 to 7 and
  pairwise cosine similarities from 28 to 21 (15 context-free + 6
  context-informed).
- **BGE-M3 excluded from comparison:** BAAI/bge-m3 was considered but excluded
  because its 1024-dim output could cause confusion with CALE's 1024-dim, it
  showed weak standalone discrimination, and it is already used by the
  indicator_clustering component for a separate purpose.

See `notebooks/00_model_comparison.ipynb` for full evidence.

### Step 3: Feature Engineering
*Not yet started.*

### Step 4: Retrieval Analysis
*Not yet started.*

### Steps 5–8: Dataset Construction and Experiments
*Not yet started.*

### Steps 9–12: Results, Ablation, Sensitivity, Failure Analysis
*Not yet started.*
