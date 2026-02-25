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
*Not yet started.*

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
