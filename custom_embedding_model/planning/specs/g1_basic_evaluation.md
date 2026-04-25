# Spec: g1 Basic Model Evaluation
**Stage:** 5 (exploration, intended to replace current NB 05)
**Notebook:** `planning/exploration/g1_basic_evaluation.ipynb`
**Date:** 2026-04-23

## Purpose

Evaluate g1 as a fine-tuned embedding model using standard data science
practices, progressing from basic model health through intrinsic task
performance to embedding space analysis to the research question. This notebook
addresses gaps in the current NB 05, which jumped directly to the research
question (misdirection/ATE) without first establishing whether g1 is a
competent model on its own terms, and which evaluated exclusively with
cosine similarity despite g1 being trained with L2 distance.

Scope is restricted to the **wndef** phrase type throughout. Cross-format
generalization (wnex) is a hypothesis testing question for Stage 6.

Once finalized, this notebook will be promoted to `notebooks/05_model_evaluation.ipynb`,
and the current NB 05 will be archived.

## Inputs

All paths relative to `custom_embedding_model/`.

**Model artifacts:**
- `models/g1/training_log.json` — per-step and per-epoch training loss, hyperparameters
- `models/g1/val_loss_results.json` — per-epoch val loss, val accuracy, val mean/median margin, train loss

**Triplet files:**
- `data/triplets/g1_train.csv` — 69,921 training triplets (columns: clue_id, definition, answer_wn, distractor_wn, anchor, positive, negative)
- `data/triplets/g1_val.csv` — 46,506 validation triplets (same columns)

**Clue data:**
- `data/filtered_split/wn_synset/clues_wn_filtered.csv` — full clue dataset with split column, definition_wn, answer_wn
- `data/filtered_split/wn_synset/clues_val.csv` — validation-split convenience subset

**Vocabulary:**
- `data/filtered_split/wn_synset/wndef/vocabulary_wndef.csv` — 53,930 words, canonical row ordering

**Embedding files (g1):**
- `data/embeddings/g1/f_common_wndef.npy` — (53930, 1024), indexed by `vocabulary_wndef.csv`
- `data/embeddings/g1/f_clue_val.npy` — (47933, 1024)
- `data/embeddings/g1/f_clue_val_index.csv` — (clue_id, definition, row)

**Embedding files (g_stock, for comparison):**
- `data/embeddings/g_stock/f_common_wndef.npy` — (53930, 1024), indexed by `vocabulary_wndef.csv`
- `data/embeddings/g_stock/f_clue.npy` — (239406, 1024), full dataset
- `data/embeddings/g_stock/f_clue_index.csv` — (clue_id, definition, row)
- Note: g_stock f_clue covers all splits. Extract validation rows by filtering the index on clue_id/definition against `clues_val.csv`.

## Outputs

- `outputs/g1_basic_evaluation-results.md` — full numerical results
- Figures in `outputs/figures/` (naming convention: `g1be_*.png`)

## Implementation details

### Environment detection and imports

Standard environment auto-detection block (local / Great Lakes / Colab).
Define DATA_DIR, EMBEDDINGS_DIR, MODELS_DIR, OUTPUT_DIR.

Key imports: pandas, numpy, scipy.stats (spearmanr), matplotlib, seaborn.
No torch or sentence-transformers needed — this is pure numpy/scipy work.

Print versions per Decision 18.

### Data loading

Load all CSV files with `keep_default_na=False, na_values=[""]`.

Load all `.npy` files and validate shapes against their index/vocabulary
files:
- `g1/f_common_wndef.npy`: assert shape[0] == len(vocabulary_wndef)
- `g1/f_clue_val.npy`: assert shape[0] == len(f_clue_val_index)
- `g_stock/f_common_wndef.npy`: assert shape[0] == len(vocabulary_wndef)
- `g_stock/f_clue.npy`: assert shape[0] == len(f_clue_index)
- All: assert shape[1] == 1024

Extract g_stock validation f_clue embeddings from the full-dataset file:
join `f_clue_index.csv` to `clues_val.csv` on (clue_id, definition) to
get the row indices, then slice `f_clue.npy`. Validate that the resulting
count matches g1's f_clue_val (47,933 rows).

### §1: Training dynamics and overfitting

How did training and validation loss change over the three epochs? Did the
model keep improving on held-out data, or did it start memorizing? Which
checkpoint generalized best?

Read `val_loss_results.json`. Extract per-epoch: train_loss, val_loss,
val_accuracy, val_mean_margin, val_median_margin.

Compute and display:
- Table: epoch, train loss, val loss, train/val loss ratio, val accuracy, val mean margin
- Identify best-generalizing checkpoint: epoch with lowest val loss (epoch 2: 0.252)
- Note that deployed model is epoch 3 (val loss 0.264, train loss 0.014, ratio 19x)

Figure (`g1be_training_dynamics.png`):
- Two-panel figure. Left: train and val loss per epoch (two lines, shared y-axis). Right: val accuracy and val mean margin per epoch (two lines, twin y-axes).

### §2: Task performance

When given a clue, the correct answer, and a distractor, how often does g1
place the clue embedding closer to the answer than to the distractor? Is it
doing this in both L2 (the metric it was trained on) and cosine (the metric we
use for research)?

#### §2a: Triplet evaluation function

Build a reusable function that, given anchor embeddings, positive
embeddings, and negative embeddings, computes:
- **Cosine triplet accuracy:** fraction where cos(anchor, positive) > cos(anchor, negative)
- **L2 triplet accuracy:** fraction where ||anchor - positive||_2 < ||anchor - negative||_2
- **Cosine mean/median margin:** cos(anchor, positive) - cos(anchor, negative)
- **L2 mean/median margin:** ||anchor - negative||_2 - ||anchor - positive||_2

For cosine, use the `rowwise_cosine` pattern from DATA.md (normalize rows,
then dot product). For L2, compute `np.linalg.norm(A - B, axis=1)`.

This function will be reused in §3d for seen/unseen stratification.

#### §2b: Validation triplet accuracy

**g1:** Resolve all 46,506 validation triplets against existing embeddings:
- Anchor: look up (clue_id, definition) in g1 f_clue_val_index → row in g1 f_clue_val.npy
- Positive: look up answer_wn in vocabulary_wndef → row in g1 f_common_wndef.npy
- Negative: look up distractor_wn in vocabulary_wndef → row in g1 f_common_wndef.npy

Report resolution rate (expect 100% with full-vocab wndef per Decision 23).
Compute and report L2 and cosine accuracy, mean margin, median margin.

**g_stock:** Same procedure using g_stock embeddings (g_stock f_clue_val
extracted from full f_clue.npy, g_stock f_common_wndef.npy).

#### §2c: Training triplet accuracy

Note explicitly that g1 f_clue_train embeddings do not exist (Decision 25
was established after g1 was trained). State that training accuracy is
inferred: with epoch 3 training loss of 0.014 and margin 1.0, the loss
formula max(0, ||a-p|| - ||a-n|| + 1.0) implies nearly all training
triplets satisfy the margin constraint, so L2 training accuracy is ~99%+.

#### §2d: Summary table

| Metric | g_stock val | g1 val | g1 train (inferred) |
|---|---|---|---|
| L2 accuracy | ... | ... | ~99%+ |
| Cosine accuracy | ... | ... | — |
| L2 mean margin | ... | ... | — |
| Cosine mean margin | ... | ... | — |

Report the L2-vs-cosine accuracy gap for g1 val. If the gap is large,
note that the model learned magnitude-based discrimination invisible to
cosine evaluation.

### §3: Embedding space geometry

How big are the embedding vectors, and did fine-tuning shrink them? Are
random word pairs more similar to each other under g1 than under g_stock
(crowding)? Is the space still using many dimensions, or has variance
collapsed into a few? Did training reshape the space differently for
vocabulary words it saw during training versus words it never saw?

#### §3a: Norm distributions

Compute L2 norms for:
- g1 f_common_wndef (53,930 words)
- g_stock f_common_wndef (53,930 words)
- g1 f_clue_val (47,933 clues)
- g_stock f_clue_val (47,933 clues)

For each, report: mean, std, min, max, P5, P95.

Table:

| Population | Model | Mean norm | Std | Min | Max | P5 | P95 |
|---|---|---|---|---|---|---|---|

Figure (`g1be_norm_distributions.png`):
- Two-panel figure. Left: wndef norms — g_stock and g1 histograms overlaid
  (alpha=0.5, distinct colors). Right: f_clue norms — same layout.
- Vertical dashed lines at means.

Interpret: Did g1 shrink magnitudes (mean norm decreased)? Did it tighten
the range (std decreased)? Is there a systematic norm gap between the
wndef and f_clue populations (relevant because L2 training computes
distances between them)?

#### §3b: Pairwise cosine among random word pairs

Sample 50,000 random distinct-row pairs from vocabulary_wndef
(random_state=42, same pairs for both models).

For each model, compute cosine similarity for all 50K pairs. Report mean,
median, std, P5, P95.

Table:

| Population | Model | Mean | Median | Std | P5 | P95 |
|---|---|---|---|---|---|---|

Figure (`g1be_pairwise_cosine.png`):
- Two-panel. Left: wndef random pairs — g_stock and g1 overlaid histograms.
  Right: f_clue random pairs — same layout. Vertical lines at means.

Repeat for f_clue_val: sample 50,000 random pairs from the 47,933
validation clue embeddings. Same reporting.

Interpret: High random-pair cosine means embeddings are crowded together —
loss of discriminability. Compare wndef and f_clue populations.

#### §3c: Effective dimensionality

For each of the four embedding matrices (g1 wndef, g_stock wndef, g1
f_clue_val, g_stock f_clue_val):
- Center the matrix (subtract column means)
- Compute singular values via `np.linalg.svd(centered, full_matrices=False, compute_uv=False)`
- Compute participation ratio: (sum(s^2))^2 / sum(s^4)
- Report total variance (sum(s^2)), effective dimensionality, and cumulative
  variance explained by top-10, top-50, top-100 components

Table:

| Population | Model | Total var | Eff. dim | Top-10 % | Top-50 % | Top-100 % |
|---|---|---|---|---|---|---|

Figure (`g1be_singular_values.png`):
- Two-panel. Left: cumulative variance explained curves for wndef (g_stock
  and g1). Right: same for f_clue. X-axis = number of components (1-200),
  y-axis = fraction of variance explained.

#### §3d: Seen/unseen vocabulary stratification

Did training reshape the embedding space differently for words the model
saw during training versus words it never encountered?

Build three word sets from `g1_train.csv` and `clues_wn_filtered.csv`:
1. **seen_wndef**: `set(answer_wn) | set(distractor_wn)` from g1_train.csv — words whose wndef phrases the model directly processed during training
2. **seen_fclue_only**: words appearing as definition_wn in training-split clues (from `clues_wn_filtered.csv` where split == 'train') that are NOT in seen_wndef — words the model saw only in f_clue context
3. **unseen**: words in `vocabulary_wndef.csv` that are in neither set

Report the sizes of all three sets and their coverage of the wndef
vocabulary.

**Norm stratification:** Report mean norm for each word set under g1 and
g_stock wndef embeddings. Did training shrink seen words' norms more than
unseen words?

**Pairwise cosine stratification:** Sample random pairs within each set
(seen_wndef, unseen). Report mean pairwise cosine for each. Are seen words
more crowded together than unseen words under g1?

**Triplet accuracy stratification:** Reuse the triplet evaluation function
from §2a. Stratify the validation triplets by the seen/unseen status of the
positive (answer_wn) and negative (distractor_wn):
- Both seen_wndef
- Both unseen
- Mixed (one seen, one unseen)

For each stratum, report L2 and cosine triplet accuracy and N triplets.
Include g_stock as baseline (g_stock saw no training triplets, so any
variation across strata reflects vocabulary properties, not training
exposure).

If g1 treats seen and unseen words very differently — higher accuracy,
more compression, different norms — that indicates the model's performance
depends on direct optimization of those word vectors during training rather
than generalized learning.

Figure (`g1be_seen_unseen.png`):
- Two-panel. Left: mean norms by word set and model (grouped bar chart).
  Right: triplet accuracy by stratum, L2 and cosine side by side, with
  g_stock baseline.

#### §3e: Compression summary

Summarize the deltas between g_stock and g1 for each measure:
- Mean norm change (wndef and f_clue)
- Pairwise cosine shift (wndef and f_clue)
- Total variance change (absolute and percentage)
- Effective dimensionality change
- Seen vs unseen differential (if any)

### §4: Structural comparison to g_stock

If two words were similar under g_stock, are they still similar under g1?
Or did fine-tuning rearrange which words are near which?

Sample 1,000 words from vocabulary_wndef (random_state=42). Compute the
full pairwise cosine matrix (1000 x 1000) for both g_stock and g1 wndef
embeddings. Extract the upper triangle (499,500 values). Compute Spearman
rank correlation between the two upper triangles.

Report: Spearman rho, p-value.

Repeat with 1,000 random f_clue_val embeddings (random_state=42).

Table:

| Population | N words | N pairs | Spearman rho | p-value |
|---|---|---|---|---|

Interpret: near-zero rho means the similarity structure was fundamentally
reorganized, not just contracted. High rho means fine-tuning preserved
relative similarities.

### §5: Does g1 exploit cryptic clue structure?

Cryptic clues contain structure unique to the genre — surface misdirection,
wordplay pointing to the answer, letter-level tricks — that a well-trained
model might learn to exploit. The ATE measures whether g1 extracts useful
information from clue context, without distinguishing which mechanism is
operating (overcoming misdirection vs discovering helpful cryptic structure).

#### §5a: T=0 and T=1 similarity distributions (cosine)

For each of the 47,933 validation pairs (clues_val.csv), compute:
- T=0: cos_sim(g(f_wndef(def)), g(f_wndef(ans)))
- T=1: cos_sim(g(f_clue(def)), g(f_wndef(ans)))

For both g_stock and g1. Report mean, median, std, P5, P95 for each
distribution.

Table:

| Distribution | Mean | Median | Std | P5 | P95 |
|---|---|---|---|---|---|
| g_stock T=0 | ... | ... | ... | ... | ... |
| g_stock T=1 | ... | ... | ... | ... | ... |
| g1 T=0 | ... | ... | ... | ... | ... |
| g1 T=1 | ... | ... | ... | ... | ... |

Compute ATE = mean(T=1 - T=0) for both models. Report: ATE, median delta,
standard error, 95% confidence interval, and % of pairs with negative delta.

#### §5b: T=0 and T=1 distances (L2)

For each of the 47,933 validation pairs, compute:
- T=0 (L2): ||g(f_wndef(def)) - g(f_wndef(ans))||_2
- T=1 (L2): ||g(f_clue(def)) - g(f_wndef(ans))||_2

For both g_stock and g1.

Note on interpretation: for L2 distance, *smaller* = more similar (opposite
of cosine where larger = more similar). Define L2 context effect as
T=1 distance − T=0 distance. Positive values mean context pushes the
definition farther from the answer.

Report means, medians, standard error, 95% CI, and % of pairs where
context increases distance.

Table:

| Metric | g_stock | g1 |
|---|---|---|
| T=0 L2 distance (mean) | ... | ... |
| T=1 L2 distance (mean) | ... | ... |
| L2 context effect (T=1 − T=0, mean) | ... | ... |
| L2 context effect (SE) | ... | ... |
| L2 context effect (95% CI) | ... | ... |
| % pairs where context increases distance | ... | ... |

#### §5c: Figures

Figure (`g1be_t0_t1_cosine.png`):
- Two-panel. Left: g_stock T=0 and T=1 cosine histograms overlaid. Right:
  g1 T=0 and T=1 cosine histograms overlaid. Same x-axis range for both
  panels.

Figure (`g1be_t0_t1_l2.png`):
- Two-panel. Left: g_stock T=0 and T=1 L2 distance histograms overlaid.
  Right: g1 T=0 and T=1 L2 distance histograms overlaid. Same x-axis range
  for both panels.

#### §5d: Interpretation

Summarize: does g1 extract useful information from cryptic clue context?
Compare the cosine ATE and L2 context effect — do they tell the same story
or different stories? Note which component (T=0 or T=1) drove the change
from g_stock to g1.

Based on the geometry findings from §3 (compression, crowding) and the
context effect from this section, state a prediction for what a retrieval
analysis would show: if we ranked all vocabulary words by similarity to
each definition, would g1 rank the true answer higher or lower than
g_stock? This prediction can be tested in a future analysis.

Point toward NB 06 for investigation of why g1 behaves as it does.

### Summary cell

- Summarize key findings from each section
- State all output files produced and their locations
- Record wall-clock runtime
- Flag what we wish we could have computed but couldn't (g1 f_clue_train
  for explicit training triplet accuracy)
- State the retrieval prediction from §5d as a testable hypothesis
- Note implications for g2 design (Decision 25, Decision 26)

## Environment

Local (CPU). No GPU work needed — all embeddings already exist.

## Notebook structure

- Use §-numbered markdown sections matching the outline above
- Environment auto-detection block for local/Great Lakes/Colab
- Version reporting cell after imports (Decision 18)
- Results file written to `outputs/g1_basic_evaluation-results.md`
- Figures saved to `outputs/figures/g1be_*.png` at 300 dpi
- `random_state=42` for all random sampling
