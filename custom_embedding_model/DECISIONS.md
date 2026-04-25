# Decisions — custom_embedding_model

These decisions are **locked in** and should not be revisited or
second-guessed without explicit team discussion. When a new decision is made,
add it here with its rationale. Claude Code should treat these as hard
constraints.

---

## Decision 1: Embedding Model

**Choice:** `gabrielloiseau/CALE-MBERT-en` (CALE, ModernBERT-based, English,
1024-dim) as g_stock. Fine-tuned variants are g_1, g_2, etc.

**Rationale:** CALE's `<t></t>` delimiter mechanism produces genuinely distinct
embeddings for a target word in context vs. the full sentence (cosine similarity
≈ 0.66), while standard sentence-transformer models produce nearly identical
embeddings for the same comparison (≈ 0.90–0.93). CALE is specifically designed
for word-sense sensitivity, which is central to our misdirection research.
Validated in `clue_misdirection/notebooks/00_model_comparison.ipynb`.

---

## Decision 2: Two-Stage Filtering

**Choice:** Structural filtering (NB 00) and WordNet filtering (NB 01) are
separate notebooks producing separate output files (`clues_filtered.csv` and
`clues_wn_filtered.csv`). Per-f filtering happens in the phrase construction
notebook (NB 02).

**Rationale:** Separating stages creates a clean forking point. If we later
adopt phrase construction strategies that do not require WordNet, we can branch
from `clues_filtered.csv` without rerunning structural filters. It also makes
the cleaning log unambiguous: we can see exactly how much each category of
constraint reduces the dataset.

---

## Decision 3: One Split on clues_wn_filtered.csv

**Choice:** The 30/20/50 train/validate/test split is assigned once on
`clues_wn_filtered.csv` at the (definition, answer) pair level using
`random_state=42`. All f-specific filtered datasets inherit split assignments
from this file by subsetting — they do not receive independent splits.

**Rationale:** Assigning one upstream split ensures that g_stock f_clue
embeddings (computed over the full `clues_wn_filtered.csv`) are correctly
split-aligned with all f-specific datasets. Applying additional f-specific
constraints reduces the dataset size but preserves the split assignments of
remaining rows. The actual resulting split fractions for each f-specific
dataset should be measured and reported in FINDINGS.md.

---

## Decision 4: Unified Vocabulary

**Choice:** Definitions and answers share a single vocabulary per f. A word
appearing as both a definition and an answer gets one entry in the vocabulary
and one set of phrase/embedding files.

**Rationale:** The research goal requires decontextualized embeddings of words
in both roles. The same f phrase and g embedding serves both roles — there is
no reason to compute separate phrases or embeddings for a word depending on
whether it is a definition or an answer in a given clue.

---

## Decision 5: Strict f Definitions — No Fallbacks

**Choice:** Each f is defined only for words where the required phrase can be
constructed without any fallback. A word either has a valid phrase for a given
f or it does not. Absent words are absent from that f's vocabulary and phrase
file and from any experiment using that f.

**Rationale:** The interpretation of what any triplet teaches the model is
always linked to the f's chosen. If f_common_wnex silently fell back to
f_common_wndef for some words, the cross-f generalization test (does g_1
generalize from wndef to wnex format?) would be confounded. Clean f definitions
are essential for clean experimental interpretation.

---

## Decision 6: Canonical Vocabulary Ordering

**Choice:** Vocabulary files use a fixed canonical row ordering established
at creation time and never changed. The `row` column value of a word is its
row index in all corresponding `.npy` embedding arrays.

**Rationale:** This allows vocabulary files to serve as indexes for embedding
arrays without any separate index file. It also means that embedding arrays
for the same vocabulary scope (e.g., `vocabulary_wndef_val.csv`) are directly
comparable across g models — row N in g_stock's array and row N in g_1's array
correspond to the same word.

---

## Decision 7: g_stock f_clue Embeddings Cover Full clues_wn_filtered.csv

**Choice:** g_stock f_clue embeddings are generated over the full
`clues_wn_filtered.csv` dataset in one pass. f-specific analyses look up their
rows from this shared index.

**Rationale:** g_stock is a fixed model — it will not be retrained. Computing
its f_clue embeddings once over the broadest usable dataset avoids returning
to Great Lakes later. Since all f-specific datasets are subsets of
`clues_wn_filtered.csv`, the shared index covers every row any analysis will
need.

If inclusion criteria are later expanded (more rows added to
`clues_wn_filtered.csv`), generate embeddings only for the new rows and append
to the existing `.npy` and index files. Do not regenerate existing rows.

---

## Decision 8: Validation-Only Embeddings for g_i During Iteration

**Choice:** During the iterative model development phase, embedding generation
for fine-tuned models (g_1, g_2, etc.) covers the validation split only. Full-
dataset embeddings for a fine-tuned model are generated only after it is
selected as the final model (Stage 7).

**Rationale:** Embedding generation takes 6–8 hours on Great Lakes. Generating
only validation-split embeddings during iteration avoids spending GPU time on
data we don't need yet. The test set must remain untouched until a final model
is chosen.

---

## Decision 9: Test Set Lockout

**Choice:** Test-split data must not be loaded, inspected, or embedded until a
final g has been chosen and that decision is documented in DECISIONS.md (as a
new entry). All model selection and hypothesis testing uses the validation
split only.

**Rationale:** Using the test set during development — even just for
inspection — risks unconscious bias in model selection decisions. The 50%
test allocation represents a significant investment in final evaluation
credibility.

---

## Decision 10: Triplet Files Share Their g Model's Name, With Explicit Split Suffix

**Choice:** Triplet CSVs are named by g model with an explicit split suffix:
`triplets/<g_name>_train.csv` for training triplets and
`triplets/<g_name>_val.csv` for validation triplets (e.g.,
`triplets/g1_train.csv` and `triplets/g1_val.csv` for g_1).

**Rationale:** The triplet design is inseparable from the model it produced,
so the filename shares the g model's name. The `_train` / `_val` suffix is
required because the file contains only rows from the named split — the
otherwise-implicit convention (no suffix = full scope) would be misleading
here. Making the split explicit in the filename eliminates ambiguity.

---

## Decision 11: Triplet Construction and Model Training in One Notebook

**Choice:** Triplet construction and model training are handled in the same
notebook (nb_03_train_<g_name>.ipynb) and script (scripts/train_<g_name>.py).

**Rationale:** Triplet construction is fast (CPU, text manipulation) and
produces a committed artifact that the training script reads immediately. There
is no benefit to separating them into different stages. The triplet CSV is
still saved as an inspectable intermediate artifact.

---

## Decision 12: Model Weights Stored in Google Drive

**Choice:** Fine-tuned model weights are stored in the shared Google Drive
folder "Research Project - NLP CCC's" (owned by Nathan). The `models/<g_name>/`
directories in the repo contain only `README.md` placeholder files with
metadata — no weights are committed to git.

**Rationale:** Model weight files are large binary artifacts unsuitable for
git. Google Drive is accessible to all team members and does not require
Git LFS setup.

---

## Decision 13: Triplet Margin Loss with α = 1.0

**Choice:** Triplet margin loss (Schroff et al., 2015) with margin α = 1.0,
following KCT's boilerplate.

**Rationale:** α = 1.0 is the value used in the initial g_1 training (NB 09)
and provides a reasonable baseline. Hyperparameter tuning of α is deferred.

---

## Decision 14: Do Not Modify Shared Directories

**Choice:** The following directories must not be modified:
- `ccc-project/data/` (shared raw data)
- `clue_misdirection/` (complete Milestone II component)
- `indicator_clustering/` (complete unsupervised component)

**Rationale:** These directories contain stable, completed work that other
analyses depend on. Any modification risks breaking existing results.

---

## Decision 15: Runtime Tracking in FINDINGS.md

**Choice:** Wall-clock runtimes for GPU steps (embedding generation, model
training) are recorded in FINDINGS.md, not in NOTEBOOKS.md.

**Rationale:** Runtimes depend on conditions (partition, row count, batch size,
sample mode) that must be recorded alongside the number to be useful.
FINDINGS.md is the natural home for empirical measurements with context.
NOTEBOOKS.md records static facts about notebooks, not empirical results.

---

## Decision 16: Article and Infinitive Stripping for WordNet Lookup

**Choice:** When looking up WordNet synsets for a definition or answer, strip
the first matching prefix from `"a "`, `"an "`, `"the "`, `"to "` (in that
order) if the initial lookup fails. Only one prefix is tried per word — the
first that matches the input. If that stripped lookup also fails, the word
has no synsets.

**Rationale:** Cryptic crossword definitions often include a leading article
or infinitive marker as part of the natural surface reading (e.g., "a shade",
"the law", "to flee"). WordNet stores headwords without these prefixes.
Stripping `"a "` was used in the Milestone II pipeline; expanding to
`"an "`, `"the "`, and `"to "` recovered 1,609 additional unique definitions
in NB 01. The prefix `"one "` was considered and rejected because "one" carries
semantic content and could cause false matches.

---

## Decision 17: Pin Critical Package Versions

**Choice:** The following packages are pinned to exact versions in
`requirements.txt` because version differences produce different results:

- `nltk==3.9.2` — different NLTK versions resolve different words to WordNet
  synsets, changing row counts (observed: NLTK 3.8.1 vs 3.9.2 produced a
  49-row difference in NB 01)
- `scikit-learn==1.8.0` — `train_test_split` with `random_state=42` must
  produce identical splits; algorithm changes between versions would
  invalidate all downstream artifacts

Other packages use `>=` minimums. When a package is promoted to an exact pin,
document the reason here.

**Rationale:** Discovered when the Verifier agent ran NB 01 under the default
`python3` kernel (NLTK 3.8.1) and produced different row counts than
Victoria's run on the `crossword` kernel (NLTK 3.9.2). Silent version
differences in packages that affect data filtering or splitting can
invalidate results without any visible error.

---

## Decision 18: Version Provenance in Notebooks and Results Files

**Choice:** Every notebook must include:

1. A **version-reporting cell** immediately after imports that prints the
   versions of all packages used in that notebook. This is a `print()`
   statement, not an assertion — it should not block execution if a
   collaborator has a slightly different environment, but it makes
   mismatches immediately visible.

2. A **versions section** in the results file
   (`outputs/<notebook-name>-results.md`) stamping the exact versions
   that produced those results.

**Rationale:** Nathan may run notebooks on Great Lakes with different package
versions. Printing versions in the notebook output makes mismatches visible
at a glance. Stamping versions in the results file creates a permanent
provenance record tied to the actual numbers.

---

## Decision 19: GPU Script Environment Provenance

**Choice:** Every GPU script that produces committed artifacts (`.npy`
embedding files, model weights, etc.) must:

1. **Print versions at startup** to stdout (captured in the SLURM log):
   Python, torch, sentence-transformers, transformers, numpy, and any other
   packages used by the script.

2. **Record the environment in FINDINGS.md** when logging the run's results.
   The FINDINGS.md entry for each GPU step must include: Python version, torch
   version (including CUDA build suffix), sentence-transformers version,
   transformers version, and the conda environment name used.

3. **Maintain `requirements-greatlakes.txt`** as a pinned snapshot of the
   Great Lakes conda environment used for GPU work. Update it whenever the
   environment changes. Generate with
   `conda list -n nlp_env --export > requirements-greatlakes.txt` or
   equivalent.

The local environment (`requirements.txt`) and the Great Lakes environment
(`requirements-greatlakes.txt`) are tracked separately because they serve
different purposes: local work is CPU-only (notebooks, hypothesis testing),
while Great Lakes work is GPU (embedding generation, model training). The
two environments share pinned versions for packages that affect data
integrity (NLTK, scikit-learn per Decision 17) but may differ on packages
that only affect computation (torch, sentence-transformers).

**Rationale:** Embedding outputs are committed artifacts that downstream
analysis depends on. If a future reproduction attempt uses different package
versions and gets different embeddings, there must be a permanent record of
what produced the originals. SLURM logs are ephemeral; FINDINGS.md and a
committed requirements file are not.

---

## Decision 20: Mean Pooling Is Canonical for CALE

**Choice:** All CALE embeddings in this project must use mean pooling over
all non-padding tokens (i.e., `SentenceTransformer.encode()` or equivalent
attention-masked mean pooling over the full hidden state). This is how CALE
was trained, published, and evaluated.

The token span extraction method used in NB 09 and `scripts/train_g1.py`
(averaging hidden states only for tokens within the `<t></t>` span) is a
non-standard extraction method that produces different embeddings (mean
cosine similarity of 0.926 vs. the canonical method). The model trained
with this method is renamed `g1_tokenspan` to distinguish it from the
corrected `g1` (to be trained with mean pooling).

**Evidence:**
- CALE's published `1_Pooling/config.json` sets `pooling_mode_mean_tokens:
  true` with all other pooling modes false
- CALE's `modules.json` contains exactly two modules: Transformer + Pooling
  (no custom concept-aligned pooling layer)
- CALE's README and model card show usage via `model.encode()`, not manual
  extraction
- `clue_misdirection/notebooks/00_model_comparison.ipynb` used
  `SentenceTransformer.encode()` for all CALE embeddings in the model
  selection evaluation that led to Decision 1
- The consistency check in `scripts/embed_val.py` measured mean cosine
  similarity of 0.926 between token span extraction and
  `SentenceTransformer.encode()` — confirming they produce substantively
  different embeddings

**Naming convention:** Models using mean pooling (the canonical method) carry
no suffix. Models using token span extraction carry the `_tokenspan` suffix.
This applies to model names (`g1` vs `g1_tokenspan`), file paths, and all
documentation references.

**Rationale:** CALE's concept alignment comes from the transformer weights
and attention mechanism (trained on concept-definition pairs with `<t></t>`
delimiters), not from a special pooling layer. The `<t></t>` delimiters guide
the attention patterns during the forward pass, and mean pooling over the
resulting hidden states produces the concept-aligned embedding. The token
span method bypasses this design. Since all prior work in this project
(model selection, g_stock f_clue embeddings) used mean pooling, it must
remain the standard for comparability.

---

## Decision 21: Validation Triplet Accuracy Uses Resolvable Subset

**Choice:** Validation triplet accuracy in NB 05 is computed over the subset
of `g1_val.csv` triplets where all three embedding lookups succeed (~27,348
of 46,506 rows, ~58.8%). The ~41.2% of triplets with unresolvable negatives
are dropped and the count is reported explicitly.

**Why negatives are unresolvable:** Distractor words in `g1_val.csv` come
from `dataset_harder.parquet` and are drawn from the full WordNet vocabulary.
Validation-split embeddings (`f_common_wndef_val.npy`) cover only the 26,152
words appearing as definitions or answers in validation-split clues (per
Decision 8: validation-only embeddings during iteration). A distractor word
that appears only in the training or test split has no g1 validation
embedding.

**Bias caveat:** The surviving triplets' negatives are words that also appear
as definitions or answers in validation clues — i.e., common crossword words.
Whether these are systematically easier or harder negatives than the dropped
distractors is unknown. However, the comparison between g_stock and g1 is
computed on the identical set of triplets, so any difficulty bias affects
both models equally and does not compromise the g_stock-vs-g1 comparison.

**Alternatives considered and rejected:**
- Re-generate embeddings for the union of `vocabulary_wndef_val.csv` and all
  distractor words (another GPU run, delays work, low payoff given 27K is
  already well-powered for a diagnostic check)
- Use g_stock full-dataset embeddings for unresolvable negatives
  (methodologically unsound — breaks the clean g_stock-vs-g1 comparison)

**Rationale:** 27,348 triplets is more than sufficient for a diagnostic
check of whether the model generalizes the training objective. The primary
research findings come from the ATE analysis (hypothesis testing), not from
triplet accuracy. Spending GPU time to recover ~19,000 additional triplets
for a diagnostic measure is not justified during iterative development.

---

## Decision 22: Full-Vocabulary wnex Embeddings for g_stock and g1

**Choice:** Generate f_common_wnex embeddings over the full wnex vocabulary
(8,360 words) for both g_stock and g1, stored as:
- `data/embeddings/g_stock/f_common_wnex.npy` — indexed by `vocabulary_wnex.csv`
- `data/embeddings/g1/f_common_wnex.npy` — indexed by `vocabulary_wnex.csv`

These supplement (do not replace) the existing val-only files
(`f_common_wnex_val.npy`, 3,008 words).

**Rationale:** NB 05 showed that g1 compressed wnex embeddings even though
it was never trained on wnex phrases. To investigate whether g1 also learned
discriminative structure in wnex space (cross-f triplet accuracy), we need
embeddings for more than just the 3,008 validation-split words — most triplet
distractors fall outside that set. The full wnex vocabulary is only 8,360
words (~12 seconds of GPU time), so Decision 8's compute-cost rationale for
val-only embeddings does not apply at this scale.

Test-split words are included in the vocabulary because individual word
embeddings carry no test-set evaluation signal (see Decision 9). Computing
a test-set ATE would require assembling (clue_id, definition, answer)
evaluation triples from test-split clues, which remains locked.

---

## Decision 23: Full-Vocabulary Embeddings for Vocabulary-Indexed Phrase Types

**Choice:** For vocabulary-indexed phrase types (f_common_wndef, f_common_wnex,
and any future f's indexed by a vocabulary file), always generate embeddings
over the full vocabulary, not just the validation subset. This amends
Decision 8, which now applies only to clue-indexed embeddings (f_clue).

Specifically, f_common_wndef embeddings (53,930 words) were generated for
both g_stock and g1, stored as:
- `data/embeddings/g_stock/f_common_wndef.npy` — indexed by `vocabulary_wndef.csv`
- `data/embeddings/g1/f_common_wndef.npy` — indexed by `vocabulary_wndef.csv`

These supplement (do not replace) the existing val-only files
(`f_common_wndef_val.npy`, 26,152 words). For future models (g2, etc.),
generate only the full-vocabulary versions — no `_val` variants.

**f_clue remains val-only during iteration.** Clue-indexed embeddings
include the clue's surface text and are keyed by (clue_id, definition).
Embedding test-split clues during iteration would violate the test-set
lockout (Decision 9). Full f_clue embeddings are generated only for the
final model (Stage 7).

**Rationale:** Vocabulary-indexed embeddings are cheap (~3 minutes for
53K words on a V100) and carry no test-set evaluation signal — they embed
individual words, not (clue, definition, answer) evaluation triples.
Val-only vocabulary embeddings created resolution gaps in triplet accuracy
evaluation (Decision 21: ~41% of validation triplets dropped) and would
have prevented fair cross-f comparisons. Generating full-vocabulary
embeddings by default eliminates these gaps for all future models. The
same reasoning as Decision 22 applies, generalized from wnex to all
vocabulary-indexed phrase types.

---

## Decision 24: Training Scripts Must Track Validation Loss

**Choice:** Every model training script must compute and log validation
loss at the end of each epoch. Specifically, the script must:

1. Accept a `--val-input` argument pointing to the validation triplet file
   (e.g., `g1_val.csv`)
2. After each epoch, run a forward pass on the full validation set with
   `model.eval()` and `torch.no_grad()` — same loss function, same margin,
   same extraction method as training
3. Log per-epoch validation loss alongside training loss to stdout and to
   the structured training log (e.g., `training_log.json`)
4. Save per-epoch model checkpoints (already standard — see `train_g1.py`)

The summary table printed at the end of training must include both
training and validation loss for each epoch, enabling immediate visual
inspection for overfitting (validation loss plateauing or increasing while
training loss decreases).

**Rationale:** The g1 training script tracked training loss per epoch
([0.470, 0.111, 0.014]) but did not compute validation loss, making it
impossible to diagnose overfitting during or after training. This omission
was identified during the g1 investigation design phase. Validation loss
tracking is standard practice in supervised learning and must not be
omitted in future training runs. For g1, validation loss was computed
retroactively from saved epoch checkpoints
(`scripts/val_loss_from_checkpoints.py`).

---

## Decision 25: Training-Split f_clue Embeddings Required for Model Evaluation

**Choice:** For each fine-tuned g_i, Stage 4 must generate training-split
f_clue embeddings (`f_clue_train.npy` + `f_clue_train_index.csv`) in
addition to validation-split f_clue embeddings. This amends Decision 8,
which previously limited fine-tuned model f_clue embeddings to the
validation split only.

Test-split f_clue embeddings remain locked until the final model is chosen
(Decision 9). Training-split vocabulary-indexed embeddings are not needed
because full-vocabulary embeddings (Decision 23) already cover all words
appearing in training-split clues.

**Rationale:** Standard model evaluation requires comparing performance on
training data versus validation data to diagnose overfitting. Training
triplet accuracy requires the anchor embeddings — g_i(f_clue(definition)) —
for training-split clues. Without these, overfitting can only be assessed
through training loss curves, which conflate the margin penalty with
classification performance. Decision 24 established that validation loss
must be tracked during training; this decision extends overfitting
diagnostics to the embedding evaluation stage, enabling direct comparison
of training versus validation triplet accuracy.

For g_stock, training-split f_clue embeddings can be extracted from the
existing full-dataset `f_clue.npy` by filtering the index file — no new
generation is needed.

---

## Decision 26: Evaluate Models Using Both Training and Research Metrics

**Choice:** Stage 5 model evaluation must report triplet accuracy and
distance/similarity distributions using both the training metric and the
research metric. For models trained with `nn.TripletMarginLoss(p=2)`, the
training metric is L2 (Euclidean) distance; the research metric is cosine
similarity. The gap between L2 and cosine triplet accuracy quantifies how
much of the model's learning is magnitude-based versus angular.

Additionally, Stage 5 must include embedding norm analysis (L2 norm
distributions, means, and standard deviations) for both vocabulary-indexed
and f_clue embeddings, comparing the fine-tuned model against g_stock.

**Rationale:** g1 was trained with `nn.TripletMarginLoss(margin=1.0, p=2)`
on unnormalized embeddings — optimizing Euclidean distance — but all
Stage 5 evaluation metrics used cosine similarity exclusively. This
train-eval metric mismatch meant that any magnitude-based discrimination
learned by the model was invisible to evaluation. Euclidean distance on
unnormalized vectors gives the model a degree of freedom (embedding
magnitude) that cosine similarity cannot detect. Evaluating in both
metrics reveals whether the model exploited this shortcut (reducing L2
distances via magnitude shrinkage rather than learning angular/semantic
structure). Norm analysis directly tests for this magnitude exploitation.

For g1, L2-based evaluation and norm analysis can be computed from existing
embedding artifacts — no new GPU work is needed.

---

## Decision 27: Training Must Include Learning Curves and Runtime Tracking

**Choice:** Every model training run must include:

1. **Learning curves.** Train multiple models on increasing subsets of the
   training data (e.g., 10%, 25%, 50%, 75%, 100%) and evaluate each on the
   full validation set. This answers "how much training data is necessary?"
   and reveals whether problems are driven by data quantity vs. data quality.

2. **Runtime estimates and actuals.** Before submitting GPU jobs, estimate
   the wall-clock time for each component in FINDINGS.md:
   - Base model training (full triplet set, all epochs)
   - Each learning curve subset training run
   - Validation loss computation per epoch (already required by Decision 24)
   - Per-epoch embedding generation if applicable

   After each job completes, record the actual runtime alongside the estimate.
   Over time, these records build a reference for planning future jobs.

3. **Structured output.** Learning curve results (subset size, train loss,
   val loss, val accuracy, val margin for each subset) must be saved to
   `models/<g_name>/learning_curve_results.json` and logged to stdout.

**Rationale:** g1's basic evaluation could not distinguish whether the model's
problems (compression, amplified context effects) were caused by training on
too much data of the wrong kind, or whether the same problems would appear
with less data. Learning curves are the standard diagnostic for data quantity
questions. They are expensive (one full training run per subset), but
planning ahead and tracking runtimes makes the compute cost predictable and
budgetable. Since learning curves are specific to a given triplet design and
phrase construction, they do not transfer across models — each g_i needs its
own. Runtime tracking ensures we can estimate costs for future models.
