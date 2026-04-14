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
selected as the final model (Stage 6).

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

## Decision 10: Triplet Files Share Their g Model's Name

**Choice:** The triplet CSV used to train g_i is named `triplets/<g_name>.csv`
(e.g., `triplets/g1.csv` for g_1).

**Rationale:** The triplet design is inseparable from the model it produced.
Naming them identically makes the relationship explicit and avoids ambiguity
when multiple triplet designs exist.

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
