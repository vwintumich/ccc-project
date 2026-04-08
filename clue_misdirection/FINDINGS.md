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

### Step 2: Embedding Generation
**Model Investigation Phase** — *Completed.* Notebook: `notebooks/00_model_comparison.ipynb`

**Embedding Generation** — *Completed.*

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

### Step 2 (continued): Phrase Construction
*Completed.* Notebook: `notebooks/02_embedding_generation.ipynb` (cells 0–21, CPU portion)

**Phrase construction approach and path distribution:**
CALE requires exactly one `<t></t>` pair per input and works best when the
target word appears only once in the text. For each unique definition or answer,
we looked up all WordNet synsets and constructed a context phrase for each synset
using a priority cascade: (1) the synset's usage example, if the target word
appears exactly once — preferred because natural sentences match CALE's training
data, used for ~22% of phrases; (2) the synset's definition text, if the target
word appears exactly once — used for ~3% of phrases; (3) a fallback format
`<t>word</t>: definition_text`, used when the word has zero occurrences in the
definition — ~75% of phrases. Each path requires the target word to appear
exactly once in the chosen text before wrapping with `<t></t>`. The fallback is
safe because the word only appears inside the delimiters.

**Unresolvable phrases:**
An "unresolvable phrase" is a specific (word, synset) combination where the
target word appears 2+ times in *both* the usage example and the definition
text, making it impossible to construct a phrase with exactly one occurrence of
the target. For example, the definition of `admiral.n.01` is "the supreme
commander of a fleet; ranks above a vice admiral and below a fleet admiral" —
"admiral" appears 3 times, so we cannot safely place a single `<t></t>` pair
around just one occurrence. In total, 234 phrases out of ~227K (0.1%) were
unresolvable and removed from the phrase files.

Most words that had unresolvable phrases still had other synsets that worked
fine. For example, "admiral" has 2 synsets; only 1 was unresolvable, so the
allsense average for "admiral" uses 1 synset instead of 2. Only 12 words had
*all* synsets unresolvable (mostly proper nouns: america, armenia, germany,
carolina, labrador, suez, sesame, betel, cain, tut). These words cannot be
embedded at all, affecting 108 rows in the clue dataset.

**Duplicate definitions in surface text:**
1,079 rows (0.45%) had the definition string appearing 2+ times in the clue
surface text (e.g., definition="letter", surface="Foreign letter coming in is
the French letter"). Since `insert_cale_delimiters` wraps only the first
occurrence, the second occurrence remains undelimited — CALE would see the
target word both inside and outside `<t></t>`, which is ambiguous and may cause
the model to attend to the wrong instance. These rows were dropped rather than
engineering a complex fix for less than half a percent of the data.

**Total cleanup:** 1,186 rows dropped (0.49%), leaving 240,211 rows for
embedding.

**Article stripping and multi-word WordNet entries:**
Step 1's data cleaning used article stripping during WordNet lookup ("a shade"
→ "shade") but stored the original definition in `clues_filtered.csv`. Step 2
creates `definition_wn` and `answer_wn` columns that mirror this heuristic with
an improvement: before stripping the article, we first try replacing spaces with
underscores (e.g., "a little" → "a_little"), since WordNet uses underscores for
multi-word entries. This recovered entries like "a_little", "a_priori",
"a_cappella" that would have been incorrectly stripped to "little", "priori",
"cappella" under the Step 1 approach. The original `definition` and `answer`
columns are preserved for surface text matching (finding the definition within
the `surface` string).

**Sense variation tracking:**
Approximately 35% of unique definitions and 46% of unique answers have only 1
usable WordNet synset. For these words, the allsense, common, and obscure
embeddings will be identical (all derived from the same single synset). The
phrase files include a `num_usable_synsets` column, and `clue_context_phrases.csv`
includes `def_num_usable_synsets` and `ans_num_usable_synsets` columns so
downstream notebooks can identify and stratify by sense variation. This is
especially relevant for the retrieval analysis (Step 4), where single-synset
words have no sense-based variation to exploit, and may affect classifier
feature importance (Steps 6–8) for features that compare common vs. obscure
embeddings.

### Step 2 (continued): Embedding Generation
*Completed.* Script: `scripts/embed_phrases.py`, submitted via `scripts/embed_phrases.sh`

The GPU embedding step loaded the three phrase CSV files produced by the CPU
portion of `02_embedding_generation.ipynb` and encoded all phrases using
`gabrielloiseau/CALE-MBERT-en` (1024-dim) via sentence-transformers. The job
ran on a Tesla V100-PCIE-16GB on the UM Great Lakes cluster in 18.5 minutes
(PyTorch 2.5.1+cu121, sentence-transformers 5.2.2).

**Output files and sizes:**

- `definition_embeddings.npy`: shape (27,385, 3, 1024) — 321 MB. Each row
  contains three 1024-dim embeddings for a unique definition: allsense average
  (mean across all WordNet synset contexts), most-common synset, and
  least-common synset.
- `answer_embeddings.npy`: shape (45,254, 3, 1024) — 530 MB. Same three-slot
  structure for unique answers.
- `clue_context_embeddings.npy`: shape (240,211, 1024) — 938 MB. One embedding
  per clue row: the definition word embedded within the clue surface using
  CALE's `<t></t>` delimiters.
- Total storage: ~1.8 GB for all embedding files.
- Three corresponding index CSVs (`definition_index.csv`, `answer_index.csv`,
  `clue_context_index.csv`) map row positions to `definition_wn` strings,
  `answer_wn` strings, and `clue_id` values respectively.

**Verification results:**

All shapes match their index files and the embedding dimension is 1024
throughout. No NaN values or all-zero rows were found in any array. Every
`definition_wn` and `answer_wn` referenced by rows in `clue_context_phrases.csv`
has a corresponding entry in the index files, confirming no embeddings are
missing for downstream feature computation. Spot-checks using cosine similarity
confirm CALE produces semantically meaningful embeddings: related words (e.g.,
plant/flower) have higher similarity than unrelated words (e.g., plant/letter),
and polysemous words like "plant" show distinct common vs. obscure embeddings.

**Embedding space statistics:**

Allsense embeddings have slightly lower mean L2 norms (~27.3–27.7) than
single-synset embeddings (~29.4), which is expected since averaging across
multiple sense vectors tends to reduce the norm. Mean cosine similarity between
common and obscure senses across definitions is 0.81 (std 0.20), and across
answers is 0.84 (std 0.19). This confirms CALE differentiates between sense
extremes, with substantial variation across words — some highly polysemous words
have cos(common, obscure) as low as 0.1, while single-synset words have
cos = 1.0 by construction. Single-synset words (where common = obscure)
account for 9,700 definitions (35.4%) and 20,626 answers (45.6%). These are
tracked via `num_usable_synsets` for downstream stratification.

### Step 3: Feature Engineering
*Completed.* Notebook: `notebooks/03_feature_engineering.ipynb`

**Merge fix for double-definition clues:** The initial merge between
`clues_filtered.csv` (241,397 rows) and `clue_context_phrases.csv` (240,211
rows) on `clue_id` alone caused a many-to-many join, inflating the working
set to 254,262 rows. Double-definition clues (e.g., "Ruined a sculpture"
→ clue_id 150 with definitions "Ruined" and "a sculpture") have multiple
rows per `clue_id` in both files. Fixed by merging on the composite key
(`clue_id`, `definition`) to produce the correct 240,211 rows.

**Feature counts:** 47 total features (1 more than the 46 originally planned):
- Context-free cosine (15): cross-word similarities mean 0.50–0.65;
  within-word similarities mean 0.69–0.92.
- Context-informed cosine (6): `cos_w1clue_w2all` (0.54 mean) is lower
  than `cos_w1all_w2all` (0.65 mean), consistent with the misdirection
  hypothesis — clue context pulls the definition embedding away from the
  answer.
- WordNet relationship (22, not the originally planned 21): 20 boolean
  two-hop types (the originally planned 19 plus `hypernym_of_hyponym` from
  Table 3) + `wn_max_path_sim` + `wn_shared_synset_count`. 50.3% of rows
  (44.1% of 127,608 unique pairs) have at least one WordNet connection —
  higher than the preliminary 31% estimate, likely because our broader
  dataset includes multi-word entries with richer WordNet coverage.
  Hyponym_of_hypernym (21.5%), hyponym (18.8%), and synonym (15.0%) are
  the most common relationship types.
- Surface (4): edit distance (mean 6.7), length ratio (mean 0.73),
  shared first letter (7.4%), character overlap ratio (mean 0.23).

**Single-synset words:** Within-word cosine features are exactly 1.0 for
words with only one usable WordNet synset (~35% of definitions, ~46% of
answers). `def_num_usable_synsets` and `ans_num_usable_synsets` are carried
as metadata for downstream stratification (Decision 19).

**Multicollinearity:** Top correlated feature pairs are among cosine
features involving the same sense types (e.g., `cos_w1obscure_w2all` ×
`cos_w1obscure_w2obscure`, r = 0.87). Expected and acceptable for
tree-based models; may affect logistic regression coefficient
interpretation.

**Output:** `data/features_all.parquet` — 240,211 rows × 60 columns
(47 features + 13 metadata).

**Pitfalls identified for Step 4 (Retrieval Analysis):**
- `clue_context_index.csv` has non-unique `clue_id`s (double-definition
  clues). NB 04 must use `clue_context_phrases.csv` or a composite key
  for clue-context embedding lookups.
- All CSV loads involving `definition_wn`, `answer_wn`, or `word` columns
  MUST use `keep_default_na=False` to prevent the word "nan" from being
  silently converted to NaN.
- ~35% of definitions have only 1 usable synset, making the Common vs
  Obscure retrieval comparison uninformative for those pairs. NB 04
  should report what percentage of pairs this affects.
- The 4×3 retrieval matrix requires 12 separate retrieval runs with
  different candidate answer matrices (Allsense, Common, Obscure on
  the answer side).

### Step 4: Retrieval Analysis
*Completed.* Notebook: `notebooks/04_retrieval_analysis.ipynb`

**Primary analysis (unique pairs):** Retrieval over 127,608 unique
(definition, answer) pairs against a candidate pool of 45,254 answers,
using CALE-MBERT-en (1024-dim) embeddings. Results for 4 definition
conditions × 3 answer conditions:

| Def Condition | Ans Condition | Top-1 | Top-10 | Top-100 | Mean Rank | Median Rank | Mean Cos Sim |
|---|---|---|---|---|---|---|---|
| Allsense | Allsense | 0.30% | 7.69% | 23.77% | 5,173 | 1,015 | 0.643 |
| Allsense | Common | 0.50% | 6.78% | 20.53% | 7,698 | 1,741 | 0.591 |
| Allsense | Obscure | 0.53% | 6.45% | 19.71% | 7,687 | 1,872 | 0.590 |
| Common | Allsense | 0.57% | 7.12% | 21.75% | 6,761 | 1,389 | 0.569 |
| Common | Common | 0.32% | 6.63% | 19.48% | 8,838 | 2,208 | 0.530 |
| Common | Obscure | 0.87% | 5.96% | 17.94% | 9,485 | 2,674 | 0.518 |
| Obscure | Allsense | 0.41% | 6.03% | 18.80% | 7,267 | 1,964 | 0.564 |
| Obscure | Common | 0.70% | 5.31% | 16.09% | 9,852 | 3,375 | 0.515 |
| Obscure | Obscure | 0.27% | 5.20% | 16.11% | 9,410 | 3,194 | 0.522 |
| Clue Context | Allsense | 0.41% | 4.40% | 16.25% | 7,401 | 2,160 | 0.542 |
| Clue Context | Common | 0.42% | 3.86% | 14.28% | 9,672 | 3,350 | 0.499 |
| Clue Context | Obscure | 0.42% | 3.66% | 13.69% | 9,811 | 3,564 | 0.497 |

**Misdirection effect:** Clue Context × Allsense (median rank 2,160) vs.
Allsense × Allsense (median rank 1,015) shows a +1,145 rank worsening.
Top-10 hit rate drops from 7.69% to 4.40% (43% relative decrease). This
directly demonstrates that embedding the definition word within the clue's
surface text pushes the representation away from the true answer — the
primary evidence for semantic misdirection.

**Allsense outperforms Common and Obscure:** On both the definition and
answer sides, the allsense-average embedding retrieves the true answer
better than either single-synset embedding. This is because averaging
across senses hedges against picking the wrong sense, providing partial
overlap with the true answer regardless of which synset was intended. Our
allsense average weights all WordNet synsets equally (not by frequency),
which artificially pulls toward obscure senses. Frequency-weighted sense
averaging is noted as a future improvement.

**Obscure definition can be worse than clue-context misdirection:** In
some cross-condition comparisons, committing to the rarest WordNet synset
hurts retrieval even more than clue-context misdirection. For example,
Obscure × Common (median rank 3,375) is worse than Clue Context × Allsense
(median rank 2,160). This suggests that when a definition word is highly
polysemous and the obscure sense diverges sharply from the answer's
meaning, the resulting embedding is pushed further away from the true
answer than the clue surface reading pushes it. Sense commitment is a
different mechanism from misdirection — it picks the wrong meaning
outright rather than being misled by surrounding context — but the
retrieval degradation can be comparable or larger.

**Supplementary all-rows analysis:** Allsense × Allsense over all 240,211
rows yields median rank 831 (vs. 1,015 for unique pairs), confirming that
frequently-reused pairs tend to be easier. This validates Decision 5's
choice of unique pairs as the more conservative primary reporting unit.

**Single-synset caveat:** ~35% of definitions and ~46% of answers have only
1 usable WordNet synset (Common = Obscure = Allsense). For the ~17.8% of
pairs where both are single-synset, all context-free conditions produce
identical ranks. The Common vs. Obscure comparisons are meaningful only
for multi-synset words.

**Scale comparison with preliminary results:** Hans's earlier work
(all-mpnet-base-v2, 8,598 candidates, 10K sample) found context-free
median rank 177.5 and context-informed median rank 684 (+506 worsening).
With CALE and 45,254 candidates, we find 1,015 → 2,160 (+1,145). Absolute
ranks are not comparable (5× larger pool), but the directional pattern —
context worsening retrieval rank — is consistent. In our analysis, clue
context roughly doubles the median rank (2.1×); Hans's preliminary result
showed a larger 3.9× increase with a smaller candidate pool.

### Steps 5 & 7: Dataset Construction
*Completed.* Notebook: `notebooks/05_dataset_construction.ipynb`

**Easy dataset (Step 5):** 480,422 rows (240,211 real + 240,211 random
distractors), all 47 features. Random distractors are trivially separable:
`cos_w1all_w2all` gap +0.218 (real 0.648 vs distractor 0.430).

**Harder dataset (Step 7):** 480,422 rows (240,211 real + 240,211 top-100
cosine-similarity distractors per Decision 6), 32 features (15 context-free
meaning features removed). The harder distractor selection works as intended:

- **Context-informed cosine gap flips negative:** `cos_w1clue_w2all` real
  0.545 vs distractor 0.615 (gap −0.070). Distractors are now *more* similar
  to definitions than real answers on this metric. Raw cosine similarity can
  no longer distinguish real from distractor — classifiers must rely on
  subtler signals.
- **WordNet gap halved but persists:** `wn_max_path_sim` gap shrinks from
  +0.288 (easy) to +0.155 (harder). Harder distractors have real WordNet
  connections to definitions (they are cosine-similar words that tend to
  share WordNet neighborhoods), but still fewer than true answer pairs. This
  remaining gap is what classifiers can exploit.
- **15 context-free meaning features correctly removed** per Decision 6 —
  they are artifacts of the cosine-similarity-based distractor selection
  method. Remaining 32 features: 6 context-informed + 22 relationship +
  4 surface.

**Feature computation:** Distractor feature computation uses
`scripts/feature_utils.py` (Decision 18) with deduplication optimization —
relationship and surface features are computed per unique (definition,
distractor_answer) pair, then broadcast to all rows sharing that pair.
Cosine features are computed per row because the clue-context embedding
varies across clues.

### Steps 6 & 8: Classification Experiments
*Completed.* Notebooks: `notebooks/06_experiments_easy.ipynb`,
`notebooks/07_experiments_harder.ipynb`. Full-data results from Great
Lakes (`scripts/run_experiments.py`, 480,422 rows per dataset).

**Easy dataset (Step 6) — full data:**

| Experiment | Model | Accuracy | F1 | ROC AUC |
|---|---|---|---|---|
| Exp 1A (47 features) | KNN | 0.863 ± 0.001 | 0.856 ± 0.002 | 0.928 ± 0.001 |
| Exp 1A | Logistic Regression | 0.869 ± 0.002 | 0.865 ± 0.002 | 0.938 ± 0.002 |
| Exp 1A | Random Forest | 0.877 ± 0.001 | 0.871 ± 0.002 | 0.945 ± 0.001 |
| Exp 1B (41 features) | KNN | 0.862 ± 0.002 | 0.856 ± 0.003 | 0.932 ± 0.002 |
| Exp 1B | Logistic Regression | 0.867 ± 0.002 | 0.863 ± 0.002 | 0.937 ± 0.002 |
| Exp 1B | Random Forest | 0.873 ± 0.001 | 0.864 ± 0.001 | 0.944 ± 0.001 |

**Δ Easy:** +0.1 to +0.4pp. Tiny as expected — sanity check passes.

**Harder dataset (Step 8) — full data:**

| Experiment | Model | Accuracy | F1 | ROC AUC |
|---|---|---|---|---|
| Exp 2A (32 features) | KNN | 0.739 ± 0.001 | 0.724 ± 0.002 | 0.805 ± 0.002 |
| Exp 2A | Logistic Regression | 0.721 ± 0.002 | 0.704 ± 0.003 | 0.775 ± 0.002 |
| Exp 2A | Random Forest | 0.757 ± 0.001 | 0.738 ± 0.002 | 0.827 ± 0.002 |
| Exp 2B (26 features) | KNN | 0.644 ± 0.003 | 0.621 ± 0.003 | 0.689 ± 0.003 |
| Exp 2B | Logistic Regression | 0.666 ± 0.003 | 0.608 ± 0.005 | 0.710 ± 0.002 |
| Exp 2B | Random Forest | 0.673 ± 0.003 | 0.627 ± 0.004 | 0.733 ± 0.004 |

**Δ Hard:** +5.5 to +9.4pp. Context features help classification
substantially across all models. KNN shows the largest delta (+9.4pp),
followed by RF (+8.4pp) and LogReg (+5.5pp). This is the opposite of
the retrieval finding (where context hurts), confirming the "either
outcome is interesting" scenario from design doc Section 8.4.

**Classifier vs. retrieval interpretation:** In the univariate retrieval
setting, clue context degrades the definition embedding's proximity to
the true answer (misdirection). But in the multivariate classifier,
context-informed features interact with relationship and surface features
to create learnable patterns — the way real pairs' embeddings shift in
context differs from how distractor pairs shift, and the classifier
exploits this.

**Group-level ablation (RF, fold 0):**

| Group Removed | Features Removed | Accuracy | Δ |
|---|---|---|---|
| None (baseline) | — | 0.740 | — |
| Context-Informed | 6 | 0.632 | −10.8pp |
| Relationship | 22 | 0.653 | −8.7pp |
| Surface | 4 | 0.734 | −0.6pp |

Context-informed features carry the most predictive weight per feature.
Relationship features are nearly as important in aggregate. Surface
features contribute minimally.

**Failure analysis (RF, fold 0, 1,040 misclassified out of 4,000 test;
sample mode):**
- Semantic near-miss: 59.9% of errors — distractors genuinely close to definition
- Surface artifact: 49.6% — model misled by string-level coincidences
- Polysemy confusion: 23.8% — highly polysemous definitions cause embedding noise
- 84.7% categorized, 15.3% uncategorized, heavy overlap between categories
- False negatives (real predicted as distractor) outnumber false positives 662:378

**Runtime:** Easy experiments (Exp 1A + 1B) took ~6.5 hours on 16 CPU
cores. Harder experiments (Exp 2A + 2B) took ~4 hours on 36 CPU cores.
Both ran on Great Lakes standard partition.

### Steps 9–12: Results, Ablation, Sensitivity, Failure Analysis
*Completed.* Notebook: `notebooks/08_results_and_evaluation.ipynb`

See Steps 6 & 8 above for results summary. Additional findings:

**Sensitivity (learning curve):** Test accuracy is flat across training
set sizes (10%–100% of fold 0 train data), indicating the performance
ceiling is driven by feature limitations rather than data quantity.
Train accuracy is ~100% throughout, indicating overfitting typical of
unrestricted-depth Random Forest.

**Best hyperparameters (RF, full data):** n_estimators=200,
min_samples_split=2, min_samples_leaf=1, max_features='log2',
max_depth=None. Changed from sample run (n_estimators=100), confirming
the value of full-data hyperparameter tuning.
