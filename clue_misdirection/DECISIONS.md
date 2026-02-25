# Team Decisions — Clue Misdirection

These decisions are **locked in** and should not be revisited or second-guessed.
Decisions 1–6 were made collaboratively by the team and/or in consultation with
Dr. Collins-Thompson. See `supervised_learning_plan_v3.docx` Section 15 for
the original decision log. Decisions 7–12 were made during pipeline planning.

---

## Decision 1: Embedding Model

**Choice:** `oskar-h/cale-modernbert-base` (CALE = Concept-Aligned Embeddings)

**Backup:** `BAAI/bge-base-en-v1.5`

**Rationale:** CALE is specifically designed to handle multiple senses of a
word, which is at the heart of our research question about semantic
misdirection. Cryptic crossword clues exploit polysemy — the model should be
sensitive to sense distinctions.

**Note:** Hans's earlier work used `all-mpnet-base-v2` (768-dim). The CALE
model also produces 768-dim embeddings (ModernBERT-base). All embedding
generation code must be updated to use the CALE model. The indicator_clustering
component uses `BAAI/bge-m3` (1024-dim) for a different purpose.

---

## Decision 2: Average Embedding Method

**Choice:** Use the single-word embedding from the model (no context sentence)
for `Word1_average` and `Word2_average`.

**Rationale:** This conflates senses in a way that depends on the model's
training data, which is fine — it represents the "uncontrolled" baseline.
The WordNet-based sense-specific embeddings (common/obscure) give us
interpretable control over which senses contribute. Comparing both is a
secondary analysis if time permits.

---

## Decision 3: Missing Values / NaN

**Choice:** Engineer all features to never produce NaN. No imputation needed.

**Implementation:**
- Cosine similarities: guaranteed non-NaN as long as embeddings are non-zero
  (ensured by the WordNet synset filter in Step 1)
- WordNet sense embeddings: fall back to single-word embedding if a synset
  lacks a definition, usage example, or representative synonym
- Relationship features: pairs with no 2-hop WordNet connection → all 19
  booleans False, max_path_similarity = 0.0, shared_synset_count = 0
- **Validate before training:** `assert not df.isnull().any().any()`

---

## Decision 4: Wordplay Type

**Choice:** Do NOT include wordplay type as a model feature.

**Rationale:** Not all clues have labels, so including it would require
discarding rows or imputing a categorical variable. More importantly, wordplay
type is a property of the clue construction, not the definition–answer semantic
relationship we're studying. It could be interesting for post-hoc analysis
(e.g., "does misdirection vary by wordplay type?") and is noted as a future
direction in the report.

---

## Decision 5: Reporting Unit

**Choice:**
- **Retrieval analysis (primary):** Report over unique (definition, answer)
  pairs. For context-informed conditions with multiple clues per pair, take
  the median rank across clues for that pair, then compute summary stats
  over unique pairs. This keeps N consistent across all conditions.
- **Retrieval analysis (supplementary):** Also report over all
  (clue, definition, answer) rows using the Average condition only, to
  discuss what is representative of CCC puzzles in general and whether
  frequently-reused pairs show different misdirection patterns. Note in the
  discussion that this view may inflate the misdirection measure.
- **Classifier:** Use all clue rows (not deduplicated) because different
  clues provide different context features (Word1_clue_context, Sentence1).

**Rationale:** Using all rows for the primary retrieval analysis would inflate
context-free results (identical embeddings → identical ranks for duplicates)
while not inflating context-informed results, making the misdirection gap
look artificially large. The supplementary all-rows analysis is still
valuable for understanding puzzle-level patterns.

---

## Decision 6: Harder Distractor Strategy

**Choice:** Select distractors by cosine similarity between `Word1_average`
(definition, no context) and `Word2_average` (answer, no context) — top-k
most similar answer words.

**Consequence:** The 15 context-free meaning features are artifacts of dataset
construction and **must be removed** from the harder dataset models. This
leaves 38 features (Exp 2A) or 25 features (Exp 2B).

**Rationale:** Random distractors make the task trivially easy (high accuracy
but uninformative). Cosine-similarity-based distractors force the model to
rely on subtler features, making the experiment genuinely informative about
whether clue context helps or hurts classification.

**Alternative not taken (future direction):** Construct distractors by matching
the WordNet relationship distribution of real pairs, which would let us retain
all 28 cosine features but remove the 21 relationship features instead.

---

## Decision 7: Cross-Validation Design

**Choice:** Stratified 5-fold GroupKFold, grouped by definition–answer pair
(or by definition word for stricter leakage prevention). Same fold assignments
across all experiments.

**Rationale:** Multiple clue rows can share the same (definition, answer) pair.
If these end up split across train and test folds, the model sees near-identical
feature vectors in training, which leaks information. GroupKFold prevents this.

---

## Decision 8: Classifier Role

**Choice:** The retrieval analysis (Step 4) is the **primary** evidence for
misdirection. The classifier (Steps 6–8) satisfies the course requirement and
provides a **complementary** multivariate view. The classifier results may or
may not replicate the retrieval finding, and either outcome is interesting.

**Rationale:** The retrieval analysis directly measures how clue context
degrades the ability to find the true answer — this is the most interpretable
signal. The classifier adds a controlled experiment with ablation, but its
results depend on how much the relationship and surface features absorb the
signal. We should not oversell the classifier as the primary evidence.

---

## Decision 9: Data Source and Filtering Scope

**Choice:** Start from `../data/clues_raw.csv` (extracted from the sqlite DB
by the indicator_clustering NB00). Do not use Hans's `clues_single_word.csv`.
Do not restrict to single-word definitions or answers — keep any row where
both the definition and answer have ≥1 synset in WordNet, even if multi-word.
For double-definition clues (where `definition` contains `/`-separated
alternatives), keep any definition that appears in WordNet.

**Rationale:** Hans's `clues_single_word.csv` applied an overly restrictive
filter that discarded multi-word entries and double-definition clues. WordNet
contains synsets for some multi-word entries, and double-definition clues are
a meaningful ~5% of the dataset. Broadening the filter preserves more data
and makes the analysis more representative of cryptic crossword puzzles as
a whole.

---

## Decision 10: Embed Surface Text, Not Raw Clue

**Choice:** When creating embeddings that require clue context
(Word1_clue_context and Sentence1), embed the `surface` text — the clue
with the trailing answer format stripped (e.g., "Plant in a garden party",
not "Plant in a garden party (5)").

**Rationale:** The answer format in parentheses (e.g., "(5)" or "(3,4)")
is metadata about the answer's length, not part of the clue's natural
language surface reading. Including it would add noise to the embedding
and is not representative of how a solver reads the clue.

---

## Decision 11: Embedding Deduplication

**Choice:** Deduplicate before embedding. Compute definition embeddings
(average, common, obscure) once per unique definition string, answer
embeddings once per unique answer string, and only compute clue-context
embeddings (Word1_clue_context, Sentence1) per row. Store with index
files that map unique strings to embedding array positions.

**Rationale:** Puzzle creators reuse (definition, answer) pairs across
different clue sentences. Computing redundant embeddings wastes GPU time.
The deduplication pattern follows the approach used in the
indicator_clustering `02_embedding_generation.ipynb`. Downstream steps
use the index files to look up the correct embedding for each row.

---

## Decision 12: No Pre-Embedding Sampling

**Choice:** Embed the entire filtered dataset. Do not sample before
creating embeddings. Take subsets downstream as needed for specific analyses.

**Rationale:** The full filtered dataset (upper bound ~100K rows) produces
embeddings under 1 GB total and takes under 20 minutes on a T4 GPU. This
is well within computational and storage constraints. Embedding everything
preserves the option for any downstream analysis without re-running the GPU
step, avoids the need to justify a sampling decision, and eliminates the
risk that a subsample misses important patterns.
