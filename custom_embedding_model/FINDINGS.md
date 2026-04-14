# Findings — custom_embedding_model

Running log of coverage measurements, compute runtimes, ATE results, and
experimental findings as the project progresses. Add entries as work
completes — do not wait until the end to record results.

For locked-in decisions, see DECISIONS.md. For the workflow, see WORKFLOW.md.

---

## Prior Work: Initial g_1 Training and Evaluation (NB 09)

These findings come from `notebooks/archive/09_learned_g_misdirection.ipynb`
(Nathan Cantwell), which represents the first attempt at fine-tuning a custom
embedding model. This work was done prior to the `custom_embedding_model`
directory and used a different dataset and split than what we are building
here. It is recorded as prior work, not as a result of the current pipeline.

### Data and setup

- Input dataset: `clue_misdirection/data/dataset_harder.parquet` (harder
  binary classification dataset from Milestone II)
- Split: 80/20 train/test at the (definition, answer) pair level
- Training set: ~190,943 triplet rows (sample mode: 37,836 rows, 20,000
  unique pairs)
- Test set: ~48,172 rows (sample mode: 20,000 rows)
- Environment: Great Lakes (GPU, CUDA)

### Triplet design (T_1)

| Component | Description | Construction |
|-----------|-------------|--------------|
| Anchor | f_clue(definition) | Definition tagged in clue surface with `<t></t>` |
| Positive | f_common_wndef(answer) | `"<t>answer</t>: <WordNet definition>"` |
| Negative | f_common_wndef(distractor) | Same format for distractor word |

Base model: `gabrielloiseau/CALE-MBERT-en` (g_stock)
Margin: α = 1.0. Mixed precision, gradient accumulation.

### ATE results (sample mode, 20K test rows)

The ATE is defined as:
mean(cos_sim(g(f_clue(def)), g(f(ans))) − cos_sim(g(f(def)), g(f(ans))))

A negative ATE indicates misdirection. The hypothesis was that g_1 would show
a *less negative* ATE than g_stock. The opposite occurred.

| Metric | g_stock | g_1 |
|--------|---------|-----|
| Decontextualized similarity T=0 (mean) | 0.548 | 0.758 |
| Contextualized similarity T=1 (mean) | 0.476 | 0.476 |
| Delta = T=1 − T=0 (ATE, mean) | −0.072 | −0.282 |
| ATE 95% CI | [−0.074, −0.071] | [−0.283, −0.281] |
| Delta median | −0.065 | −0.281 |
| % pairs with negative delta | 76.9% | 100.0% |

**Key observation:** The T=1 (clue-context) similarity stayed almost identical
between g_stock and g_1 (0.476 vs. 0.476). The T=0 (decontextualized)
similarity jumped dramatically (0.548 → 0.758). g_1 pulled decontextualized
f_common_wndef phrases much closer together in embedding space, making the
ATE more negative rather than less.

### Retrieval-based ATE (sample mode)

| Metric | g_stock | g_1 |
|--------|---------|-----|
| Median rank T=0 (decontextualized) | 618 | 821 |
| Median rank T=1 (clue-context) | 936 | 1240 |
| Rank delta (T=1 − T=0) | +318 | +419 |
| % pairs worsened by context | 55.6% | 53.6% |

### Interpretation

g_1 did not learn to counteract misdirection. The evidence suggests it learned
to compress f_common_wndef phrases ("word: WordNet definition" format) closer
together in embedding space — a form of format-specific overfitting. Two
contributing factors identified:

1. **Unnatural phrase format:** f_common_wndef uses a dictionary-entry format
   (`"<t>word</t>: definition"`) that CALE was not trained on. The model may
   have keyed on this format signature rather than semantic content.

2. **Positive target:** The positive was f_common_wndef(answer), not
   f_common_wndef(definition). This pulls the clue-contextualized definition
   embedding toward the answer's decontextualized embedding — a conceptually
   indirect objective that may not be well-suited to counteracting misdirection.

### Planned investigation

Step A: Reproduce g_1 exactly using the new pipeline.
Step B: Test the formatting hypothesis — compute g_1(f_common_wnex(word)) for
validation-set words and compare to g_stock(f_common_wnex(word)). If g_1
also compresses wnex phrases (even though it was not trained on them), that
suggests semantic generalization. If not, that confirms format-specific
overfitting.

---

## Stage 0: Structural Filtering

*Not yet run.*

---

## Stage 1: WordNet Filtering and Split Assignment

**Notebook:** `01_wn_filtering_and_split.ipynb` — completed 2026-04-13
**Environment:** Local, `crossword` kernel (NLTK 3.9.2, scikit-learn 1.8.0)

### Coverage measurements

| Metric | Value |
|--------|-------|
| clues_filtered.csv rows | 457,262 |
| clues_wn_filtered.csv rows | 239,406 |
| Fraction retained after WN filter | 52.4% |
| Unique (definition, answer) pairs | 150,805 |
| Train pairs (target ~30%) | 45,241 (30.0%) |
| Validate pairs (target ~20%) | 30,161 (20.0%) |
| Test pairs (target ~50%) | 75,403 (50.0%) |
| Train rows | 72,107 (30.1%) |
| Validate rows | 47,933 (20.0%) |
| Test rows | 119,366 (49.9%) |
| vocabulary.csv words | 53,930 |
| vocabulary_val.csv words | 26,152 |
| Val vocab as fraction of full vocab | 48.5% |

### Article-stripping recovery

Expanded from Milestone II's `"a "` only to `"a "`, `"an "`, `"the "`,
`"to "` (Decision 16). Recovery counts for unique definitions:

| Prefix | Unique defs recovered | Unique answers recovered |
|--------|----------------------|-------------------------|
| `"a"` | 1,579 | 43 |
| `"an"` | 257 | 1 |
| `"the"` | 615 | 231 |
| `"to"` | 737 | 13 |

The three new prefixes recovered 1,609 additional unique definitions beyond
what `"a "` alone would have captured.

### Vocabulary overlap

| Category | Count |
|----------|-------|
| Words appearing as both definition and answer | 18,200 |
| Words appearing only as definition | 8,965 |
| Words appearing only as answer | 26,765 |

### Version discrepancy finding

Running NB 01 under the default `python3` kernel (NLTK 3.8.1) vs. the
`crossword` kernel (NLTK 3.9.2) produced a 49-row difference (239,455 vs.
239,406). Root cause: different NLTK versions resolve slightly different
sets of words to WordNet synsets. Led to Decision 17 (pin NLTK and
scikit-learn) and Decision 18 (version provenance in notebooks).

---

## Stage 2: Phrase Construction

**Notebook:** `02_phrase_construction_wn.ipynb` — completed 2026-04-13
**Environment:** Local, `crossword` kernel (NLTK 3.9.2, pandas 3.0.0, numpy 2.3.5)

### f_clue coverage

| Metric | Value |
|--------|-------|
| clues_wn_filtered.csv rows (input) | 239,406 |
| Rows with valid f_clue phrase | 239,406 (100.0%) |
| Rows dropped | 0 (0.0%) |

`tag_definition_in_surface` succeeded on every row — no ambiguous definition
placements in the wn_synset-filtered dataset.

### f_common_wndef coverage

| Metric | Value |
|--------|-------|
| vocabulary.csv words (input) | 53,930 |
| Words with valid f_common_wndef phrase | 53,930 (100.0%) |
| Self-referential phrases | 1,139 (2.1%) |
| clues_wndef_filtered.csv rows | 239,406 (100.0% of clues_wn_filtered) |
| vocabulary_wndef_val.csv words | 26,152 |
| Train fraction | 72,107 (30.1%) |
| Validate fraction | 47,933 (20.0%) |
| Test fraction | 119,366 (49.9%) |

100% coverage is expected: every word in vocabulary.csv has at least one synset
(the NB 01 filter), and synset[0] always has a definition. wndef vocabulary
equals the full vocabulary; wndef-filtered clues equal the full clue set.

1,139 words (2.1%) have self-referential definitions — the word appears
untagged in its own WordNet definition text (e.g., "admit: declare to be true
or admit the existence…"). These are flagged with `self_ref=True` in
`f_common_wndef.csv` for downstream subsetting but are not filtered out.

### f_common_wnex coverage

| Metric | Value |
|--------|-------|
| vocabulary.csv words (input) | 53,930 |
| Words with valid f_common_wnex phrase | 8,360 (15.5%) |
| clues_wnex_filtered.csv rows | 24,327 (10.2% of clues_wn_filtered) |
| vocabulary_wnex_val.csv words | 3,008 |
| Train fraction | 7,075 (29.1%) |
| Validate fraction | 4,825 (19.8%) |
| Test fraction | 12,427 (51.1%) |

Low coverage is expected: many WordNet synsets lack usage examples, and among
those with examples, the target word must appear exactly once for unambiguous
tagging. wnex is a strict subset of wndef (8,360 words in both, 0 in wnex
only, 45,570 in wndef only).

### Cross-f comparison

| Metric | Value |
|--------|-------|
| Words in wndef but not wnex | 45,570 |
| Words in wnex but not wndef | 0 |
| Words in both | 8,360 |
| Rows in clues_wndef_filtered but not clues_wnex_filtered | 215,079 |

---

## Stage 1d: g_stock f_clue Embedding Generation

**Script:** `scripts/embed_f_clue_gstock.py` — completed 2026-04-13
**Environment:** Great Lakes, `nlp_env` conda environment

| Metric | Value |
|--------|-------|
| Total rows embedded | 239,406 |
| Embedding array shape | (239406, 1024) |
| Embedding dtype | float32 |
| L2 norm range | [24.4009, 34.1953] |
| NaN values | 0 |
| All-zero rows | 0 |
| Encoding rate | 675 phrases/sec (batch_size=64) |
| Encoding time | 354.7s (5 min 51 sec) |
| Great Lakes partition | gpu (gl1002) |
| Date run | 2026-04-13 |

### Output files

| File | Size | Location |
|------|------|----------|
| `f_clue.npy` | 936 MB | `data/embeddings/g_stock/` |
| `f_clue_index.csv` | 4.9 MB | `data/embeddings/g_stock/` |

### Environment versions

| Package | Version |
|---------|---------|
| Python | 3.12.12 |
| torch | 2.5.1+cu121 |
| sentence-transformers | 5.2.2 |
| transformers | 4.52.3 |
| numpy | 2.3.5 |
| conda environment | nlp_env |

### Notes

- Used `SentenceTransformer.encode()` (not manual AutoModel extraction).
- `normalize_embeddings=False` — raw magnitudes preserved; downstream ATE
  computation normalizes as needed.
- Encountered `np.save()` temp file naming bug: numpy auto-appends `.npy`,
  causing atomic rename to fail. Files were saved successfully but required
  manual rename on Great Lakes. Bug fixed in commit c3653b9.

---

## Stage 3: g_1 Training

*Not yet run.*

| Metric | Value |
|--------|-------|
| Triplet file | data/triplets/g1.csv |
| Training rows | — |
| Base model | gabrielloiseau/CALE-MBERT-en |
| Margin α | 1.0 |
| Learning rate | — |
| Epochs | — |
| Batch size | — |
| Final training loss | — |
| Great Lakes partition | — |
| Wall-clock runtime | — |
| Date trained | — |
| Google Drive path | — |

---

## Stage 4: g_1 Embedding Generation (Validation Split)

*Not yet run.*

| Metric | Value |
|--------|-------|
| Embeddings generated | f_common_wndef_val, f_common_wnex_val, f_clue_val |
| vocabulary_wndef_val words | — |
| vocabulary_wnex_val words | — |
| Validation clues embedded | — |
| Great Lakes partition | — |
| Wall-clock runtime | — |
| Date run | — |

---

## Stage 5: Hypothesis Testing

*Not yet run.*

### Step A — g_1 Reproduction (Validation Set ATE)

| Metric | g_stock | g_1 |
|--------|---------|-----|
| T=0 mean similarity | — | — |
| T=1 mean similarity | — | — |
| ATE (mean delta) | — | — |
| ATE 95% CI | — | — |
| % pairs with negative delta | — | — |

### Step B — Formatting Hypothesis Test

**Hypothesis:** g_1 learned to compress f_common_wndef phrases (format-specific
overfitting) rather than learning something semantically meaningful.

**Test:** Compare cos_sim(g_1(f_common_wnex(word)), g_stock(f_common_wnex(word)))
against the g_stock baseline for words in vocabulary_wnex_val.

| Metric | g_stock baseline | g_1 |
|--------|-----------------|-----|
| Mean cos_sim(f_common_wndef, f_common_wnex) for same word | — | — |
| Fraction of wnex vocab tested | — | — |
| wnex vocab words tested | — | — |
| Interpretation | — | — |

---

## Stage 6: Final Evaluation

*Locked — do not populate until final g is chosen and documented in DECISIONS.md.*
