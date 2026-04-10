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

*Not yet run.*

### Coverage measurements (to be filled in)

| Metric | Value |
|--------|-------|
| clues_filtered.csv rows | — |
| clues_wn_filtered.csv rows | — |
| Fraction retained after WN filter | — |
| Unique (definition, answer) pairs | — |
| Train pairs (target ~30%) | — |
| Validate pairs (target ~20%) | — |
| Test pairs (target ~50%) | — |
| vocabulary.csv words | — |
| vocabulary_val.csv words | — |
| Val vocab as fraction of full vocab | — |

---

## Stage 2: Phrase Construction

*Not yet run.*

### f_clue coverage (to be filled in)

| Metric | Value |
|--------|-------|
| clues_wn_filtered.csv rows | — |
| Rows with valid f_clue phrase | — |
| Rows dropped (definition appears 2+ times in surface) | — |

### f_common_wndef coverage (to be filled in)

| Metric | Value |
|--------|-------|
| vocabulary.csv words | — |
| Words with valid f_common_wndef phrase | — |
| clues_wndef_filtered.csv rows | — |
| Fraction of clues_wn_filtered.csv retained | — |
| Train/validate/test fractions (actual) | — |

### f_common_wnex coverage (to be filled in)

| Metric | Value |
|--------|-------|
| vocabulary.csv words | — |
| Words with valid f_common_wnex phrase | — |
| Fraction of full vocabulary with wnex phrase | — |
| clues_wnex_filtered.csv rows | — |
| Fraction of clues_wn_filtered.csv retained | — |
| Train/validate/test fractions (actual) | — |

---

## Stage 1d: g_stock f_clue Embedding Generation

*Not yet run.*

| Metric | Value |
|--------|-------|
| Total rows embedded | — |
| Embedding array shape | — |
| Great Lakes partition | — |
| Wall-clock runtime | — |
| Date run | — |

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
