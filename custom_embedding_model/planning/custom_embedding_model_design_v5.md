# Design Document: custom_embedding_model
## Fine-Tuning CALE for Cryptic Crossword Misdirection

**Authors:** Victoria Winters, Nathan Cantwell
**Advisor:** Dr. Kevyn Collins-Thompson (University of Michigan)
**Date:** April 2026
**Version:** 5
**Status:** Active — data preparation and initial experiments in progress

---

## 1. Background and Motivation

This document describes the current phase of our research into semantic
misdirection in cryptic crossword clues, building on prior work completed
in a Milestone II project (SIADS 696, Winter 2026). That work investigated
how much the surface reading of a cryptic crossword clue misleads
embedding-based models trying to connect the definition to the answer.
The key findings were:

- **Retrieval analysis:** Embedding a definition word *in the context of
  its cryptic clue* (rather than out of context) roughly doubles its median
  retrieval rank among ~45,000 candidate answer words. Clue context hurts
  retrieval in about 70% of cases. Retrieval was best using an embedding that
  averaged across *all* WordNet senses of the word — suggesting that polysemy
  is central to how misdirection works.

- **Binary classification:** Building a classifier to distinguish true
  definition–answer pairs from distractor pairs revealed that distractor
  construction method greatly affects model performance. In the harder dataset,
  distractors were slightly closer in embedding space to the definition than
  the true answers — a consequence of our construction method — and the model
  learned and exploited this characteristic. This highlighted the importance
  of distractor design and the difficulty of isolating a clean misdirection
  signal in a supervised setting.

These analyses are complete. The current phase takes a different approach:
rather than measuring misdirection with a fixed off-the-shelf model, we
investigate whether we can fine-tune our embedding model (CALE) using triplet
margin loss so that it becomes more adept at embedding a clue-contextualized
definition as a sense more aligned with the clue's answer.

The longer-term goal — using the resulting custom embeddings to build a model
that measures misdirection more directly — is out of scope for this directory
and will live in a separate component. This project focuses entirely on the
fine-tuning step.

---

## 2. The Two-Step Chain: f and g

Understanding this project requires a clear picture of how we get from a word
to an embedding. There are two distinct steps.

**Step 1 — Phrase construction: f**

A phrase construction strategy (**f**) takes a word and produces a passage of
text with the target word tagged using CALE's `<t></t>` delimiter mechanism:

> f_common_wndef("fleet") → `"<t>fleet</t>: a group of ships"`

The choice of f is not a neutral preprocessing step. Different f's encode
different assumptions about which sense of a word to capture and how naturally
to express it. The interpretation of what any triplet teaches the model is
always tied to the f's chosen — f must be designed with the desired triplet
interpretation in mind.

**Step 2 — Embedding: g**

An embedding model (**g**) takes a tagged passage and produces a 1024-dimensional
vector. The full chain is:

> word → f(word) → g(f(word)) → 1024-dim embedding

The unmodified CALE model is called **g_stock**. Fine-tuned variants are
**g_1**, **g_2**, etc.

**f_clue: the clue-contextualized case**

**f_clue** is a special phrase construction strategy that operates on clues
rather than vocabulary words. It takes a clue's surface text and wraps the
definition word(s) in `<t></t>` delimiters:

> f_clue("Parties broken for sea-faring group", "sea-faring group")
> → `"Parties broken for <t>sea-faring group</t>"`

The embedding g(f_clue(definition)) reflects how the clue's misleading surface
reading colors the meaning of the definition. This is the "treatment" condition
in our misdirection analysis.

---

## 3. Why CALE?

Our embedding model is `gabrielloiseau/CALE-MBERT-en` (CALE = Concept-Aligned
Embeddings, ModernBERT-based, English, 1024 dimensions). CALE is specifically
designed to be sensitive to word sense: its `<t></t>` delimiter pair focuses
the embedding on a target word within a passage, rather than producing a
general sentence-level embedding. This behavior was validated in the Milestone
II work — CALE produces genuinely distinct embeddings for a target word in
context versus the full sentence (cosine similarity ≈ 0.66), while standard
sentence-transformer models produce nearly identical embeddings for the same
comparison (cosine similarity ≈ 0.90–0.93).

Because CALE was trained on naturalistic text, it already places semantically
similar words nearby in its embedding space. Fine-tuning with triplet margin
loss adjusts that space for our specific task.

A known limitation of CALE is that bare-word embeddings (without context) do
not discriminate well between related and unrelated words. This is why we
always construct a phrase for every word we embed, rather than embedding words
directly.

---

## 4. Data Pipeline

### 4.1 Source

George Ho's cryptic crossword clue dataset (660,613 clues), available at
https://cryptics.georgeho.org/data.db. The raw data was previously extracted
to `clues_raw.csv` and lives in the shared `ccc-project/data/` directory.
**This directory must not be modified.**

### 4.2 Shared Upstream Pipeline (ccc-project/notebooks/)

Two notebooks and one utility module run at the `ccc-project/` level and
produce shared artifacts consumed by this component and any future components.
They are documented in `ccc-project/WORKFLOW.md`.

**`puzzle_metadata.ipynb`** extracts structured metadata for every puzzle by
parsing `puzzle_name`, `source_url`, and `clue_number`. Its output —
`ccc-project/data/puzzle_metadata.csv` — is a standalone file keyed on
`clue_id`, containing `publisher`, `series`, `setter`, `puzzle_no`,
`puzzle_date`, `clue_no`, and `clue_direction`. It is independent of all
filtering and can be joined into any downstream dataset when needed. See
`DATA_RAW.md` §4 for the full extraction logic.

**`structural_filtering.ipynb`** filters `clues_raw.csv` to clues satisfying
CCC structural constraints. Loads only `clue_id`, `clue`, `answer`, and
`definition` from the raw file.

Filtering steps:
1. Remove rows where `definition` or `answer` is null
2. Validate that `answer` matches the length/format code in `clue`
3. Parse double-definition clues: split `definition` on `/`; retain only
   candidate definitions that appear as intact whole words in the surface
   (accepting `<word>'s` as a valid match for `<word>`); keep the clue only
   if at least one valid definition appears at the start or end of the surface;
   expand to one row per valid definition
4. **Bracket diagnostic** (not a filter): after all other steps, count and
   display surviving rows containing `[`; document the decision in `DECISIONS.md`

Output columns: `clue_id`, `surface`, `definition`, `answer`.

Separating structural filtering into a shared notebook means any future
component can branch from `clues_filtered.csv` without rerunning these filters.

**`clue_utils.py`** is a shared Python module containing the
definition-finding and delimiter-placement logic used by both
`structural_filtering.ipynb` and `02_phrase_construction.ipynb`. It ensures
that the matching logic applied during filtering and the logic applied during
f_clue phrase construction are identical — critical for correctness when
apostrophe-s handling is involved. Key functions:

- `find_definition_in_surface(definition, surface)` — word-boundary match,
  accepts `<word>'s` for `<word>`, returns span or `None`
- `tag_definition_in_surface(definition, surface)` — wraps the matched span
  in `<t></t>`, preserving original capitalization and punctuation, returns
  tagged string or `None`

This module is a committed, versioned artifact. Changes to it may invalidate
existing `f_clue.csv` files and must be documented in `DECISIONS.md`.

### 4.3 Component-Specific Filtering (custom_embedding_model/notebooks/)

**`01_wn_filtering_and_split.ipynb`** reads `clues_filtered.csv` and applies
WordNet coverage as a further constraint: both the definition and answer must
have at least one WordNet synset. This produces the baseline dataset for all
WordNet-based phrase construction strategies, stored under
`data/filtered_split/wn_synset/`.

The directory name reflects the nature of the constraint: `filtered_split`
indicates this is a subset of `clues_filtered.csv` with a train/validate/test
split assigned; `wn_synset` names the specific constraint (both definition and
answer have at least one WordNet synset). Future constraints would produce
sibling directories (e.g. `filtered_split/dict_example/`) if needed.

Output: `data/filtered_split/wn_synset/clues_wn_filtered.csv`

### 4.4 Columns

`clues_filtered.csv` output schema (fixed column order):

| Column | Description |
|--------|-------------|
| `clue_id` | From George Ho sqlite; not unique after multi-definition expansion |
| `surface` | Clue text with answer format stripped; used for f_clue phrase construction |
| `definition` | Single valid definition substring (one row per valid definition) |
| `answer` | Answer string, uppercase |

Puzzle provenance metadata (`publisher`, `series`, `setter`, `puzzle_no`,
`clue_no`, `clue_direction`) lives in `puzzle_metadata.csv` and is joined
in downstream when needed.

Intermediate columns (`clue`, `answer_format`, `surface_normalized`,
`num_definitions`, `source`) are not output. They serve their purpose during
filtering and are not needed downstream. Normalization for definition-in-surface
matching is handled internally by `clue_utils.py`, which operates on `surface`
directly and never requires a separately stored normalized form.

### 4.5 Train / Validate / Test Split

We apply a 30 / 20 / 50 train / validate / test split before any phrase
construction or embedding. The test set must not be touched until we have
settled on a final model.

The split is performed at the level of unique (definition, answer) pairs —
multiple clues can share the same pair, and all rows sharing a pair must land
in the same split. The split is stored as a column in `clues_wn_filtered.csv`,
assigned once using `random_state=42`, and treated as a fixed committed
artifact.

The ratio reflects the experimental nature of this work: 50% test ensures
robust final evaluation; 30% train is sufficient for triplet training; 20%
validate supports iterative development.

### 4.6 Unified Vocabulary

Because definitions and answers are frequently reused across many clues, we
build vocabularies of unique words rather than working row-by-row. All
vocabulary files use a fixed canonical row ordering established at creation
and never changed — the row number of a word in a vocabulary file is its row
index into any corresponding embedding array.

Vocabulary files live in their respective subset directories:

| File | Location | Contents |
|------|----------|----------|
| `vocabulary.csv` | `wn_synset/` | All words with ≥1 WordNet synset |
| `vocabulary_val.csv` | `wn_synset/` | Validation-split words from vocabulary.csv |
| `vocabulary_wndef.csv` | `wn_synset/wndef/` | Words with a valid WN definition phrase |
| `vocabulary_wndef_val.csv` | `wn_synset/wndef/` | Validation-split subset |
| `vocabulary_wnex.csv` | `wn_synset/wnex/` | Words with a valid WN usage example |
| `vocabulary_wnex_val.csv` | `wn_synset/wnex/` | Validation-split subset |

### 4.7 Coverage Measurement as a First-Class Activity

WordNet coverage is imperfect and uneven. As we apply progressively stricter
constraints — requiring usage example availability, requiring multiple synsets,
requiring both definition and answer to satisfy a constraint simultaneously —
the dataset shrinks. We do not know in advance how much.

Coverage measurement is therefore a first-class research activity, not a
preprocessing footnote. At every stage where a constraint reduces the dataset,
we record:

- How many unique words and clue rows remain, and what fraction of the
  previous stage that represents
- Whether the remaining data is large enough to be meaningful for training

These numbers are recorded in the notebook summary cell and in `FINDINGS.md`.
They directly inform decisions about whether to proceed with a given phrase
construction strategy or to seek richer external resources.

---

## 5. Phrase Construction Strategies: the f Functions

Each f takes a word from the vocabulary and produces a tagged passage suitable
for CALE embedding, representing a specific chosen sense of the word.

**Strict f definitions — no fallbacks.** An f is defined only for words where
the required phrase can be constructed without any fallback. If f_common_wnex
requires a WordNet usage example and none exists for a word, that word is
absent from `vocabulary_wnex.csv` and from any experiment requiring
f_common_wnex. This is essential for clean interpretation: when we compare
results across f's, each f must mean exactly one thing.

All phrases are committed data artifacts — generated once, saved, and reused.

### 5.1 Naming Convention

`f_<sense>_<construction>`, where:
- **sense** — which meaning of the word the phrase captures
- **construction** — how the passage is built

### 5.2 Planned f Functions

**f_clue** — *Clue-contextualized definition*

The definition word tagged within the clue's surface text. Indexed by
(clue_id, definition) rather than by word. Generated for the full wn_synset
clue scope — it does not further filter rows. Lives in `clue_phrases/` rather
than a subset directory, reflecting that it belongs to the full `wn_synset`
scope rather than any stricter constraint.

**Output:** `data/filtered_split/wn_synset/clue_phrases/f_clue.csv`

**f_common_wndef** — *Most common WordNet sense, "word: definition" format*

For each vocabulary word, look up its most frequent WordNet sense (index 0)
and construct `"<t>word</t>: <WordNet definition text>"`. Defined for every
word with at least one synset. Lives in `wndef/` alongside the vocabulary
files and filtered clue file for this constraint.

**Output:** `data/filtered_split/wn_synset/wndef/f_common_wndef.csv`

This was the format used in g_1 (see §6), with known limitations discussed
there.

**f_common_wnex** — *Most common WordNet sense, WordNet usage example*

For each vocabulary word, look up its most frequent WordNet sense and use the
WordNet usage example as the passage, with the target word wrapped in
`<t></t>`. Defined only for words with a valid usage example where the target
word appears exactly once. Lives in `wnex/` alongside its vocabulary files and
filtered clue file.

**Output:** `data/filtered_split/wn_synset/wnex/f_common_wnex.csv`

Coverage is expected to be substantially lower than f_common_wndef.

**f_3 and beyond (TBD)** — if WordNet coverage is insufficient, we will
investigate richer resources in order of increasing complexity: modern
dictionary resources (more expansive than WordNet, but without its structured
semantic network), then LLM-generated phrases (maximum flexibility, but
requiring careful prompt design and reproducibility considerations). Each new
strategy would live in its own subdirectory under `filtered_split/wn_synset/`
(or a sibling scope directory if the filtering constraint also changes). Any
phrases generated will be committed artifacts.

---

## 6. Triplet Margin Loss and Fine-Tuning

### 6.1 What is Triplet Margin Loss?

Triplet margin loss (Schroff et al., CVPR 2015) trains an embedding model by
showing it three examples at a time: an **Anchor**, a **Positive** (which
should be close to the Anchor in embedding space), and a **Negative** (which
should be far from the Anchor). The loss function is:

> Loss = max(0, ‖z_anchor − z_positive‖ − ‖z_anchor − z_negative‖ + α)

We use α = 1.0. The sentence-transformers library handles phrase-to-embedding
conversion internally during training — no pre-computed embedding artifacts
are needed or produced for the training set. Training produces one artifact:
the updated model weights.

### 6.2 Three Distinct Computational Activities

**Training** — fine-tuning g_stock on triplets to produce g_i. Runs on GPU
(Great Lakes SLURM). Produces model weights saved to Google Drive.

**Embedding generation** — applying a trained g_i to phrase artifacts to
produce `.npy` embedding arrays. Also GPU-intensive, also a SLURM script.
Produces embedding artifacts scp'd back locally.

**Hypothesis testing** — loading saved embedding artifacts and computing
cosine similarities, ATEs, and other statistics. Lightweight, CPU, runs
locally in a notebook. This is where scientific questions get answered.

The distinction between training loss monitoring and hypothesis testing matters:
training loss tells you the model is adjusting to the training data, but says
nothing interpretable about *what* it learned. Hypothesis testing — using
validation-set embeddings to compute specific statistics — reveals whether the
model learned what we intended, or something else.

### 6.3 The Initial Triplet Design (T_1) and What We Learned

The initial fine-tuning work producing **g_1** used the following triplet
design, trained from g_stock on `dataset_harder.parquet`:

| Component | Description | Example |
|-----------|-------------|---------|
| **Anchor** | f_clue(definition) | `"Parties broken for <t>sea-faring group</t>"` |
| **Positive** | f_common_wndef(answer) | `"<t>fleet</t>: a group of ships"` |
| **Negative** | f_common_wndef(distractor) | `"<t>crew</t>: the people on a ship"` |

**What we expected:** g_1 should show a *less negative* ATE than g_stock —
meaning clue context should hurt definition-to-answer similarity less.

**What actually happened:** The ATE became *more* negative under g_1:

|  | g_stock | g_1 |
|--|---------|-----|
| Decontextualized similarity T=0 (mean) | 0.548 | 0.758 |
| Contextualized similarity T=1 (mean) | 0.476 | 0.476 |
| ATE (mean delta T=1 − T=0) | −0.072 | −0.282 |
| % pairs with negative delta | 76.9% | 100.0% |

The clue-contextualized similarity (T=1) stayed nearly identical. The
decontextualized similarity (T=0) jumped dramatically. g_1 pulled
f_common_wndef phrases together in embedding space — the opposite of the
intended effect.

### 6.4 Understanding What Happened

Reflecting on T_1 reveals implicit assumptions in the original triplet design.
The real triplet was:

> T_1 = (g_stock(f_clue(def)), g_stock(f_common_wndef(ans)), g_stock(f_common_wndef(dstr)))

Three issues stand out:

**Unnatural phrase format.** `"<t>word</t>: <WordNet definition>"` is not
naturalistic text. CALE was trained on natural sentences; it may have responded
to this format as a surface pattern and compressed all f_common_wndef phrases
together — which the T=0 jump (0.548 → 0.758) suggests.

**Sense commitment.** f_common_wndef always uses the most common WordNet sense.
If the most common sense of the answer is not the sense relevant to the clue,
the Positive is pointing the model in a potentially misleading direction.

**Positive and Negative design.** In FaceNet, the Positive was a real image
of the same person as the Anchor — semantically clear. In T_1, the Positive
is the answer word, not the definition word. A more principled design might
pull the clue-contextualized definition toward a decontextualized embedding of
the *same word* (the definition), using the answer only to select the relevant
sense. Distractors as Negatives also have a somewhat opaque interpretation when
selected by cosine similarity to the answer embedding.

### 6.5 Planned Investigations

**Step A — Reproduce g_1**

Reproduce the T_1 training procedure exactly as implemented in NB 09 (same
hyperparameters, data, triplet structure) to produce a reproducible g_1 as
a baseline. All future comparisons depend on this. Track GPU runtime.

**Step B — Test the formatting hypothesis**

*Hypothesis:* g_1 learned to compress f_common_wndef phrases together rather
than learning anything semantically meaningful.

*Primary test:* Compute the ATE on the validation set under g_1 and g_stock.
For each (clue, definition, answer) pair, compute:

> delta = cos_sim(g(f_clue(def)), g(f_common_wndef(ans))) − cos_sim(g(f_common_wndef(def)), g(f_common_wndef(ans)))

Compare the distribution of deltas between g_stock and g_1.

*Cross-f generalization test:* Compute g_1(f_common_wnex(word)) for
validation-set words with valid f_common_wnex phrases, and compare cosine
similarities to g_stock(f_common_wnex(word)). If g_1 also pulls f_common_wnex
phrases together — even though it was never trained on them — that is evidence
of semantic generalization. If not, that is evidence of format-specific
overfitting. Report the size of this subset alongside the results.

*Baseline:* Measure cosine similarities between f_common_wndef and
f_common_wnex phrases under g_stock, for words that have both. This
establishes how format-sensitive stock CALE already is before fine-tuning.

**Step C — Design an improved triplet (T_2, future work)**

Based on Steps A and B, design a corrected triplet where all three elements
focus on the definition word, with the answer used only to guide sense
selection. A leading candidate:

> T_2 = (f_clue(def), f(def, answer-aligned sense), f(def, answer-misaligned sense))

This may require f to take two arguments — f(definition, answer) — where the
answer informs sense selection but is not itself embedded. Details are deferred
pending Steps A and B.

---

## 7. Embeddings: What Gets Computed When

Embedding generation is time-intensive (6–8 hours on Great Lakes GPU). We are
deliberate about what gets computed at each stage.

**Phrases:** Generated for the full dataset upfront, CPU only, committed
artifacts. Cheap to generate; expensive to regenerate if lost.

**Training-set embeddings:** Never saved. The training loop computes embeddings
internally for each batch and discards them after backpropagation.

**Validation-set embeddings:** Computed once per g_i being evaluated:
- g_i(f(word)) for all words in the appropriate validation vocabulary file
- g_i(f_clue(def)) for all (clue, definition) pairs in `clues_val.csv`

**g_stock f_clue embeddings:** Because g_stock will not be iterated on, we
generate g_stock(f_clue(def)) for the **full dataset** in one pass, rather
than just the validation set. This avoids a return trip to Great Lakes for
test-set f_clue embeddings later.

**g_i f_clue embeddings (iterative models):** Validation set only, named with
`_val` suffix. Full-dataset embeddings for any g_i are generated only if that
model is selected as final.

**Test-set embeddings:** Generated once, only after settling on a final g.
The test set must not be touched until then.

**Model weights:** Saved after each training run to the shared Google Drive
folder "Research Project - NLP CCC's" (owned by Nathan). Each
`models/<g_name>/` directory in the repo contains a `README.md` recording the
Google Drive path, HuggingFace base model identifier and version hash,
training script and hyperparameters, triplet file used, date trained, and
wall-clock runtime.

---

## 8. File Organization and Naming Conventions

### 8.1 Indexing and Triplet Provenance

An embedding array (`.npy` file) has no built-in labels. To look up the
embedding for a specific word or clue, we use an index.

For **vocabulary-based embeddings**, the vocabulary file itself is the index —
the row number of a word in `vocabulary_wndef.csv` is its row in
`f_common_wndef.npy`, and so on. No separate index file is needed.

For **f_clue embeddings**, the identifier is a composite key (clue_id,
definition), so an explicit `_index.csv` file is always stored alongside the
`.npy` file.

**Triplet provenance.** A triplet file spans multiple subset directories and
f's — the anchor comes from `clue_phrases/`, the positive and negative from
`wndef/`, `wnex/`, or another subset. The training rows are the intersection
of clues with valid phrases under all three f's used. This intersection is not
recoverable from the directory structure alone.

Each triplet file therefore has a companion `<g_name>_meta.json` that records:
- Which f was used for each triplet role (anchor, positive, negative)
- Source paths of the phrase files used
- Number of rows in the triplet file
- Split used (always `'train'`) and `random_state`

This metadata file is the authoritative provenance record for the triplet
dataset. `models/<g_name>/README.md` references it rather than duplicating it.

### 8.2 Naming Conventions

- **No suffix** = full dataset scope
- **`_val` suffix** = validation split only

Applies to both vocabulary files and embedding files. Phrase files are always
generated for the full dataset; `_val` is used only for embeddings.

Phrase files are named starting with `f_` (e.g. `f_clue.csv`,
`f_common_wndef.csv`). Each phrase file lives in the subset directory that
corresponds to its filtering constraint. The `f_` prefix makes phrase files
immediately identifiable within a flat subset directory alongside clue and
vocabulary files.

Notebook files start with a two-digit stage number and no prefix:
`01_wn_filtering_and_split.ipynb`, `02_phrase_construction.ipynb`, etc.

### 8.3 Directory Structure

```
ccc-project/
├── data/                                   # SHARED — do not modify
│   ├── data.sqlite3
│   ├── clues_raw.csv
│   ├── puzzle_metadata.csv                 # Produced by puzzle_metadata.ipynb
│   ├── clues_filtered.csv                  # Produced by structural_filtering.ipynb
│   └── publisher_lookup.csv
├── notebooks/
│   ├── puzzle_metadata.ipynb
│   ├── structural_filtering.ipynb
│   └── clue_utils.py                       # Shared definition-finding + delimiter logic
└── custom_embedding_model/
    ├── data/
    │   ├── filtered_split/
    │   │   └── wn_synset/                  # Rows where def + answer have ≥1 WN synset
    │   │       ├── clues_wn_filtered.csv   # Full scope, with split column
    │   │       ├── clues_val.csv           # Validation rows (convenience subset)
    │   │       ├── vocabulary.csv          # All wn_synset words; IS the index
    │   │       ├── vocabulary_val.csv      # Validation-split subset; IS the index
    │   │       ├── clue_phrases/           # f_clue only; no further row filtering
    │   │       │   └── f_clue.csv          # Indexed by (clue_id, definition)
    │   │       ├── wndef/                  # Words with a valid WN definition phrase
    │   │       │   ├── clues_wndef_filtered.csv
    │   │       │   ├── vocabulary_wndef.csv
    │   │       │   ├── vocabulary_wndef_val.csv
    │   │       │   └── f_common_wndef.csv
    │   │       └── wnex/                   # Words with a valid WN usage example
    │   │           ├── clues_wnex_filtered.csv
    │   │           ├── vocabulary_wnex.csv
    │   │           ├── vocabulary_wnex_val.csv
    │   │           └── f_common_wnex.csv
    │   ├── triplets/                       # Spans subset dirs and f's; one file per g
    │   │   ├── g1_train.csv
    │   │   └── g1_train_meta.json          # Provenance: f per role, source paths, row counts
    │   └── embeddings/                     # Organized by g model, not by subset
    │       ├── g_stock/
    │       │   ├── f_clue.npy              # Full wn_synset clues
    │       │   ├── f_clue_index.csv        # (clue_id, definition, row)
    │       │   ├── f_common_wndef.npy      # Indexed by wndef/vocabulary_wndef.csv
    │       │   └── f_common_wnex.npy       # Indexed by wnex/vocabulary_wnex.csv
    │       └── g1/                         # Validation-split embeddings only
    │           ├── f_clue_val.npy
    │           ├── f_clue_val_index.csv    # (clue_id, definition, row)
    │           ├── f_common_wndef_val.npy  # Indexed by wndef/vocabulary_wndef_val.csv
    │           └── f_common_wnex_val.npy   # Indexed by wnex/vocabulary_wnex_val.csv
    ├── models/
    │   ├── g_stock/
    │   │   └── README.md                   # HuggingFace ID + version hash
    │   └── g1/
    │       └── README.md                   # Google Drive path, hyperparams, date trained
    ├── notebooks/
    │   └── archive/
    │       └── 09_learned_g_misdirection.ipynb  # Reference only
    ├── scripts/
    ├── outputs/
    ├── CLAUDE.md
    ├── WORKFLOW.md
    ├── DATA.md
    ├── NOTEBOOKS.md
    ├── DECISIONS.md
    └── FINDINGS.md
```

### 8.4 Runtime Tracking

All notebooks and scripts record wall-clock runtimes for computationally
significant steps. This enables better planning for future GPU allocations
and identifying bottlenecks.

---

## 9. Relationship to Prior Work

This project builds directly on the Milestone II `clue_misdirection` component:

- **Embedding model:** Same CALE model (`gabrielloiseau/CALE-MBERT-en`, 1024-dim)
- **Data source:** Same raw input (`clues_raw.csv`); structural filtering logic
  is carried forward and reorganized into the shared upstream pipeline
- **Phrase construction:** f_common_wndef and f_common_wnex are related to
  the phrase construction logic in `clue_misdirection/02_embedding_generation.ipynb`,
  which used a priority cascade (usage example → definition → fallback). Here
  each strategy is an explicit, named f with no hidden fallbacks.
- **ATE framework:** The misdirection Average Treatment Effect measure was
  introduced in NB 09 and carries forward as the primary evaluation measure.

The `clue_misdirection` and `indicator_clustering` directories are complete
and must not be modified.

---

## 10. Open Questions and Deferred Decisions

- **Validation vocabulary fraction:** What fraction of each full vocabulary
  file appears in the validation split? To be measured after the split is
  built — informs whether embedding validation-only vocabularies saves
  meaningful GPU time.

- **WordNet example coverage:** How many words in the full vocabulary have
  valid f_common_wnex phrases? How many clue rows remain when both definition
  and answer must have valid examples? Informs whether f_common_wnex is
  viable as a training f.

- **f_3 and beyond:** Contingent on WordNet coverage measurements.

- **Improved triplet structure (T_2):** To be designed after Steps A and B
  are complete.

- **f(definition, answer):** The improved triplet may require f to take two
  arguments, where the answer informs sense selection. How to define
  "answer-aligned sense" rigorously is an open question.

- **Evaluation metrics for the final model:** What does success look like
  beyond the ATE? TBD.

- **Misdirection measurement model:** Out of scope for this directory; will
  live in a separate component.

---

## References

Schroff, F., Kalenichenko, D., and Philbin, J. (2015). FaceNet: A Unified
Embedding for Face Recognition and Clustering. *Proceedings of the IEEE
Conference on Computer Vision and Pattern Recognition (CVPR)*, Boston,
7–12 June 2015, pp. 815–823.
