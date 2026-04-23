# Spec: POS and WordNet Sense Census

**Stage:** Exploration (not part of the numbered pipeline)
**Notebook:** `planning/exploration/pos_wordnet_census.ipynb`
**Date:** 2026-04-20
**Status:** Approved

## Purpose

Descriptive census of part-of-speech distributions and WordNet sense
properties across the vocabulary, training triplets, and validation pairs.
This notebook establishes facts about the data that inform Stage 6
hypothesis testing — specifically, it characterizes the scale of design
issues DI-2 (POS-biased sense selection) and DI-3 (unreliable frequency
ordering) and provides the POS composition of training and evaluation data.

The notebook also serves a forward-looking purpose: understanding sense
availability and lemma count distributions informs the design of future
f's that might select senses more carefully.

## Inputs

All paths relative to `custom_embedding_model/`.

- `data/filtered_split/wn_synset/vocabulary.csv` — full vocabulary
  (53,930 words). Columns: `word`, `row`.
- `data/filtered_split/wn_synset/clues_wn_filtered.csv` — full clue
  dataset with split column. Columns include `clue_id`, `surface`,
  `definition`, `answer`, `definition_wn`, `answer_wn`, `split`.
- `data/filtered_split/wn_synset/clues_val.csv` — validation-split
  clues (47,933 rows). Same columns plus `row_id`.
- `data/triplets/g1_train.csv` — training triplets (69,921 rows).
  Columns: `clue_id`, `definition`, `answer_wn`, `distractor_wn`,
  `anchor`, `positive`, `negative`.
- WordNet (via NLTK) — for synset lookups, POS, lemma counts.

## Outputs

- `outputs/pos_wordnet_census-results.md` — all numerical results
- `outputs/figures/pos_vocab_distribution.png` — vocabulary POS breakdown
  and sense reliability
- `outputs/figures/pos_triplet_composition.png` — training triplet POS
  composition by role (includes confusion-matrix heatmap for POS mismatch)
- `outputs/figures/pos_validation_composition.png` — validation pair POS
  composition and T=0/T=1 POS alignment
- `outputs/pos_mismatch_examples.md` — collected examples where multi-word
  contextual POS could not be determined (for review)

## Implementation Details

### §0 — Setup

Standard imports (pandas, numpy, matplotlib, seaborn, pathlib, time).
NLTK imports: `from nltk.corpus import wordnet as wn`.
spaCy imports: `import spacy; nlp = spacy.load("en_core_web_sm")`.
Environment auto-detection. Version reporting cell (including NLTK and
spaCy versions).

Define `DATA_DIR`, `OUTPUT_DIR`, `FIGURE_DIR` using pathlib relative to
the notebook location (`planning/exploration/`).

### §1 — Load data

1. Load `vocabulary.csv` with `keep_default_na=False, na_values=[""]`.
   Assert 53,930 rows.

2. Load `clues_wn_filtered.csv` with `keep_default_na=False,
   na_values=[""]`. Subset to training split (`split == 'train'`) for
   later use.

3. Load `clues_val.csv` with `keep_default_na=False, na_values=[""]`.
   Assert 47,933 rows.

4. Load `g1_train.csv` with `keep_default_na=False, na_values=[""]`.
   Assert 69,921 rows.

### §2 — WordNet census: full vocabulary

For each word in `vocabulary.csv`, look up WordNet synsets using the
same lookup procedure as NB 01 (lowercase, underscore conversion for
multi-word entries, article stripping per Decision 16). For each word,
record:

- `n_synsets`: total number of synsets across all POS
- `n_synsets_n`, `n_synsets_v`, `n_synsets_a`, `n_synsets_r`: synsets
  per POS (noun, verb, adj/satellite adj, adverb)
- `sense0_pos`: POS of `synsets(word)[0]`
- `sense0_lemma_count`: sum of lemma counts for the word's lemma(s) in
  `synsets(word)[0]`. Specifically:
  `sum(l.count() for l in synsets(word)[0].lemmas() if l.name() == word)`
- `max_lemma_count_any_sense`: highest lemma count across ALL synsets
  for this word (regardless of POS)
- `max_lemma_count_same_pos`: highest lemma count among synsets sharing
  sense[0]'s POS
- `has_nonzero_count`: whether any sense of this word has a nonzero
  lemma count
- `sense0_is_max_within_pos`: whether sense[0] has the highest (or
  tied-highest) lemma count among synsets of the same POS
- `higher_freq_other_pos`: whether any synset in a different POS has a
  strictly higher lemma count than sense[0]

Store as a DataFrame (`vocab_census`) for reuse in later sections.

**Report:**

a) **POS distribution of sense[0]:**
   Count and percentage for each POS category (n, v, a, s, r). Note:
   WordNet separates adjectives (a) from satellite adjectives (s); report
   both individually and combined as "adj".

b) **Sense availability:**
   - Distribution of `n_synsets` (mean, median, quartiles, max)
   - How many words have synsets in multiple POS categories?
   - How many words have only noun synsets? Only verb? etc.

c) **Lemma count reliability:**
   - How many words have `has_nonzero_count == True`? (i.e., any
     frequency evidence at all)
   - Among words with nonzero counts: how many have
     `sense0_is_max_within_pos == True`? (sense[0] is actually the
     most frequent sense of its POS)
   - How many have `higher_freq_other_pos == True`? (a different POS
     has demonstrably higher frequency)
   - Crosstab: `sense0_pos` × `has_nonzero_count` (how reliable is
     each POS category?)

d) **Visualization:**
   - Bar chart of sense[0] POS distribution (noun, verb, adj, adverb)
   - Histogram or CDF of `n_synsets` per word
   - Annotated heatmap: sense[0] POS (rows) × reliability category
     (columns: reliable / arbitrary), cells annotated with counts and
     colored by magnitude

   Save primary figure to `outputs/figures/pos_vocab_distribution.png`
   (300 dpi). Use a multi-panel layout combining all three visualizations.

### §3 — Contextual POS of definitions in clue surfaces

This section determines the POS of each definition as it appears within
its clue surface — the contextual POS relevant to the f_clue anchor
(in training) and the T=1 condition (in evaluation).

#### Procedure

For each clue row (used in both §4 and §5), determine the contextual POS
of the definition word using `definition_wn` (not the raw `definition`
column). Using `definition_wn` effectively applies the same article
stripping as NB 01, increasing the number of single-word definitions.

1. **Strip `<t>` and `</t>` delimiters** from the f_clue phrase to
   recover clean surface text. (Or equivalently, use the `surface`
   column directly.)

2. **POS-tag the surface** using spaCy (`en_core_web_sm`).

3. **Locate `definition_wn`** in the tagged surface. Replace underscores
   with spaces for matching multi-word WordNet entries.

4. **Assign contextual POS:**

   - **Single-word `definition_wn`** (no underscores): Take the spaCy
     POS tag of that token directly. This is the contextual POS.

   - **Multi-word `definition_wn`** (contains underscores): Take the
     POS of the **last word** of the matched span. Then validate:
     - Look up `wn.synsets(definition_wn)` — if there is exactly one
       synset AND its POS matches the last-word POS tag → **accept**.
     - Otherwise → mark as **undetermined**. Collect these rows
       (surface, definition, definition_wn, tagged POS, WordNet POS)
       for manual review.

5. **Also record** the WordNet sense[0] POS of `definition_wn` (from
   the vocab census) for every row, for comparison with the contextual
   POS.

#### Handling notes

- **Capitalization:** Definitions at the start of the surface are
  capitalized. spaCy may tag these as proper nouns (PROPN). If
  `definition_wn` appears sentence-initially, lowercase the first
  character of the surface before tagging. Document this choice.

- **Coverage:** Report the fraction of rows where contextual POS was
  successfully determined vs. marked undetermined. Proceed with the
  confident subset for downstream statistics. Collect undetermined
  examples in a separate output for Victoria to review.

#### Output

Store the contextual POS assignments as a column that can be joined
into both training triplet and validation pair analyses. Save the
undetermined examples to `outputs/pos_mismatch_examples.md`.

**Runtime note:** POS-tagging all clue surfaces (~240K rows for the full
dataset, or ~70K training + ~48K validation if done per-split) with spaCy
may take several minutes. Use `nlp.pipe(surfaces, batch_size=1000)` for
efficiency and time this step.

### §4 — POS of training triplets

For each row in `g1_train.csv`, classify the POS of all three roles.

To get `definition_wn` for training triplet rows: join `g1_train.csv`
to `clues_wn_filtered.csv` on `(clue_id, definition)` to obtain the
`definition_wn` and `surface` columns needed for contextual POS tagging.

**Anchor (f_clue — contextual POS):**
- Use the §3 procedure to determine contextual POS of `definition_wn`
  within the clue surface.
- Also record the WordNet sense[0] POS of `definition_wn` from the
  vocab census.

**Positive (f_common_wndef — answer):**
- Look up `answer_wn` in the vocab census → `sense0_pos`,
  `has_nonzero_count`, `sense0_is_max_within_pos`

**Negative (f_common_wndef — distractor):**
- Look up `distractor_wn` in the vocab census → `sense0_pos`,
  `has_nonzero_count`, `sense0_is_max_within_pos`

**Report:**

a) **Per-role POS distribution (noun vs not-noun):**
   - Anchor contextual POS: % noun, % verb, % adj, % other,
     % undetermined
   - Anchor WordNet sense[0] POS: same breakdown (for comparison with
     contextual)
   - Positive (answer) WordNet sense[0] POS: % noun, % with nonzero
     counts, % where sense[0] is max within POS
   - Negative (distractor) WordNet sense[0] POS: same

b) **POS mismatch (anchor contextual vs anchor WordNet sense[0]):**
   - How often do these disagree?
   - Crosstab: contextual POS × WordNet sense[0] POS

c) **Triplet-level composition:**
   - What fraction of triplets have positive AND negative both noun?
   - What fraction have all wndef-representable roles (positive +
     negative) as noun?
   - How does the anchor contextual POS distribute when positive and
     negative are both nouns vs. when at least one is not?

d) **Sense reliability in training:**
   - What fraction of triplets have both positive and negative with
     `has_nonzero_count == True`?
   - What fraction have at least one role with arbitrary ordering
     (zero counts)?
   - What fraction have a role where `higher_freq_other_pos == True`?
     (trained on a noun sense when a more frequent non-noun sense
     exists)

e) **Visualization:**
   - Grouped bar chart: POS distribution by triplet role
     (anchor-contextual, anchor-WordNet, positive, negative)
   - Confusion-matrix heatmap for §4b: contextual POS (rows) × WordNet
     sense[0] POS (columns), cells annotated with counts, colored by
     magnitude. The diagonal represents agreement; off-diagonal cells
     show the scale and direction of POS mismatches.
   - Save to `outputs/figures/pos_triplet_composition.png` (300 dpi).
     Use a multi-panel layout combining both visualizations.

### §5 — POS of validation pairs

For each row in `clues_val.csv`, classify POS for the evaluation
components.

**Clue-contextualized definition (f_clue, used in T=1):**
- Use the §3 procedure on `definition_wn` within the surface.
- Record contextual POS.

**Decontextualized definition (f_common_wndef, used in T=0):**
- Look up `definition_wn` in vocab census → `sense0_pos`,
  `has_nonzero_count`, `sense0_is_max_within_pos`

**Answer (f_common_wndef, used in both T=0 and T=1):**
- Look up `answer_wn` in vocab census → same fields

**Report:**

a) **Per-component POS distribution (noun vs not-noun):**
   - Contextualized definition (from POS tagger): % noun, % verb,
     % adj, % other, % undetermined
   - Decontextualized definition (WordNet sense[0]): % noun, % with
     nonzero counts
   - Answer (WordNet sense[0]): % noun, % with nonzero counts

b) **Pair-level composition for T=0 (def_wndef × answer_wndef):**
   - Crosstab: definition sense[0] POS × answer sense[0] POS
   - What fraction of pairs are noun-noun?
   - Condense to: noun-noun, noun-other, other-noun, other-other

c) **POS mismatch for T=1 (contextual def POS vs answer sense[0] POS):**
   - Crosstab: definition contextual POS × answer sense[0] POS
   - How often does the contextualized definition POS agree with the
     answer's sense[0] POS?

d) **Sense reliability in evaluation:**
   - What fraction of pairs have both definition_wn and answer_wn with
     `has_nonzero_count == True`?
   - What fraction have at least one arbitrary-ordering word?

e) **Visualization:**
   - Annotated 2×2 heatmap for §5b: definition sense[0] POS (noun /
     not-noun, rows) × answer sense[0] POS (noun / not-noun, columns),
     cells annotated with counts and percentages of total pairs. This
     communicates the noun-noun dominance pattern at a glance.
   - Grouped bar chart: POS distribution by evaluation component
     (contextualized definition, decontextualized definition, answer)
   - Save to `outputs/figures/pos_validation_composition.png` (300 dpi).
     Use a multi-panel layout combining both visualizations.

### §6 — Summary cell

- Key numbers from the vocabulary census (sense[0] POS distribution,
  reliability rates)
- Training triplet POS composition (fraction noun across roles,
  fraction with arbitrary sense selection, contextual POS mismatch rate)
- Validation pair POS composition (fraction noun-noun pairs, contextual
  POS mismatch rate)
- Implications for DI-2 and DI-3: how large is the noun-dominance issue
  in our actual training and evaluation data?
- Forward-looking: what does sense availability look like for designing
  future f's?
- Wall-clock runtime
- Figures produced and locations

### §7 — Write results file

Write `outputs/pos_wordnet_census-results.md` containing all tables and
summary statistics from §2-§5, organized by section, with version stamps.

## Environment

Local (CPU). Requires:
- `spacy` with `en_core_web_sm` model for POS tagging
- `nltk` (3.9.2) with WordNet corpus

**spaCy installation (if needed):**
```bash
pip install spacy
python -m spacy download en_core_web_sm
```

If spaCy is not available in the `crossword` kernel, NLTK's `pos_tag`
(Penn Treebank tags) is an acceptable alternative — just document which
tagger was used and map Penn Treebank tags to universal POS categories
for consistent reporting.

## Notebook structure

- Use §-numbered markdown sections before each logical block
- Include environment auto-detection for local/Great Lakes/Colab
- Standard notebook header:
  - Primary author: Victoria
  - Builds on: `01_wn_filtering_and_split.ipynb` (Victoria — WordNet
    filtering and vocabulary construction),
    `03_train_g1.ipynb` (Victoria/Nathan — triplet construction)
  - Environment: Local
- Write results file to `outputs/pos_wordnet_census-results.md`
