# Spec: ATE Breakdown by Wordplay Type

**Stage:** Exploration (not part of the numbered pipeline)
**Notebook:** `planning/exploration/wordplay_ate_breakdown.ipynb`
**Date:** 2026-04-19
**Status:** Approved (revised 2026-04-19)

## Purpose

Investigate whether g_stock and g1 show different misdirection patterns
(ATE) on clues with different algorithmically verifiable wordplay types.
This is an advisor-requested exploratory analysis, not a pipeline notebook.

The notebook has three parts: (1) a descriptive landscape of wordplay type
frequencies within the validation set, (2) a structural comparison between
standard and double-definition clues, and (3) ATE breakdowns by letterplay
type within standard clues — first all 10 individual types, then grouped
categories.

**Double definitions as a point of interest:** Double-definition clues have
a fundamentally different structure from standard cryptic clues. Instead of
the fodder/indicator/definition structure of a normal cryptic clue, a double
definition concatenates two or more definitions for the answer — each
pointing to a *different sense* of the answer word (e.g., "Friendly drink"
→ CORDIAL, where "friendly" and "drink" are unrelated senses). The
concatenated surface may read as something else entirely. This means the
misdirection mechanism is structurally different: instead of wordplay
pulling the surface away from the definition, the surface's coherent reading
emerges from jamming together definitions of different senses. Whether this
produces more or less ATE than standard clues is an open empirical question
— we investigate it without a directional prediction.

## Inputs

All paths relative to `custom_embedding_model/` unless noted.

- `data/filtered_split/wn_synset/clues_val.csv` — validation-split clues
  (47,933 rows). Columns include `clue_id`, `surface`, `definition`,
  `answer`, `definition_wn`, `answer_wn`.
- `../../data/wordplay_metadata.csv` — one row per unique `clue_id` with
  boolean wordplay type columns. Join key: `clue_id` (many-to-one from
  clues_val).
- `data/embeddings/g_stock/f_clue_val.npy` — g_stock f_clue val embeddings
  (47933, 1024). Index: `data/embeddings/g_stock/f_clue_val_index.csv`.
- `data/embeddings/g1/f_clue_val.npy` — g1 f_clue val embeddings
  (47933, 1024). Index: `data/embeddings/g1/f_clue_val_index.csv`.
- `data/embeddings/g_stock/f_common_wndef.npy` — g_stock wndef embeddings,
  full vocabulary (53930, 1024). Index: `data/filtered_split/wn_synset/wndef/vocabulary_wndef.csv`.
- `data/embeddings/g1/f_common_wndef.npy` — g1 wndef embeddings, full
  vocabulary (53930, 1024). Index: same vocabulary file.
- `data/filtered_split/wn_synset/wndef/vocabulary_wndef.csv` — canonical
  vocabulary ordering for wndef embeddings.

## Outputs

- `outputs/figures/wp_cooccurrence_heatmap.png` — co-occurrence heatmap of
  wordplay types in validation set
- `outputs/figures/wp_ate_individual_letterplay.png` — ATE for all 10
  individual letterplay types (standard clues only)
- `outputs/figures/wp_ate_grouped_letterplay.png` — ATE for grouped
  letterplay categories (standard clues only)
- `outputs/figures/wp_ate_structural.png` — ATE for structural comparison
  (all validation vs standard vs double-def)
- `outputs/figures/wp_double_def_comparison.png` — double-def vs standard
  delta distributions
- `outputs/wordplay_ate_breakdown-results.md` — all numerical results

## Implementation Details

### §0 — Setup

Standard imports (pandas, numpy, matplotlib, seaborn, pathlib, time).
Environment auto-detection. Version reporting cell.

Define `DATA_DIR`, `EMBED_DIR`, `OUTPUT_DIR`, `FIGURE_DIR` using pathlib
relative to the notebook location (`planning/exploration/`).

### §1 — Load data

1. Load `clues_val.csv` with `keep_default_na=False, na_values=[""]`.
   Report row count (expect 47,933).

2. Load `wordplay_metadata.csv` with `keep_default_na=False, na_values=[""]`.
   Report row count.

3. Join on `clue_id` (left join from clues_val). Every row in clues_val
   should match — assert no nulls in any wordplay column after join.
   Report joined row count (should equal clues_val row count).

4. Load `vocabulary_wndef.csv` with `keep_default_na=False, na_values=[""]`.
   Build a word→row lookup dict.

5. Load embedding arrays and index files:
   - g_stock f_clue_val.npy + index
   - g1 f_clue_val.npy + index
   - g_stock f_common_wndef.npy
   - g1 f_common_wndef.npy

6. Assert all array shapes match expected sizes from FINDINGS.md.

### §2 — Wordplay type landscape (validation set)

**Per-type frequencies in the validation set:**

The 11 boolean columns from wordplay_metadata are:
`anagram_single_word`, `anagram_consec_words`, `hidden_fwd`, `hidden_rev`,
`selection_alt`, `selection_alt_rev`, `selection_firsts`,
`selection_firsts_rev`, `selection_lasts`, `selection_lasts_rev`,
`double_def`.

For each type, count how many *rows* in the joined validation data have
that type True. Report as a table with count and percentage of total
validation rows. Note: because clue_id is non-unique (double-def expansion),
a clue_id with double_def=True contributes multiple rows.

Also report:
- Rows with at least one type True (count and %)
- Rows with no type True (count and %)

**Co-occurrence heatmap:**

Compute a symmetric matrix where cell (i, j) is the number of *unique
clue_ids* in the validation set where both type i and type j are True.
Diagonal = count of unique clue_ids with that type True.

Plot as a heatmap using seaborn with:
- Annotated cell values
- A diverging or sequential colormap (use `cmap='YlOrRd'` or similar)
- Type names as axis labels (abbreviated if needed for readability)
- Title: "Wordplay Type Co-occurrence (Validation Set, unique clue_ids)"

Save to `outputs/figures/wp_cooccurrence_heatmap.png` (300 dpi).

### §3 — Define analysis categories

Cryptic crossword clues fall into two structural types:

- **Standard clues** contain one definition, an indicator, and fodder. The
  algorithmically detected letterplay types (anagram, hidden, selection,
  etc.) describe what the fodder/indicator mechanism does.
- **Double-definition clues** concatenate two or more definitions for the
  answer, each pointing to a different sense. There is no fodder or
  indicator. `double_def` is a structural type, not a letterplay type.

This two-level taxonomy drives the category definitions below. All
letterplay categories are intersected with `not_double_def` so they
describe standard clues only. The few clues that are both `double_def=True`
and have a detected letterplay type (e.g., 22 `anagram_consec` ×
`double_def` co-occurrences) appear in the double_def structural group
but are excluded from the letterplay breakdown.

Define each category as a boolean Series over the joined validation
DataFrame (row-level, not clue_id-level).

**Define the standard-clue mask first:**

`is_standard = ~joined["double_def"]`

**Level 1 — Structural categories (plotted in Figure 3):**

| Category name | Definition |
|---------------|------------|
| `standard` | `is_standard` |
| `double_def` | `joined["double_def"]` |

**Level 2a — Individual letterplay types within standard clues (Figure 1):**

All 10 letterplay columns, each intersected with `is_standard`:

| Category name | Mask |
|---------------|------|
| `anagram_consec` | `is_standard & joined["anagram_consec_words"]` |
| `anagram_single` | `is_standard & joined["anagram_single_word"]` |
| `hidden_fwd` | `is_standard & joined["hidden_fwd"]` |
| `hidden_rev` | `is_standard & joined["hidden_rev"]` |
| `selection_alt` | `is_standard & joined["selection_alt"]` |
| `selection_alt_rev` | `is_standard & joined["selection_alt_rev"]` |
| `selection_firsts` | `is_standard & joined["selection_firsts"]` |
| `selection_firsts_rev` | `is_standard & joined["selection_firsts_rev"]` |
| `selection_lasts` | `is_standard & joined["selection_lasts"]` |
| `selection_lasts_rev` | `is_standard & joined["selection_lasts_rev"]` |

Baseline for this figure: `no_letterplay` (standard clues with no detected
letterplay).

**Level 2b — Grouped letterplay categories within standard clues (Figure 2):**

| Category name | Definition |
|---------------|------------|
| `any_anagram` | `is_standard & (anagram_single_word \| anagram_consec_words)` |
| `any_hidden` | `is_standard & (hidden_fwd \| hidden_rev)` |
| `any_reversal` | `is_standard & (hidden_rev \| selection_alt_rev \| selection_firsts_rev \| selection_lasts_rev)` |
| `any_selection` | `is_standard & (selection_alt \| selection_alt_rev \| selection_firsts \| selection_firsts_rev \| selection_lasts \| selection_lasts_rev)` |
| `any_letterplay` | `is_standard & (any of the 10 letterplay columns True)` |
| `no_letterplay` | `is_standard & (all 10 letterplay columns False)` |

Baseline for this figure: `no_letterplay` (standard clues with no detected
letterplay).

Report the row count (N) for each category. Flag any category with N < 50
as too small for reliable ATE estimation (still compute it, but mark it in
the results table).

### §4 — ATE computation

**Reusable computation function:**

Write a function `compute_ate(df_subset, model_name, ...)` that takes a
subset of the joined validation DataFrame and computes:

For each row in the subset:
1. Look up `definition_wn` in `vocabulary_wndef.csv` → row index → wndef
   embedding for that model
2. Look up `answer_wn` in `vocabulary_wndef.csv` → row index → wndef
   embedding for that model
3. Look up (clue_id, definition) in the f_clue_val index → row index →
   f_clue_val embedding for that model
4. Compute T=0 = cosine_similarity(wndef(definition_wn), wndef(answer_wn))
5. Compute T=1 = cosine_similarity(f_clue(definition), wndef(answer_wn))
6. Compute delta = T=1 − T=0

A row is "resolvable" if all three lookups succeed (definition_wn in vocab,
answer_wn in vocab, (clue_id, definition) in f_clue index). Since we are
using full-vocab wndef embeddings (Decision 23), all vocab lookups should
succeed. Report any unresolvable rows.

Return a dict with:
- `n_total`: rows in subset
- `n_resolved`: rows where all lookups succeeded
- `t0_mean`: mean T=0
- `t1_mean`: mean T=1
- `ate_mean`: mean delta
- `ate_median`: median delta
- `ate_ci_lo`, `ate_ci_hi`: 95% bootstrap CI on mean delta (1000 bootstrap
  samples, random_state=42)
- `pct_negative`: % of deltas that are negative

**Apply to all categories:**

For each of the categories defined in §3, compute ATE for both g_stock and
g1. Store results in a DataFrame with columns: `category`, `model`, `n`,
`t0_mean`, `t1_mean`, `ate_mean`, `ate_median`, `ate_ci_lo`, `ate_ci_hi`,
`pct_negative`.

The baseline for letterplay figures is `no_letterplay` (standard clues with
no detected letterplay). This cleanly isolates the effect of each letterplay
type against the complement within standard clues.

Print the results table, sorted by category name then model.

### §5 — ATE visualization

Three dot plots, each using the same visual conventions:
- X-axis: ATE (mean delta)
- Y-axis: category names
- Two markers per row at the same y-position (no vertical jitter): g_stock
  (blue circle) and g1 (orange triangle), connected by a thin light-gray
  line
- Horizontal error bars showing 95% CI
- A dashed vertical line at x=0 (no misdirection)
- N annotated next to each row
- Categories with N < 50 shown in lighter color / dashed error bars

**Figure 1 — Individual letterplay types (standard clues only):**

Y-axis rows (top to bottom): `no_letterplay` (baseline), then the 10
individual letterplay types ordered by descending N.

Save to `outputs/figures/wp_ate_individual_letterplay.png` (300 dpi).

**Figure 2 — Grouped letterplay categories (standard clues only):**

Y-axis rows (top to bottom): `no_letterplay` (baseline), `any_anagram`,
`any_hidden`, `any_reversal`, `any_selection`, `any_letterplay`.

Save to `outputs/figures/wp_ate_grouped_letterplay.png` (300 dpi).

**Figure 3 — Structural comparison:**

Y-axis rows: `standard`, `double_def`.

Save to `outputs/figures/wp_ate_structural.png` (300 dpi).

### §6 — Structural comparison: double-def vs standard

This section accompanies Figure 3 with a deeper look at the distributional
difference between the two structural types.

**Structural context:** In a standard cryptic clue, the tagged definition's
context is wordplay (fodder + indicator). In a double-def clue, the tagged
definition's context is one or more *other* definitions of the same answer,
each typically pointing to a different sense. The surface reading formed by
concatenating these definitions may be misleading in its own way — not
through wordplay, but through the juxtaposition of unrelated senses creating
a coherent-seeming phrase. Whether this produces more, less, or comparable
misdirection as measured by ATE is an open question.

**Distribution comparison:**

For the `double_def` and `standard` groups, plot overlapping histograms (or
KDE curves) of per-row delta values, for g_stock. Use alpha transparency so
both distributions are visible. Repeat for g1 as a second subplot
(vertically stacked: g_stock on top, g1 below).

Save to `outputs/figures/wp_double_def_comparison.png` (300 dpi).

**Narrative in markdown cell:**

After the plot, include a markdown cell discussing the result:
- Do double-def clues show a different ATE distribution than standard clues?
- If so, in which direction, and what might explain it given the structural
  differences?
- If not, what does that tell us about what the ATE is capturing?

### §7 — Summary cell

Standard summary cell:
- Total validation rows; standard vs double-def split
- Per-category sample sizes
- Key structural finding: how double-def ATE compares to standard
- Key letterplay findings: which types show the most/least misdirection
  under g_stock? Does g1 improve ATE for any type?
- Figures produced and their locations
- Wall-clock runtime

### §8 — Write results file

Write `outputs/wordplay_ate_breakdown-results.md` containing:
- Full ATE results table (all categories × both models), organized by
  level (structural, then individual letterplay, then grouped letterplay)
- Per-type validation set frequencies
- Structural comparison summary (double-def vs standard)
- Version stamps

## Environment

Local (CPU). No GPU required. All embeddings are pre-computed.

## Notebook structure

- Use §-numbered markdown sections before each logical block
- Include environment auto-detection for local/Great Lakes/Colab
- Standard notebook header:
  - Primary author: Victoria
  - Builds on: `05_model_evaluation.ipynb` (Victoria — ATE computation
    methodology), `wordplay_metadata.ipynb` (Victoria — wordplay type
    detection)
  - Environment: Local
- Write results file to `outputs/wordplay_ate_breakdown-results.md`
