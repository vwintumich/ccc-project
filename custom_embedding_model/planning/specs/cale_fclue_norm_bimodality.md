# Spec: CALE f_clue Norm Bimodality Investigation

**Stage:** Exploration (not part of numbered pipeline)
**Notebook:** `planning/exploration/cale_fclue_norm_bimodality.ipynb`
**Date:** 2026-04-24

## Purpose

During the g1 basic evaluation, we observed that the g_stock f_clue (val) L2
norm distribution is visibly bimodal, with peaks near 29.5 and 31.5. This
notebook investigates the source of the bimodality through systematic
subsetting, rules out computational error, characterizes which embedding
dimensions drive the two modes, and explores whether the pattern propagates
into cosine similarity or ATE-relevant measures.

This is a CALE model characterization — it documents an idiosyncrasy of how
the pretrained CALE model processes tagged text in clue context, analogous to
the WordNet sense-selection idiosyncrasies documented elsewhere. The finding
does not invalidate prior analyses but should be understood and tracked.

## Inputs

- `data/embeddings/g_stock/f_clue.npy` (239,406 × 1024)
- `data/embeddings/g_stock/f_clue_index.csv`
- `data/embeddings/g_stock/f_common_wndef.npy` (53,930 × 1024)
- `data/embeddings/g1/f_clue_val.npy` (47,933 × 1024)
- `data/embeddings/g1/f_clue_val_index.csv`
- `data/embeddings/g1/f_common_wndef.npy` (53,930 × 1024) — Decision 23
  full-vocab
- `data/filtered_split/wn_synset/clues_wn_filtered.csv` — full dataset
  (all splits) for the full-dataset norm check
- `data/filtered_split/wn_synset/clues_val.csv`
- `data/filtered_split/wn_synset/wndef/vocabulary_wndef.csv`
- `data/filtered_split/wn_synset/clue_phrases/f_clue.csv`
- `../../data/id_map.csv` — maps `clue_id` → `puzzle_id`
- `../../data/puzzle_metadata.csv` — maps `puzzle_id` → `publisher`

## Outputs

- `outputs/cale_fclue_norm_bimodality-results.md`
- `outputs/figures/fclue_norm_bimodality_*.png` (4–6 figures, naming TBD
  based on section content)

No new data artifacts (CSVs, .npy files) are produced.

## Implementation details

### §1 — What does embedding norm bimodality mean?

Markdown-only conceptual section. No code. Cover:

1. **What L2 norm represents.** The L2 norm of an embedding is its magnitude
   — how far the vector sits from the origin. Under CALE, each f_clue
   embedding represents a definition word contextualized by its cryptic clue.
   The norm reflects how "strongly" the model activates in response to that
   input.

2. **What two norm peaks imply.** A bimodal norm distribution means CALE
   processes these inputs in two distinct regimes, producing embeddings of
   systematically different magnitudes. If the two groups don't map to any
   observable text property (word identity, clue length, publisher), this
   reflects internal model behavior — something about the self-attention
   patterns, not something about the text.

3. **Relationship to L2 distance and cosine similarity.** L2 distance
   between two vectors depends on both their directions AND their magnitudes:
   ‖a − b‖² = ‖a‖² + ‖b‖² − 2‖a‖‖b‖cos θ. So norm differences directly
   affect L2 distances. Cosine similarity depends only on direction
   (cos θ = a·b / ‖a‖‖b‖) — it normalizes out the magnitude. This means:
   - The bimodality will show up in any L2 measure involving f_clue
     (including T=1 L2 distance).
   - It should NOT show up in cosine similarity — UNLESS the two norm groups
     also differ in direction.
   - The T=0 measures (which use only f_wndef, not f_clue) cannot be affected
     by f_clue norm bimodality at all.

4. **What it means that g1 eliminates the bimodality.** g1 compressed all
   norms into a tighter range (f_clue std: 1.20 → 0.75). Fine-tuning erased
   CALE's two-regime behavior as part of its global compression of the
   embedding space. This was not a targeted fix — g1 steamrolled all norm
   variation.

5. **Relationship to T=0 cosine bimodality.** The g_stock T=0 cosine
   similarity distribution also shows irregular multi-modal structure. T=0
   uses only f_wndef embeddings, so it cannot share a cause with the f_clue
   norm bimodality. Any T=0 bimodality reflects the semantic properties of
   definition-answer word pairs (how similar they are under g_stock's
   pretrained representations), not how CALE processes clue context. These
   are independent phenomena.

### §2 — Reproducing the bimodal distributions

First, confirm the bimodality is not a validation-set artifact by comparing
the full-dataset g_stock f_clue norms (239,406 rows, all splits) against the
validation-only slice (47,933 rows). Plot both as overlaid normalized
histograms (60 bins, same x-axis range). Both should show the same bimodal
structure. Print text histograms for each to verify the peak locations match.
If they do, the rest of the notebook proceeds on the validation slice only
(since that's where we have g1 embeddings for comparison).

Then reproduce the relevant panels from the g1 basic evaluation, focused on
the bimodality. Three plots:

1. **f_clue norm histogram** (adapted from `g1be_norm_distributions.png`
   right panel). Show g_stock and g1 overlaid. Use the same visual encoding
   as the basic evaluation (diagonal hatch for L2, model colors). Use 60 bins.
   This is the primary exhibit.

2. **T=0/T=1 cosine** (adapted from `g1be_t0_t1_cosine.png` left panel,
   g_stock only). Show the T=0 outline distribution to confirm the
   multi-modal structure Victoria observed. Annotate or label the apparent
   modes.

3. **T=0/T=1 L2** (adapted from `g1be_t0_t1_l2.png` left panel, g_stock
   only). Check whether T=1 L2 also shows bimodality (expected, since T=1
   L2 depends on f_clue norms).

Print a text-based histogram of f_clue norms (25 bins, range 25–34) for
precise identification of the peak locations and valley.

Data loading pattern:
- Load g_stock f_clue full-dataset embeddings (239,406 rows).
- Extract the validation slice (47,933 rows) aligned to g1's
  f_clue_val_index, using the same key-matching approach as the basic
  evaluation notebook. Store as `embeddings[("g_stock", "f_clue_val")]`.
- Load g1 f_clue_val.npy (47,933 rows).
- Load g_stock and g1 f_common_wndef.npy (53,930 rows each).
- Compute norms: `np.linalg.norm(emb, axis=1)` for f_clue_val under both
  models.
- For T=0/T=1: resolve (definition, answer) pairs from clues_val against
  vocabulary_wndef. Compute rowwise cosine and L2 for both T=0 and T=1.

### §3 — Definition position as the primary structural driver

The definition in a cryptic clue appears at either the start or end of the
surface (occasionally in the middle). When the definition is at the start,
the tagged f_clue phrase looks like `<t>Definition</t> rest of clue`. When
at the end: `rest of clue <t>definition</t>`. Transformer attention patterns
are position-sensitive, so the same word tagged in different positions may
produce embeddings with different properties.

**Analysis:**

1. Classify each validation clue by definition position: start
   (`surface.lower().startswith(definition.lower())`), end
   (`surface.lower().endswith(definition.lower())`), or middle (neither).
   Report counts.

2. Plot separate norm histograms for start-definitions and end-definitions
   (60 bins, same x-axis range). Use a two-panel figure.
   Expected finding: end-definitions are unimodal; start-definitions are
   bimodal with the same two-peak structure as the overall distribution.

3. Print text histograms for each group to confirm the visual impression.

4. Report norm summary stats (mean, std, P25, P75) for each position group.

### §4 — Ruling out alternative explanations

For start-definitions specifically (since they carry the bimodality), test
whether any observable text property explains which mode a clue lands in.
Split start-definitions at the valley (norm ≈ 30.5) into a "lower mode" and
"upper mode" group, then compare:

1. **Surface word count:** mean for each group.
2. **Definition character length:** mean for each group.
3. **Definition subword token count:** tokenize definitions with the MBERT
   tokenizer (`bert-base-multilingual-cased`) and compare means. Also produce
   separate norm histograms for 1-token and 2-token definitions to show the
   bimodality persists within each.
4. **Publisher:** join validation clues to `id_map.csv` on `clue_id` to get
   `puzzle_id`, then join `puzzle_metadata.csv` on `puzzle_id` to get
   `publisher` (see DATA_RAW.md §4.3 for the standard join pattern). Report
   mean norm and std by publisher. Produce separate norm histograms for the
   top 2–3 publishers by count.
5. **Word identity:** count how many unique definition words appear in both
   modes (2,834 out of 8,728 from our investigation). Show that the same
   word lands in different modes depending on the clue. Compute
   within-word std vs. overall std (expected: within-word is ~85% of overall,
   confirming context drives the variation).
6. **wndef norm correlation:** correlate each clue's f_clue norm with its
   definition word's wndef norm. Expected: weak positive correlation
   (ρ ≈ 0.125), confirming the word itself contributes little.

For each comparison, the key claim to support is: **none of these variables
cleanly separate the two modes.** The differences are negligible.

### §5 — Dimension-level characterization

Instead of looking for bimodal individual dimensions, identify which
dimensions have systematically different mean values between the two norm
groups. This tells us which dimensions "assign" a clue to the lower or upper
mode.

**Analysis (start-definitions only):**

1. Split at the valley (norm ≈ 30.5) into lower-mode and upper-mode groups.

2. Compute the mean embedding vector for each group: `mean_lower` (shape
   1024,) and `mean_upper` (shape 1024,).

3. Compute `delta = mean_upper - mean_lower` for each dimension.

4. **Dimension profile plot:** Plot `delta` across all 1024 dimensions as a
   bar/stem chart. This shows which dimensions contribute most to separating
   the two norm groups. Sort by absolute delta for a ranked view (show top 20
   in a table as well).

5. **Effect size:** For the top 10 dimensions by |delta|, compute Cohen's d
   (delta / pooled_std) to quantify how much each dimension separates the
   groups relative to within-group variability.

6. **Concentration check:** What fraction of the total norm² difference is
   explained by the top 10, top 50, and top 100 dimensions? Compute:
   `sum(mean_upper[top_k]² - mean_lower[top_k]²) / (mean_norm_upper² -
   mean_norm_lower²)`. If a small number of dimensions explain most of the
   difference, report that. If it's distributed across many dimensions,
   report that instead.

7. **Sanity check:** Confirm that the top-delta dimension is dim 379
   (the highest norm-correlated dimension from our investigation, r ≈ 0.54).

### §6 — Does the bimodality propagate to cosine similarities?

The f_clue norm bimodality necessarily affects L2 distances involving f_clue
but should not affect cosine similarity — unless the two norm groups also
differ directionally. Test this.

**Analysis (start-definitions only, using validation pairs with valid wndef
entries for both definition and answer):**

1. **T=1 cosine stratified by norm group:** Compute T=1 cosine similarity
   = rowwise_cosine(f_clue_emb, wndef_ans_emb) for each clue. Split into
   lower-mode (norm < 30.5) and upper-mode (norm ≥ 30.5). Compare the T=1
   cosine distributions (mean, std, histogram). If they're similar, cosine
   successfully normalizes out the norm difference. If they differ, the two
   norm groups also differ directionally.

2. **T=1 L2 stratified by norm group:** Same split, but T=1 L2 distance.
   Expected: the two groups WILL differ (L2 depends on norms), and the
   direction of the difference should match (higher-norm group → larger L2
   distances, all else equal).

3. **T=0 cosine — independent phenomenon:** Briefly confirm that the T=0
   cosine distribution's irregular shape is NOT related to f_clue norms.
   T=0 uses only wndef embeddings, so there should be no correlation
   between a clue's f_clue norm and its T=0 cosine. Compute Spearman
   correlation to verify (expected: near zero). Note that any multi-modal
   structure in T=0 cosine is a separate phenomenon reflecting the
   distribution of definition-answer semantic distances under g_stock.

4. **ATE stratified by norm group:** Compute ATE (T=1 − T=0 cosine) for
   each norm group. If the ATEs are similar, the norm bimodality does not
   introduce a confound into our primary evaluation metric. If they differ,
   document the magnitude and direction.

### §7 — Effect of fine-tuning

Briefly show that g1 eliminates the bimodality.

1. Overlay g1 and g_stock f_clue norm histograms (reproducing §2 panel 1)
   with added annotation highlighting the bimodal vs. unimodal structure.

2. For g1 f_clue norms, run the same definition-position split as §3:
   start-defs vs. end-defs. Expected: both are unimodal under g1.

3. Connect to the compression findings from the evaluation report: g1's
   global compression reduced norm std from 1.20 to 0.75, collapsing both
   modes into one.

### §8 — Summary

Markdown cell summarizing:
- The bimodality is real, not a computational error
- It is driven by the clue context (not the definition word) and is
  strongest for start-of-surface definitions
- No observable text property explains which mode a start-definition lands in
- It is a distributed CALE model behavior involving many embedding dimensions
- Whether it propagates to cosine similarity (from §6 results)
- Whether it affects ATE measurements (from §6 results)
- g1 eliminates it as a side effect of global compression
- The T=0 cosine irregularity is a separate, unrelated phenomenon

## Environment

Local (CPU). All data is already available locally. MBERT tokenizer download
is the only network dependency (for §4 subword analysis); it can be cached.

## Notebook structure

- Use §-numbered markdown sections before each logical block.
- Include environment auto-detection for local/Great Lakes/Colab.
- Follow FIGURE_STANDARDS.md: model colors (blue = g_stock, orange = g1),
  diagonal hatch for L2 data, solid fill for cosine.
- Save all figures to `outputs/figures/` with prefix `fclue_bimodal_`.
- Write results to `outputs/cale_fclue_norm_bimodality-results.md`.
- Header: primary author Victoria, builds on g1_basic_evaluation (Victoria),
  AI assistance Claude/Claude Code.
