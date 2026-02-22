# OPEN_QUESTIONS.md — Unresolved Decisions

This document tracks questions that have not yet been settled. Before making implementation choices in any of these areas, check this document and consult the team and/or faculty advisor if needed. When a question is resolved, move the resolution to FINDINGS_AND_DECISIONS.md and remove it here.

Last updated: February 21, 2026

---

## Clustering

### Q1: What granularity of clusters should we target? *(PARTIALLY RESOLVED)*

**The question:** Our results range from 18 clusters (DBSCAN on single-word indicators) to 353 clusters (HDBSCAN on all verified indicators). The labeled types in the data suggest k=8. Our theoretical analysis suggests k=3, k=4, k=8, k=14, or an unknown k based on conceptual metaphors. Which is most useful to report?

**Options:**
- k=3 (highest-level abstraction: fodder→chunk, chunk→chunk, chunk+chunk→chunk)
- k=4 (by operation type)
- k=8 (by labeled wordplay type)
- k=14+ (all Minute Cryptic types)
- Let the algorithm determine k (DBSCAN/HDBSCAN approach) and interpret what it finds
- Target multiple granularities in a hierarchical structure and report at all levels

**Advisor guidance (KCT, Feb 8):** "The question is not what exact number of clusters, but what nature of relationships do we want to capture." He recommended starting with k=8 (the labeled types) and characterizing which clusters are tight vs. diffuse, then building up. The number can emerge from a principled evaluation criterion — for example, what k produces the most stable, interpretable clusters across multiple runs.

**Partially resolved (Feb 21, 2026):** The unconstrained analysis (Stage 4) shows no natural k=8 level. k=10 is a local silhouette optimum, but metrics improve monotonically up to k=250. The data's natural structure is finer-grained than any coarse k. The remaining question shifts to Notebook 05: can domain knowledge (seed words, connectivity constraints) impose meaningful coarse structure that doesn't emerge naturally? If constrained clustering at k=7 or k=8 produces interpretable wordplay-type clusters despite the lack of natural support, that is itself a finding about the role of expert knowledge.

---

### Q2: How should we handle indicators that appear under multiple wordplay types?

**The question:** Many indicator strings appear under multiple wordplay types (e.g., "about" is container, reversal, and anagram). When we create a list of unique indicators, we lose this multi-label information. How do we handle this for clustering evaluation?

**Options:**
- Keep indicators as unique strings; accept that a single point may belong to multiple true clusters (evaluate with soft metrics)
- Keep multiple instances of the same indicator string (one per wordplay-type occurrence), accepting that identical strings will cluster together
- Use only indicators that appear under exactly ONE wordplay type for evaluation
- Use the multi-label structure to inform the connectivity matrix in constrained clustering

**What we know:** The same indicator appearing under multiple types is not noise — it is meaningful that "about" works for container, reversal, and anagram. This is a linguistic reality about CCC vocabulary.

**Current direction:** Unresolved. Need a decision before evaluation notebooks are finalized.

---

### ~~Q3: How should we set HDBSCAN/DBSCAN epsilon?~~

**RESOLVED (Feb 21, 2026).** Pairwise distance analysis completed in Notebook 04. Epsilon candidates selected from distance distribution percentiles (13 values from 0.0 to 2.68, with fine-grained resolution in the 0-to-5th-percentile transition zone). Sensitivity analysis shows a sharp, abrupt transition from 282 clusters (eps=0) to 3-4 clusters (eps≥1.5) with no stable middle ground. See FINDINGS_AND_DECISIONS.md Stage 4 section for full results.

---

### Q4: Should indicator frequency (count in dataset) be used as a feature?

**The question:** "In" appears 1,487 times in the dataset; many indicators appear only once. Should the frequency of an indicator be included as an additional feature alongside its embedding?

**Options:**
- No — frequency is a property of the dataset, not of the indicator itself, and may just reinforce the class imbalance
- Yes — frequency reflects how commonly accepted/used an indicator is, which may be semantically meaningful
- Use log-frequency as a weighting when computing cluster centroids or when evaluating cluster quality
- Use frequency as a confidence weight (KCT's "confidence-based feature" suggestion)

**Advisor guidance (KCT, Feb 8):** "General principle in ML: you see a feature, and then there's a feature that captures how much of that thing there was... Confidence based feature allows the classifier to understand when it can rely on certain features more." This suggests frequency could be a useful meta-feature.

**Current direction:** Unresolved. Low priority; try after baseline clustering is working.

---

### Q5: Should we do a two-stage clustering (fine-grained then coarse-grained)? *(DEPRIORITIZED)*

**The question:** We noticed that many HDBSCAN clusters seem to group morphologically similar variants of the same indicator (e.g., "contributing", "contributes to", "contributing to"). We wondered whether an initial round of clustering to group these near-synonyms, followed by a second round to group the resulting macro-clusters into wordplay types, would be more effective.

**Options:**
- Two-stage: cluster similar variants first, then cluster macro-clusters
- Single-stage: let the embedding model handle variant similarity and directly cluster into wordplay types
- Use a logistic regression pass first (per KCT) to identify which n-grams are most discriminative per wordplay type, use those as data-derived seeds, then cluster

**Advisor guidance (KCT, Feb 8):** Suggested a logistic regression approach — "regularized logistic regression, one classifier per wordplay, learns to classify that indicator class vs the rest. Maybe 5 lines of code. Easy to interpret. Get a sparse set of weights on those 1,2,3-grams." This could be the basis for the second stage.

**Deprioritized (Feb 21, 2026):** Stage 4 dendrograms and metrics show a continuum of merge distances rather than distinct hierarchical levels, making two-stage clustering (coarse then fine) less motivated. The data does not naturally split into a coarse level followed by a fine level — merge distances increase gradually. Notebook 05's constrained clustering at k=7 and k=34 tests whether expert-defined levels have merit even if they don't emerge naturally. If constrained results are poor, this approach could be revisited.

---

### Q6: Which seed set should be used for constrained clustering?

**The question:** We have five seed sets in Wordplay_Seeds.xlsx (cc_for_dummies_ho_6, cc_for_dummies_ALL, minute_cryptic_ho_7, minute_cryptic_ALL, conceptual_groups). Each corresponds to a different clustering philosophy.

**Options:**
- cc_for_dummies_ho_6: Simple, covers only 6 of 8 Ho types, from a single source
- minute_cryptic_ho_7: Covers 7 Ho types including alternation, more granular subcategories
- conceptual_groups: Most theoretically motivated; seeds organized by conceptual metaphor rather than wordplay type; explicitly acknowledges multi-type membership
- Try multiple seed sets and compare results
- Derive seeds from the data using logistic regression (KCT suggestion) and compare to expert-sourced seeds

**Advisor guidance (KCT, Feb 8):** "Expert knowledge is helpful. Shouldn't matter whether seed words come from the data or outside sources. Could do both approaches: one from experts, one from data. See where they agree."

Seeds do not need to be unique across wordplay types.

**Current direction:** Start with cc_for_dummies_ho_6 for simplicity. Try minute_cryptic_ho_7 for comparison. Use conceptual_groups if targeting a conceptual-metaphor level of clustering.

---

### Q11: What does the k=10 local silhouette optimum represent?

**The question:** Agglomerative clustering at k=10 shows a local silhouette optimum (0.299) — silhouette jumps from 0.272 at k=8, peaks at k=10, then drops to 0.281 at k=11 before resuming the monotonic upward trend. This is the only evidence for any coarse-level structure in the data. Is this an interpretable grouping or an artifact of embedding geometry?

**Steps needed:**
1. When labels are applied in Notebook 05/06, inspect the 10 agglomerative clusters to see if they correspond to any interpretable taxonomy
2. Check whether the 10 clusters map to a meaningful grouping (e.g., the k=4 operation-type level plus some splits, or a partial separation of the 8 wordplay types)
3. Compare the k=10 cluster boundaries to the centroid dendrogram to understand which merges are being avoided at k=10 that k=8 forces

**Current direction:** Open. Low priority until label-based evaluation in Notebook 05/06. If k=10 maps to a meaningful taxonomy, it becomes a candidate for the "best coarse clustering" in the report.

---

## Evaluation

### Q7: How do we evaluate clusters when there is no clean ground truth?

**The question:** The labeled wordplay types are not a clean ground truth because (a) indicators can belong to multiple types, (b) the labels come from blog parsing and have noise, (c) the "correct" number of clusters is itself a research question.

**Options:**
- Intrinsic metrics only: silhouette score, Davies-Bouldin, Calinski-Harabasz
- Extrinsic evaluation: use cluster labels as features in a downstream task and measure whether richer cluster structure improves performance
- Qualitative evaluation: extract representative/centroid indicators from each cluster and have team members interpret them
- Visualizations: 2D UMAP scatter plots, colored by cluster and/or by known wordplay type

**Advisor guidance (KCT, Feb 8):** Task-based evaluation of cluster quality — the value of a clustering can be assessed by whether it reveals meaningful structure when you examine representative members of each cluster and compare them to known wordplay types.

**Required (per course rubric):** At least two visualizations per unsupervised method. Sensitivity analysis on the best model.

**Current direction:** Use all of the above. Intrinsic metrics provide a quick comparison; qualitative inspection and visualization are necessary for the report narrative.

---

### Q8: How should we investigate and report noise points?

**The question:** HDBSCAN threw out 4,212 points (33.4%) as noise at eps=0. Are these genuinely anomalous indicators, or artifacts of the clustering parameters?

**Steps needed:**
1. Examine the noise point indicators qualitatively — are they unusual words, multi-word phrases, foreign words?
2. Vary epsilon and min_cluster_size to see how much noise changes
3. Try DBSCAN (which does not have the hierarchical structure of HDBSCAN) to see if it absorbs some noise
4. Report noise as a finding, not a problem to hide

**Current direction:** Still open. Must address before results are final.

**Evidence (Feb 21, 2026):** The Stage 4 epsilon sweep shows noise drops rapidly with increasing epsilon (33.4% at eps=0 → 10.3% at eps=0.43 → 0.9% at eps=0.78 → 0% at eps=1.07), but silhouette also drops sharply (0.631 → -0.118 → -0.186 → -0.120). Many noise points form visible clumps in the 2D UMAP projection but fall below min_cluster_size=10 or lack sufficient density in 10D. Investigation needed in Notebook 05/06: are noise points genuinely ambiguous indicators, multi-label indicators that don't belong to any single type, or rare/unusual vocabulary? Understanding the noise is important for the report narrative.

---

## Data Representation

### ~~Q9: How should ground-truth labels be output from the data cleaning stage?~~

**RESOLVED (Feb 19, 2026).** See FINDINGS_AND_DECISIONS.md. Output is `verified_clues_labeled.csv` with one row per verified (clue_id, indicator) pair. Includes both Ho blog labels and algorithmic ground truth. A per-unique-indicator summary can be derived via groupby.

---

### Q10: Should we use WordNet in the embedding or clustering pipeline?

**The question:** WordNet is a high-precision lexical graph. Several integration approaches have been discussed for clustering specifically.

**Options:**
- Use WordNet synsets to expand seed words (add synonyms/hypernyms to each seed list before constrained clustering)
- Use SenseBERT or ARES (models that integrate WordNet into the embedding space) instead of BGE-M3
- Use PyKEEN for graph-based path distance as a complement to cosine similarity when evaluating cluster coherence

**Advisor guidance (KCT, Feb 1):** Discussed WordNet as a high-precision tool for understanding word relationships. For clustering, the most relevant application is seed word expansion — using WordNet to grow a small expert seed list into a richer set of related indicators.

**Current direction:** Low priority. Start with BGE-M3 embeddings and expert seed lists. Consider WordNet expansion of seeds only if constrained clustering results are poor.
