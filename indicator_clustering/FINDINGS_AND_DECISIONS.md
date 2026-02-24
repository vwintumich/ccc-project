# FINDINGS_AND_DECISIONS.md — Empirical Results and Advisor Guidance

This document is append-only. Add new findings with dates. Do not rewrite or delete earlier entries.
Older entries at the bottom reflect earlier states of understanding that may have been superseded.

---

## February 20, 2026 — Dimensionality Reduction (Notebook 03)

**FINDING: PCA captures very little structure in BGE-M3 embeddings.**
The scree plot shows no sharp elbow. The top principal component explains only ~4% of variance, 10 components explain 22.3%, and 100 components reach only ~70%. This confirms the embedding space is highly nonlinear and PCA is inadequate as a primary reduction method.

**FINDING: UMAP reveals clear local cluster structure that PCA misses.**
The 2D UMAP projection shows many small dense clumps connected by sparser regions, with isolated clusters at the periphery. The 2D PCA projection shows a diffuse, structureless blob. UMAP with cosine metric is the correct choice for dimensionality reduction before clustering.

**FINDING: UMAP clumps likely reflect morphological variant groups, not wordplay types.**
Because indicators were embedded without stemming or lemmatization (per the settled decision that the embedding model handles morphological variation), variant groups like "contribute to / contributes to / contributing / contributing in / contributing to / contribution from / contribution to / contributors to" receive nearly identical embeddings and form tight local neighborhoods in UMAP space. This explains why earlier HDBSCAN runs found 353 fine-grained clusters — they are likely capturing variant-level groupings rather than wordplay-type-level groupings. This is consistent with OPEN_QUESTIONS.md Q5 (two-stage clustering) and suggests that a second pass may be needed to aggregate variant-level clusters into higher-level wordplay-type clusters.

**FINDING: Overall UMAP shape supports the primary hypothesis.**
The indicator embedding space forms one large connected mass rather than 8 cleanly separated islands. This is consistent with the hypothesis that indicators will not cluster cleanly into 8 wordplay types, but meaningful local structure exists for density-based methods to discover.

**SETTLED: Dimensionality reduction parameters.**
UMAP: n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42. These are starting values and have not been systematically tuned (see OPEN_QUESTIONS.md). PCA is retained as a baseline comparison only.

### Stage 4: Clustering (Notebook 04 — Unconstrained Exploration)

**Research question:** "What structure emerges when we let the algorithms find it without guidance?"

**Methods:** HDBSCAN with epsilon sensitivity sweep (13 epsilon values from 0.0 to 2.68); agglomerative clustering with Ward's linkage across 17 k values (k=4 to k=250); Ward's linkage dendrograms (truncated full-data and centroid-based). No labels or domain knowledge were used in this notebook. Wordplay type labels (Ho and GT) and seed words are introduced in Notebook 05.

#### Key Findings

**FINDING 1: HDBSCAN — No stable intermediate level exists.**
The refined epsilon sweep (with 6 additional candidates filling the 0-to-5th-percentile gap) confirms an abrupt transition. At eps=0: 282 clusters, 33.4% noise, silhouette 0.631. By eps=0.43: 62 clusters, 10.3% noise, silhouette -0.118. By eps=0.78: 11 clusters, 0.9% noise, silhouette -0.186. There is no plateau — no epsilon value produces a moderate number of clusters (8-20) with both low noise and positive silhouette. The transition from fine-grained to collapsed is genuinely abrupt. The high silhouette at eps=0 is partly an artifact of discarding 33% of points as noise (per KCT's warning). HDBSCAN grades itself only on the easy cases.

**FINDING 2: Agglomerative — Metrics improve monotonically with k; no elbow.**
Silhouette score rises from 0.246 (k=4) to 0.431 (k=250) with no plateau. Davies-Bouldin falls from 1.456 to 0.884 over the same range. The data's natural structure is finer than any k we tested. There is no single "correct" number of clusters. Exception: k=10 is a local silhouette optimum. Silhouette jumps from 0.272 (k=8) to 0.299 (k=10), then drops to 0.281 (k=11) before resuming the upward trend. This suggests that k=10 carves along natural boundaries more cleanly than its neighbors. This is the only evidence for any coarse-level structure in the data. Calinski-Harabasz peaks near k=6 and declines with larger k — this reflects the metric's known bias toward fewer clusters, not genuine evidence for coarse structure.

**FINDING 3: Dendrograms — No natural cut at k=8.**
The truncated full-data dendrogram shows a strong 2-way split at the top (final merge distance ~225 vs. penultimate ~140), a rough 3-4 way split around distance 115-140, and densely packed fine-grained merges below distance 50. There is no gap in merge distances corresponding to 8 clusters. The k=34 centroid dendrogram shows clear sibling pairs (clusters merging at distances 1.0-2.0), larger sub-groups forming around distance 3-5, and a major split around distance 10. The gradual increase in merge distances with no dramatic gap confirms that the data does not naturally organize into a small number of well-separated groups.

**FINDING 4: Qualitative inspection — Fine-grained clusters are semantically coherent.**
At k=8, clusters are broad mixtures — centroids pull together indicators from diverse semantic themes. At k=10, clusters become slightly more focused. At k=34 and k=250, centroid-nearest indicators share recognizable conceptual themes (movement/mixing words, hearing/speaking words, containment/placement words). This is consistent with the hypothesis that the natural granularity is at the conceptual metaphor level. HDBSCAN clusters at eps=0 show similar coherence: the 15 largest clusters include groups centered on "bringing back / return" (reversal), "reinvented / rebuilt" (anagram-change), "concealed / hidden" (hidden/container), "going up / rising" (reversal-upward), "being heard / listened" (homophone).

#### Settled Decisions from Stage 4

**SETTLED: No natural k=8 grouping exists.** Neither HDBSCAN nor agglomerative clustering recovers 8 wordplay types without guidance. This is a finding, not a failure.

**SETTLED: k=10 is the local coarse optimum** for agglomerative clustering (silhouette-based).

**SETTLED: Silhouette and Davies-Bouldin improve monotonically with finer k** — metrics alone cannot select a preferred granularity.

**SETTLED: Calinski-Harabasz favors low k due to structural bias** — do not use it to argue for coarse granularity.

**SETTLED: The dendrogram shows no clean cut point** at any k value — merge distances increase gradually.

**SETTLED: Best HDBSCAN run by silhouette is eps=0** (282 clusters, 33.4% noise) — but this is the most aggressive run that excludes the most points.

**SETTLED: Labels and domain knowledge are deferred to Notebook 05** for constrained clustering and targeted experiments.

**SETTLED: Pairwise distance analysis procedure.**
Computed Euclidean pairwise distances on 2,000 random samples from 10D UMAP embeddings. Median distance 3.55, range 0.003-8.51. Epsilon candidates selected from percentiles of this distribution, with 6 fine-grained candidates in the 0-to-5th-percentile transition zone plus 6 coarse candidates from 5th-25th percentiles. This procedure follows KCT's requirement (Feb 15) for principled epsilon selection.

### Stage 5: Constrained and Targeted Experiments (Notebook 05)

**Research question:** "Does expert knowledge improve clustering, and do theoretically motivated subsets behave as predicted?"

**Methods:** Label-based evaluation of NB 04 results (Ho and GT overlays, per-cluster type distribution heatmaps); constrained agglomerative clustering with seed-word connectivity matrices (MC7 k=7 and CG34 k=34); subset experiments (homophone vs. reversal, hidden+container+insertion, anagram sub-clustering).

**This is the first notebook where domain knowledge enters the clustering pipeline.**

#### Key Findings

**FINDING 1: The Ho type overlay is the most informative visualization in the project.**

Spatial distribution of types in UMAP space reveals a clear hierarchy of separability:
- **Homophone** is the most concentrated type — a tight cluster in the bottom-right of the UMAP projection, well-separated from everything else.
- **Reversal** forms several tight sub-clusters in the left half, moderately concentrated.
- **Container and insertion** have nearly identical spatial distributions — both spread across the center-right, thoroughly intermixed with each other. This visually confirms they share indicator vocabulary.
- **Hidden** partially overlaps with container/insertion (shared placement metaphors) but also has a few distinct patches.
- **Anagram** blankets the entire space (51% of indicators), with internal sub-structure corresponding to different conceptual metaphors.
- **Deletion** concentrates in the upper-center, partially overlapping with anagram.
- **Alternation** is sparse and scattered (only 216 indicators).

**FINDING 2: Unconstrained clusters do NOT correspond to wordplay types.**

Per-cluster Ho type distribution heatmaps confirm:
- At k=8: average purity 0.563. Five of 8 clusters are anagram-dominated. No cluster cleanly captures homophone, hidden, or alternation.
- At k=10: average purity 0.655. The two extra clusters help — one captures homophone (0.78 purity) and one captures reversal (0.90), but the remaining 8 are still mixed.
- At k=34: average purity 0.652. Multiple clusters per type, but anagram still dominates many clusters. Some clean captures: reversal clusters at 0.96 and 0.90 purity, homophone at 0.78.
- HDBSCAN eps=0.0: average purity 0.750 (20 largest clusters). Fine-grained clusters are purer, but 282 clusters is far more than the 8-type taxonomy.

**FINDING 3: Constrained clustering provides marginal improvement at best.**

| Run | k | Seeds | Silhouette | Davies-Bouldin | Avg Purity |
|-----|---|-------|-----------|----------------|------------|
| Unconstrained | 8 | None | 0.272 | 1.267 | 0.563 |
| Constrained MC7 | 7 | minute_cryptic | 0.264 | 1.211 | 0.598 |
| Unconstrained | 34 | None | 0.322 | 1.068 | 0.652 |
| Constrained CG34 | 34 | conceptual_groups | 0.324 | 1.039 | 0.671 |

MC7 (k=7): The constrained run is slightly better on purity (0.598 vs 0.563) but slightly worse on silhouette. Cluster 1 (n=2,929) became a catch-all absorbing homophone, hidden, and insertion indicators. Only cluster 5 achieved a non-anagram dominant type (container+insertion at 0.46/0.34). The seeds were too few (82 matched) to overcome anagram's 51% base rate. sklearn issued a warning about 13 disconnected components in the connectivity matrix — some seed groups were isolated from the main kNN graph.

CG34 (k=34): Marginal improvement across all metrics. The conceptual metaphor groupings are compatible with the embedding geometry but do not dramatically improve upon what unconstrained Ward's already discovers. This is actually a positive finding: the BGE-M3 embeddings naturally organize indicators by conceptual metaphor without needing expert guidance.

**FINDING 4: The 4A/4B ARI contrast is the strongest quantitative result.**

| Experiment | Method | ARI vs Ho | Silhouette |
|-----------|--------|----------|-----------|
| 4A: Homophone vs Reversal | Agglomerative k=2 | **0.611** | 0.394 |
| 4B: Hidden+Container+Insertion | Agglomerative k=3 | **0.045** | 0.213 |

The 13.5x ratio between 4A and 4B ARI confirms the conceptual metaphor prediction: types with distinct metaphorical bases (hearing vs. direction) separate cleanly, while types sharing placement/containment metaphors are inseparable. This is the single most citable finding for the report.

The 4B scatter plot visually confirms total intermixing — the k=3 clusters carve spatial regions, but those regions do not correspond to the three Ho types. The clusters instead organize by sub-metaphor (likely surrounding vs. consuming vs. segment/piece vocabulary).

**FINDING 5: Anagram has rich conceptual sub-structure.**

HDBSCAN found 149 sub-clusters within the 6,610 anagram indicators (36.8% noise). Qualitative inspection of the largest sub-clusters reveals clear conceptual metaphor themes:
- Cluster 21 (n=256): "reinvented", "redesign", "rebuilt" → **repair**
- Cluster 114 (n=91): "strangely", "bizarrely", "peculiar" → **incorrectness**
- Cluster 123 (n=85): "waltzing", "rippling", "swashbuckling" → **movement**
- Cluster 143 (n=83): "dreadfully", "awful", "abominable" → **disorder/badness**
- Cluster 78 (n=50): "cooks", "toast", "cooked" → **tamper**
- Cluster 23 (n=64): "changes", "change of", "to alter" → **transformation**
- Cluster 43 (n=50): "hybrid", "blend of", "mixture" → **mixing**

Agglomerative k=8 on the anagram subset produced interpretable sub-clusters with silhouette 0.316 — comparable to the full-data k=34 silhouette (0.322).

This confirms that the conceptual metaphor hierarchy from DOMAIN_KNOWLEDGE.md is not just a theoretical framework but an empirically observable organizing principle in the embedding space.

#### Settled Decisions from Stage 5

**SETTLED: The 8-type wordplay taxonomy cannot be fully recovered by clustering on indicator semantics alone.** Container, insertion, and hidden are inseparable (ARI=0.045). The best achievable taxonomy has at most 6 distinguishable groups (merging these three).

**SETTLED: Constrained clustering with seed words provides marginal benefit.** Seeds are compatible with the natural structure but mostly redundant. The BGE-M3 embeddings already capture the relevant semantic organization.

**SETTLED: Homophone and reversal are the most separable types.** ARI=0.611 for the binary homophone/reversal experiment.

**SETTLED: Anagram indicators organize by conceptual metaphor internally.** Sub-clustering reveals repair, incorrectness, movement, disorder, tamper, transformation, and mixing themes — validating the conceptual_groups taxonomy.

**SETTLED: k=10 is interpretable.** The two extra clusters (vs k=8) capture homophone and reversal, which are the two most spatially concentrated types. This explains the local silhouette spike at k=10 observed in NB 04.

---

## February 8, 2026 — Current State of Findings

### Data Cleaning

**SETTLED: Verification method (Victoria's checksum)**
An indicator is verified if the intact phrase appears in BOTH the blog commentary AND the clue text. This filters misparsings and leaves 14,196 verified unique indicators (down from 15,735 unique). This is the canonical dataset for all downstream work.

**SETTLED: No stemming or lemmatization before embedding**
KCT confirmed (Feb 1 meeting): "The right embeddings will take care of that for you." Do not reduce indicators to word stems before embedding. The BGE-M3 model will handle morphological variation in semantic space.

**SETTLED: Multi-word expressions (MWE) are acceptable**
It is okay to keep multi-word indicator phrases. Do not reduce to single-word indicators before embedding. KCT confirmed that passing 1-, 2-, and 3-gram indicators into the sentence transformer is appropriate.

**SETTLED: No domain-knowledge-based filtering**
When cleaning data, use only objective, algorithmic criteria. Subjectively rejecting indicators because they "seem wrong" introduces bias. KCT: "The better way is to use external cues / sources of evidence to justify removing something."

**SETTLED: Ground-truth labels output format (Q9 resolved, Feb 19)**
`verified_clues_labeled.csv` contains one row per verified (clue_id, indicator) pair (76,015 rows). Columns: `clue_id`, `indicator`, `wordplay_ho` (Ho blog label), `wordplay_gt` (algorithmic ground truth using priority: hidden > reversal > alternation > anagram, null if no pattern fires or answer < 4 letters), `wordplay_gt_all` (all patterns that fired), `answer_letter_count`, `label_match` (Ho vs GT agreement). Multi-label indicators appear as multiple rows with different `wordplay_ho` values (310 such pairs). Label match rate is 92.6% where GT exists (19,803 of 76,015 rows). A per-unique-indicator summary can be derived via groupby.

**SETTLED: Deduplicated indicator list for embedding input (Feb 19)**
`verified_indicators_unique.csv` contains one row per unique indicator string (12,622 rows). No wordplay labels — just the indicator column. This is the sole input to Stage 2 embedding generation. Produced by `01_data_cleaning_Victoria.ipynb`.

**SETTLED: Pipeline output files (Feb 19)**

| Stage | File | Rows / Shape | Description | Produced by |
|-------|------|-------------|-------------|-------------|
| 0 | `clues_raw.csv` | 660,613 | All clues with answer, definition, source URL | `00_data_extraction_Victoria.ipynb` |
| 0 | `indicators_raw.csv` | 15,735 | Unique indicators with wordplay type and clue IDs | `00_data_extraction_Victoria.ipynb` |
| 0 | `indicators_by_clue_raw.csv` | 93,867 | One row per clue with indicator columns | `00_data_extraction_Victoria.ipynb` |
| 0 | `indicators_consolidated_raw.csv` | 1 | All indicators per wordplay type (newline-separated) | `00_data_extraction_Victoria.ipynb` |
| 0 | `charades_raw.csv` | — | Charade components with clue IDs | `00_data_extraction_Victoria.ipynb` |
| 0 | `charades_by_clue_raw.csv` | — | One row per clue-charade pair | `00_data_extraction_Victoria.ipynb` |
| 1 | `verified_indicators_unique.csv` | 12,622 | Deduplicated unique indicator strings (sole input to Stage 2) | `01_data_cleaning_Victoria.ipynb` |
| 1 | `verified_clues_labeled.csv` | 76,015 | One row per (clue_id, indicator) pair with Ho + GT labels | `01_data_cleaning_Victoria.ipynb` |
| 2 | `embeddings_bge_m3_all.npy` | (12622, 1024) | BGE-M3 embeddings for all unique indicators | `02_embedding_generation_Victoria.ipynb` |
| 2 | `indicator_index_all.csv` | 12,622 | Row number → indicator string mapping for the .npy | `02_embedding_generation_Victoria.ipynb` |

Row `i` in `indicator_index_all.csv` corresponds to row `i` in `embeddings_bge_m3_all.npy`. To get labels for an indicator, join `indicator_index_all.csv` with `verified_clues_labeled.csv` on the `indicator` column.

### Embeddings

**SETTLED: Primary embedding model is BAAI/bge-m3**
This is the BGE-M3 model from SentenceTransformer. It produces 1024-dimensional embeddings. It is a CALE-family model (Concept-Aligned Language Embeddings) — CALE is not a specific tool but a family of models trained to distinguish word senses contextually.

**SETTLED: CALE is an approach, not a specific tool**
KCT (Feb 8 meeting): "CALE is a term for a family of pretrained models to distinguish different senses of words... It's not a specific tool, but a family of models pretrained to distinguish by sense." BGE-M3 qualifies as a CALE-family model. KCT recommended: `model = SentenceTransformer('all-mpnet-base-v2')` as a solid alternative.

**SETTLED: CALE-style contextualization approach**
To use CALE-style contextualization on an indicator within its clue context, surround the target word with asterisks and pass the whole sentence:
```python
highlighted_text = sentence.replace(target_word, f"*{target_word}*")
embedding = model.encode(highlighted_text)
```
This causes the model to use context to correctly disambiguate the target word.

**OPEN: Whether to embed indicators with clue context or in isolation**
Embedding an indicator in isolation captures its general semantic meaning. Embedding it within its clue context (using the CALE asterisk technique) captures its contextually disambiguated meaning. It is unresolved which approach produces better clustering. See OPEN_QUESTIONS.md.

**OPEN: Whether frequency/count of an indicator should be used as a feature**
KCT (Feb 8): "General principle in ML: you see a feature, and then there's a feature that captures how much of that thing there was... Confidence based feature allows the classifier to understand when it can rely on certain features more." The frequency with which an indicator appears in the dataset could be included as an additional feature alongside the embedding. This has not been tried yet.

### Dimensionality Reduction

**SETTLED: Purpose of each reduction stage**
Dimensionality reduction serves three distinct purposes in this pipeline:
1. **Before clustering:** Reduce noise and improve clustering performance (UMAP to ~10 dimensions is common)
2. **For visualization:** Reduce to 2D for scatter plots (UMAP or t-SNE; UMAP is generally preferred for preservation of global structure)
3. **For computational tractability:** Large embedding matrices are expensive; reduction speeds up downstream steps

**SETTLED: Embed unique indicators only, without clue context**
Contextualized embeddings (indicator embedded within its clue surface using the CALE asterisk technique) were explored and ruled out. Two reasons: (1) embedding an indicator within a specific clue instance ties the embedding to that instance rather than to the indicator as a general token, making it impossible to have a single canonical embedding per unique indicator; (2) the correct working dataset is the 14,196 unique verified indicators, not the 90,000+ instance-level rows. Do not use any notebook or embedding file that was generated from the instance-level dataset or with clue context. The BGE-M3 embeddings must be computed on unique indicators only, in isolation.
PCA is a linear method and may not preserve the nonlinear structure of semantic embeddings. UMAP preserves both local and global structure better for high-dimensional embeddings. Use PCA for comparison/baseline only.

**OPEN: UMAP parameters (n_neighbors, min_dist, n_components)**
These have not been systematically tuned. The current pipeline uses 10 dimensions for clustering input and 2 dimensions for visualization. See OPEN_QUESTIONS.md.

### Clustering Results to Date

**FINDING (Nathan, single-word indicators, Feb 8):**
Comparison of three methods on single-word indicators:

| Method | Clusters Found | Silhouette | Davies-Bouldin |
|--------|---------------|------------|----------------|
| KMeans (best k=50) | 50 | 0.033 | 3.995 |
| Hierarchical (best k=5) | 5 | 0.039 | 2.319 |
| DBSCAN | 18 | 0.754 | 0.437 |

DBSCAN dramatically outperforms KMeans and hierarchical clustering. 18 clusters were found automatically — more than the 8 labeled types, fewer than 353. However: KCT noted that DBSCAN's high silhouette score may partly reflect leaving out too many noise points rather than truly finding better clusters. Examine noise points before concluding the score reflects genuine structure.

**FINDING (Victoria, all 14,196 verified indicators, Feb 8):**
Pipeline: BGE-M3 embeddings (1024-dim) → UMAP reduction (10-dim) → HDBSCAN clustering

| Metric | Result |
|--------|--------|
| Clusters found | 353 |
| Silhouette score | 0.304 |
| Noise points | 4,076 (29% of data) |

Example semantically coherent clusters:
- Cluster 192 (reversal): "yields up", "when turning up", "when rising", "when climbing"
- Cluster 141 (anagram): "being reorganised", "following reorganisation", "is redesigned"
- Cluster 327 (anagram): "cheeky", "clumsily", "cranky", "crazily"
- Tightest clusters by cohesion: "filling for" (0.004), "buried in" (0.005)

The 353 clusters are semantically coherent but far more granular than the 8 labeled wordplay types. Interpreting these as fine-grained conceptual metaphor groupings (rather than wordplay types) may be the right framing.

**OPEN: What granularity of clusters is most useful?**
KCT posed this directly: "18 clusters (DBSCAN) vs 353 (HDBSCAN) vs 8 labeled types — what granularity is most useful?" This is a central research question. See OPEN_QUESTIONS.md.

---

## Advisor Guidance: Clustering Methods

**KCT on HDBSCAN epsilon tuning (Feb 15 meeting):**
"Did you try different values of epsilon to get the best silhouette score? Results are very sensitive to epsilon. Take a look at typical pairwise distances, get a distribution, pick a value of epsilon based on that. Try different values of epsilon. Results can vary by scaling."

Procedure to follow before running HDBSCAN:
1. Compute pairwise distances for a sample of embeddings
2. Plot the distribution to understand the scale
3. Choose epsilon candidates based on the distribution (e.g., 10th, 25th, 50th percentiles)
4. Run HDBSCAN with multiple epsilon values and compare silhouette scores
5. Do a sensitivity analysis: if clustering changes dramatically with small epsilon changes, the structure is fragile

**KCT on hierarchical (agglomerative) clustering (Feb 15 meeting):**
"Use Ward's method. It uses more global statistics about the cluster, so doesn't form strange elongated clusters."

For constrained clustering with seed words, use the connectivity matrix approach:
- Seed words that should be in the same cluster → "must link" constraint
- Words that must NOT be in the same cluster → "cannot link" constraint
- Not all scikit-learn clustering supports constraints, but hierarchical clustering does
- The connectivity matrix defines which points should be neighbors/in the same cluster
- Ward's method checks the connectivity matrix first and adds a penalty to the merge score as needed

**KCT on cluster granularity (Feb 8 meeting):**
"Starting simple and looking at the 8 types you have data for. Form clusters based on those. Then doing an analysis: which clusters are more tightly focused, which more 'slippery' or vague?"

Recommended approach:
- Phase 1: Cluster embedding vectors. Start with known types. Characterize which are closer/tighter/more diffuse.
- Phase 2: Augment embeddings by adding features, possibly related to the context of the clue.

**KCT on dendrogram interpretation:**
"Hierarchical clustering gives you a tree. Each time something gets merged, it gets merged at a particular distance. You can say 'I want to look at all the clusters at an interesting semantic distance.' Y axis of a dendrogram is the merge distance. If you set the distance big, you get higher level clusters." Use Ward's method and interpret the dendrogram to find a natural cut point.

**KCT on sensitivity analysis:**
"One helpful strategy is to see how stable the clustering is to perturbations in the data. If you change one or two critical parameters, showing a more robust, confident result tells you there's a stable underlying structure."

---

## February 5, 2026 — Earlier Findings

**Victoria's verification breakthrough:** Checking that indicators appear intact in the clue text as well as the blog commentary eliminates misparsings. This produced verified_indicators.csv and verified_indicators_one_word.csv.

**POS tagging exploration (Sahana):** Explored SentenceTransformer with POS tagging — probably not worth the effort. You can take an embedding and weight each part based on POS (noun vs. adjective), but the consensus is the embedding handles this implicitly.

---

## February 1, 2026 — Earlier Findings and Decisions

**KCT on WordNet and graph-based methods:**
WordNet is a high-precision lexical graph. PyKEEN is a Python package that can measure path distance between two words in the graph. This is the graph-based complement to the embedding approach. Primarily useful for seed word expansion — adding synonyms/hypernyms from WordNet to expand a seed list before constrained clustering.
