# Results: 05 — Model Evaluation (Pre-Hypothesis Testing)

**Date:** 2026-04-14  
**Environment:** Local

## Versions

- pandas:     3.0.0
- numpy:      2.3.5
- scipy:      1.17.0
- matplotlib: 3.10.8
- seaborn:    0.13.2

## Scope

Canonical mean-pooling models only: `g_stock` and `g1`. The `_tokenspan` variants are out of scope per Decision 20.

## §2 — Validation triplet accuracy

Validation triplet file: `data/triplets/g1_val.csv` (47,933 rows).
- anchors resolved:   47,933 / 47,933
- positives resolved: 47,933 / 47,933
- negatives resolved: 28,775 / 47,933
- all three resolved: 27,348 / 47,933 (57.1%) — used for the accuracy table below

Note: 19,158 distractor_wn values are absent from vocabulary_wndef_val.csv (validation-split wndef vocabulary). Distractors are drawn from the full WordNet vocabulary; only those also present in the validation split can be resolved to a g1 / g_stock validation embedding. See DECISIONS.md Decision 21 and `planning/questions/05_model_evaluation-questions.md`.

**Bias caveat** (per DECISIONS.md Decision 21): the surviving triplets' negatives are words that also appear as definitions or answers in validation clues — i.e., common crossword words. Whether these are systematically easier or harder negatives than the dropped distractors is unknown. However, the comparison between `g_stock` and `g1` is computed on the identical set of triplets, so any difficulty bias affects both models equally and does not compromise the `g_stock`-vs-`g1` comparison.

| Metric | g_stock | g1 |
|---|---|---|
| Triplet accuracy (% correct) | 39.5934 | 88.5220 |
| Mean margin (cos_pos - cos_neg) | -0.0509 | 0.1163 |
| Median margin | -0.0424 | 0.1128 |
| % triplets with margin > 0.1 | 19.1934 | 55.2399 |
| % triplets with margin > 0.5 | 0.2011 | 0.0037 |
| N validation triplets evaluated | 27348.0000 | 27348.0000 |

Figure: `outputs/figures/05_val_triplet_accuracy.png`

## §3a — Mean pairwise cosine among random word pairs

Sampled 50,000 random distinct-row pairs per (model, phrase) with random_state=42. Same pairs used for both models within a phrase type.

| Model | Phrase type | Mean | Median | Std | P5 | P95 |
|---|---|---|---|---|---|---|
| g_stock | f_common_wndef_val | 0.3976 | 0.3876 | 0.1181 | 0.2211 | 0.6097 |
| g1 | f_common_wndef_val | 0.5708 | 0.5713 | 0.0744 | 0.4474 | 0.6926 |
| g_stock | f_common_wnex_val | 0.2994 | 0.2807 | 0.1292 | 0.1187 | 0.5435 |
| g1 | f_common_wnex_val | 0.5055 | 0.5044 | 0.0648 | 0.4012 | 0.6139 |

## §3b — Embedding variance and effective dimensionality

| Model | Phrase type | Total var | Eff. dim | Top-10 var % | Top-50 var % | Top-100 var % |
|---|---|---|---|---|---|---|
| g_stock | f_common_wndef_val | 13454020.00 | 43.59 | 38.10 | 64.03 | 75.12 |
| g1 | f_common_wndef_val | 8720421.00 | 48.68 | 33.00 | 67.84 | 81.72 |
| g_stock | f_common_wnex_val | 1943473.12 | 47.67 | 34.42 | 60.60 | 72.78 |
| g1 | f_common_wnex_val | 1228944.00 | 77.49 | 26.93 | 58.67 | 74.77 |

Figures: `outputs/figures/05_collapse_pairwise_cosine.png`, `outputs/figures/05_collapse_singular_values.png`

## §4 — T=0 and T=1 similarity distributions

Evaluation pairs assembled from clues_val.csv (47,933 rows):
- dropped: (clue_id, definition) not in f_clue index: 0
- dropped: definition_wn not in vocabulary_wndef_val:   0
- dropped: answer_wn not in vocabulary_wndef_val:       0
- kept:   47,933 (100.0%)

| Distribution | Mean | Median | Std | P5 | P95 |
|---|---|---|---|---|---|
| g_stock T=0 | 0.5762 | 0.5903 | 0.1769 | 0.2828 | 0.8424 |
| g_stock T=1 | 0.5130 | 0.5162 | 0.1640 | 0.2452 | 0.7716 |
| g1 T=0 | 0.7146 | 0.7162 | 0.0698 | 0.5964 | 0.8265 |
| g1 T=1 | 0.5906 | 0.5933 | 0.0707 | 0.4704 | 0.7020 |

ATE preview (deferred to Stage 6):
- g_stock ATE (mean of T=1 - T=0): -0.0632
- g1 ATE      (mean of T=1 - T=0): -0.1240

Figure: `outputs/figures/05_t0_t1_distributions.png`

## §5 — RSA (Spearman correlation of pairwise cosines)

| Phrase type | N words sampled | N pair values | Spearman rho | p-value |
|---|---|---|---|---|
| f_common_wndef_val | 1000 | 499500 | 0.112159 | 0.000000 |
| f_common_wnex_val | 1000 | 499500 | 0.074685 | 0.000000 |
