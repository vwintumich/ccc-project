# Results: 05 — Model Evaluation (Pre-Hypothesis Testing)

**Date:** 2026-04-19  
**Environment:** Local

## Versions

- pandas:     3.0.0
- numpy:      2.3.5
- scipy:      1.17.0
- matplotlib: 3.10.8
- seaborn:    0.13.2

## Scope

Canonical mean-pooling models only: `g_stock` and `g1`. The `_tokenspan` variants are out of scope per Decision 20.

Triplet accuracy (§2, §3) uses full-vocabulary wndef and wnex embeddings. Collapse (§4), T=0/T=1 (§5, §6), and RSA (§7) use validation-only embeddings (Decision 9 model-selection discipline).

## §2 — Validation triplet accuracy (wndef, full vocabulary)

Validation triplet file: `data/triplets/g1_val.csv` (46,506 rows). Resolved against `vocabulary_wndef.csv` (53,930 words).
- anchors resolved:   46,506 / 46,506
- positives resolved: 46,506 / 46,506
- negatives resolved: 46,506 / 46,506
- all three resolved: 46,506 / 46,506 (100.0%) — used for the accuracy table below

| Metric | g_stock | g1 |
|---|---|---|
| Triplet accuracy (% correct) | 38.7778 | 89.9583 |
| Mean margin (cos_pos - cos_neg) | -0.0540 | 0.1249 |
| Median margin | -0.0441 | 0.1215 |
| % triplets with margin > 0.1 | 17.9848 | 58.7236 |
| % triplets with margin > 0.5 | 0.1634 | 0.0108 |
| N validation triplets evaluated | 46506.0000 | 46506.0000 |

Figure: `outputs/figures/05_val_triplet_accuracy.png`

## §3 — Cross-f triplet accuracy (matched comparison)

Matched comparison: same triplet subset evaluated under wndef and wnex embeddings. A triplet enters the matched set if both its `answer_wn` and `distractor_wn` are present in `vocabulary_wnex.csv`.

Wnex resolution breakdown across 46,506 val triplets:
- both answer and distractor in wnex: 2,985 (6.4%) — the matched subset
- answer in wnex, distractor not:     6,994 (15.0%)
- distractor in wnex, answer not:     5,694 (12.2%)
- neither in wnex:                    30,833 (66.3%)

| Metric | g_stock (wndef) | g1 (wndef) | g_stock (wnex) | g1 (wnex) |
|---|---|---|---|---|
| Triplet accuracy (% correct) | 45.5946 | 88.3082 | 40.2680 | 67.2027 |
| Mean margin (cos_pos - cos_neg) | -0.0227 | 0.1085 | -0.0516 | 0.0413 |
| Median margin | -0.0171 | 0.1102 | -0.0421 | 0.0414 |
| % triplets with margin > 0.1 | 21.7420 | 53.5008 | 19.6650 | 26.6332 |
| N triplets | 2985.0000 | 2985.0000 | 2985.0000 | 2985.0000 |

Figure: `outputs/figures/05_crossf_triplet_accuracy.png`

## §4a — Mean pairwise cosine among random word pairs (val-only)

Sampled 50,000 random distinct-row pairs per (model, phrase) with random_state=42. Same pairs used for both models within a phrase type.

| Model | Phrase type | Mean | Median | Std | P5 | P95 |
|---|---|---|---|---|---|---|
| g_stock | f_common_wndef_val | 0.3976 | 0.3876 | 0.1181 | 0.2211 | 0.6097 |
| g1 | f_common_wndef_val | 0.5708 | 0.5713 | 0.0744 | 0.4474 | 0.6926 |
| g_stock | f_common_wnex_val | 0.2994 | 0.2807 | 0.1292 | 0.1187 | 0.5435 |
| g1 | f_common_wnex_val | 0.5055 | 0.5044 | 0.0648 | 0.4012 | 0.6139 |

## §4b — Embedding variance and effective dimensionality (val-only)

| Model | Phrase type | Total var | Eff. dim | Top-10 var % | Top-50 var % | Top-100 var % |
|---|---|---|---|---|---|---|
| g_stock | f_common_wndef_val | 13454020.00 | 43.59 | 38.10 | 64.03 | 75.12 |
| g1 | f_common_wndef_val | 8720421.00 | 48.68 | 33.00 | 67.84 | 81.72 |
| g_stock | f_common_wnex_val | 1943473.12 | 47.67 | 34.42 | 60.60 | 72.78 |
| g1 | f_common_wnex_val | 1228944.00 | 77.49 | 26.93 | 58.67 | 74.77 |

Figures: `outputs/figures/05_collapse_pairwise_cosine.png`, `outputs/figures/05_collapse_singular_values.png`

## §5 — T=0 and T=1 similarity distributions (wndef, val-only)

Evaluation pairs assembled from clues_val.csv (47,933 rows):
- dropped: (clue_id, definition) not in f_clue index: 0
- dropped: definition_wn not in vocabulary_wndef_val:  0
- dropped: answer_wn     not in vocabulary_wndef_val:  0
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

Figure: `outputs/figures/05_t0_t1_wndef_distributions.png`

## §6 — T=0 and T=1 similarity distributions (wnex, val-only)

Evaluation pairs assembled from clues_val.csv (47,933 rows):
- rows where definition_wn is in wnex vocab: 13,227 (27.6%)
- rows where answer_wn     is in wnex vocab: 7,734 (16.1%)
- rows where BOTH are in wnex vocab (kept):  4,825 (10.1%)

| Distribution | Mean | Median | Std | P5 | P95 |
|---|---|---|---|---|---|
| g_stock T=0 | 0.4946 | 0.4885 | 0.1938 | 0.1896 | 0.8103 |
| g_stock T=1 | 0.4855 | 0.4896 | 0.1755 | 0.2007 | 0.7592 |
| g1 T=0 | 0.5901 | 0.5901 | 0.0807 | 0.4598 | 0.7230 |
| g1 T=1 | 0.5474 | 0.5481 | 0.0708 | 0.4319 | 0.6625 |

ATE preview (deferred to Stage 6):
- g_stock ATE (mean of T=1 - T=0): -0.0091
- g1 ATE      (mean of T=1 - T=0): -0.0427

Note: the wndef ATE in §5 and the wnex ATE here are computed on different (overlapping but not identical) subsets of clues_val. §6d below makes the comparison on a matched subset.

Figure: `outputs/figures/05_t0_t1_wnex_distributions.png`

### §6d — Matched ATE comparison (wndef vs wnex on identical pairs)

Restricted to the 4,825 clues_val pairs that resolve under both wndef_val and wnex_val. The wndef ATE on this matched subset is the cleanest wndef-vs-wnex comparison.

| Phrase format | g_stock ATE | g1 ATE | N pairs |
|---|---|---|---|
| wndef | -0.0706 | -0.1342 | 4825 |
| wnex | -0.0091 | -0.0427 | 4825 |

## §7 — RSA (Spearman correlation of pairwise cosines, val-only)

| Phrase type | N words sampled | N pair values | Spearman rho | p-value |
|---|---|---|---|---|
| f_common_wndef_val | 1000 | 499500 | 0.112159 | 0.000000 |
| f_common_wnex_val | 1000 | 499500 | 0.074685 | 0.000000 |
