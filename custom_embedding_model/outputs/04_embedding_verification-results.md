# Results: 04 — Embedding Verification

**Date:** 2026-04-14  
**Environment:** Local

## Versions

- pandas: 3.0.0
- numpy:  2.3.5

## Overall verdict

**PASS**

| Criterion | Result |
|---|---|
| All shapes match expected | PASS |
| No NaN anywhere | PASS |
| No all-zero rows | PASS |
| f_clue indexes consistent | PASS |
| No off-diagonal cell >= 0.999 | PASS |

Max off-diagonal mean cosine across all three matrices: **0.9212**

### Pattern-consistency warnings

- f_clue_val: same-weights/diff-pool pairs disagree (std=0.1114) — ('g_stock', 'g_stock_tokenspan')=0.9212 vs ('g1', 'g1_tokenspan')=0.6983
- f_common_wnex_val: same-weights/diff-pool pairs disagree (std=0.1042) — ('g_stock', 'g_stock_tokenspan')=0.9151 vs ('g1', 'g1_tokenspan')=0.7067

## §2 — Per-file integrity

| model | phrase | shape | shape_ok | has_nan | n_zero | L2_min | L2_mean | L2_max |
|---|---|---|---|---|---|---|---|---|
| g_stock | f_clue_val | 47933 x 1024 | True | False | 0 | 25.2485 | 30.6002 | 33.7204 |
| g_stock | f_common_wndef_val | 26152 x 1024 | True | False | 0 | 23.5365 | 29.1415 | 33.6164 |
| g_stock | f_common_wnex_val | 3008 x 1024 | True | False | 0 | 26.3180 | 30.3338 | 33.2035 |
| g_stock_tokenspan | f_clue_val | 47933 x 1024 | True | False | 0 | 23.9008 | 35.3654 | 38.1381 |
| g_stock_tokenspan | f_common_wndef_val | 26152 x 1024 | True | False | 0 | 23.8445 | 33.9466 | 37.8931 |
| g_stock_tokenspan | f_common_wnex_val | 3008 x 1024 | True | False | 0 | 24.1150 | 35.1279 | 37.9414 |
| g1 | f_clue_val | 47933 x 1024 | True | False | 0 | 25.5613 | 28.9786 | 31.6818 |
| g1 | f_common_wndef_val | 26152 x 1024 | True | False | 0 | 24.0279 | 27.8044 | 30.6785 |
| g1 | f_common_wnex_val | 3008 x 1024 | True | False | 0 | 25.5698 | 28.7370 | 31.0049 |
| g1_tokenspan | f_clue_val | 47933 x 1024 | True | False | 0 | 25.2691 | 36.3807 | 38.5271 |
| g1_tokenspan | f_common_wndef_val | 26152 x 1024 | True | False | 0 | 24.1656 | 31.8432 | 34.8947 |
| g1_tokenspan | f_common_wnex_val | 3008 x 1024 | True | False | 0 | 25.6869 | 35.9666 | 38.3528 |

### f_clue_val index consistency

Reference: `g_stock` (47,933 rows)

| Other model | Matches reference |
|---|---|
| g_stock_tokenspan | True |
| g1 | True |
| g1_tokenspan | True |

## §3 — f_clue_val pairwise cosine (mean)

| model | g_stock | g_stock_tokenspan | g1 | g1_tokenspan |
|---|---|---|---|---|
| g_stock | 1.0000 | 0.9212 | 0.2961 | 0.2747 |
| g_stock_tokenspan | 0.9212 | 1.0000 | 0.2880 | 0.2823 |
| g1 | 0.2961 | 0.2880 | 1.0000 | 0.6983 |
| g1_tokenspan | 0.2747 | 0.2823 | 0.6983 | 1.0000 |

### Per-pair detail

| model_a | model_b | mean | median | min | std |
|---|---|---|---|---|---|
| g_stock | g_stock_tokenspan | 0.9212 | 0.9257 | 0.7347 | 0.0283 |
| g_stock | g1 | 0.2961 | 0.2965 | 0.0661 | 0.0593 |
| g_stock | g1_tokenspan | 0.2747 | 0.2753 | 0.0020 | 0.0664 |
| g_stock_tokenspan | g1 | 0.2880 | 0.2873 | 0.0477 | 0.0592 |
| g_stock_tokenspan | g1_tokenspan | 0.2823 | 0.2821 | -0.0247 | 0.0691 |
| g1 | g1_tokenspan | 0.6983 | 0.7034 | 0.3428 | 0.0529 |

## §4 — f_common_wndef_val pairwise cosine (mean)

| model | g_stock | g_stock_tokenspan | g1 | g1_tokenspan |
|---|---|---|---|---|
| g_stock | 1.0000 | 0.9033 | 0.3913 | 0.3734 |
| g_stock_tokenspan | 0.9033 | 1.0000 | 0.4130 | 0.4285 |
| g1 | 0.3913 | 0.4130 | 1.0000 | 0.8074 |
| g1_tokenspan | 0.3734 | 0.4285 | 0.8074 | 1.0000 |

### Per-pair detail

| model_a | model_b | mean | median | min | std |
|---|---|---|---|---|---|
| g_stock | g_stock_tokenspan | 0.9033 | 0.9074 | 0.7214 | 0.0303 |
| g_stock | g1 | 0.3913 | 0.3918 | 0.1639 | 0.0572 |
| g_stock | g1_tokenspan | 0.3734 | 0.3745 | 0.1437 | 0.0595 |
| g_stock_tokenspan | g1 | 0.4130 | 0.4148 | 0.1187 | 0.0629 |
| g_stock_tokenspan | g1_tokenspan | 0.4285 | 0.4298 | 0.1233 | 0.0707 |
| g1 | g1_tokenspan | 0.8074 | 0.8092 | 0.6593 | 0.0208 |

## §5 — f_common_wnex_val pairwise cosine (mean)

| model | g_stock | g_stock_tokenspan | g1 | g1_tokenspan |
|---|---|---|---|---|
| g_stock | 1.0000 | 0.9151 | 0.3264 | 0.3119 |
| g_stock_tokenspan | 0.9151 | 1.0000 | 0.3173 | 0.3183 |
| g1 | 0.3264 | 0.3173 | 1.0000 | 0.7067 |
| g1_tokenspan | 0.3119 | 0.3183 | 0.7067 | 1.0000 |

### Per-pair detail

| model_a | model_b | mean | median | min | std |
|---|---|---|---|---|---|
| g_stock | g_stock_tokenspan | 0.9151 | 0.9193 | 0.7835 | 0.0287 |
| g_stock | g1 | 0.3264 | 0.3277 | 0.0773 | 0.0702 |
| g_stock | g1_tokenspan | 0.3119 | 0.3123 | 0.1043 | 0.0697 |
| g_stock_tokenspan | g1 | 0.3173 | 0.3198 | 0.0972 | 0.0709 |
| g_stock_tokenspan | g1_tokenspan | 0.3183 | 0.3172 | 0.0735 | 0.0728 |
| g1 | g1_tokenspan | 0.7067 | 0.7129 | 0.4305 | 0.0553 |

## Runtime

- Load 12 .npy + 4 index CSV + 2 vocab CSV: 0.8s
- §3 f_clue_val pairwise cosine:           1.0s
- §4 f_common_wndef_val pairwise cosine:   0.4s
- §5 f_common_wnex_val pairwise cosine:    0.0s
