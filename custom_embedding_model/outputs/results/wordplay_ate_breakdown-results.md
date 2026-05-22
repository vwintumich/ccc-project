# Wordplay ATE Breakdown — Results

Generated: 2026-04-20  
Notebook: `planning/exploration/wordplay_ate_breakdown.ipynb`  
Random seed: 42  
Bootstrap samples: 1000  
Validation rows: 47,933  
Unique validation clue_ids: 47,653  
Standard clues (is_standard): 43,403  
Double-def clues: 4,530  

## Per-type validation-set frequencies

- Rows with ≥1 type detected: 9,565 (20.0%)
- Rows with no type detected: 38,368 (80.0%)

| wordplay_type | n_rows | pct_rows | unique_clue_ids |
| --- | --- | --- | --- |
| double_def | 4,530 | 9.45% | 4,250 |
| anagram_consec_words | 2,125 | 4.43% | 2,123 |
| hidden_fwd | 1,503 | 3.14% | 1,503 |
| anagram_single_word | 563 | 1.17% | 562 |
| hidden_rev | 414 | 0.86% | 413 |
| selection_firsts | 238 | 0.50% | 238 |
| selection_alt | 190 | 0.40% | 190 |
| selection_lasts | 29 | 0.06% | 29 |
| selection_alt_rev | 25 | 0.05% | 25 |
| selection_firsts_rev | 5 | 0.01% | 5 |
| selection_lasts_rev | 4 | 0.01% | 4 |

## ATE by category × model

Δ = cos(g(f_clue(def)), g(wndef(ans))) − cos(g(wndef(def)), g(wndef(ans)))  
CI = 95% bootstrap CI on mean Δ. `small_n` marks categories with fewer than 50 rows. All letterplay rows are restricted to standard clues (is_standard).

### Structural

| category | model | n_total | t0_mean | t1_mean | ate_mean | ate_median | ate_ci_lo | ate_ci_hi | pct_negative | small_n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| standard | g_stock | 43,403 | +0.5875 | +0.5199 | -0.0676 | -0.0607 | -0.0688 | -0.0664 | 72.9% |  |
| standard | g1 | 43,403 | +0.7163 | +0.5892 | -0.1271 | -0.1260 | -0.1278 | -0.1263 | 94.1% |  |
| double_def | g_stock | 4,530 | +0.4680 | +0.4467 | -0.0212 | -0.0137 | -0.0245 | -0.0177 | 56.0% |  |
| double_def | g1 | 4,530 | +0.6984 | +0.6036 | -0.0948 | -0.0959 | -0.0973 | -0.0922 | 86.5% |  |

### Individual letterplay (standard clues only)

| category | model | n_total | t0_mean | t1_mean | ate_mean | ate_median | ate_ci_lo | ate_ci_hi | pct_negative | small_n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no_letterplay | g_stock | 38,368 | +0.5861 | +0.5182 | -0.0679 | -0.0612 | -0.0693 | -0.0666 | 73.0% |  |
| no_letterplay | g1 | 38,368 | +0.7163 | +0.5856 | -0.1307 | -0.1296 | -0.1316 | -0.1299 | 94.7% |  |
| anagram_consec | g_stock | 2,101 | +0.6238 | +0.5531 | -0.0707 | -0.0687 | -0.0763 | -0.0653 | 73.7% |  |
| anagram_consec | g1 | 2,101 | +0.7119 | +0.6202 | -0.0916 | -0.0922 | -0.0952 | -0.0886 | 88.6% |  |
| hidden_fwd | g_stock | 1,494 | +0.5745 | +0.5150 | -0.0596 | -0.0472 | -0.0663 | -0.0533 | 70.7% |  |
| hidden_fwd | g1 | 1,494 | +0.7195 | +0.6176 | -0.1019 | -0.0998 | -0.1061 | -0.0976 | 89.9% |  |
| anagram_single | g_stock | 557 | +0.5923 | +0.5255 | -0.0668 | -0.0524 | -0.0776 | -0.0558 | 71.1% |  |
| anagram_single | g1 | 557 | +0.7238 | +0.6217 | -0.1020 | -0.1019 | -0.1089 | -0.0950 | 88.7% |  |
| hidden_rev | g_stock | 412 | +0.5810 | +0.5209 | -0.0601 | -0.0511 | -0.0729 | -0.0476 | 70.1% |  |
| hidden_rev | g1 | 412 | +0.7099 | +0.6010 | -0.1089 | -0.1028 | -0.1170 | -0.1016 | 91.5% |  |
| selection_firsts | g_stock | 237 | +0.5868 | +0.5291 | -0.0577 | -0.0468 | -0.0746 | -0.0431 | 72.6% |  |
| selection_firsts | g1 | 237 | +0.7194 | +0.6051 | -0.1144 | -0.1103 | -0.1248 | -0.1037 | 91.6% |  |
| selection_alt | g_stock | 189 | +0.5740 | +0.5194 | -0.0546 | -0.0411 | -0.0729 | -0.0352 | 69.8% |  |
| selection_alt | g1 | 189 | +0.7197 | +0.6123 | -0.1074 | -0.1101 | -0.1191 | -0.0954 | 89.9% |  |
| selection_lasts | g_stock | 29 | +0.5717 | +0.5045 | -0.0672 | -0.0647 | -0.1244 | -0.0113 | 79.3% | yes |
| selection_lasts | g1 | 29 | +0.7049 | +0.5733 | -0.1315 | -0.1258 | -0.1645 | -0.0952 | 96.6% | yes |
| selection_alt_rev | g_stock | 25 | +0.5712 | +0.5124 | -0.0588 | -0.0247 | -0.1268 | +0.0054 | 68.0% | yes |
| selection_alt_rev | g1 | 25 | +0.7382 | +0.5990 | -0.1392 | -0.1082 | -0.1715 | -0.1057 | 96.0% | yes |
| selection_firsts_rev | g_stock | 5 | +0.4653 | +0.4686 | +0.0032 | +0.0570 | -0.1897 | +0.2037 | 40.0% | yes |
| selection_firsts_rev | g1 | 5 | +0.6663 | +0.5726 | -0.0937 | -0.0781 | -0.1697 | -0.0177 | 80.0% | yes |
| selection_lasts_rev | g_stock | 4 | +0.4528 | +0.3751 | -0.0777 | -0.0094 | -0.2237 | +0.0000 | 75.0% | yes |
| selection_lasts_rev | g1 | 4 | +0.7483 | +0.5355 | -0.2128 | -0.2146 | -0.2372 | -0.1865 | 100.0% | yes |

### Grouped letterplay (standard clues only)

| category | model | n_total | t0_mean | t1_mean | ate_mean | ate_median | ate_ci_lo | ate_ci_hi | pct_negative | small_n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no_letterplay | g_stock | 38,368 | +0.5861 | +0.5182 | -0.0679 | -0.0612 | -0.0693 | -0.0666 | 73.0% |  |
| no_letterplay | g1 | 38,368 | +0.7163 | +0.5856 | -0.1307 | -0.1296 | -0.1316 | -0.1299 | 94.7% |  |
| any_anagram | g_stock | 2,658 | +0.6172 | +0.5473 | -0.0699 | -0.0651 | -0.0750 | -0.0651 | 73.2% |  |
| any_anagram | g1 | 2,658 | +0.7144 | +0.6206 | -0.0938 | -0.0936 | -0.0968 | -0.0909 | 88.6% |  |
| any_hidden | g_stock | 1,902 | +0.5761 | +0.5162 | -0.0598 | -0.0474 | -0.0656 | -0.0539 | 70.7% |  |
| any_hidden | g1 | 1,902 | +0.7174 | +0.6140 | -0.1034 | -0.1003 | -0.1072 | -0.0997 | 90.2% |  |
| any_reversal | g_stock | 446 | +0.5780 | +0.5185 | -0.0595 | -0.0455 | -0.0728 | -0.0464 | 69.7% |  |
| any_reversal | g1 | 446 | +0.7113 | +0.6000 | -0.1113 | -0.1054 | -0.1191 | -0.1039 | 91.7% |  |
| any_selection | g_stock | 488 | +0.5776 | +0.5207 | -0.0568 | -0.0464 | -0.0698 | -0.0448 | 71.5% |  |
| any_selection | g1 | 488 | +0.7192 | +0.6046 | -0.1146 | -0.1106 | -0.1217 | -0.1072 | 91.4% |  |
| any_letterplay | g_stock | 5,035 | +0.5981 | +0.5332 | -0.0649 | -0.0571 | -0.0685 | -0.0615 | 72.2% |  |
| any_letterplay | g1 | 5,035 | +0.7159 | +0.6165 | -0.0994 | -0.0983 | -0.1017 | -0.0972 | 89.5% |  |

## Structural comparison: double-def vs standard

### g_stock

- standard   (n=43,403): mean Δ = -0.0676, median Δ = -0.0607, % Δ<0 = 72.9%
- double_def (n=4,530): mean Δ = -0.0212, median Δ = -0.0137, % Δ<0 = 56.0%

### g1

- standard   (n=43,403): mean Δ = -0.1271, median Δ = -0.1260, % Δ<0 = 94.1%
- double_def (n=4,530): mean Δ = -0.0948, median Δ = -0.0959, % Δ<0 = 86.5%

## T=0 and T=1 component differences: structural

95% Welch CI on the difference of component means: `double_def − standard`. `excludes_zero` = yes when the CI does not contain 0.

| component | model | n_dd | n_std | mean_dd | mean_std | diff | ci_lo | ci_hi | excludes_zero |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| t0 | g_stock | 4,530 | 43,403 | +0.4680 | +0.5875 | -0.1195 | -0.1247 | -0.1144 | yes |
| t0 | g1 | 4,530 | 43,403 | +0.6984 | +0.7163 | -0.0179 | -0.0201 | -0.0157 | yes |
| t1 | g_stock | 4,530 | 43,403 | +0.4467 | +0.5199 | -0.0732 | -0.0780 | -0.0684 | yes |
| t1 | g1 | 4,530 | 43,403 | +0.6036 | +0.5892 | +0.0144 | +0.0123 | +0.0165 | yes |

## T=0 and T=1 component differences: individual letterplay

95% Welch CI on the difference of component means: each letterplay type − `no_letterplay` (standard clues only). Rows ordered by descending N_type. `small_n` marks categories with fewer than 50 rows.

| category | component | model | n_type | n_baseline | mean_type | mean_baseline | diff | ci_lo | ci_hi | excludes_zero | small_n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anagram_consec | t0 | g1 | 2,101 | 38,368 | +0.7119 | +0.7163 | -0.0045 | -0.0076 | -0.0013 | yes |  |
| anagram_consec | t0 | g_stock | 2,101 | 38,368 | +0.6238 | +0.5861 | +0.0377 | +0.0304 | +0.0449 | yes |  |
| anagram_consec | t1 | g1 | 2,101 | 38,368 | +0.6202 | +0.5856 | +0.0346 | +0.0318 | +0.0375 | yes |  |
| anagram_consec | t1 | g_stock | 2,101 | 38,368 | +0.5531 | +0.5182 | +0.0348 | +0.0282 | +0.0415 | yes |  |
| hidden_fwd | t0 | g1 | 1,494 | 38,368 | +0.7195 | +0.7163 | +0.0031 | -0.0004 | +0.0067 |  |  |
| hidden_fwd | t0 | g_stock | 1,494 | 38,368 | +0.5745 | +0.5861 | -0.0116 | -0.0206 | -0.0025 | yes |  |
| hidden_fwd | t1 | g1 | 1,494 | 38,368 | +0.6176 | +0.5856 | +0.0320 | +0.0287 | +0.0353 | yes |  |
| hidden_fwd | t1 | g_stock | 1,494 | 38,368 | +0.5150 | +0.5182 | -0.0032 | -0.0117 | +0.0053 |  |  |
| anagram_single | t0 | g1 | 557 | 38,368 | +0.7238 | +0.7163 | +0.0074 | +0.0012 | +0.0136 | yes |  |
| anagram_single | t0 | g_stock | 557 | 38,368 | +0.5923 | +0.5861 | +0.0062 | -0.0081 | +0.0204 |  |  |
| anagram_single | t1 | g1 | 557 | 38,368 | +0.6217 | +0.5856 | +0.0361 | +0.0303 | +0.0419 | yes |  |
| anagram_single | t1 | g_stock | 557 | 38,368 | +0.5255 | +0.5182 | +0.0072 | -0.0061 | +0.0206 |  |  |
| hidden_rev | t0 | g1 | 412 | 38,368 | +0.7099 | +0.7163 | -0.0064 | -0.0126 | -0.0002 | yes |  |
| hidden_rev | t0 | g_stock | 412 | 38,368 | +0.5810 | +0.5861 | -0.0051 | -0.0222 | +0.0120 |  |  |
| hidden_rev | t1 | g1 | 412 | 38,368 | +0.6010 | +0.5856 | +0.0154 | +0.0087 | +0.0220 | yes |  |
| hidden_rev | t1 | g_stock | 412 | 38,368 | +0.5209 | +0.5182 | +0.0027 | -0.0135 | +0.0189 |  |  |
| selection_firsts | t0 | g1 | 237 | 38,368 | +0.7194 | +0.7163 | +0.0031 | -0.0046 | +0.0108 |  |  |
| selection_firsts | t0 | g_stock | 237 | 38,368 | +0.5868 | +0.5861 | +0.0007 | -0.0208 | +0.0222 |  |  |
| selection_firsts | t1 | g1 | 237 | 38,368 | +0.6051 | +0.5856 | +0.0194 | +0.0104 | +0.0285 | yes |  |
| selection_firsts | t1 | g_stock | 237 | 38,368 | +0.5291 | +0.5182 | +0.0109 | -0.0096 | +0.0315 |  |  |
| selection_alt | t0 | g1 | 189 | 38,368 | +0.7197 | +0.7163 | +0.0034 | -0.0060 | +0.0127 |  |  |
| selection_alt | t0 | g_stock | 189 | 38,368 | +0.5740 | +0.5861 | -0.0121 | -0.0391 | +0.0149 |  |  |
| selection_alt | t1 | g1 | 189 | 38,368 | +0.6123 | +0.5856 | +0.0267 | +0.0172 | +0.0362 | yes |  |
| selection_alt | t1 | g_stock | 189 | 38,368 | +0.5194 | +0.5182 | +0.0012 | -0.0244 | +0.0269 |  |  |
| selection_lasts | t0 | g1 | 29 | 38,368 | +0.7049 | +0.7163 | -0.0115 | -0.0361 | +0.0131 |  | yes |
| selection_lasts | t0 | g_stock | 29 | 38,368 | +0.5717 | +0.5861 | -0.0144 | -0.0640 | +0.0351 |  | yes |
| selection_lasts | t1 | g1 | 29 | 38,368 | +0.5733 | +0.5856 | -0.0123 | -0.0332 | +0.0087 |  | yes |
| selection_lasts | t1 | g_stock | 29 | 38,368 | +0.5045 | +0.5182 | -0.0137 | -0.0643 | +0.0369 |  | yes |
| selection_alt_rev | t0 | g1 | 25 | 38,368 | +0.7382 | +0.7163 | +0.0218 | -0.0028 | +0.0465 |  | yes |
| selection_alt_rev | t0 | g_stock | 25 | 38,368 | +0.5712 | +0.5861 | -0.0149 | -0.0748 | +0.0450 |  | yes |
| selection_alt_rev | t1 | g1 | 25 | 38,368 | +0.5990 | +0.5856 | +0.0134 | -0.0150 | +0.0417 |  | yes |
| selection_alt_rev | t1 | g_stock | 25 | 38,368 | +0.5124 | +0.5182 | -0.0058 | -0.0727 | +0.0611 |  | yes |
| selection_firsts_rev | t0 | g1 | 5 | 38,368 | +0.6663 | +0.7163 | -0.0501 | -0.1218 | +0.0217 |  | yes |
| selection_firsts_rev | t0 | g_stock | 5 | 38,368 | +0.4653 | +0.5861 | -0.1208 | -0.3001 | +0.0585 |  | yes |
| selection_firsts_rev | t1 | g1 | 5 | 38,368 | +0.5726 | +0.5856 | -0.0130 | -0.0955 | +0.0694 |  | yes |
| selection_firsts_rev | t1 | g_stock | 5 | 38,368 | +0.4686 | +0.5182 | -0.0497 | -0.2083 | +0.1089 |  | yes |
| selection_lasts_rev | t0 | g1 | 4 | 38,368 | +0.7483 | +0.7163 | +0.0319 | +0.0149 | +0.0489 | yes | yes |
| selection_lasts_rev | t0 | g_stock | 4 | 38,368 | +0.4528 | +0.5861 | -0.1333 | -0.3175 | +0.0508 |  | yes |
| selection_lasts_rev | t1 | g1 | 4 | 38,368 | +0.5355 | +0.5856 | -0.0501 | -0.0828 | -0.0175 | yes | yes |
| selection_lasts_rev | t1 | g_stock | 4 | 38,368 | +0.3751 | +0.5182 | -0.1431 | -0.2348 | -0.0515 | yes | yes |

## Version stamps

- pandas: 3.0.0
- numpy: 2.3.5
- matplotlib: 3.10.8
- seaborn: 0.13.2
