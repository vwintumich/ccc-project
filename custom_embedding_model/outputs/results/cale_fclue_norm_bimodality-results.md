# CALE f_clue Norm Bimodality Investigation — Results

Generated: 2026-04-27

## Versions

- **python:** 3.12.12
- **numpy:** 2.3.5
- **pandas:** 3.0.0
- **scipy:** 1.17.0
- **matplotlib:** 3.10.8
- **seaborn:** 0.13.2

## §2 — Full dataset vs validation slice norms

| Distribution                   |      N |    Mean |    Std |      P5 |     P25 |     P50 |     P75 |     P95 |
|:-------------------------------|-------:|--------:|-------:|--------:|--------:|--------:|--------:|--------:|
| g_stock f_clue (full, 239,406) | 239406 | 30.6077 | 1.2024 | 28.6396 | 29.6478 | 30.6674 | 31.6345 | 32.3553 |
| g_stock f_clue (val,  47,933)  |  47933 | 30.6002 | 1.2029 | 28.6387 | 29.6386 | 30.6500 | 31.6276 | 32.3554 |

## §2 — g_stock vocabulary norms (contrast to f_clue)

| Distribution                               |    Mean |    Std |      P5 |     P50 |     P95 |
|:-------------------------------------------|--------:|-------:|--------:|--------:|--------:|
| g_stock f_wndef (53,930)                   | 29.1093 | 1.3706 | 26.8265 | 29.1194 | 31.3000 |
| g_stock f_wnex  (8,360)                    | 30.3636 | 1.1011 | 28.3845 | 30.4863 | 31.9767 |
| g_stock f_wndef restricted to wnex (8,360) | 28.7728 | 1.4419 | 26.4609 | 28.7342 | 31.1151 |

Both vocabulary distributions are visibly unimodal, confirming the bimodality is specific to the clue-context setting.

## §2 — Cross-format directional alignment (wndef vs wnex, same words)

| Model   |    N |   Mean |   Median |    Std |     P5 |    P95 |
|:--------|-----:|-------:|---------:|-------:|-------:|-------:|
| g_stock | 8360 | 0.7196 |   0.7370 | 0.1092 | 0.5113 | 0.8642 |
| g1      | 8360 | 0.6329 |   0.6339 | 0.0674 | 0.5223 | 0.7413 |

## §2 — T=0 / T=1 distribution summary (both models)

| Model   | Metric   |    Mean |   Median |    Std |      P5 |     P95 |
|:--------|:---------|--------:|---------:|-------:|--------:|--------:|
| g_stock | T0_cos   |  0.5762 |   0.5903 | 0.1769 |  0.2828 |  0.8424 |
| g_stock | T1_cos   |  0.5130 |   0.5162 | 0.1640 |  0.2452 |  0.7716 |
| g_stock | T0_l2    | 26.3968 |  26.5512 | 6.0612 | 16.3019 | 35.6480 |
| g_stock | T1_l2    | 29.1448 |  29.4296 | 5.2611 | 20.2302 | 37.2384 |
| g1      | T0_cos   |  0.7146 |   0.7162 | 0.0698 |  0.5964 |  0.8265 |
| g1      | T1_cos   |  0.5906 |   0.5933 | 0.0707 |  0.4704 |  0.7020 |
| g1      | T0_l2    | 20.6338 |  20.6703 | 2.6552 | 16.2187 | 24.9511 |
| g1      | T1_l2    | 25.5286 |  25.5120 | 2.2997 | 21.8051 | 29.3307 |

## §3 — Norm stats by definition position (g_stock)

| Position   |     N |    Mean |    Std |     P25 |     P50 |     P75 |
|:-----------|------:|--------:|-------:|--------:|--------:|--------:|
| start      | 26964 | 30.7685 | 1.2149 | 29.7640 | 31.0198 | 31.7762 |
| end        | 20459 | 30.3672 | 1.1489 | 29.5140 | 30.2670 | 31.2766 |
| middle     |   510 | 31.0469 | 1.0583 | 30.4568 | 31.2257 | 31.7851 |

## §4 — Top publishers by start-def count

| publisher        |    N |    mean |    std |
|:-----------------|-----:|--------:|-------:|
| Daily Telegraph  | 8151 | 30.7636 | 1.2151 |
| The Times        | 4641 | 30.7796 | 1.2179 |
| The Hindu        | 3557 | 30.9081 | 1.1887 |
| Guardian         | 3462 | 30.7303 | 1.2219 |
| Financial Times  | 2944 | 30.7169 | 1.2168 |
| Independent      | 1319 | 30.7567 | 1.2275 |
| Sunday Telegraph |  896 | 30.6069 | 1.2044 |
| National Post    |  552 | 30.7107 | 1.2226 |
| Observer         |  521 | 30.7090 | 1.2214 |
| Sunday Times     |  457 | 30.8452 | 1.2132 |

- Unique start-def definition words: **8,848**
- Appearing in both modes: **2,818 (31.8%)**
- Within-word mean std / overall std: **0.789**
- Spearman rho(wndef_norm, fclue_norm): **0.1304**

## §4 — Wordplay type stratification (all definitions)

| Wordplay Type        |   N (True) | Fit Type    |     mu1 |     mu2 |   pi1 (True) |   pi1 (False) |   delta_pi1 |
|:---------------------|-----------:|:------------|--------:|--------:|-------------:|--------------:|------------:|
| double_def           |      22849 | free        | 29.1771 | 31.2362 |       0.3222 |        0.5155 |     -0.1933 |
| not double_def       |     216557 | free        | 29.6431 | 31.6393 |       0.5155 |        0.3222 |      0.1933 |
| any letterplay       |      24675 | free        | 29.4891 | 31.5847 |       0.4616 |        0.5058 |     -0.0441 |
| no letterplay        |     191882 | free        | 29.6611 | 31.6491 |       0.5231 |        0.4096 |      0.1134 |
| anagram_single_word  |       2922 | free        | 29.1821 | 31.4650 |       0.3744 |        0.5028 |     -0.1284 |
| selection_lasts      |        167 | constrained | 29.6208 | 31.5964 |       0.6035 |        0.5026 |      0.1009 |
| selection_firsts     |       1113 | free        | 29.9752 | 31.8263 |       0.5837 |        0.5026 |      0.0810 |
| hidden_fwd           |       7554 | free        | 29.5612 | 31.6550 |       0.4452 |        0.5048 |     -0.0596 |
| selection_alt        |        909 | free        | 29.6984 | 31.7774 |       0.5459 |        0.5011 |      0.0448 |
| hidden_rev           |       2031 | free        | 29.8441 | 31.8115 |       0.5354 |        0.5010 |      0.0344 |
| anagram_consec_words |      10181 | free        | 29.3736 | 31.5184 |       0.4736 |        0.5031 |     -0.0294 |
| selection_alt_rev    |        109 | constrained | 29.6208 | 31.5964 |       0.5127 |        0.5025 |      0.0101 |

## §4 — Wordplay type stratification (start-definitions only)

| Wordplay Type        |   N (True) | Fit Type    |     mu1 |     mu2 |   pi1 (True) |   pi1 (False) |   delta_pi1 |
|:---------------------|-----------:|:------------|--------:|--------:|-------------:|--------------:|------------:|
| anagram_single_word  |       1629 | free        | 29.2380 | 31.4710 |       0.2674 |        0.4342 |     -0.1669 |
| double_def           |      11793 | free        | 29.2549 | 31.2694 |       0.3196 |        0.4358 |     -0.1162 |
| not double_def       |     122477 | free        | 29.6370 | 31.7008 |       0.4358 |        0.3196 |      0.1162 |
| any letterplay       |      14155 | free        | 29.5276 | 31.6283 |       0.3638 |        0.4411 |     -0.0773 |
| no letterplay        |     108322 | free        | 29.6491 | 31.7135 |       0.4456 |        0.3619 |      0.0837 |
| selection_firsts     |        758 | free        | 29.8838 | 31.8265 |       0.5415 |        0.4314 |      0.1101 |
| hidden_fwd           |       4766 | free        | 29.5742 | 31.6687 |       0.3382 |        0.4352 |     -0.0970 |
| selection_lasts      |        102 | constrained | 29.6208 | 31.5964 |       0.5227 |        0.4326 |      0.0901 |
| anagram_consec_words |       5131 | free        | 29.3556 | 31.5565 |       0.3614 |        0.4347 |     -0.0733 |
| selection_alt        |        504 | free        | 29.7572 | 31.8270 |       0.4557 |        0.4317 |      0.0240 |
| selection_alt_rev    |         56 | constrained | 29.6208 | 31.5964 |       0.4192 |        0.4318 |     -0.0126 |
| hidden_rev           |       1333 | free        | 29.8098 | 31.8459 |       0.4432 |        0.4318 |      0.0114 |

## §5 — Top 10 dimensions by |delta|

|      Dim |   delta |   mean_lower |   mean_upper |   pooled_std |   cohens_d |
|---------:|--------:|-------------:|-------------:|-------------:|-----------:|
| 379.0000 |  1.2699 |      -6.4137 |      -5.1438 |       1.3469 |     0.9428 |
| 195.0000 |  0.8224 |       4.1015 |       4.9239 |       2.2414 |     0.3669 |
| 963.0000 |  0.4643 |       4.9137 |       5.3780 |       2.3121 |     0.2008 |
|  83.0000 | -0.4121 |      -0.8648 |      -1.2768 |       0.8235 |    -0.5004 |
| 724.0000 |  0.3723 |       0.6814 |       1.0537 |       1.0964 |     0.3396 |
| 921.0000 |  0.3673 |       0.7839 |       1.1511 |       0.7727 |     0.4753 |
| 734.0000 | -0.3670 |      -0.2007 |      -0.5677 |       0.9993 |    -0.3672 |
| 162.0000 |  0.3657 |       0.2661 |       0.6318 |       0.6932 |     0.5276 |
| 927.0000 | -0.3394 |      -0.1569 |      -0.4963 |       0.8169 |    -0.4155 |
| 159.0000 | -0.3269 |      -0.9252 |      -1.2521 |       0.8196 |    -0.3988 |

### Concentration of norm-squared difference by top-k dims

|        k |   partial_diff |   fraction_of_total |
|---------:|---------------:|--------------------:|
|  10.0000 |         1.3070 |              0.0234 |
|  50.0000 |        12.4897 |              0.2235 |
| 100.0000 |        20.2184 |              0.3618 |

- Rank of dim 379 by |delta|: **0**

## §6 — T=0 / T=1 / ATE stratified by norm group (start-defs)

| Metric   | Group   |     N |    Mean |    Std |
|:---------|:--------|------:|--------:|-------:|
| T1_cos   | lower   | 10703 |  0.5229 | 0.1628 |
| T1_cos   | upper   | 16261 |  0.5188 | 0.1652 |
| T1_l2    | lower   | 10703 | 28.2054 | 5.1954 |
| T1_l2    | upper   | 16261 | 29.5059 | 5.3719 |
| T0_cos   | lower   | 10703 |  0.5760 | 0.1747 |
| T0_cos   | upper   | 16261 |  0.5661 | 0.1784 |
| ATE_cos  | lower   | 10703 | -0.0531 | 0.1337 |
| ATE_cos  | upper   | 16261 | -0.0472 | 0.1369 |

- Spearman rho(f_clue norm, T=0 cosine):   **-0.0345**
- Spearman rho(f_clue norm, T=1 cosine):   **-0.0257**
- Spearman rho(f_clue norm, ATE cosine):   **0.0193**

## §7 — g_stock vs g1 f_clue norms by definition position

| Model   | Position   |     N |    Mean |    Std |
|:--------|:-----------|------:|--------:|-------:|
| g_stock | start      | 26964 | 30.7685 | 1.2149 |
| g_stock | end        | 20459 | 30.3672 | 1.1489 |
| g1      | start      | 26964 | 28.8771 | 0.7390 |
| g1      | end        | 20459 | 29.1181 | 0.7516 |

## GMM fit summary (all distributions)

| figure                          | distribution                                                 |     mu1 |   sigma1 |   weight1 |     mu2 |   sigma2 |   weight2 |
|:--------------------------------|:-------------------------------------------------------------|--------:|---------:|----------:|--------:|---------:|----------:|
| fclue_bimodal_by_position       | g_stock f_clue norms (start-defs)                            | 29.6056 |   0.7693 |    0.4347 | 31.6628 |   0.5629 |    0.5653 |
| fclue_bimodal_by_position       | g_stock f_clue norms (end-defs)                              | 29.6164 |   0.7368 |    0.5850 | 31.4256 |   0.7077 |    0.4150 |
| fclue_bimodal_by_subword_tokens | g_stock f_clue norms (1-token start-defs)                    | 29.5883 |   0.7737 |    0.4297 | 31.6771 |   0.5590 |    0.5703 |
| fclue_bimodal_by_subword_tokens | g_stock f_clue norms (2-token start-defs)                    | 29.6855 |   0.7504 |    0.4271 | 31.7194 |   0.5296 |    0.5729 |
| fclue_bimodal_by_source         | g_stock f_clue norms (start-defs, publisher=Daily Telegraph) | 29.5995 |   0.7680 |    0.4341 | 31.6564 |   0.5656 |    0.5659 |
| fclue_bimodal_by_source         | g_stock f_clue norms (start-defs, publisher=The Times)       | 29.5626 |   0.7573 |    0.4179 | 31.6535 |   0.5556 |    0.5821 |
| fclue_bimodal_by_source         | g_stock f_clue norms (start-defs, publisher=The Hindu)       | 29.7150 |   0.7795 |    0.4064 | 31.7249 |   0.5677 |    0.5936 |
| fclue_bimodal_by_wordplay       | g_stock f_clue norms (wordplay=double_def, True)             | 29.1771 |   0.8601 |    0.3222 | 31.2362 |   0.6479 |    0.6778 |
| fclue_bimodal_by_wordplay       | g_stock f_clue norms (wordplay=not double_def, True)         | 29.6431 |   0.7367 |    0.5155 | 31.6393 |   0.5933 |    0.4845 |
| fclue_bimodal_by_wordplay       | g_stock f_clue norms (wordplay=any letterplay, True)         | 29.4891 |   0.7257 |    0.4616 | 31.5847 |   0.5691 |    0.5384 |
| fclue_bimodal_by_wordplay       | g_stock f_clue norms (wordplay=no letterplay, True)          | 29.6611 |   0.7361 |    0.5231 | 31.6491 |   0.5951 |    0.4769 |
| fclue_bimodal_by_wordplay       | g_stock f_clue norms (wordplay=anagram_single_word, True)    | 29.1821 |   0.6838 |    0.3744 | 31.4650 |   0.5314 |    0.6256 |
| fclue_bimodal_by_wordplay       | g_stock f_clue norms (wordplay=anagram_consec_words, True)   | 29.3736 |   0.7029 |    0.4736 | 31.5184 |   0.5731 |    0.5264 |
| fclue_bimodal_by_wordplay       | g_stock f_clue norms (wordplay=hidden_fwd, True)             | 29.5612 |   0.7153 |    0.4452 | 31.6550 |   0.5382 |    0.5548 |
| fclue_bimodal_by_wordplay       | g_stock f_clue norms (wordplay=hidden_rev, True)             | 29.8441 |   0.7266 |    0.5354 | 31.8115 |   0.5240 |    0.4646 |
| fclue_bimodal_by_wordplay       | g_stock f_clue norms (wordplay=selection_alt, True)          | 29.6984 |   0.7263 |    0.5459 | 31.7774 |   0.5499 |    0.4541 |
| fclue_bimodal_by_wordplay       | g_stock f_clue norms (wordplay=selection_firsts, True)       | 29.9752 |   0.6807 |    0.5837 | 31.8263 |   0.5719 |    0.4163 |
| fclue_bimodal_by_wordplay       | g_stock f_clue norms (wordplay=selection_lasts, True)        | 29.6208 |   0.7578 |    0.6035 | 31.5964 |   0.6079 |    0.3965 |
| fclue_bimodal_propagation       | T=1 cosine (lower norm group)                                |  0.3960 |   0.1075 |    0.4925 |  0.6461 |   0.1009 |    0.5075 |
| fclue_bimodal_propagation       | T=1 cosine (upper norm group)                                |  0.3902 |   0.1089 |    0.4975 |  0.6462 |   0.0998 |    0.5025 |
| fclue_bimodal_propagation       | T=1 L2 (lower norm group)                                    | 23.9923 |   3.6236 |    0.4606 | 31.8024 |   3.2766 |    0.5394 |
| fclue_bimodal_propagation       | T=1 L2 (upper norm group)                                    | 25.0034 |   3.6142 |    0.4522 | 33.2223 |   3.3690 |    0.5478 |
| fclue_bimodal_g1_by_position    | g1 f_clue norms (start-defs)                                 | 28.3567 |   0.6810 |    0.4057 | 29.2322 |   0.5399 |    0.5943 |
| fclue_bimodal_g1_by_position    | g1 f_clue norms (end-defs)                                   | 28.5703 |   0.6650 |    0.4126 | 29.5030 |   0.5405 |    0.5874 |

## Figures

- `figures/fclue_bimodal_norm_full_vs_val.png`
- `figures/fclue_bimodal_vocab_norms.png`
- `figures/fclue_bimodal_cross_format_cosine.png`
- `figures/fclue_bimodal_norm_overlay.png`
- `figures/fclue_bimodal_t0_t1.png`
- `figures/fclue_bimodal_by_position.png`
- `figures/fclue_bimodal_by_subword_tokens.png`
- `figures/fclue_bimodal_by_source.png`
- `figures/fclue_bimodal_by_wordplay.png`
- `figures/fclue_bimodal_dimensions.png`
- `figures/fclue_bimodal_propagation.png`
- `figures/fclue_bimodal_g1_by_position.png`
