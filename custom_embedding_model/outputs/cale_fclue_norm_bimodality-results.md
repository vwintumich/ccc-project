# CALE f_clue Norm Bimodality Investigation — Results

Generated: 2026-04-24

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

## Figures

- `figures/fclue_bimodal_norm_full_vs_val.png`
- `figures/fclue_bimodal_vocab_norms.png`
- `figures/fclue_bimodal_norm_overlay.png`
- `figures/fclue_bimodal_t0_t1.png`
- `figures/fclue_bimodal_by_position.png`
- `figures/fclue_bimodal_by_subword_tokens.png`
- `figures/fclue_bimodal_by_source.png`
- `figures/fclue_bimodal_dimensions.png`
- `figures/fclue_bimodal_propagation.png`
- `figures/fclue_bimodal_g1_by_position.png`
