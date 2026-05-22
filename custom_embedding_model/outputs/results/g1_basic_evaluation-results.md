# g1 Basic Model Evaluation — Results

Generated: 2026-04-24

## Versions

- **python:** 3.12.12
- **numpy:** 2.3.5
- **pandas:** 3.0.0
- **scipy:** 1.17.0
- **matplotlib:** 3.10.8
- **seaborn:** 0.13.2

## §2 — Training dynamics

|   epoch |   train_loss |   val_loss |   ratio (val/train) |   val_accuracy |   val_mean_margin |   val_median_margin |
|--------:|-------------:|-----------:|--------------------:|---------------:|------------------:|--------------------:|
|  1.0000 |       0.4703 |     0.3046 |              0.6476 |         0.8721 |            0.0534 |              0.0503 |
|  2.0000 |       0.1114 |     0.2521 |              2.2624 |         0.8966 |            0.0884 |              0.0846 |
|  3.0000 |       0.0137 |     0.2642 |             19.3506 |         0.8996 |            0.1249 |              0.1215 |

Best-generalizing checkpoint: epoch 2.  Deployed checkpoint: epoch 3.

## §3 — Task performance

### §3b — Validation triplet accuracy (wndef, full-vocab)

| model   |   n_triplets |   l2_accuracy |   cos_accuracy |   l2_mean_margin |   l2_median_margin |   cos_mean_margin |   cos_median_margin |
|:--------|-------------:|--------------:|---------------:|-----------------:|-------------------:|------------------:|--------------------:|
| g_stock |        46506 |        0.3896 |         0.3878 |          -1.6142 |            -1.4031 |           -0.0540 |             -0.0441 |
| g1      |        46506 |        0.9021 |         0.8996 |           3.8900 |             3.8036 |            0.1249 |              0.1215 |

### §3d — Summary table

| Metric             |   g_stock val |   g1 val | g1 train (infer)   |
|:-------------------|--------------:|---------:|:-------------------|
| L2 accuracy        |     0.389606  | 0.902099 | >=0.986            |
| Cosine accuracy    |     0.387778  | 0.899583 | —                  |
| L2 mean margin     |    -1.61423   | 3.89004  | —                  |
| Cosine mean margin |    -0.0539886 | 0.124856 | —                  |

## §4 — Embedding space geometry

### §4a — Norm distributions

| Population           | Model   |    Mean |    Std |     Min |     Max |      P5 |     P95 |
|:---------------------|:--------|--------:|-------:|--------:|--------:|--------:|--------:|
| wndef vocab (53,930) | g_stock | 29.1093 | 1.3706 | 23.5365 | 33.6164 | 26.8265 | 31.3000 |
| f_clue val (47,933)  | g_stock | 30.6002 | 1.2029 | 25.2485 | 33.7204 | 28.6387 | 32.3554 |
| wndef vocab (53,930) | g1      | 27.9319 | 0.8580 | 24.0275 | 30.6785 | 26.4850 | 29.2722 |
| f_clue val (47,933)  | g1      | 28.9786 | 0.7544 | 25.5613 | 31.6818 | 27.6432 | 30.1399 |

### §4b — Pairwise cosine among random word pairs (N=50,000)

| Population   | Model   |   Mean |   Median |    Std |     P5 |    P95 |
|:-------------|:--------|-------:|---------:|-------:|-------:|-------:|
| wndef vocab  | g_stock | 0.4119 |   0.4028 | 0.1186 | 0.2338 | 0.6249 |
| wndef vocab  | g1      | 0.5611 |   0.5605 | 0.0745 | 0.4400 | 0.6852 |
| f_clue val   | g_stock | 0.3166 |   0.3024 | 0.1168 | 0.1511 | 0.5316 |
| f_clue val   | g1      | 0.4520 |   0.4528 | 0.0644 | 0.3449 | 0.5564 |

### §4c — Effective dimensionality

| Population   | Model   |   Total var |   Eff. dim |   Top-10 % |   Top-50 % |   Top-100 % |
|:-------------|:--------|------------:|-----------:|-----------:|-----------:|------------:|
| wndef vocab  | g_stock | 26980812.00 |      41.06 |      38.97 |      64.83 |       75.68 |
| f_clue val   | g_stock | 30692136.00 |      52.34 |      35.01 |      61.04 |       72.79 |
| wndef vocab  | g1      | 18555818.00 |      45.42 |      33.37 |      67.69 |       81.63 |
| f_clue val   | g1      | 22085598.00 |     103.17 |      22.37 |      56.72 |       73.05 |

### §4d — Seen/unseen stratification

Norms by word set:

| Word set        | Model   |     N |   Mean norm |    Std |
|:----------------|:--------|------:|------------:|-------:|
| seen_wndef      | g_stock | 36082 |     29.1533 | 1.3622 |
| seen_wndef      | g1      | 36082 |     27.8918 | 0.8675 |
| seen_fclue_only | g_stock |  5054 |     29.0228 | 1.3705 |
| seen_fclue_only | g1      |  5054 |     28.0859 | 0.7652 |
| unseen          | g_stock | 12794 |     29.0195 | 1.3882 |
| unseen          | g1      | 12794 |     27.9843 | 0.8560 |

Pairwise cosine (seen_wndef vs unseen):

| Word set   | Model   |   N pairs |   Mean cos |   Median |    Std |
|:-----------|:--------|----------:|-----------:|---------:|-------:|
| seen_wndef | g_stock |     50000 |     0.4092 |   0.3977 | 0.1198 |
| seen_wndef | g1      |     50000 |     0.5645 |   0.5644 | 0.0758 |
| unseen     | g_stock |     50000 |     0.4302 |   0.4227 | 0.1171 |
| unseen     | g1      |     50000 |     0.5567 |   0.5546 | 0.0738 |

Triplet accuracy stratified by pos/neg exposure:

| Stratum                | Model   |     N |   L2 accuracy |   Cos accuracy |   L2 mean margin |   Cos mean margin |
|:-----------------------|:--------|------:|--------------:|---------------:|-----------------:|------------------:|
| both seen_wndef        | g_stock | 37943 |        0.3905 |         0.3899 |          -1.6418 |           -0.0549 |
| both seen_wndef        | g1      | 37943 |        0.9034 |         0.9011 |           3.9070 |            0.1253 |
| both unseen            | g_stock |   477 |        0.4046 |         0.3878 |          -1.3485 |           -0.0428 |
| both unseen            | g1      |   477 |        0.8931 |         0.8826 |           3.5778 |            0.1127 |
| mixed (one seen_wndef) | g_stock |  8086 |        0.3847 |         0.3779 |          -1.5004 |           -0.0504 |
| mixed (one seen_wndef) | g1      |  8086 |        0.8964 |         0.8936 |           3.8287 |            0.1236 |

### §4e — Compression summary

| Population   |   Δ mean norm |   Δ mean pair cos |   Δ total variance |   % total variance |   Δ effective dim |
|:-------------|--------------:|------------------:|-------------------:|-------------------:|------------------:|
| wndef vocab  |       -1.1774 |            0.1492 |      -8424994.0000 |           -31.2259 |            4.3592 |
| f_clue val   |       -1.6216 |            0.1354 |      -8606538.0000 |           -28.0415 |           50.8328 |

## §5 — Structural comparison to g_stock

| Population   |   N words |   N pairs |   Spearman rho |   p-value |
|:-------------|----------:|----------:|---------------:|----------:|
| wndef vocab  |      1000 |    499500 |        0.1353  |         0 |
| f_clue val   |      1000 |    499500 |        0.06656 |         0 |

## §6 — Context effects

### §6a — Cosine T=0 / T=1 distributions

| Distribution   |   Mean |   Median |    Std |     P5 |    P95 |
|:---------------|-------:|---------:|-------:|-------:|-------:|
| g_stock T=0    | 0.5762 |   0.5903 | 0.1769 | 0.2828 | 0.8424 |
| g_stock T=1    | 0.5130 |   0.5162 | 0.1640 | 0.2452 | 0.7716 |
| g1 T=0         | 0.7146 |   0.7162 | 0.0698 | 0.5964 | 0.8265 |
| g1 T=1         | 0.5906 |   0.5933 | 0.0707 | 0.4704 | 0.7020 |

### §6a — Cosine ATE

| Model   |     N |   ATE (mean) |   Median Δ |     SE |   95% CI lo |   95% CI hi |   % Δ < 0 |
|:--------|------:|-------------:|-----------:|-------:|------------:|------------:|----------:|
| g_stock | 47933 |      -0.0632 |    -0.0562 | 0.0006 |     -0.0644 |     -0.0620 |   71.2724 |
| g1      | 47933 |      -0.1240 |    -0.1235 | 0.0004 |     -0.1248 |     -0.1233 |   93.4096 |

### §6b — L2 T=0 / T=1 and context effect

|                                     |    g_stock |         g1 |
|:------------------------------------|-----------:|-----------:|
| T=0 L2 distance (mean)              | 26.3968    | 20.6338    |
| T=1 L2 distance (mean)              | 29.1448    | 25.5286    |
| L2 context effect (T=1 - T=0, mean) |  2.74796   |  4.89475   |
| SE                                  |  0.0203117 |  0.0134421 |
| 95% CI lo                           |  2.70815   |  4.8684    |
| 95% CI hi                           |  2.78777   |  4.9211    |
| % pairs context increases distance  | 76.5527    | 95.4019    |

### §6d — Cosine vs L2 context effect summary

| Model   |   Cosine ATE (T=1 - T=0) |   Cosine mean T=0 |   Cosine mean T=1 |   L2 context (T=1 - T=0) |   L2 mean T=0 |   L2 mean T=1 |
|:--------|-------------------------:|------------------:|------------------:|-------------------------:|--------------:|--------------:|
| g_stock |                  -0.0632 |            0.5762 |            0.5130 |                   2.7480 |       26.3968 |       29.1448 |
| g1      |                  -0.1240 |            0.7146 |            0.5906 |                   4.8948 |       20.6338 |       25.5286 |

## Figures

- `figures/g1be_training_dynamics.png`
- `figures/g1be_norm_distributions.png`
- `figures/g1be_pairwise_cosine.png`
- `figures/g1be_singular_values.png`
- `figures/g1be_seen_unseen.png`
- `figures/g1be_seen_unseen_cosine.png`
- `figures/g1be_t0_t1_cosine.png`
- `figures/g1be_t0_t1_l2.png`
