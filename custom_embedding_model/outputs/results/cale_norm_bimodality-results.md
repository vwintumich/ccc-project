# CALE L2 Norm Bimodality Survey (g_stock) — Results

Generated: 2026-04-30

## Versions

- **python:** 3.12.12
- **numpy:** 2.3.5
- **pandas:** 3.0.0
- **scipy:** 1.17.0
- **sklearn:** 1.8.0
- **matplotlib:** 3.10.8
- **seaborn:** 0.13.2

## §1 — Norm summary (all g_stock populations)

| Population          |      N |   Mean norm |   Std norm |     Min |     Max |
|:--------------------|-------:|------------:|-----------:|--------:|--------:|
| f_clue (full)       | 239406 |     30.6077 |     1.2024 | 24.4009 | 34.1953 |
| f_clue (val)        |  47933 |     30.6002 |     1.2029 | 25.2485 | 33.7204 |
| f_wndef (full)      |  53930 |     29.1093 |     1.3706 | 23.5365 | 33.6164 |
| f_wndef (val)       |  26152 |     29.1415 |     1.3713 | 23.5365 | 33.6164 |
| f_wnex (full)       |   8360 |     30.3636 |     1.1011 | 26.3180 | 33.4561 |
| f_clue, def ∈ wnex  |  79801 |     30.5489 |     1.2221 | 24.4009 | 34.1953 |
| f_wndef, wnex words |   8360 |     28.7728 |     1.4419 | 24.0539 | 32.8197 |

## §2 — ΔBIC and Ashman's D (all populations)

| Population | N | BIC(1) | BIC(2) | ΔBIC | Ashman's D | μ1 | σ1 | π1 | μ2 | σ2 | π2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| f_clue (full) | 239,406 | 767664.4 | 738858.6 | 28805.7 | 2.88 | 29.6208 | 0.7578 | 0.5013 | 31.5964 | 0.6079 | 0.4987 |
| f_clue, def ∈ wnex | 79,801 | 258502.1 | 248893.9 | 9608.2 | 2.88 | 29.5254 | 0.7637 | 0.4907 | 31.5345 | 0.6263 | 0.5093 |
| f_clue (val) | 47,933 | 153757.0 | 148031.3 | 5725.8 | 2.88 | 29.6247 | 0.7567 | 0.5062 | 31.6000 | 0.6078 | 0.4938 |
| f_wndef (full) | 53,930 | 187073.2 | 186673.0 | 400.3 | 1.88 | 28.1759 | 1.0418 | 0.5028 | 30.0532 | 0.9534 | 0.4972 |
| f_wnex (full) | 8,360 | 25352.3 | 24963.9 | 388.4 | 2.11 | 29.3723 | 0.8472 | 0.3929 | 31.0051 | 0.6964 | 0.6071 |
| f_wndef (val) | 26,152 | 90750.5 | 90538.2 | 212.3 | 1.90 | 28.2168 | 1.0410 | 0.5108 | 30.1072 | 0.9416 | 0.4892 |
| f_wndef, wnex words | 8,360 | 29862.1 | 29722.5 | 139.6 | 2.12 | 27.8072 | 1.0023 | 0.5406 | 29.9092 | 0.9774 | 0.4594 |

## §2 — Bimodal populations (two-criterion test)

| Population          |      N |    ΔBIC |   Ashman's D | Bimodal?   |
|:--------------------|-------:|--------:|-------------:|:-----------|
| f_clue (full)       | 239406 | 28805.7 |         2.88 | Yes        |
| f_clue, def ∈ wnex  |  79801 |  9608.2 |         2.88 | Yes        |
| f_clue (val)        |  47933 |  5725.8 |         2.88 | Yes        |
| f_wndef (full)      |  53930 |   400.3 |         1.88 | No         |
| f_wnex (full)       |   8360 |   388.4 |         2.11 | Yes        |
| f_wndef (val)       |  26152 |   212.3 |         1.90 | No         |
| f_wndef, wnex words |   8360 |   139.6 |         2.12 | Yes        |

Bimodal criteria: ΔBIC > 10 (Kass & Raftery 1995, "very strong" evidence) AND Ashman's D > 2 (Ashman, Bird & Zepf 1994, "cleanly separated" components).

ΔBIC scales with sample size, so at large N it flags any minor deviation from normality. Ashman's D is sample-size-independent and provides the effect-size filter that distinguishes meaningful bimodality from sample-size artefact.

## §2 — Full vs val equivalence

| Phrase type   | Split   |      N |    D |      μ1 |     σ1 |      μ2 |     σ2 |
|:--------------|:--------|-------:|-----:|--------:|-------:|--------:|-------:|
| f_clue        | full    | 239406 | 2.88 | 29.6208 | 0.7578 | 31.5964 | 0.6079 |
| f_clue        | val     |  47933 | 2.88 | 29.6247 | 0.7567 | 31.6000 | 0.6078 |
| f_wndef       | full    |  53930 | 1.88 | 28.1759 | 1.0418 | 30.0532 | 0.9534 |
| f_wndef       | val     |  26152 | 1.90 | 28.2168 | 1.0410 | 30.1072 | 0.9416 |

Full and val agree on Ashman's D and on all four GMM component parameters. The annotated display figure (§2d) therefore shows only the full-dataset version of each phrase type.

## §2 — GMM parameters (bimodal populations)

| Population          |      N |    ΔBIC |   Ashman's D |      μ1 |     σ1 |     π1 |      μ2 |     σ2 |     π2 |
|:--------------------|-------:|--------:|-------------:|--------:|-------:|-------:|--------:|-------:|-------:|
| f_clue (full)       | 239406 | 28805.7 |         2.88 | 29.6208 | 0.7578 | 0.5013 | 31.5964 | 0.6079 | 0.4987 |
| f_clue, def ∈ wnex  |  79801 |  9608.2 |         2.88 | 29.5254 | 0.7637 | 0.4907 | 31.5345 | 0.6263 | 0.5093 |
| f_clue (val)        |  47933 |  5725.8 |         2.88 | 29.6247 | 0.7567 | 0.5062 | 31.6000 | 0.6078 | 0.4938 |
| f_wnex (full)       |   8360 |   388.4 |         2.11 | 29.3723 | 0.8472 | 0.3929 | 31.0051 | 0.6964 | 0.6071 |
| f_wndef, wnex words |   8360 |   139.6 |         2.12 | 27.8072 | 1.0023 | 0.5406 | 29.9092 | 0.9774 | 0.4594 |

## §3 — Is L2 norm a property of the word?

- **ICC(1,1):** 0.1378  (weak word-level consistency)
- **Words with ≥2 appearances:** 17,207
- **Clue rows covered:** 239,406
- **Avg group size (k):** 13.33
- **Median within-word std:** 1.0482
- **Overall population std:** 1.2036
- **Cross-format Spearman ρ (mean f_clue vs f_wndef):** 0.257
- **Cross-format Spearman ρ (mean f_clue vs f_wnex):** 0.205
- **Cross-format Spearman ρ (f_wndef vs f_wnex):** 0.158
- **N words in cross-format comparison:** 5,540

## §4 — Surface + wordplay features predicting f_clue L2 norm

- **Original 6-feature R²:** 0.0403
- **New 15-feature R²:** 0.0413
- **Incremental R² from wordplay:** 0.0010
- **Adjusted R² (15 features):** 0.0412
- **Variance unexplained (1 − R², 15 features):** 0.9587
- **N rows:** 239,406

| Feature | β (standardized) | 95% CI | Raw coefficient |
|---|---|---|---|
| t_position | -0.1867 | [-0.2040, -0.1693] | -0.4705 |
| p_punctuation | -0.1268 | [-0.1320, -0.1216] | -0.1909 |
| t_tokens | -0.1209 | [-0.1331, -0.1088] | -0.1403 |
| t_p_ratio | +0.0639 | [+0.0510, +0.0769] | +0.5535 |
| p_tokens | +0.0530 | [+0.0428, +0.0632] | +0.0179 |
| double_def | -0.0320 | [-0.0375, -0.0264] | -0.1088 |
| t_capitalized | +0.0269 | [+0.0099, +0.0439] | +0.0543 |
| anagram_consec_words | -0.0160 | [-0.0207, -0.0112] | -0.0792 |
| hidden_fwd | +0.0129 | [+0.0081, +0.0176] | +0.0736 |
| hidden_rev | +0.0102 | [+0.0055, +0.0150] | +0.1117 |
| selection_firsts | +0.0081 | [+0.0034, +0.0129] | +0.1193 |
| anagram_single_word | -0.0048 | [-0.0096, -0.0001] | -0.0441 |
| selection_alt_rev | +0.0038 | [-0.0009, +0.0085] | +0.1774 |
| selection_alt | +0.0028 | [-0.0019, +0.0075] | +0.0455 |
| selection_lasts | -0.0027 | [-0.0074, +0.0021] | -0.1006 |

## §5 — Directional structure in f_clue embeddings

- **N pairs sampled:** 50,000
- **Mean cosine similarity:** 0.3179
- **Median cosine similarity:** 0.3035
- **Std:** 0.1168

### §5a — Centroid direction comparison across the bimodal halves

- **Median norm split point:** 30.6674
- **Lower-half size (norm < median):** 119,703
- **Upper-half size (norm ≥ median):** 119,703
- **Lower-half centroid pre-normalization L2 norm:** 0.5595
- **Upper-half centroid pre-normalization L2 norm:** 0.5737
- **Centroid-direction cosine similarity:** 0.9777

Centroid pre-normalization L2 norm reflects directional concentration within the half (close to 1 = tightly clustered around the mean direction; close to 0 = directionally diffuse). The centroid-direction cosine answers whether the two halves face the same direction.

Figure: `outputs/figures/norm_bimodality_fclue_pairwise_cosine.png`

## Figures

- `outputs/figures/norm_bimodality_survey_raw.png` — raw 3×3 KDE grid, all 7 populations grouped by phrase type (rows) × variant (columns), shared x-axis
- `outputs/figures/norm_bimodality_survey_gstock.png` — annotated 3×2 display: full-dataset (left) vs wnex-aligned subset (right) for each phrase type, with GMM overlays on bimodal panels and mean lines on unimodal panels
- `outputs/figures/norm_bimodality_icc_strip.png` — per-word f_clue norm strips for ~25 words spanning the f_clue mean-norm range, with f_wndef diamond and f_wnex triangle markers and the f_clue GMM mode reference lines
- `outputs/figures/norm_bimodality_cross_format_scatter.png` — 1×3 panel showing all three pairwise cross-format norm correlations (mean f_clue vs f_wndef, mean f_clue vs f_wnex, f_wndef vs f_wnex) on the wnex vocabulary, each with OLS trend and Spearman ρ annotation
- `outputs/figures/norm_bimodality_surface_features_binned.png` — binned-mean ± 95% CI panels, one per surface feature
- `outputs/figures/norm_bimodality_wordplay_features_binned.png` — 9-panel bar chart of mean L2 norm by True/False for each wordplay feature, with rescaled y-axes for visibility of small effects
- `outputs/figures/norm_bimodality_surface_features_coefficients.png` — standardized OLS coefficient dot-and-whisker plot
- `outputs/figures/norm_bimodality_fclue_pairwise_cosine.png` — pairwise cosine similarity KDE on 50,000 random pairs of unit-normalized f_clue embeddings
