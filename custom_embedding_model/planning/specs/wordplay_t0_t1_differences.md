# Spec: T=0 and T=1 Difference CIs for Wordplay Breakdown

**Stage:** Exploration (revision to existing notebook)
**Notebook:** `planning/exploration/wordplay_ate_breakdown.ipynb`
**Date:** 2026-04-20
**Status:** Draft

## Purpose

Add between-category confidence intervals on T=0 and T=1 component means to
the existing wordplay ATE breakdown notebook. The current notebook reports
T=0 and T=1 means per category but provides CIs only on the composite ATE.
We need CIs on the *difference* between categories to determine whether
observed T=0 and T=1 patterns are statistically meaningful — in particular,
whether different clue types show genuinely different contextualized (T=1)
or decontextualized (T=0) similarity under g_stock and g1.

## Inputs

No new inputs. Uses the same data already loaded in §1 of the existing
notebook: validation clues, wordplay metadata, full-vocab wndef embeddings,
and f_clue_val embeddings for both g_stock and g1.

## Outputs

Updates to existing outputs:

- `outputs/wordplay_ate_breakdown-results.md` — add two new tables:
  structural T=0/T=1 differences and letterplay T=0/T=1 differences
- `outputs/figures/wp_t0_t1_structural.png` — new figure (T=0 and T=1
  difference dot plot for double_def vs standard)
- `outputs/figures/wp_t0_t1_individual_letterplay.png` — new figure (T=0
  and T=1 difference dot plot for each individual letterplay type vs
  no_letterplay)

## Implementation Details

### New section: §6a — T=0 and T=1 difference CIs

Insert a new section between the current §6 (structural comparison) and §7
(summary). Renumber downstream sections as needed.

**Method: Welch-style confidence interval on the difference of means.**

For two independent groups A and B:

```
diff = mean_A − mean_B
SE = sqrt(var_A / N_A + var_B / N_B)
CI = diff ± 1.96 × SE
```

where `var_A` and `var_B` are sample variances of the per-row T=0 (or T=1)
values within each group. This does not assume equal variances.

**Computation:**

Define a function `compute_component_diff(mask_a, mask_b, model, component)`
where:
- `mask_a`, `mask_b` are boolean arrays selecting the two groups
- `model` is `"g_stock"` or `"g1"`
- `component` is `"t0"` or `"t1"`

The function indexes into the precomputed `per_row[model]["t0"]` or
`per_row[model]["t1"]` vectors (already computed in §4), computes the
difference of means and the Welch CI, and returns a dict:

```python
{
    "component": "t0" or "t1",
    "model": model name,
    "group_a": category name,
    "group_b": baseline name,
    "n_a": int,
    "n_b": int,
    "mean_a": float,
    "mean_b": float,
    "diff": float,        # mean_a - mean_b
    "se": float,
    "ci_lo": float,
    "ci_hi": float,
    "excludes_zero": bool  # True if CI does not contain 0
}
```

**Structural comparisons (double_def − standard):**

Compute for all 4 combinations: (T=0, T=1) × (g_stock, g1).

Group A = `categories["double_def"]`, Group B = `categories["standard"]`.

Print a formatted table with columns: component, model, N_dd, N_std,
mean_dd, mean_std, diff, CI, excludes_zero.

**Letterplay comparisons (each individual type − no_letterplay):**

For each of the 10 individual letterplay types, compute for all 4
combinations: (T=0, T=1) × (g_stock, g1).

Group A = `categories[letterplay_name]`, Group B =
`categories["no_letterplay"]`.

Print a formatted table with columns: category, component, model, N_type,
N_baseline, mean_type, mean_baseline, diff, CI, excludes_zero, small_n.

Flag categories with N < 50 as `small_n` (consistent with existing
convention).

### New figures

**Figure: `wp_t0_t1_structural.png`**

A dot plot with 4 rows: T=0 g_stock, T=0 g1, T=1 g_stock, T=1 g1. Each
row shows the difference (double_def − standard) with a horizontal CI bar.
Dashed vertical line at diff = 0. Use the same color convention as the
existing plots (blue = g_stock, orange = g1). Distinguish T=0 vs T=1 by
row label and/or marker style. Title: "T=0 and T=1 differences:
double_def − standard (95% Welch CI)".

Save to `outputs/figures/wp_t0_t1_structural.png` (300 dpi).

**Figure: `wp_t0_t1_individual_letterplay.png`**

Two side-by-side subplots (or vertically stacked): one for T=0 differences,
one for T=1 differences. Each subplot has the 10 individual letterplay types
as y-axis rows (ordered by descending N, consistent with the existing
individual letterplay figure). Each row shows two markers (g_stock and g1)
with CI bars, connected by a thin gray line. Dashed vertical line at
diff = 0. Small-N categories at reduced alpha.

Title: "T=0 and T=1 differences: letterplay type − no_letterplay
(95% Welch CI)".

Save to `outputs/figures/wp_t0_t1_individual_letterplay.png` (300 dpi).

### Updates to existing sections

**§7 (Summary cell):** Add a bullet summarizing the T=0/T=1 difference
findings. Specifically note:
- Which structural and letterplay T=0/T=1 differences have CIs excluding
  zero
- Whether the T=1 divergence under g1 (observed in the raw means) holds up
  with CIs

**§8 (Results file):** Add two new sections to
`outputs/wordplay_ate_breakdown-results.md`:

```
## T=0 and T=1 component differences: structural

[table with: component, model, n_dd, n_std, mean_dd, mean_std, diff, ci_lo,
ci_hi, excludes_zero]

## T=0 and T=1 component differences: individual letterplay

[table with: category, component, model, n_type, n_baseline, mean_type,
mean_baseline, diff, ci_lo, ci_hi, excludes_zero, small_n]
```

### Markdown narrative

Include a markdown cell after the computation discussing:
- Which differences are statistically meaningful (CI excludes zero)
- The pattern across models: do g_stock and g1 show the same between-category
  structure on T=0 and T=1, or does fine-tuning change which categories
  differ?
- Whether the T=1 divergence under g1 between letterplay types is confirmed
  or was an artifact of looking at raw means without uncertainty

## Environment

Local (CPU). No new data loading required. All computation uses the
precomputed `per_row` vectors from §4.

## Notebook structure

This is a revision to an existing notebook. The new section should be
inserted after the current §6 and before the current §7 (summary) and §8
(results file). Renumber as needed. Maintain the existing §-numbered
markdown section convention.
