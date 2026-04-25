# Figure Visual Standards — custom_embedding_model

All figures in this component must use the encoding conventions below to
ensure consistency across notebooks, exploration figures, and the evaluation
report. When a figure shows any of the dimensions listed here, use the
specified encoding — do not invent new color/style mappings.

## Dimension encodings

### Model identity → Color

| Model | Color | Hex |
|-------|-------|-----|
| g_stock | Blue | `#1f77b4` |
| g1 | Orange | `#ff7f0e` |

For future models (g2, etc.), extend with colorblind-safe colors that are
perceptually distinct from blue and orange. Green (`#2ca02c`) and red
(`#d62728`) are reserved for training dynamics. Good candidates: purple
(`#9467bd`), teal (`#17becf`).

### Treatment condition → Shade + fill style

For overlaid distribution plots (histograms, KDE), use fill style to
distinguish treatment conditions:

| Condition | Fill style | Shade |
|-----------|-----------|-------|
| T=0 (no clue context) | Outline only (unfilled, `histtype='step'`) | Full model color |
| T=1 (clue context) | Filled | Full model color |

T=0 is the "empty" baseline, T=1 is the "filled" treatment. Use the model's
full color for both — the filled vs outline distinction provides the contrast.

For bar charts or other non-overlay contexts where fill style doesn't apply,
use light/deep shades:

| Model | T=0 (light) | T=1 (deep) |
|-------|-------------|------------|
| g_stock | `#aec7e8` | `#1f77b4` |
| g1 | `#ffbb78` | `#ff7f0e` |

### Distance metric → Fill pattern

| Metric | Pattern |
|--------|---------|
| Cosine similarity | Solid fill |
| L2 distance | Diagonal hatch (`//`) |

L2 is always hatched, even when appearing alone — so any bar or histogram is
instantly recognizable as L2 without reading axis labels. Cosine is always
solid.

### Training dynamics → Green / Red

Training and validation curves use green and red, following standard ML
convention (green = training performance, red = validation performance):

| Data source | Color | Hex |
|-------------|-------|-----|
| Training | Green | `#2ca02c` |
| Validation | Red | `#d62728` |

These colors must not be reused for model identity (blue/orange) or treatment
condition. They apply to any figure showing per-epoch metrics: loss curves,
accuracy curves, margin curves, etc.

## General figure conventions

- Save to `outputs/figures/` as PNG at 300 dpi
- Use `random_state=42` for any sampling
- Use readable axis labels (not raw column names)
- Include a legend when more than one visual encoding is present
- When comparing g_stock and g1 in paired panels, use the same
  axis range for both panels so the reader can compare directly
