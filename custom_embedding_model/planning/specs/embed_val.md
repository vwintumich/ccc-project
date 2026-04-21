# Spec: Validation-split embedding generation script

**Stage:** 4 (also covers g_stock vocabulary embeddings needed for Stages 5/6)
**Script:** `scripts/embed_val.py` + `scripts/embed_val_gstock.sh` + `scripts/embed_val_g1.sh`
**Date:** 2026-04-13
**Status:** Draft

## Purpose

Generate validation-split embeddings for a given model (g_stock or g_1) across
all three phrase types: f_clue, f_common_wndef, and f_common_wnex. This is a
single reusable script, run once per model, that produces the embedding arrays
Stages 5 and 6 need for ATE computation and cross-f generalization testing.

## Inputs

All paths relative to `custom_embedding_model/`.

**Phrase files (contain the text to embed):**
- `data/filtered_split/wn_synset/clue_phrases/f_clue.csv` — has `split` column; filter to `split == 'validate'`
- `data/filtered_split/wn_synset/wndef/f_common_wndef.csv` — one row per vocabulary word; filter to validation words
- `data/filtered_split/wn_synset/wnex/f_common_wnex.csv` — one row per vocabulary word; filter to validation words

**Vocabulary files (define which words are in the validation split):**
- `data/filtered_split/wn_synset/wndef/vocabulary_wndef_val.csv` — 26,152 words
- `data/filtered_split/wn_synset/wnex/vocabulary_wnex_val.csv` — 3,008 words

**Model weights:**
- g_stock: load from HuggingFace ID `gabrielloiseau/CALE-MBERT-en`
- g_1: load from local path `models/g1/model/`

## Outputs

All written to `data/embeddings/<g_name>/` where `<g_name>` is specified via
`--output-dir`.

| File | Index file | Expected shape |
|------|-----------|----------------|
| `f_clue_val.npy` | `f_clue_val_index.csv` (clue_id, definition, row) | (~47,933, 1024) |
| `f_common_wndef_val.npy` | `vocabulary_wndef_val.csv` (word ordering = row ordering) | (26,152, 1024) |
| `f_common_wnex_val.npy` | `vocabulary_wnex_val.csv` (word ordering = row ordering) | (3,008, 1024) |

Estimated sizes: ~187 MB + ~102 MB + ~12 MB = ~301 MB per model.

## Implementation details

### CLI arguments

```
--model-path    Path or HuggingFace ID for the model to use
--output-dir    Output directory (e.g., data/embeddings/g_stock or data/embeddings/g1)
--data-dir      Root of filtered_split data (default: data/filtered_split/wn_synset)
--batch-size    Encoding batch size (default: 64)
--sample        If > 0, take first N rows of each phrase set (smoke testing)
```

### Model loading

The training script (`train_g1.py`) saves g_1 using `AutoModel.save_pretrained()`
plus `tokenizer.save_pretrained()`. This produces raw transformer weights
without the SentenceTransformer wrapper config (no `modules.json`). The
existing `embed_f_clue_gstock.py` uses `SentenceTransformer.encode()` for
g_stock.

**Design consideration for the Coder:** The script must use a loading and
extraction method that works for both g_stock (HuggingFace SentenceTransformer
model) and g_1 (AutoModel save). Two options:

1. **Use AutoModel + manual concept-aligned extraction for everything.** Port
   the `extract_concept_embedding()` function from `train_g1.py`, run it
   under `torch.no_grad()` in eval mode. This is guaranteed to match what
   g_1 learned during training. For g_stock, load with
   `AutoModel.from_pretrained("gabrielloiseau/CALE-MBERT-en")`.

2. **Use SentenceTransformer for everything.** Copy the CALE sentence-
   transformers config files into the g_1 model directory so
   SentenceTransformer can load it. Simpler encoding code but requires
   verifying that SentenceTransformer's pooling matches the manual extraction.

**Recommendation:** Option 1 is safer. The manual extraction function is
already written and tested, and it avoids any ambiguity about whether
SentenceTransformer's pooling matches the training-time extraction. The
performance cost is trivial for validation-set sizes.

**Important consistency note:** The existing `g_stock/f_clue.npy` (239,406
rows, full dataset) was generated with `SentenceTransformer.encode()`. If
this script uses AutoModel extraction, the Coder should add a verification
step: load a small sample of the existing `g_stock/f_clue.npy`, re-embed
those same phrases with AutoModel extraction, and assert that cosine
similarity is > 0.999. If the methods disagree, flag this as a blocking
issue. Print the verification result to stdout.

### Embedding procedure

For each of the three phrase types, in sequence:

**1. f_clue_val:**
- Load `f_clue.csv` with `keep_default_na=False`
- Filter to `split == 'validate'`
- Extract the `phrase` column as input texts
- Encode with the model
- Build index DataFrame with columns (clue_id, definition, row)
- Save `f_clue_val.npy` and `f_clue_val_index.csv` atomically

**2. f_common_wndef_val:**
- Load `vocabulary_wndef_val.csv` with `keep_default_na=False`
- Load `f_common_wndef.csv` with `keep_default_na=False`
- Inner-join on `word` to get phrases for validation words only
- **Assert** the join result has exactly `len(vocabulary_wndef_val)` rows
- **Assert** the row ordering after join matches `vocabulary_wndef_val.csv`
  ordering (the .npy array must be indexed by the vocabulary file)
- Encode the `phrase` column
- Save `f_common_wndef_val.npy` atomically

**3. f_common_wnex_val:**
- Same procedure as wndef, using `vocabulary_wnex_val.csv` and
  `f_common_wnex.csv`
- Save `f_common_wnex_val.npy` atomically

### Validation checks (for each embedding array)

- Shape matches expected (n_rows, 1024)
- No NaN values
- No all-zero rows
- Print L2 norm range

### Atomic saves

Follow the pattern from `embed_f_clue_gstock.py`: write to `.tmp.npy` then
rename. Remember that `np.save()` auto-appends `.npy`, so temp names must
end in `.npy` to avoid double extensions.

### Version and environment reporting

Per Decision 19: print Python, torch, sentence-transformers (if used),
transformers, numpy versions at startup. Print a summary block at the end
with row counts, shapes, file sizes, encoding times, and total runtime.

### SLURM scripts

Create two SLURM submission scripts:

**`scripts/embed_val_gstock.sh`:**
- Job name: `embed_val_gstock`
- Same resource allocation as `embed_f_clue_gstock.sh` (1 GPU, 32G RAM, 4 CPUs)
- Wall time: 1 hour (conservative; expect ~5 min)
- Fix the two known SLURM issues:
  - Use `source activate nlp_env` instead of `conda activate nlp_env`
  - Add `export PYTHONUNBUFFERED=1` before the python command
- Pass `--model-path gabrielloiseau/CALE-MBERT-en --output-dir data/embeddings/g_stock`

**`scripts/embed_val_g1.sh`:**
- Same resources and fixes
- Pass `--model-path models/g1/model --output-dir data/embeddings/g1`

Both scripts should include header comments documenting:
- What to verify before submitting (data files uploaded, logs/ dir exists)
- The submit command
- scp commands for retrieving output files afterward (use full absolute paths
  with `/home/vwinters/ccc-project/custom_embedding_model/...`)

## Data files needed on Great Lakes

Before submitting these jobs, the following files must be present on Great
Lakes at `/home/vwinters/ccc-project/custom_embedding_model/`:

- `data/filtered_split/wn_synset/clue_phrases/f_clue.csv`
- `data/filtered_split/wn_synset/wndef/f_common_wndef.csv`
- `data/filtered_split/wn_synset/wndef/vocabulary_wndef_val.csv`
- `data/filtered_split/wn_synset/wnex/f_common_wnex.csv`
- `data/filtered_split/wn_synset/wnex/vocabulary_wnex_val.csv`
- `data/embeddings/g_stock/f_clue.npy` (for the consistency verification step)
- `data/embeddings/g_stock/f_clue_index.csv` (for the consistency verification)
- `models/g1/model/` (for the g_1 run only; already present from training)

## Environment

Great Lakes (GPU). Both jobs can be submitted simultaneously if two GPU
slots are available — they are independent.

## Estimated runtime

Based on the Stage 1d benchmark (675 phrases/sec on V100 with batch_size=64):
- f_clue_val (~47,933 phrases): ~71s
- f_common_wndef_val (26,152 phrases): ~39s
- f_common_wnex_val (3,008 phrases): ~5s
- Model loading: ~30s

Total per model: ~3-5 min. Both models combined: ~10 min.
