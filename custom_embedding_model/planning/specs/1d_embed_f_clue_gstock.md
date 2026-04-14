# Spec: g_stock f_clue Embedding Generation

**Stage:** 1d
**Scripts:** `scripts/embed_f_clue_gstock.py`, `scripts/embed_f_clue_gstock.sh`
**Date:** 2026-04-13
**Status:** Draft

## Purpose

Encode all f_clue phrases from the full `clues_wn_filtered.csv` dataset using
g_stock (unmodified CALE). This produces the f_clue embedding array and its
companion index file, which are used by all downstream hypothesis testing
(Stage 5). Computed once for g_stock and never regenerated unless the phrase
file changes.

## Inputs

- `data/filtered_split/wn_synset/clue_phrases/f_clue.csv` — 239,406 rows;
  columns: `clue_id`, `definition`, `split`, `phrase`

## Outputs

All outputs go to `data/embeddings/g_stock/`:

| File | Description | Shape / Schema |
|------|-------------|----------------|
| `f_clue.npy` | Dense embedding array, float32, unnormalized | (239406, 1024) |
| `f_clue_index.csv` | Row-to-key mapping | columns: `clue_id`, `definition`, `row` |

## Implementation details

### Embedding approach: SentenceTransformer.encode()

This script uses the `sentence-transformers` library's high-level
`model.encode()` API, **not** the manual `AutoModel` + token-offset
extraction approach from NB 09.

CALE is distributed as a sentence-transformers model with a built-in
pooling layer trained to perform concept-aligned extraction: when the
input text contains `<t></t>` delimiters, the pooling layer automatically
averages only the hidden states of the tokens within the delimited span.
Calling `model.encode()` invokes this trained pooling layer.

NB 09 bypassed `sentence-transformers` entirely and reimplemented the
extraction manually using `AutoModel`, character-to-token offset mapping,
and hand-written span averaging. That manual approach is more complex,
harder to verify, and may diverge subtly from CALE's trained pooling
behavior. It is also necessary for training (where gradients must flow
through the extraction), but for inference it is unnecessary.

The validated `clue_misdirection` embedding pipeline
(`clue_misdirection/scripts/embed_phrases.py`) used
`SentenceTransformer.encode()` successfully on ~240K phrases.

### `scripts/embed_f_clue_gstock.py`

#### Command-line arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--input` | (required) | Path to `f_clue.csv` |
| `--output-dir` | (required) | Directory for output files |
| `--batch-size` | `64` | Encoding batch size |
| `--sample` | `0` | If > 0, embed only this many rows (for testing) |

#### §1 — Imports and configuration

Standard imports: `argparse`, `pathlib`, `time`, `numpy`, `pandas`, `torch`.

From sentence-transformers: `SentenceTransformer`.

Set reproducibility seeds (`numpy`, `torch`; seed 42). Auto-detect device
(CUDA > CPU). Print device, GPU name if available, torch version, and
sentence-transformers version.

Model identifier: `gabrielloiseau/CALE-MBERT-en`.

#### §2 — Load phrase data

Load `f_clue.csv` with `keep_default_na=False, na_values=[""]`.

Assert expected columns: `clue_id`, `definition`, `split`, `phrase`.
Assert no null values in `phrase`.

If `--sample` > 0, take the first N rows (deterministic subset for testing).

Print row count.

#### §3 — Load model

```python
model = SentenceTransformer(MODEL_NAME)
```

Verify embedding dimension:
```python
assert model.get_sentence_embedding_dimension() == 1024
```

Print model name, embedding dimension, and whether GPU is being used.

#### §4 — Encode phrases

Encode all phrases in one call. `model.encode()` handles batching and
GPU transfer internally:

```python
phrases = df["phrase"].tolist()

embeddings = model.encode(
    phrases,
    batch_size=args.batch_size,
    show_progress_bar=True,
    normalize_embeddings=False,  # Save raw embeddings; downstream code
)                                # normalizes as needed for cosine similarity
embeddings = np.array(embeddings, dtype=np.float32)
```

**`normalize_embeddings=False`:** This preserves the raw embedding
magnitudes in the saved `.npy` file. Downstream ATE computation normalizes
vectors itself inside `rowwise_cosine()`. Pre-normalizing would discard
magnitude information that may be useful for other analyses (e.g.,
detecting degenerate embeddings, comparing L2 norms across models).
`False` is the default, but stating it explicitly makes the choice visible.

Assert output shape is `(N, 1024)`.
Assert no NaN values.
Assert no all-zero rows (every embedding should have nonzero L2 norm).

Print encoding time.

#### §5 — Build index file

Construct the index DataFrame from the input data:

| Column | Source |
|--------|--------|
| `clue_id` | from f_clue.csv |
| `definition` | from f_clue.csv |
| `row` | 0-indexed position matching the embedding array |

Assert `len(index) == embeddings.shape[0]`.

#### §6 — Save outputs atomically

Create the output directory if it doesn't exist.

Save to temporary paths first, then rename to prevent corrupt partial
files if the job is killed mid-write:

```python
# np.save() auto-appends ".npy" when the path doesn't already end in it,
# so temp names must end in ".npy" to avoid a double extension.
tmp_npy = output_dir / "f_clue.tmp.npy"
tmp_csv = output_dir / "f_clue_index.tmp.csv"

np.save(tmp_npy, embeddings)
index_df.to_csv(tmp_csv, index=False)

tmp_npy.rename(output_dir / "f_clue.npy")
tmp_csv.rename(output_dir / "f_clue_index.csv")
```

#### §7 — Summary

Print:
- Total rows embedded
- Embedding array shape and dtype
- Output file paths and sizes (MB)
- Total wall-clock runtime
- Any warnings encountered

### `scripts/embed_f_clue_gstock.sh`

SLURM submission script. Key parameters:

```bash
#!/bin/bash
#SBATCH --job-name=embed_fclue_gstock
#SBATCH --account=<account>
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/embed_fclue_gstock_%j.out

module load python/3.10
# Activate the appropriate conda environment or venv

python scripts/embed_f_clue_gstock.py \
    --input data/filtered_split/wn_synset/clue_phrases/f_clue.csv \
    --output-dir data/embeddings/g_stock \
    --batch-size 64
```

**Notes for Victoria and Nathan:**
- Replace `<account>` with the actual Great Lakes account.
- Adjust `module load` and environment activation to match your setup.
- The `logs/` directory must exist before submission (`mkdir -p logs`).
- 4 hours is conservative for ~239K phrases at batch size 64. The
  `clue_misdirection` pipeline embedded a similar number of phrases in
  10-20 minutes on a V100/A40. Padding the time limit avoids job kills.
- After the job completes, `scp` the two output files back to your local
  machine:
  ```
  scp <user>@greatlakes-xfer.arc-ts.umich.edu:<path>/data/embeddings/g_stock/f_clue.npy \
      custom_embedding_model/data/embeddings/g_stock/
  scp <user>@greatlakes-xfer.arc-ts.umich.edu:<path>/data/embeddings/g_stock/f_clue_index.csv \
      custom_embedding_model/data/embeddings/g_stock/
  ```

## Environment

Great Lakes (GPU). Requires CUDA-capable PyTorch and `sentence-transformers`.
The CALE model (`gabrielloiseau/CALE-MBERT-en`) will be downloaded from
HuggingFace on first use — ensure internet access or pre-cache the model.

## Verification

After running, verify locally:
- `f_clue.npy` shape is `(239406, 1024)` and dtype is `float32`
- `f_clue_index.csv` has 239,406 rows and columns `clue_id`, `definition`,
  `row`
- Row values in the index are contiguous from 0 to 239,405
- No NaN values and no all-zero rows in the embedding array
- Spot-check: load both files, pick a known clue_id, verify the embedding
  is a non-zero 1024-dim vector with reasonable magnitude
