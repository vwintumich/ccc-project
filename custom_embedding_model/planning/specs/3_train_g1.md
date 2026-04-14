# Spec: g_1 Triplet Construction and Training (Step A)

**Stage:** 3
**Notebook:** `notebooks/03_train_g1.ipynb`
**Scripts:** `scripts/train_g1.py`, `scripts/train_g1.sh`
**Date:** 2026-04-13
**Status:** Draft

## Purpose

Reproduce the T_1 triplet design from NB 09 using our new pipeline's data
artifacts, then fine-tune g_stock to produce g_1. This is **Step A** from the
design document (§6.5): a faithful reproduction of the initial training
procedure, creating a reproducible baseline that all future model comparisons
depend on.

We expect g_1 to exhibit the same failure pattern observed in NB 09 — ATE
becoming *more* negative due to format-specific compression of f_common_wndef
phrases. Confirming this failure under our new pipeline's cleaner data
preparation is a prerequisite for Step B (diagnosing the failure) and
Step C (designing an improved triplet).

## Relationship to NB 09

This spec reproduces T_1's **triplet design and training procedure** on our
pipeline's data. The table below summarizes what is the same and what differs:

| Aspect | NB 09 | NB 03 (this spec) |
|--------|-------|--------------------|
| **Triplet design** | Anchor=f_clue(def), Pos=f_common_wndef(ans), Neg=f_common_wndef(distractor) | **Same** |
| **Distractor source** | Pre-built in `dataset_harder.parquet` (cosine-similarity top-100 from M2 NB 05) | **Same distractors**, looked up from `dataset_harder.parquet` by (clue_id, definition_wn) |
| **Anchor phrase construction** | Built on-the-fly with regex in NB 09's `build_cale_anchor()` | Pre-built in `f_clue.csv` using `clue_utils.tag_definition_in_surface()` (same logic, validated in NB 02) |
| **Positive/Negative phrase construction** | Built on-the-fly with `build_decontext_phrase()`: `"<t>word</t>: WN definition"`, with bare `<t>word</t>` fallback for words without synsets | Pre-built in `f_common_wndef.csv`: `"<t>word</t>: WN definition"` for all words (no fallback needed — all words have synsets per NB 01 filtering) |
| **Source dataset** | `dataset_harder.parquet` (240,211 real pairs from M2 pipeline) | `clues_wn_filtered.csv` (239,406 rows from our pipeline), intersected with `dataset_harder.parquet` for distractor assignments |
| **Split** | 80/20 train/test at pair level | 30/20/50 train/validate/test at pair level (Decision 3) |
| **Full training rows** | 192,039 (before sampling) | ~69,921 (training split ∩ harder ∩ all phrases valid) |
| **NB 09 sample mode** | 37,593 rows from 20,000 sampled pairs | Not applicable — we train on the full ~69,921 |
| **Training approach** | `AutoModel` + manual `extract_concept_embedding` | **Same**: `AutoModel` + manual concept-aligned extraction (required for gradient flow) |
| **Hyperparameters** | lr=2e-5, margin=1.0, epochs=3, batch=32, AdamW(wd=0.01), linear warmup (10%), grad clip 1.0, mixed precision (fp16) | **Same** |
| **Phrase files** | Constructed on-the-fly, not saved | Pre-built, committed artifacts in `data/filtered_split/` |
| **Triplet file** | Not saved as a separate artifact | Saved as `data/triplets/g1.csv` + `g1_meta.json` (Decision 10) |

### Dataset size comparison

NB 09's published results used SAMPLE_MODE=True, which randomly sampled
20,000 unique (definition, answer) pairs from the 102,086 available training
pairs, yielding 37,593 training triplet rows. The full (unsampled) training
set was 192,039 rows.

Our pipeline produces **69,921 training triplet rows** — about 1.9× NB 09's
sampled training set and 0.36× its full training set. The smaller total is
because our split allocates only 30% to training (vs NB 09's 80%). This is
intentional: our 50% test allocation is an investment in final evaluation
credibility (Decision 9).

69,921 triplets is sufficient for reproduction. If the T_1 failure pattern
(format-specific compression) occurs even with 37K examples, it will occur
with 70K. The question is not whether the model can learn from this data, but
*what* it learns — and that is determined by the triplet design, not the
dataset size.

## Inputs

### For triplet construction (NB 03, local)

| File | Path (relative to `custom_embedding_model/`) | Purpose |
|------|------|---------|
| `clues_wn_filtered.csv` | `data/filtered_split/wn_synset/clues_wn_filtered.csv` | Training-split rows; provides (clue_id, definition, answer, definition_wn, answer_wn, split) |
| `f_clue.csv` | `data/filtered_split/wn_synset/clue_phrases/f_clue.csv` | Anchor phrases; indexed by (clue_id, definition) |
| `f_common_wndef.csv` | `data/filtered_split/wn_synset/wndef/f_common_wndef.csv` | Positive and Negative phrases; indexed by word |
| `dataset_harder.parquet` | `../../clue_misdirection/data/dataset_harder.parquet` | Distractor assignments; label=0 rows provide (clue_id, definition_wn) → distractor_wn mapping |

### For model training (train_g1.py, Great Lakes)

| File | Path | Purpose |
|------|------|---------|
| `g1.csv` | `data/triplets/g1.csv` | Training triplets (anchor, positive, negative phrase text) |

The training script reads only the triplet CSV. It does not need access to
the source phrase files or `dataset_harder.parquet`.

## Outputs

### From NB 03 (local)

| File | Path | Description |
|------|------|-------------|
| `g1.csv` | `data/triplets/g1.csv` | Training triplets; ~69,921 rows |
| `g1_meta.json` | `data/triplets/g1_meta.json` | Provenance metadata |
| Results file | `outputs/03_train_g1-results.md` | Coverage statistics, comparison to NB 09, triplet examples |

### From train_g1.py (Great Lakes)

| File | Location | Description |
|------|----------|-------------|
| Model weights | Google Drive: `custom_embedding_models/g1/` | Full fine-tuned CALE model (saved via `model.save_pretrained()`) |
| `README.md` | `models/g1/README.md` | Pointer to weights, hyperparameters, date, runtime |

## Implementation details

### Part 1: Notebook `03_train_g1.ipynb` (local, CPU)

This notebook constructs the triplet file and provides inspection and
statistics. It does not train the model.

#### §0 — Header and imports

Standard header per CLAUDE.md. Primary author: Victoria. Builds on:
`archive/09_learned_g_misdirection.ipynb` (Nathan — triplet design, training
loop).

Imports: `pathlib`, `pandas`, `numpy`, `json`, `time`.

Version reporting cell: print pandas, numpy versions.

Environment auto-detection. Define paths:
- `DATA_DIR` → `data/`
- `WN_DIR` → `data/filtered_split/wn_synset/`
- `TRIPLET_DIR` → `data/triplets/`
- `HARDER_PATH` → `../../clue_misdirection/data/dataset_harder.parquet`

#### §1 — Load source data

Load with `keep_default_na=False, na_values=[""]`:

1. `clues_wn_filtered.csv` — filter to `split == 'train'`
2. `f_clue.csv` — build lookup dict: `(clue_id, definition) → phrase`
3. `f_common_wndef.csv` — build lookup dict: `word → phrase`
4. `dataset_harder.parquet` — filter to `label == 0`, keep only
   `['clue_id', 'definition_wn', 'answer_wn']`, rename `answer_wn` to
   `distractor_wn`

Print row counts for each.

Assert `f_clue` lookup has 239,406 entries (full dataset).
Assert `f_common_wndef` lookup has 53,930 entries (full vocabulary).

#### §2 — Join training rows with distractors

Merge `wn_train` (training-split rows) with the distractor lookup on
`['clue_id', 'definition_wn']` using an inner join.

```python
wn_train = clues_wn[clues_wn['split'] == 'train'].copy()
merged = wn_train.merge(dist_lookup, on=['clue_id', 'definition_wn'], how='inner')
```

Report:
- Training rows before join: 72,107
- Training rows after join (have distractor): expected ~70,415
- Rows lost: expected ~1,692 (2.3%)

Explain in markdown: rows are lost because `dataset_harder.parquet` was built
from a slightly different upstream filtering pipeline (Milestone II). The
lost rows are clues that passed our NB 01 WordNet filter but were not present
in the Milestone II dataset. This is a small and acceptable loss.

#### §3 — Look up phrases for all three triplet roles

For each row in the merged DataFrame:

- **Anchor**: look up `(clue_id, definition)` in f_clue lookup
- **Positive**: look up `answer_wn` in f_common_wndef lookup
- **Negative**: look up `distractor_wn` in f_common_wndef lookup

```python
merged['anchor'] = merged.apply(
    lambda r: f_clue_lookup.get((r['clue_id'], r['definition'])), axis=1
)
merged['positive'] = merged['answer_wn'].map(f_wndef_lookup)
merged['negative'] = merged['distractor_wn'].map(f_wndef_lookup)
```

Drop rows where any phrase is missing. Report:
- Rows with all three phrases: expected ~69,921
- Rows lost to missing anchor: expected 0 (f_clue has 100% coverage)
- Rows lost to missing positive: expected 0 (all answers are in wndef vocab)
- Rows lost to missing negative: expected ~494 (222 distractor words not in
  our vocabulary_wndef — these are words that passed Milestone II's filtering
  but not our stricter NB 01 WordNet filter)

#### §4 — Build and save triplet file

Select output columns for `g1.csv`:

| Column | Type | Source |
|--------|------|--------|
| `clue_id` | int | from merged |
| `definition` | str | from merged (original case) |
| `answer_wn` | str | from merged |
| `distractor_wn` | str | from merged |
| `anchor` | str | f_clue phrase |
| `positive` | str | f_common_wndef phrase for answer |
| `negative` | str | f_common_wndef phrase for distractor |

Save to `data/triplets/g1.csv` with `index=False`.

Assertions:
- No null values in any column
- All anchor phrases contain `<t>` and `</t>`
- All positive phrases start with `<t>`
- All negative phrases start with `<t>`
- No rows where positive == negative (answer and distractor are different words)
- `split` column is NOT included (triplet file is training-only by definition)

#### §5 — Save provenance metadata

Build and save `data/triplets/g1_meta.json`:

```json
{
    "g_name": "g1",
    "triplet_design": "T_1",
    "description": "Step A reproduction of NB 09 triplet design",
    "anchor_f": "f_clue",
    "anchor_source": "data/filtered_split/wn_synset/clue_phrases/f_clue.csv",
    "positive_f": "f_common_wndef",
    "positive_source": "data/filtered_split/wn_synset/wndef/f_common_wndef.csv",
    "negative_f": "f_common_wndef",
    "negative_source": "data/filtered_split/wn_synset/wndef/f_common_wndef.csv",
    "distractor_source": "../../clue_misdirection/data/dataset_harder.parquet",
    "distractor_method": "cosine-similarity top-100 from Milestone II NB 05",
    "split": "train",
    "n_rows": <actual count>,
    "n_unique_clue_ids": <actual count>,
    "n_unique_pairs": <actual count>,
    "random_state": 42,
    "date_created": "<today>"
}
```

#### §6 — Inspection and examples

Display 5 example triplets showing all three phrases in full, so Victoria
can visually verify the construction.

Print statistics:
- Total triplet rows
- Unique clue_ids
- Unique (definition_wn, answer_wn) pairs
- Unique answer words
- Unique distractor words
- Overlap: words appearing as both answer and distractor across different rows

#### §7 — Comparison to NB 09

Print a side-by-side comparison table (matching the table in this spec's
"Relationship to NB 09" section) showing:
- NB 09 full training size: 192,039 rows from 102,086 pairs
- NB 09 sampled training size: 37,593 rows from 20,000 pairs
- Our training size: <actual> rows from <actual> pairs
- Ratio: our size / NB 09 sampled size

#### §8 — Summary cell

Per CLAUDE.md conventions. Report:
- Input file sizes
- Join statistics (rows at each step, loss fractions)
- Final triplet file size and location
- Key observation: this is a faithful T_1 reproduction; the only differences
  from NB 09 are upstream filtering (our NB 01 vs. M2 pipeline) and split
  proportions (30/20/50 vs. 80/20)

#### §9 — Results file

Write `outputs/03_train_g1-results.md` containing:
- Version stamps
- All coverage statistics from §2–§7
- The comparison table from §7
- File paths and sizes of all outputs

---

### Part 2: Script `scripts/train_g1.py` (Great Lakes, GPU)

This script reads the triplet CSV and fine-tunes g_stock to produce g_1.
It reproduces NB 09's training procedure (§4–§5).

#### Command-line arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--input` | (required) | Path to `triplets/g1.csv` |
| `--output-dir` | (required) | Directory for model weights |
| `--epochs` | `3` | Number of training epochs |
| `--batch-size` | `32` | Training batch size |
| `--lr` | `2e-5` | Learning rate |
| `--margin` | `1.0` | Triplet margin α |
| `--grad-accum` | `1` | Gradient accumulation steps |
| `--seed` | `42` | Random seed |
| `--sample` | `0` | If > 0, sample this many unique pairs for a test run |

#### §1 — Imports and configuration

Standard imports: `argparse`, `pathlib`, `time`, `json`, `numpy`, `pandas`,
`torch`, `torch.nn`.

From transformers: `AutoModel`, `AutoTokenizer`.
From torch.amp: `autocast`, `GradScaler`.

Print versions at startup per Decision 19: Python, torch (including CUDA
build suffix), transformers, numpy, pandas.

Set seeds: `numpy`, `torch`, `torch.cuda`.
Auto-detect device (CUDA > CPU).
Print device, GPU name, VRAM.

Model identifier: `gabrielloiseau/CALE-MBERT-en`.

#### §2 — Load triplet data

Load `g1.csv` with `keep_default_na=False, na_values=[""]`.

Assert columns: `clue_id`, `definition`, `answer_wn`, `distractor_wn`,
`anchor`, `positive`, `negative`.

Assert no null values.

If `--sample > 0`: build unique pair keys from `(definition, answer_wn)`,
sample `--sample` pairs using `random_state=seed`, filter to those pairs.
Print sampled row count.

Print total training rows.

#### §3 — PyTorch Dataset and DataLoader

Implement `TripletDataset` class (same as NB 09 §3.1):

```python
class TripletDataset(Dataset):
    def __init__(self, dataframe):
        self.anchors   = dataframe['anchor'].values
        self.positives = dataframe['positive'].values
        self.negatives = dataframe['negative'].values

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, idx):
        return {
            'anchor':   self.anchors[idx],
            'positive': self.positives[idx],
            'negative': self.negatives[idx],
        }
```

Create DataLoader with `shuffle=True`, `drop_last=True`, `num_workers=0`.

Print dataset size and batches per epoch.

#### §4 — Model loading and concept-aligned extraction

Load model and tokenizer:

```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.gradient_checkpointing_enable()
model = model.to(device)
model.train()
```

Implement `extract_concept_embedding(model, tokenizer, texts, device)`
exactly as in NB 09 §4.2:

1. Tokenize batch with `padding=True, truncation=True, max_length=128`
2. Forward pass to get `last_hidden_state`
3. For each text in batch:
   a. Find `<t>` and `</t>` character offsets
   b. Map character offsets to token indices using `token_to_chars()`
   c. Average hidden states of tokens within the delimited span
   d. Fallback to mean pooling if no delimiters found (defensive)
4. Return tensor of shape `(batch_size, 1024)`

**Why AutoModel, not SentenceTransformer:** The embedding generation script
(Stage 1d) uses `SentenceTransformer.encode()` because it is inference-only.
Training requires `AutoModel` because gradients must flow through the
extraction — specifically through the token selection and averaging that
implements concept-aligned embedding. `SentenceTransformer`'s training API
abstracts away this step, and its pooling layer may not match the manual
extraction exactly. For Step A faithfulness, we use the same approach as
NB 09.

Print model parameter count (total and trainable).

#### §5 — Training loop

Match NB 09 §5.1–§5.2 exactly:

**Loss function:**
```python
triplet_loss_fn = nn.TripletMarginLoss(margin=args.margin, p=2)
```

**Optimizer:**
```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=args.lr,
    weight_decay=0.01,
)
```

**Scheduler:** Linear warmup (10% of total steps) then linear decay to 0.

**Mixed precision:** `GradScaler` + `autocast(device_type='cuda', dtype=torch.float16)`.

**Gradient clipping:** `clip_grad_norm_(model.parameters(), max_norm=1.0)`.

**Training loop:** For each epoch, for each batch:
1. Forward pass (all three phrase types) inside `autocast` context
2. Compute triplet loss
3. Scale and backward
4. Unscale, clip, step, zero_grad (with `set_to_none=True`)
5. Delete intermediate tensors explicitly (memory management)
6. Log loss every 100 steps

**Epoch checkpoints:** Save `model_epoch{n}.pt` after each epoch containing
model state dict, optimizer state dict, epoch number, and average loss.
These are temporary checkpoints for recovery — the final model is saved
properly in §6.

Print per-epoch average loss and wall-clock time.

#### §6 — Save final model

After training completes, save the final model using HuggingFace's
`save_pretrained()`:

```python
model.save_pretrained(output_dir / 'model')
tokenizer.save_pretrained(output_dir / 'model')
```

This saves in a format loadable by both `AutoModel.from_pretrained()` and
`SentenceTransformer()` (for downstream embedding generation).

Also save a `training_log.json` with the per-step loss history.

Print:
- Total wall-clock runtime
- Final epoch loss
- Output directory contents and sizes
- All hyperparameters used

#### §7 — Summary output

Print a summary block to stdout (captured in SLURM log) containing all
information needed for `models/g1/README.md` and FINDINGS.md:
- Base model and version
- Triplet file used
- Total training rows
- All hyperparameters
- Per-epoch loss
- Total runtime
- GPU type and partition
- Python/torch/transformers versions

---

### Part 3: SLURM script `scripts/train_g1.sh`

```bash
#!/bin/bash
#SBATCH --job-name=train_g1
#SBATCH --account=siads696w26_class
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/train_g1_%j.out

conda activate nlp_env

python scripts/train_g1.py \
    --input data/triplets/g1.csv \
    --output-dir models/g1 \
    --epochs 3 \
    --batch-size 32 \
    --lr 2e-5 \
    --margin 1.0
```

**Notes:**
- 8 hours is conservative. NB 09 trained 3 epochs on ~37K rows (sample
  mode) in under 2 hours. Our ~70K rows at batch_size=32 should complete
  in 2–4 hours. The padding avoids job kills.
- `logs/` directory must exist before submission (`mkdir -p logs`).
- After completion, upload model weights from `models/g1/model/` to Google
  Drive folder "Research Project - NLP CCC's" under
  `custom_embedding_models/g1/`.
- Then scp is not needed for model weights (they go to Drive, not back to
  local). But do scp back the SLURM log and `training_log.json` for
  record-keeping.

---

### Part 4: Post-training local tasks

After the training job completes and weights are uploaded to Google Drive:

1. **Create `models/g1/README.md`** with metadata from the SLURM log output
   (hyperparameters, runtime, date, Drive path, versions).

2. **Update FINDINGS.md** Stage 3 section with:
   - Triplet file row count
   - Training hyperparameters
   - Per-epoch loss values
   - Wall-clock runtime
   - Great Lakes partition and GPU type
   - Environment versions per Decision 19

## Environment

- **NB 03 (triplet construction):** Local, CPU, `crossword` kernel.
  Only needs pandas, numpy, pyarrow (for parquet).
- **train_g1.py:** Great Lakes, GPU. Requires CUDA-capable PyTorch,
  `transformers`, `numpy`, `pandas`. The CALE model will be downloaded
  from HuggingFace on first use.

## Notebook structure

NB 03 uses §-numbered markdown sections (§0–§9). Includes environment
auto-detection. Writes results to `outputs/03_train_g1-results.md`.

The training script uses §-numbered comment banners for each logical block,
matching NB 09's style.
