# Spec: g1 training script (mean pooling, canonical)

**Stage:** 3
**Script:** `scripts/train_g1.py` + `scripts/train_g1.sh`
**Date:** 2026-04-14
**Status:** Draft

## Purpose

Fine-tune g_stock using triplet margin loss to produce g1 — the corrected
version of g1_tokenspan that uses CALE's canonical mean pooling (Decision 20)
instead of token span extraction. Same triplets, same hyperparameters,
different extraction method.

## Relationship to train_g1_tokenspan.py

This script is a modified copy of `scripts/train_g1_tokenspan.py`. The ONLY
substantive change is the extraction method used in the training loop:

- **train_g1_tokenspan.py:** calls `extract_concept_embedding()` — averages
  hidden states within the `<t></t>` span only
- **train_g1.py:** calls `extract_meanpool_embedding()` — attention-masked
  mean over all non-padding tokens (matching `SentenceTransformer.encode()`)

Everything else must remain identical: CLI arguments, data loading, Dataset
class, optimizer, scheduler, loss function, checkpointing, logging. This
ensures the two models differ ONLY in extraction method, making the Stage 5/6
comparison clean.

## Inputs

- `data/triplets/g1_train.csv` — same triplet file used by g1_tokenspan
  (69,921 training rows)

## Outputs

Written to `--output-dir` (default: `models/g1`):
- `model/` — HuggingFace `save_pretrained()` format (weights + tokenizer)
- `model_epoch{n}.pt` — per-epoch recovery checkpoints
- `training_log.json` — per-step loss, hyperparameters, versions

## Implementation details

### Start from train_g1_tokenspan.py

Copy `scripts/train_g1_tokenspan.py` to `scripts/train_g1.py` and make the
following targeted changes. Do NOT rewrite the script — preserve the structure,
comments, and all non-extraction logic verbatim.

### 1. Update the module docstring

Replace the description to say this uses mean pooling (canonical, Decision 20)
rather than concept-aligned token span extraction. Update the usage examples:

```
python scripts/train_g1.py \
    --input data/triplets/g1_train.csv \
    --output-dir models/g1 \
    --epochs 3 --batch-size 32 --lr 2e-5 --margin 1.0
```

Smoke test:
```
python scripts/train_g1.py \
    --input data/triplets/g1_train.csv \
    --output-dir models/g1_smoke \
    --epochs 1 --batch-size 8 --sample 200
```

### 2. Replace the extraction function

Remove `find_delimiter_char_offsets()` and `extract_concept_embedding()`.
Replace with `extract_meanpool_embedding()`:

```python
def extract_meanpool_embedding(model, tokenizer, texts, device, max_length=128):
    """Mean-pooled embedding matching SentenceTransformer.encode() behavior.

    Averages last_hidden_state over all non-padding tokens using the
    attention mask — CALE's canonical pooling (Decision 20). The <t></t>
    delimiters are still present in the input and guide the attention
    patterns during the forward pass; the pooling itself just averages
    the resulting hidden states.

    Gradients flow through this operation, enabling fine-tuning.
    """
    encoded = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True,
        max_length=max_length,
    ).to(device)
    outputs = model(**encoded)
    hidden_states = outputs.last_hidden_state        # (batch, seq_len, dim)
    mask = encoded["attention_mask"].unsqueeze(-1).to(hidden_states.dtype)
    summed = (hidden_states * mask).sum(dim=1)       # (batch, dim)
    counts = mask.sum(dim=1)                         # (batch, 1)
    return summed / counts                           # (batch, dim)
```

This is the same function as in `embed_val.py` §3, but with the note that
gradients flow through it (since it will be called during training, not
under `torch.no_grad()`).

### 3. Update the training loop calls

In §6, replace all three calls to `extract_concept_embedding(...)` with
`extract_meanpool_embedding(...)`:

```python
z_anchor = extract_meanpool_embedding(
    model, tokenizer, batch["anchor"], device, args.max_length
)
z_positive = extract_meanpool_embedding(
    model, tokenizer, batch["positive"], device, args.max_length
)
z_negative = extract_meanpool_embedding(
    model, tokenizer, batch["negative"], device, args.max_length
)
```

### 4. Update the section header comment for §5

Change from "Concept-aligned embedding extraction" to
"Mean-pooled embedding extraction (canonical)" and update the explanatory
comment to reference Decision 20 instead of NB 09.

### 5. Update the summary block

Change `models/g1/README.md` references to reflect g1 (not g1_tokenspan).
The summary should print `Pooling method: meanpool`.

### 6. Everything else unchanged

Preserve exactly:
- CLI arguments (§2)
- Data loading and validation (§3)
- TripletDataset class (§4)
- Training loop structure, optimizer, scheduler, loss function (§6)
- Gradient checkpointing, mixed precision, gradient accumulation
- Checkpointing logic (per-epoch .pt + final save_pretrained)
- Training log JSON format
- Version reporting

## SLURM wrapper

Create `scripts/train_g1.sh` modeled on `scripts/train_g1_tokenspan.sh`:

- Job name: `train_g1`
- Same resources (1 GPU, 32G, 4 CPUs, 4h wall)
- `source activate nlp_env` (not conda activate)
- `export PYTHONUNBUFFERED=1`
- Command: `python scripts/train_g1.py --input data/triplets/g1_train.csv --output-dir models/g1`
- Header comments: pre-submission checklist, submit command, scp commands for
  retrieving outputs

## Environment

Great Lakes (GPU). Expected runtime: ~49 min (same as g1_tokenspan — same
data size and hyperparameters; mean pooling is slightly cheaper than token
span extraction but the difference is negligible).

## Testing

Run a `--sample 200 --epochs 1` smoke test before the full run to verify
the mean pooling extraction works correctly during training.
