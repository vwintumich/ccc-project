# Spec: Validation embedding script revision — add pooling flag

**Stage:** 4
**Script:** `scripts/embed_val.py` (revise in place), plus new/revised SLURM wrappers
**Date:** 2026-04-14
**Status:** Draft
**Supersedes:** `planning/specs/embed_val.md`

## Purpose

Revise `embed_val.py` to support two embedding extraction methods via a
`--pooling` flag: `meanpool` (canonical CALE usage per Decision 20) and
`tokenspan` (NB 09 historical method). This single script serves all four
embedding runs needed for Stage 5 hypothesis testing.

## Context

The existing `embed_val.py` uses only the token span extraction method
(ported from `train_g1.py`). The consistency check against
`g_stock/f_clue.npy` (generated via `SentenceTransformer.encode()`) failed
with mean cosine similarity 0.926, confirming the two methods produce
different embeddings. Decision 20 established mean pooling as canonical.
We need both methods to evaluate g1_tokenspan fairly against its own
baseline and to evaluate the corrected g1 against the canonical baseline.

## Inputs

Unchanged from the original spec. All paths relative to
`custom_embedding_model/`.

**Phrase files:**
- `data/filtered_split/wn_synset/clue_phrases/f_clue.csv`
- `data/filtered_split/wn_synset/wndef/f_common_wndef.csv`
- `data/filtered_split/wn_synset/wnex/f_common_wnex.csv`

**Vocabulary files:**
- `data/filtered_split/wn_synset/wndef/vocabulary_wndef_val.csv` (26,152 words)
- `data/filtered_split/wn_synset/wnex/vocabulary_wnex_val.csv` (3,008 words)

**Model weights:**
- g_stock: `gabrielloiseau/CALE-MBERT-en` (HuggingFace ID)
- g1_tokenspan: `models/g1_tokenspan/model/` (local path on Great Lakes)
- g1: `models/g1/model/` (local path on Great Lakes — does not exist yet)

**Consistency reference:**
- `data/embeddings/g_stock/f_clue.npy` + `f_clue_index.csv`

## Outputs

Per run, written to `--output-dir`:

| File | Index | Shape |
|------|-------|-------|
| `f_clue_val.npy` | `f_clue_val_index.csv` | (~47,933, 1024) |
| `f_common_wndef_val.npy` | `vocabulary_wndef_val.csv` | (26,152, 1024) |
| `f_common_wnex_val.npy` | `vocabulary_wnex_val.csv` | (3,008, 1024) |

Four separate output directories for the four runs:
- `data/embeddings/g_stock_tokenspan/`
- `data/embeddings/g1_tokenspan/`
- `data/embeddings/g_stock/` (adds `_val` files alongside existing full-scope files)
- `data/embeddings/g1/` (created after g1 is trained — Phase 6)

## Implementation details

### CLI changes

Add one new required argument:

```
--pooling       Required. Choices: "meanpool", "tokenspan".
                meanpool = attention-masked mean over all non-padding tokens
                           (matches SentenceTransformer.encode()).
                tokenspan = average hidden states within <t></t> span only
                            (NB 09 / train_g1_tokenspan.py method).
```

Remove `--skip-verify`. The consistency check logic changes (see below).

### Extraction functions

**Keep** the existing `extract_concept_embedding()` and
`find_delimiter_char_offsets()` functions unchanged. They are used when
`--pooling tokenspan`.

**Add** a new `extract_meanpool_embedding()` function:

```python
def extract_meanpool_embedding(model, tokenizer, texts, device, max_length=128):
    """Mean-pool embedding matching SentenceTransformer.encode() behavior.

    Averages last_hidden_state over all non-padding tokens using the
    attention mask. This is CALE's canonical pooling (Decision 20).
    """
    encoded = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True,
        max_length=max_length,
    ).to(device)
    outputs = model(**encoded)
    hidden_states = outputs.last_hidden_state        # (batch, seq_len, dim)
    mask = encoded["attention_mask"].unsqueeze(-1)    # (batch, seq_len, 1)
    summed = (hidden_states * mask).sum(dim=1)        # (batch, dim)
    counts = mask.sum(dim=1)                          # (batch, 1)
    return summed / counts                            # (batch, dim)
```

### Routing in encode_phrases()

`encode_phrases()` currently calls `extract_concept_embedding()` directly.
Change it to accept a `pooling` parameter and dispatch:

```python
def encode_phrases(model, tokenizer, phrases, device, batch_size, max_length,
                   pooling):
    ...
    if pooling == "tokenspan":
        vecs = extract_concept_embedding(...)
    elif pooling == "meanpool":
        vecs = extract_meanpool_embedding(...)
    ...
```

Pass `args.pooling` through from `main()`.

### Consistency check changes

The consistency check compares this script's extraction against the existing
`g_stock/f_clue.npy` (which was produced with `SentenceTransformer.encode()`,
i.e., mean pooling).

**New logic:** Run the consistency check automatically when **both** conditions
are met:
1. `--pooling meanpool`
2. The reference files exist (`g_stock/f_clue.npy` and `f_clue_index.csv`)

Skip it otherwise (tokenspan will not match; fine-tuned model weights will
not match). Remove the `--skip-verify` flag entirely — the script decides
automatically.

Print a clear message either way:
- If running: "Consistency check: verifying meanpool extraction matches existing g_stock/f_clue.npy ..."
- If skipping due to tokenspan: "Consistency check skipped: tokenspan extraction is expected to differ from g_stock/f_clue.npy (produced with mean pooling)."
- If skipping due to missing files: "Consistency check skipped: reference files not found."

### Docstring update

Update the module docstring to reflect the `--pooling` flag and the four
usage patterns:

```
# g_stock with mean pooling (canonical baseline):
python scripts/embed_val.py \
    --model-path gabrielloiseau/CALE-MBERT-en \
    --output-dir data/embeddings/g_stock \
    --pooling meanpool --batch-size 64

# g_stock with token span extraction (baseline for g1_tokenspan comparison):
python scripts/embed_val.py \
    --model-path gabrielloiseau/CALE-MBERT-en \
    --output-dir data/embeddings/g_stock_tokenspan \
    --pooling tokenspan --batch-size 64

# g1_tokenspan:
python scripts/embed_val.py \
    --model-path models/g1_tokenspan/model \
    --output-dir data/embeddings/g1_tokenspan \
    --pooling tokenspan --batch-size 64

# g1 (after training):
python scripts/embed_val.py \
    --model-path models/g1/model \
    --output-dir data/embeddings/g1 \
    --pooling meanpool --batch-size 64
```

### Print the pooling method in the summary block

Add `Pooling method: meanpool` or `Pooling method: tokenspan` to the summary
block at the end, so the SLURM log permanently records which method was used.

### Everything else unchanged

All other behavior stays the same: atomic saves, validation checks, version
reporting, `--sample` flag, per-f embedding routines, progress printing.

## SLURM wrappers

### Existing files to revise

**`scripts/embed_val_gstock.sh`** — revise to add `--pooling meanpool`.
Update header comments. This produces canonical g_stock val embeddings.

**`scripts/embed_val_g1.sh`** — **delete** (or leave for later). The g1
model (mean pooling) does not exist yet. Will be created in Phase 6 after
g1 is trained.

### New files to create

**`scripts/embed_val_g_stock_tokenspan.sh`:**
- Job name: `embed_val_gstock_ts`
- `--model-path gabrielloiseau/CALE-MBERT-en`
- `--output-dir data/embeddings/g_stock_tokenspan`
- `--pooling tokenspan`
- Same resources as existing wrappers (1 GPU, 32G, 4 CPUs, 1h wall)
- `source activate nlp_env` and `export PYTHONUNBUFFERED=1`

**`scripts/embed_val_g1_tokenspan.sh`:**
- Job name: `embed_val_g1_ts`
- `--model-path models/g1_tokenspan/model`
- `--output-dir data/embeddings/g1_tokenspan`
- `--pooling tokenspan`
- Same resources

Both should include header comments with:
- Pre-submission checklist (files to verify, mkdir -p logs)
- Submit command
- scp commands for retrieving outputs (full paths with
  `/home/vwinters/ccc-project/custom_embedding_model/...`)

## Data files needed on Great Lakes

Before submitting any job, these must be present:
- `data/filtered_split/wn_synset/clue_phrases/f_clue.csv`
- `data/filtered_split/wn_synset/wndef/f_common_wndef.csv`
- `data/filtered_split/wn_synset/wndef/vocabulary_wndef_val.csv`
- `data/filtered_split/wn_synset/wnex/f_common_wnex.csv`
- `data/filtered_split/wn_synset/wnex/vocabulary_wnex_val.csv`
- `data/embeddings/g_stock/f_clue.npy` (for meanpool consistency check)
- `data/embeddings/g_stock/f_clue_index.csv` (for meanpool consistency check)

Additionally for g1_tokenspan:
- `models/g1_tokenspan/model/` (weights from training)

## Environment

Great Lakes (GPU). All embedding jobs are independent and can be submitted
simultaneously. Expected runtime: ~3-5 min each.

## Testing

Before submitting full jobs, run a `--sample 50` smoke test locally or on
Great Lakes to verify both pooling paths work:

```bash
python scripts/embed_val.py \
    --model-path gabrielloiseau/CALE-MBERT-en \
    --output-dir /tmp/smoke_meanpool \
    --pooling meanpool --batch-size 16 --sample 50

python scripts/embed_val.py \
    --model-path gabrielloiseau/CALE-MBERT-en \
    --output-dir /tmp/smoke_tokenspan \
    --pooling tokenspan --batch-size 16 --sample 50
```
