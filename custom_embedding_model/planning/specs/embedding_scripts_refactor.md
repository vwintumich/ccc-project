# Spec: Embedding scripts refactor — embedding_utils.py, embed_clue.py, embed_vocab.py

**Stage:** 4 (Embedding Generation)
**Scripts:** `scripts/embedding_utils.py`, `scripts/embed_clue.py`, `scripts/embed_vocab.py`
**Date:** 2026-04-19
**Status:** Draft

## Purpose

Replace the two existing embedding scripts (`embed_f_clue_gstock.py` and
`embed_val.py`) with a cleaner two-script architecture that separates
clue-contextualized embeddings from decontextualized vocabulary embeddings,
with shared utility code factored into a common module. The immediate
motivation is generating full-vocabulary wnex embeddings for g_stock and g1
(8,360 words each), but the refactor is designed for long-term reuse across
future models and phrase types.

### What replaces what

| Old script | New equivalent | Notes |
|---|---|---|
| `embed_f_clue_gstock.py` | `embed_clue.py` | Generalizes from g_stock-only to any model; adds split filtering and pooling flag |
| `embed_val.py` (f_clue portion) | `embed_clue.py` | Split filtering via `--split` flag replaces hardcoded `validate` filter |
| `embed_val.py` (wndef/wnex portions) | `embed_vocab.py` | Vocabulary file passed as argument instead of hardcoded val paths |

## Inputs

These are the data files the scripts will read. No new data files are
created by this spec — the scripts operate on existing phrase and
vocabulary files.

### embed_clue.py reads:
- `data/filtered_split/wn_synset/clue_phrases/f_clue.csv`
  - Columns: `clue_id`, `definition`, `split`, `phrase`
  - 239,417 data rows (239,406 train+val+test clues, all with valid f_clue phrases)

### embed_vocab.py reads (passed as CLI arguments):
- A vocabulary CSV with columns `word`, `row` (row is 0-indexed contiguous)
- A phrase CSV with columns including `word`, `phrase`
- Examples of valid (vocab, phrase) pairs:
  - `wnex/vocabulary_wnex.csv` (8,360 words) + `wnex/f_common_wnex.csv`
  - `wnex/vocabulary_wnex_val.csv` (3,008 words) + `wnex/f_common_wnex.csv`
  - `wndef/vocabulary_wndef_val.csv` (26,152 words) + `wndef/f_common_wndef.csv`

## Outputs

### New embedding files (the immediate job)
- `data/embeddings/g_stock/f_common_wnex.npy` — (8360, 1024) float32, indexed by `vocabulary_wnex.csv`
- `data/embeddings/g1/f_common_wnex.npy` — (8360, 1024) float32, indexed by `vocabulary_wnex.csv`

### Verification outputs (confirm new scripts reproduce old scripts)
Seven verification runs reproducing every existing embedding artifact. Each
run compares the new output against the existing `.npy` file using rowwise
cosine similarity. Expected: mean cosine > 0.9999 for same-model
same-pooling reproductions (any deviation is floating-point non-determinism
only). Verification results printed to stdout (captured in SLURM logs).

| Verification run | Script | Model | Arguments | Existing file to compare against |
|---|---|---|---|---|
| V1 | `embed_clue.py` | g_stock (HF) | `--split all --pooling meanpool` | `g_stock/f_clue.npy` |
| V2 | `embed_clue.py` | g_stock (HF) | `--split validate --pooling meanpool` | `g_stock/f_clue_val.npy` |
| V3 | `embed_clue.py` | g1 | `--split validate --pooling meanpool` | `g1/f_clue_val.npy` |
| V4 | `embed_vocab.py` | g_stock (HF) | wndef val vocab + wndef phrases | `g_stock/f_common_wndef_val.npy` |
| V5 | `embed_vocab.py` | g_stock (HF) | wnex val vocab + wnex phrases | `g_stock/f_common_wnex_val.npy` |
| V6 | `embed_vocab.py` | g1 | wndef val vocab + wndef phrases | `g1/f_common_wndef_val.npy` |
| V7 | `embed_vocab.py` | g1 | wnex val vocab + wnex phrases | `g1/f_common_wnex_val.npy` |

**Note on V1:** The existing `g_stock/f_clue.npy` was produced by
`embed_f_clue_gstock.py` using `SentenceTransformer.encode()`, while
`embed_clue.py` will use `AutoModel` + manual mean pooling. The existing
`embed_val.py` already verified these two methods produce mean cosine >
0.999 (see FINDINGS.md Stage 4). Expect the V1 comparison to show mean
cosine ~0.999+ rather than ~1.0, because the extraction methods are
numerically different despite being mathematically equivalent. All other
verification runs (V2–V7) compare AutoModel-to-AutoModel, so they should
show mean cosine ~1.0.

### New SLURM wrappers
- `scripts/embed_wnex_full_gstock.sh` — g_stock full wnex vocabulary
- `scripts/embed_wnex_full_g1.sh` — g1 full wnex vocabulary
- `scripts/verify_embedding_scripts.sh` — runs all seven verification jobs sequentially

### Archive
After verification passes, move to `scripts/archive/`:
- `embed_f_clue_gstock.py`
- `embed_f_clue_gstock.sh`
- `embed_val.py`
- `embed_val_g1.sh`
- `embed_val_g1_tokenspan.sh`
- `embed_val_gstock.sh`
- `embed_val_gstock_tokenspan.sh`

---

## Implementation details

### scripts/embedding_utils.py

Shared module imported by both `embed_clue.py` and `embed_vocab.py`. Contains
only format-agnostic embedding machinery — no knowledge of clue IDs,
vocabulary files, splits, or phrase types. No CLI argument parsing.

**Functions to include:**

1. **`print_environment()`**
   - Prints Python, torch, transformers, numpy, pandas versions and CUDA
     info to stdout (Decision 19). Called at the top of each script's
     `main()`.

2. **`load_model(model_path: str, device: torch.device) -> tuple[AutoModel, AutoTokenizer]`**
   - Loads `AutoModel` and `AutoTokenizer` from `model_path` (HuggingFace
     ID or local path).
   - Moves model to `device`, calls `model.eval()`.
   - Asserts `model.config.hidden_size == 1024`.
   - Prints model load time.
   - Returns `(model, tokenizer)`.

3. **`extract_meanpool_embedding(model, tokenizer, texts, device, max_length) -> torch.Tensor`**
   - Attention-masked mean pooling over all non-padding tokens (Decision 20
     canonical). Ported from `embed_val.py` §3.
   - Must be called inside `torch.no_grad()`.
   - Returns tensor of shape `(batch, 1024)`.

4. **`extract_concept_embedding(model, tokenizer, texts, device, max_length) -> torch.Tensor`**
   - Token span extraction: averages hidden states for tokens whose
     character span lies inside `<t></t>`. Ported from `embed_val.py` §3.
   - Includes the `find_delimiter_char_offsets()` helper (can be a
     module-level function or nested).
   - Must be called inside `torch.no_grad()`.
   - Returns tensor of shape `(batch, 1024)`.

5. **`encode_phrases(model, tokenizer, phrases, device, batch_size, max_length, pooling) -> np.ndarray`**
   - Batched encoding loop. Dispatches to `extract_meanpool_embedding` or
     `extract_concept_embedding` based on `pooling` string (`"meanpool"` or
     `"tokenspan"`).
   - Calls `model.eval()`, wraps in `torch.no_grad()`.
   - Moves each batch result to CPU as float32 immediately.
   - Prints progress every 20 batches.
   - Returns `np.ndarray` of shape `(N, 1024)`, dtype float32.

6. **`validate_embeddings(embeddings: np.ndarray, n_expected: int, label: str) -> None`**
   - Asserts shape is `(n_expected, 1024)`.
   - Asserts no NaN values.
   - Asserts no all-zero rows (L2 norm > 0).
   - Prints L2 norm range.
   - Ported from `embed_val.py` §5.

7. **`save_npy_atomic(array: np.ndarray, path: Path) -> None`**
   - Writes to `.tmp.npy`, then renames. The temp name must already end in
     `.npy` because `np.save()` auto-appends `.npy` (see commit c3653b9).
   - Ported from `embed_val.py` §4.

8. **`save_csv_atomic(df: pd.DataFrame, path: Path) -> None`**
   - Writes to `.tmp.csv`, then renames.
   - Ported from `embed_val.py` §4.

**Do not include:** CLI argument parsing, consistency check logic,
split-filtering logic, vocabulary-joining logic, output naming logic.
These belong in the individual scripts.

---

### scripts/embed_clue.py

Embeds clue-contextualized (f_clue) phrases for a given model and split.

**CLI arguments:**

| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `--model-path` | str | yes | — | HuggingFace ID or local model directory |
| `--pooling` | str | yes | — | `meanpool` or `tokenspan` |
| `--split` | str | yes | — | `train`, `validate`, or `all` |
| `--output-dir` | Path | yes | — | Directory for output files |
| `--f-clue-csv` | Path | no | `data/filtered_split/wn_synset/clue_phrases/f_clue.csv` | Path to f_clue.csv |
| `--batch-size` | int | no | 64 | Encoding batch size |
| `--max-length` | int | no | 128 | Tokenizer max_length |
| `--sample` | int | no | 0 | If > 0, use first N rows (smoke test) |
| `--verify-against` | Path | no | None | Path to existing `.npy` file for consistency check |

**Main logic:**

1. Call `embedding_utils.print_environment()`.
2. Parse CLI arguments.
3. Load model via `embedding_utils.load_model()`.
4. Load `f_clue.csv` with `keep_default_na=False, na_values=[""]`.
   Assert required columns: `clue_id`, `definition`, `split`, `phrase`.
   Assert no null phrases.
5. Filter by split:
   - `--split train` → `df[df["split"] == "train"]`
   - `--split validate` → `df[df["split"] == "validate"]`
   - `--split all` → no filter (use all rows, including test)
   - Print row count before and after filtering.
6. If `--sample > 0`, take first N rows.
7. Encode via `embedding_utils.encode_phrases()`.
8. Run `embedding_utils.validate_embeddings()`.
9. Build index DataFrame: columns `clue_id`, `definition`, `row` (0-indexed).
10. Determine output filenames from `--split`:
    - `train` → `f_clue_train.npy`, `f_clue_train_index.csv`
    - `validate` → `f_clue_val.npy`, `f_clue_val_index.csv`
    - `all` → `f_clue.npy`, `f_clue_index.csv`
11. Save via `embedding_utils.save_npy_atomic()` and
    `embedding_utils.save_csv_atomic()`.
12. **Consistency check** (only if `--verify-against` is provided):
    - Load the reference `.npy` file.
    - Load the reference index CSV from the same directory (same stem +
      `_index.csv`, or for `f_clue.npy` → `f_clue_index.csv`). If the
      reference is `f_clue_val.npy`, look for `f_clue_val_index.csv` in
      the same directory.
    - Match rows by `(clue_id, definition)` between the new index and
      the reference index.
    - Compute rowwise cosine similarity on all matched rows (or a sample
      of 500 if there are more than 500 matches).
    - Print: number of matched rows, mean cosine, min cosine, max cosine.
    - Assert mean cosine > 0.999. Print PASS or FAIL.
13. Print summary: model path, pooling, split, row count, output paths
    with file sizes, encoding time, total wall-clock time.

---

### scripts/embed_vocab.py

Embeds decontextualized vocabulary phrases for a given model, vocabulary
file, and phrase file.

**CLI arguments:**

| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `--model-path` | str | yes | — | HuggingFace ID or local model directory |
| `--pooling` | str | yes | — | `meanpool` or `tokenspan` |
| `--vocab-file` | Path | yes | — | Vocabulary CSV (columns: `word`, `row`) |
| `--phrase-file` | Path | yes | — | Phrase CSV (columns include: `word`, `phrase`) |
| `--output-file` | Path | yes | — | Output `.npy` path |
| `--batch-size` | int | no | 64 | Encoding batch size |
| `--max-length` | int | no | 128 | Tokenizer max_length |
| `--sample` | int | no | 0 | If > 0, use first N rows (smoke test) |
| `--verify-against` | Path | no | None | Path to existing `.npy` file for consistency check |

**Main logic:**

1. Call `embedding_utils.print_environment()`.
2. Parse CLI arguments.
3. Load model via `embedding_utils.load_model()`.
4. Load vocabulary CSV with `keep_default_na=False, na_values=[""]`.
   Assert required columns: `word`, `row`.
   Assert `row` is contiguous 0..N-1 (canonical ordering check).
   Print vocabulary size.
5. Load phrase CSV with `keep_default_na=False, na_values=[""]`.
   Assert required columns: `word`, `phrase`.
   Assert no null phrases.
6. Left-join vocabulary onto phrases on `word`.
   Assert join is lossless: every vocabulary word must have a phrase
   (no NaN in phrase column after join). Print failure count and assert 0.
   Assert post-join ordering matches vocabulary `row` ordering (compare
   `joined["word"].values` against `vocab["word"].values`).
   Assert post-join length equals vocabulary length.
7. If `--sample > 0`, take first N rows.
8. Encode via `embedding_utils.encode_phrases()`.
9. Run `embedding_utils.validate_embeddings()`.
10. Save via `embedding_utils.save_npy_atomic()`.
    (No index CSV needed — the vocabulary file IS the index, per Decision 6.)
11. **Consistency check** (only if `--verify-against` is provided):
    - Load the reference `.npy` file.
    - Assert reference shape[0] matches the vocabulary length (they must be
      indexed by the same vocabulary file).
    - Compute rowwise cosine similarity on all rows (or a sample of 500 if
      vocabulary > 500 words).
    - Print: number of rows compared, mean cosine, min cosine, max cosine.
    - Assert mean cosine > 0.999. Print PASS or FAIL.
12. Print summary: model path, pooling, vocab file, phrase file, vocabulary
    size, output path with file size, encoding time, total wall-clock time.

---

### SLURM wrappers

All `.sh` files follow the existing pattern (see `embed_val_g1.sh`):
- SLURM header: job name, account `siads696w26_class`, partition `gpu`,
  1 GPU, 4 CPUs, 32G mem, 1-hour wall time, log to `logs/`
- `source activate nlp_env`
- `export PYTHONUNBUFFERED=1`
- Python command
- Comments: required files, scp commands for transferring output

#### scripts/embed_wnex_full_gstock.sh

```bash
python scripts/embed_vocab.py \
    --model-path gabrielloiseau/CALE-MBERT-en \
    --pooling meanpool \
    --vocab-file data/filtered_split/wn_synset/wnex/vocabulary_wnex.csv \
    --phrase-file data/filtered_split/wn_synset/wnex/f_common_wnex.csv \
    --output-file data/embeddings/g_stock/f_common_wnex.npy
```

#### scripts/embed_wnex_full_g1.sh

```bash
python scripts/embed_vocab.py \
    --model-path models/g1/model \
    --pooling meanpool \
    --vocab-file data/filtered_split/wn_synset/wnex/vocabulary_wnex.csv \
    --phrase-file data/filtered_split/wn_synset/wnex/f_common_wnex.csv \
    --output-file data/embeddings/g1/f_common_wnex.npy
```

#### scripts/verify_embedding_scripts.sh

Runs all seven verification jobs sequentially in a single SLURM submission.
Request 2-hour wall time to accommodate all seven runs. Each verification
run uses `--verify-against` to compare against the existing artifact.

```bash
# V1: g_stock f_clue all (meanpool) — compare against SentenceTransformer output
python scripts/embed_clue.py \
    --model-path gabrielloiseau/CALE-MBERT-en \
    --pooling meanpool --split all \
    --output-dir data/embeddings/g_stock_verify \
    --verify-against data/embeddings/g_stock/f_clue.npy

# V2: g_stock f_clue validate (meanpool)
python scripts/embed_clue.py \
    --model-path gabrielloiseau/CALE-MBERT-en \
    --pooling meanpool --split validate \
    --output-dir data/embeddings/g_stock_verify \
    --verify-against data/embeddings/g_stock/f_clue_val.npy

# V3: g1 f_clue validate (meanpool)
python scripts/embed_clue.py \
    --model-path models/g1/model \
    --pooling meanpool --split validate \
    --output-dir data/embeddings/g1_verify \
    --verify-against data/embeddings/g1/f_clue_val.npy

# V4: g_stock wndef val (meanpool)
python scripts/embed_vocab.py \
    --model-path gabrielloiseau/CALE-MBERT-en \
    --pooling meanpool \
    --vocab-file data/filtered_split/wn_synset/wndef/vocabulary_wndef_val.csv \
    --phrase-file data/filtered_split/wn_synset/wndef/f_common_wndef.csv \
    --output-file data/embeddings/g_stock_verify/f_common_wndef_val.npy \
    --verify-against data/embeddings/g_stock/f_common_wndef_val.npy

# V5: g_stock wnex val (meanpool)
python scripts/embed_vocab.py \
    --model-path gabrielloiseau/CALE-MBERT-en \
    --pooling meanpool \
    --vocab-file data/filtered_split/wn_synset/wnex/vocabulary_wnex_val.csv \
    --phrase-file data/filtered_split/wn_synset/wnex/f_common_wnex.csv \
    --output-file data/embeddings/g_stock_verify/f_common_wnex_val.npy \
    --verify-against data/embeddings/g_stock/f_common_wnex_val.npy

# V6: g1 wndef val (meanpool)
python scripts/embed_vocab.py \
    --model-path models/g1/model \
    --pooling meanpool \
    --vocab-file data/filtered_split/wn_synset/wndef/vocabulary_wndef_val.csv \
    --phrase-file data/filtered_split/wn_synset/wndef/f_common_wndef.csv \
    --output-file data/embeddings/g1_verify/f_common_wndef_val.npy \
    --verify-against data/embeddings/g1/f_common_wndef_val.npy

# V7: g1 wnex val (meanpool)
python scripts/embed_vocab.py \
    --model-path models/g1/model \
    --pooling meanpool \
    --vocab-file data/filtered_split/wn_synset/wnex/vocabulary_wnex_val.csv \
    --phrase-file data/filtered_split/wn_synset/wnex/f_common_wnex.csv \
    --output-file data/embeddings/g1_verify/f_common_wnex_val.npy \
    --verify-against data/embeddings/g1/f_common_wnex_val.npy
```

After verification passes, the `_verify` directories can be deleted.

---

## Environment

**Implementation:** Local (the scripts are Python files, not notebooks).

**Execution:** Great Lakes (GPU). All embedding jobs and verification runs
are submitted via SLURM.

### SLURM considerations

- All jobs use partition `gpu`, 1 GPU, 4 CPUs, 32G RAM.
- The verification script runs seven encoding passes sequentially — request
  2-hour wall time.
- The two wnex full-vocabulary jobs are independent and can be submitted
  in parallel. Each needs ~5 minutes.
- The g1 model weights must be present on Great Lakes at
  `models/g1/model/`. The g_stock model is downloaded from HuggingFace
  automatically.

### Post-job file transfers

After the wnex full-vocabulary jobs complete, scp the new files back:
```
scp vwinters@greatlakes-xfer.arc-ts.umich.edu:/home/vwinters/ccc-project/custom_embedding_model/data/embeddings/g_stock/f_common_wnex.npy \
    /Users/victoria/Desktop/MADS/ccc-project/custom_embedding_model/data/embeddings/g_stock/

scp vwinters@greatlakes-xfer.arc-ts.umich.edu:/home/vwinters/ccc-project/custom_embedding_model/data/embeddings/g1/f_common_wnex.npy \
    /Users/victoria/Desktop/MADS/ccc-project/custom_embedding_model/data/embeddings/g1/
```

After verification, scp the SLURM log for the verification job back to
`logs/` locally for the record.

---

## Post-completion steps

### Archive old scripts

Create `scripts/archive/` and move:
- `embed_f_clue_gstock.py`
- `embed_f_clue_gstock.sh`
- `embed_val.py`
- `embed_val_g1.sh`
- `embed_val_g1_tokenspan.sh`
- `embed_val_gstock.sh`
- `embed_val_gstock_tokenspan.sh`

### Update project documentation

- **NOTEBOOKS.md:** Update the Stage 1d and Stage 4 script tables to
  reference the new scripts. Add a note that the old scripts are archived
  in `scripts/archive/`. Add `embedding_utils.py` to the Scripts table.
- **DATA.md:** Add entries for the new full-vocabulary wnex embedding files
  (`g_stock/f_common_wnex.npy` and `g1/f_common_wnex.npy`), documenting
  their shape, index file, and which script produced them.
- **FINDINGS.md:** Add a brief entry for the wnex full-vocabulary embedding
  generation (runtime, shape, date).

### Clean up verification artifacts

Delete the `data/embeddings/g_stock_verify/` and
`data/embeddings/g1_verify/` directories after confirming all seven
verification runs passed.
