# Spec: Full-vocabulary wndef embeddings for g_stock and g1

**Stage:** 4
**Scripts:** `scripts/embed_wndef_full_gstock.sh`, `scripts/embed_wndef_full_g1.sh`
**Date:** 2026-04-19
**Status:** Draft

## Purpose

Generate full-vocabulary f_common_wndef embeddings (53,930 words) for both
g_stock and g1. Currently we only have validation-subset wndef embeddings
(26,152 words). The full-vocabulary versions are needed for two reasons:

1. **Cross-f triplet accuracy (matched comparison).** To fairly compare
   wndef vs. wnex triplet accuracy in NB 05, we need to evaluate the
   exact same set of triplets under both phrase types. The binding
   constraint for the matched set is the wnex vocabulary (8,360 words);
   since wnex ⊂ wndef, any word in vocabulary_wnex is guaranteed to be
   in vocabulary_wndef — but some may not be in vocabulary_wndef_val.
   Full-vocab wndef embeddings eliminate this gap.

2. **Improved wndef triplet resolution.** The existing NB 05 §2 drops
   ~41% of validation triplets because distractor words are absent from
   vocabulary_wndef_val (Decision 21). With full-vocab wndef embeddings,
   resolution should jump to ~99%+ (the only remaining dropouts are the
   ~222 distractor words not in our vocabulary at all, per
   g1_train_meta.json).

The same rationale as Decision 22 (full-vocab wnex) applies:
individual word embeddings carry no test-set evaluation signal, so
including train/test-split vocabulary words does not violate Decision 9.

## Inputs

- `data/filtered_split/wn_synset/wndef/vocabulary_wndef.csv` (53,930 rows)
- `data/filtered_split/wn_synset/wndef/f_common_wndef.csv` (53,930 rows)
- g_stock model: `gabrielloiseau/CALE-MBERT-en` (HuggingFace)
- g1 model: `models/g1/model/` (fine-tuned weights on Great Lakes)

## Outputs

- `data/embeddings/g_stock/f_common_wndef.npy` — shape (53930, 1024),
  indexed by `vocabulary_wndef.csv`
- `data/embeddings/g1/f_common_wndef.npy` — shape (53930, 1024),
  indexed by `vocabulary_wndef.csv`

These supplement (do not replace) the existing val-only files
(`f_common_wndef_val.npy`, 26,152 rows).

## Implementation details

Two SLURM wrapper scripts, modeled exactly on the existing
`embed_wnex_full_gstock.sh` and `embed_wnex_full_g1.sh`. Each calls
`scripts/embed_vocab.py` with the wndef vocabulary and phrase files.

### Script 1: `scripts/embed_wndef_full_gstock.sh`

```
python scripts/embed_vocab.py \
    --model-path gabrielloiseau/CALE-MBERT-en \
    --pooling meanpool \
    --vocab-file data/filtered_split/wn_synset/wndef/vocabulary_wndef.csv \
    --phrase-file data/filtered_split/wn_synset/wndef/f_common_wndef.csv \
    --output-file data/embeddings/g_stock/f_common_wndef.npy
```

SLURM settings (identical to the wnex scripts):
- `--job-name=embed_wndef_full_gstock`
- `--account=siads696w26_class`
- `--partition=gpu`
- `--gpus=1`
- `--cpus-per-task=4`
- `--mem=32G`
- `--time=01:00:00`
- `--output=logs/embed_wndef_full_gstock_%j.out`
- `source activate nlp_env`
- `export PYTHONUNBUFFERED=1`

### Script 2: `scripts/embed_wndef_full_g1.sh`

```
python scripts/embed_vocab.py \
    --model-path models/g1/model \
    --pooling meanpool \
    --vocab-file data/filtered_split/wn_synset/wndef/vocabulary_wndef.csv \
    --phrase-file data/filtered_split/wn_synset/wndef/f_common_wndef.csv \
    --output-file data/embeddings/g1/f_common_wndef.npy
```

SLURM settings (same as above, with `--job-name=embed_wndef_full_g1`
and `--output=logs/embed_wndef_full_g1_%j.out`).

### Header comments

Follow the same documentation pattern as the wnex scripts. Each script
should include:
- Purpose: what this job produces and why
- A note that this supplements (not replaces) the existing val-only file
- Decision 22 rationale reference (full-vocab word embeddings carry no
  test-set signal)
- Pre-submission checklist: `mkdir -p logs`, verify input files exist
- scp commands for transferring outputs back to local machine
  (use full absolute paths per agent memory):
  - `.npy` file from Great Lakes to local `data/embeddings/<model>/`
  - SLURM log from Great Lakes to local `logs/`

### Expected runtime

At ~380 phrases/sec (the rate observed for the wnex jobs), 53,930 phrases
should take ~142 seconds (~2.5 minutes). The 1-hour wall time is
conservative. The two jobs are independent and can be submitted
simultaneously.

### scp paths

After completion, transfer outputs:

**g_stock:**
```
scp vwinters@greatlakes-xfer.arc-ts.umich.edu:/home/vwinters/ccc-project/custom_embedding_model/data/embeddings/g_stock/f_common_wndef.npy \
    /Users/victoria/Desktop/MADS/ccc-project/custom_embedding_model/data/embeddings/g_stock/

scp vwinters@greatlakes-xfer.arc-ts.umich.edu:/home/vwinters/ccc-project/custom_embedding_model/logs/embed_wndef_full_gstock_<jobid>.out \
    /Users/victoria/Desktop/MADS/ccc-project/custom_embedding_model/logs/
```

**g1:**
```
scp vwinters@greatlakes-xfer.arc-ts.umich.edu:/home/vwinters/ccc-project/custom_embedding_model/data/embeddings/g1/f_common_wndef.npy \
    /Users/victoria/Desktop/MADS/ccc-project/custom_embedding_model/data/embeddings/g1/

scp vwinters@greatlakes-xfer.arc-ts.umich.edu:/home/vwinters/ccc-project/custom_embedding_model/logs/embed_wndef_full_g1_<jobid>.out \
    /Users/victoria/Desktop/MADS/ccc-project/custom_embedding_model/logs/
```

## Environment

Great Lakes (GPU). Two independent SLURM jobs.

## Post-completion

After transferring the files locally:
1. Verify shapes: each `.npy` should be (53930, 1024)
2. Add a DECISIONS.md entry (Decision 23) documenting the choice to
   generate full-vocab wndef embeddings, paralleling Decision 22
3. Add FINDINGS.md entries recording the runtime, shapes, and
   environment versions from the SLURM logs
4. Update DATA.md to register the new embedding files
5. Proceed to Step 2: revise NB 05 with the cross-f triplet accuracy
   analysis (separate spec)
