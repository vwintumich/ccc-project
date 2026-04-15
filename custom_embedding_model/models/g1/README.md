# Model: g1

**Status:** trained 2026-04-14 — weights on Great Lakes

**Extraction method:** Mean pooling (canonical). This model uses CALE's
standard attention-masked mean pooling over all non-padding tokens, matching
`SentenceTransformer.encode()` behavior. See Decision 20.

## Base model

- HuggingFace ID: `gabrielloiseau/CALE-MBERT-en`

## Weights location

- Great Lakes: `/home/vwinters/ccc-project/custom_embedding_model/models/g1/model/`

Weights are **not** committed to this repo (Decision 12).

## Training details

- Triplet file: `data/triplets/g1_train.csv`
- Triplet design: T_1 (same triplets as g1_tokenspan — different extraction method)
- Training script: `scripts/train_g1.py`
- SLURM script: `scripts/train_g1.sh`

Hyperparameters:
  - margin: 1.0 (Decision 13)
  - learning_rate: 2e-5
  - epochs: 3
  - batch_size: 32
  - weight_decay: 0.01
  - grad_accum: 1
  - warmup_fraction: 0.1
  - max_length: 128
  - random_state: 42

Date trained: 2026-04-14
Runtime: 43.5 min (0.72 h) on Great Lakes gpu partition, Tesla V100-PCIE-16GB
Per-epoch loss: [0.470, 0.111, 0.014]

Environment versions (per Decision 19):
  - Python: 3.12.12
  - torch: 2.5.1+cu121
  - transformers: 4.57.6
  - numpy: 2.3.5
  - pandas: 3.0.0
