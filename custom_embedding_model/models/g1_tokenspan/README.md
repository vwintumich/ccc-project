# Model: g1_tokenspan

**Status:** trained 2026-04-13 — weights on Great Lakes

**Extraction method:** Token span (non-standard). This model averages hidden
states only within the `<t></t>` span, rather than using CALE's canonical
mean pooling over all tokens. See Decision 20 for details. Compare against
`g_stock_tokenspan` for a fair baseline.

## Base model

- HuggingFace ID: `gabrielloiseau/CALE-MBERT-en`

## Weights location

- Great Lakes: `/home/vwinters/ccc-project/custom_embedding_model/models/g1_tokenspan/model/`

Weights are **not** committed to this repo (Decision 12).

## Training details

- Triplet file: `data/triplets/g1.csv`
- Triplet design: T_1 (NB 09 reproduction, Step A)
- Training script: `scripts/train_g1_tokenspan.py`
- SLURM script: `scripts/train_g1_tokenspan.sh`

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

Date trained: 2026-04-13
Runtime: 49.0 min (0.82 h) on Great Lakes gpu partition, Tesla V100-PCIE-16GB
Per-epoch loss: [0.488, 0.104, 0.013]

Environment versions (per Decision 19):
  - Python: 3.12.12
  - torch: 2.5.1+cu121
  - transformers: 4.57.6
  - numpy: 2.3.5
  - pandas: 3.0.0
