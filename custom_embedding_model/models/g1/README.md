# Model: g1

**Status:** placeholder — weights not yet trained. Fill in the sections below
from the `SUMMARY` block in the Great Lakes SLURM log after the
`scripts/train_g1.sh` job completes.

## Base model

- HuggingFace ID: `gabrielloiseau/CALE-MBERT-en`
- Version/commit hash: *(pin from the downloaded snapshot after training)*

## Weights location

- Google Drive: "Research Project - NLP CCC's" (owned by Nathan)
- Path: `custom_embedding_models/g1/`

Weights are **not** committed to this repo (Decision 12). After training on
Great Lakes, upload the contents of `models/g1/model/` (the HuggingFace
`save_pretrained()` directory produced by `scripts/train_g1.py`) to the
Google Drive path above.

## Training details

- Triplet file: `data/triplets/g1.csv`
- Triplet design: T_1 (NB 09 reproduction, Step A)
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

Date trained: *(fill in after run)*
Runtime: *(fill in: hours on Great Lakes, partition, GPU type)*
Per-epoch loss: *(fill in from training_log.json)*

Environment versions (per Decision 19):
  - Python: *(fill in)*
  - torch: *(fill in, including CUDA build suffix)*
  - transformers: *(fill in)*
  - numpy: *(fill in)*
  - pandas: *(fill in)*
