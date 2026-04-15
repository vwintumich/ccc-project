# Results: 03 — g_1 Triplet Construction (Step A)

## Versions

- pandas: 3.0.0
- numpy:  2.3.5

## Input files

- `clues_wn_filtered.csv`: 239,406 rows total, 72,107 in training split
- `f_clue.csv`: 239,406 anchor phrases (full wn_synset scope)
- `f_common_wndef.csv`: 53,930 positive/negative phrases
- `dataset_harder.parquet`: 480,422 rows total, 240,211 label=0 distractor rows

## Join and coverage statistics

- Training rows before distractor join: 72,107
- Training rows after distractor join: 70,415 (97.65% retained)
- Rows lost to distractor join: 1,692 (2.35%)
- Rows lost to missing anchor phrase: 0
- Rows lost to missing positive phrase: 0
- Rows lost to missing negative phrase: 494
- Unique distractor words absent from `vocabulary_wndef.csv`: 222
- Final triplet rows: 69,921

## Triplet-set structure

- Unique clue_ids: 69,298
- Unique (definition, answer_wn) pairs: 43,924
- Unique answer words: 24,464
- Unique distractor words: 26,735
- Words appearing as both answer and distractor (across different rows): 15,117


## Validation triplets (for NB 05 evaluation)

- Validation rows before distractor join: 47,933
- Validation rows after distractor join: 46,847 (97.73% retained)
- Rows lost to distractor join: 1,086 (2.27%)
- Rows lost to missing anchor phrase: 0
- Rows lost to missing positive phrase: 0
- Rows lost to missing negative phrase: 341
- Final validation triplet rows: 46,506
- Unique clue_ids (val): 46,248
- Unique (definition, answer_wn) pairs (val): 29,261
- Unique answer words (val): 18,947
- Unique distractor words (val): 22,396

## Comparison to NB 09

| Dataset | Pairs | Triplet rows |
|---|---:|---:|
| NB 09 full (unsampled) | 102,086 | 192,039 |
| NB 09 sampled (published) | 20,000 | 37,593 |
| g_1 (this notebook) | 43,924 | 69,921 |

- g_1 / NB 09 sampled ratio: 1.86×
- g_1 / NB 09 full ratio: 0.36×

## Outputs

- `data/triplets/g1_train.csv` — 14.7 MB, 69,921 rows
- `data/triplets/g1_val.csv` — 9.7 MB, 46,506 rows
- `data/triplets/g1_train_meta.json` — 1.1 KB (includes train and validation counts)

## Runtime

- Input load: 1.2s
- Notebook end-to-end: a few seconds (CPU, text manipulation only)

## Interpretation

This is a faithful reproduction of NB 09's T_1 triplet design on our pipeline's cleaner data. The only material differences from NB 09 are upstream filtering (our NB 01 WordNet filter vs. Milestone II's pipeline) and split proportions (30/20/50 vs. 80/20, per Decision 3). The anchor, positive, and negative roles and their phrase constructions match NB 09 exactly — so if the T_1 failure pattern (format-specific compression of f_common_wndef phrases) recurs here, it is attributable to the triplet design itself and not to any data-preparation artifact.
