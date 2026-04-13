# Results: 02 — WordNet Phrase Construction


## Versions

- pandas: 3.0.0
- numpy: 2.3.5
- nltk: 3.9.2
- WordNet corpus: 3.0

## f_clue Coverage

- Input rows (`clues_wn_filtered.csv`): 239,406
- Rows with valid f_clue phrase: 239,406 (100.0%)
- Rows dropped: 0 (0.0%)

## f_common_wndef Coverage

- Full vocabulary size: 53,930
- Words with valid phrase: 53,930 (100.0%)
- Self-referential phrases: 1,139 (2.1%)
- `clues_wndef_filtered.csv` rows: 239,406 (100.0% of clues_wn_filtered)
- `vocabulary_wndef.csv`: 53,930 words
- `vocabulary_wndef_val.csv`: 26,152 words
- Split fractions in clues_wndef_filtered:
  - Train: 72,107 (30.1%)
  - Validate: 47,933 (20.0%)
  - Test: 119,366 (49.9%)

## f_common_wnex Coverage

- Full vocabulary size: 53,930
- Words with valid phrase: 8,360 (15.5%)
- `clues_wnex_filtered.csv` rows: 24,327 (10.2% of clues_wn_filtered)
- `vocabulary_wnex.csv`: 8,360 words
- `vocabulary_wnex_val.csv`: 3,008 words
- Split fractions in clues_wnex_filtered:
  - Train: 7,075 (29.1%)
  - Validate: 4,825 (19.8%)
  - Test: 12,427 (51.1%)

## Cross-f Comparison

- Words in wndef but not wnex: 45,570
- Words in wnex but not wndef: 0
- Words in both: 8,360
- Rows in clues_wndef_filtered but not clues_wnex_filtered: 215,079

## Runtime

- f_clue construction: 8.0s
- f_common_wndef construction: 3.3s
- f_common_wnex construction: 1.2s
- Total notebook: 14.9s
