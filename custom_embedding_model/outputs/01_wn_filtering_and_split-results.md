# Results: 01 — WordNet Filtering and Split Assignment


## Versions

- pandas: 3.0.0
- numpy: 2.3.5
- scikit-learn: 1.8.0
- nltk: 3.9.2
- WordNet corpus: 3.0

## Coverage

- Input rows (`clues_filtered.csv`): 457,262
- Output rows (`clues_wn_filtered.csv`): 239,406
- Fraction retained: 52.4%
- Total rows dropped: 217,856

### Article-stripping recovery (unique definitions)

- No strip needed: 41,217 (19.7%)
- Recovered by "a": 1,579 (0.8%)
- Recovered by "an": 257 (0.1%)
- Recovered by "the": 615 (0.3%)
- Recovered by "to": 737 (0.4%)
- No synsets found: 164,474 (78.7%)

### Article-stripping recovery (unique answers)

- No strip needed: 63,097 (67.7%)
- Recovered by "a": 43 (0.0%)
- Recovered by "an": 1 (0.0%)
- Recovered by "the": 231 (0.2%)
- Recovered by "to": 13 (0.0%)
- No synsets found: 29,757 (31.9%)

## Split Statistics

### Unique (definition, answer) pairs: 150,805

- Train: 45,241 (30.0%)
- Validate: 30,161 (20.0%)
- Test: 75,403 (50.0%)

### Rows per split

- Train: 72,107 (30.1%)
- Validate: 47,933 (20.0%)
- Test: 119,366 (49.9%)

## Vocabulary Statistics

- `vocabulary.csv`: 53,930 words
- `vocabulary_val.csv`: 26,152 words
- Validation vocabulary as fraction of full: 48.5%

### Overlap

- Words appearing as both definition and answer: 18,200
- Words appearing only as definition: 8,965
- Words appearing only as answer: 26,765

## Runtime

- WordNet lookups: 4.1s
- Total notebook: 5.6s
