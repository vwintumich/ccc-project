# POS and WordNet Sense Census — Results

Generated: 2026-04-21  
Notebook: `planning/exploration/pos_wordnet_census.ipynb`  
Vocabulary rows: 53,930  
Validation rows: 47,933  
Training triplet rows: 69,921  

## Versions

- pandas: 3.0.0
- numpy: 2.3.5
- matplotlib: 3.10.8
- seaborn: 0.13.2
- nltk: 3.9.2 (WordNet 3.0)
- spacy: 3.8.14 (en_core_web_sm, POS tagger)

## §2 — Vocabulary census

### sense[0] POS distribution (raw WN labels)
| pos | count | fraction |
| --- | --- | --- |
| n (noun) | 37,672 | 69.9% |
| v (verb) | 9,734 | 18.0% |
| a (adj) | 1,639 | 3.0% |
| s (sat_adj) | 3,623 | 6.7% |
| r (adv) | 1,262 | 2.3% |

### sense[0] POS distribution (combined adj)
| pos | count | fraction |
| --- | --- | --- |
| noun | 37,672 | 69.9% |
| verb | 9,734 | 18.0% |
| adj (a+s) | 5,262 | 9.8% |
| adv | 1,262 | 2.3% |

### Sense availability
- mean=2.75, median=2, q1=1, q3=3, max=75
- Words with synsets in ≥ 2 POS categories: 11,311 (21.0%)
- Only noun: 28,525 (52.9%)
- Only verb: 7,787 (14.4%)
- Only adj (a+s): 5,045 (9.4%)
- Only adv: 1,262 (2.3%)

### Lemma count reliability
- Any nonzero lemma count: 14,761 (27.4%)
- Among those: sense[0] is max within POS = 14,721 (99.7%)
- Among those: higher-freq sense in other POS = 2,926 (19.8%)

### Three-way reliability breakdown
- Unambiguous (n_synsets == 1):       25,258 (46.8%)
- Frequency-confirmed:                10,733 (19.9%)
- Arbitrary (multi-synset, no proof): 17,939 (33.3%)

### Reliability × sense[0] POS
| pos | unambiguous | frequency-confirmed | arbitrary |
| --- | --- | --- | --- |
| noun | 18,665 | 6,952 | 12,055 |
| verb | 2,870 | 2,292 | 4,572 |
| adj (a+s) | 2,800 | 1,224 | 1,238 |
| adv | 923 | 265 | 74 |

## §3 — Contextual POS coverage

- validation (n=47,933): determined=46,150 (96.3%), undetermined=1,498 (3.1%), span_not_found=285 (0.6%)
- training (n=69,921): determined=67,254 (96.2%), undetermined=2,301 (3.3%), span_not_found=366 (0.5%)
- Unique-pair undetermined examples saved to `outputs/pos_mismatch_examples.md`

## §4 — Training triplet POS

### Per-role POS distribution (fraction of triplets)
| pos | anchor_contextual | anchor_wn_sense0 | positive_wn_sense0 | negative_wn_sense0 |
| --- | --- | --- | --- | --- |
| n | 62.0% | 79.2% | 73.5% | 70.6% |
| v | 16.9% | 11.5% | 14.9% | 17.7% |
| a | 14.0% | 7.5% | 10.0% | 10.2% |
| r | 2.4% | 1.8% | 1.6% | 1.5% |
| other | 0.8% | 0.0% | 0.0% | 0.0% |
| undetermined | 3.8% | 0.0% | 0.0% | 0.0% |

### Anchor contextual vs. WN sense[0] crosstab
| contextual↓ / wn→ | n | v | a | r |
| --- | --- | --- | --- | --- |
| n | 42,375 | 558 | 414 | 2 |
| v | 6,700 | 4,991 | 160 | 0 |
| a | 4,770 | 1,008 | 4,033 | 1 |
| r | 575 | 43 | 218 | 865 |
| other | 417 | 29 | 75 | 20 |
- Determined-row agreement: 52,264 / 67,254 (77.7%)

### Triplet-level composition
- Both positive and negative are nouns: 43,147 (61.7%)
- Positive is noun: 51,415 (73.5%)
- Negative is noun: 49,377 (70.6%)

### Sense reliability
- Both pos and neg have nonzero counts: 14,640 (20.9%)
- At least one role arbitrary: 55,281 (79.1%)
- At least one role has higher-freq other-POS sense: 24,111 (34.5%)

### Three-way reliability (triplet-level)
- All three roles unambiguous or frequency-confirmed: 28,118 (40.2%)
- At least one role arbitrary: 41,803 (59.8%)

#### Per-role reliability (fraction of triplets)
| category | anchor (definition_wn) | positive (answer_wn) | negative (distractor_wn) |
| --- | --- | --- | --- |
| unambiguous | 14.6% | 32.6% | 26.1% |
| frequency-confirmed | 69.6% | 40.8% | 32.8% |
| arbitrary | 15.9% | 26.6% | 41.1% |

## §5 — Validation pair POS

### Per-component POS distribution (fraction of pairs)
| pos | contextual_def_T1 | wn_sense0_def_T0 | wn_sense0_answer |
| --- | --- | --- | --- |
| n | 61.5% | 78.7% | 73.2% |
| v | 16.6% | 11.4% | 14.9% |
| a | 14.9% | 8.1% | 10.2% |
| r | 2.5% | 1.7% | 1.7% |
| other | 0.8% | 0.0% | 0.0% |
| undetermined | 3.7% | 0.0% | 0.0% |

### T=0 pair composition (def sense[0] POS × ans sense[0] POS)
| def↓ / ans→ | n | v | a | r |
| --- | --- | --- | --- | --- |
| n | 32,477 | 3,437 | 1,681 | 125 |
| v | 1,436 | 3,362 | 663 | 18 |
| a | 1,045 | 317 | 2,459 | 75 |
| r | 141 | 14 | 110 | 573 |

- Condensed 2x2 noun-noun: 32,477 (67.8%); noun-other: 5,243 (10.9%); other-noun: 2,622 (5.5%); other-other: 7,591 (15.8%)

### T=1 pair composition (contextual def POS × ans sense[0] POS)
| ctx↓ / ans→ | n | v | a | r |
| --- | --- | --- | --- | --- |
| n | 26,894 | 1,927 | 630 | 21 |
| v | 3,822 | 3,455 | 645 | 38 |
| a | 3,002 | 935 | 3,170 | 35 |
| r | 333 | 88 | 234 | 547 |
| other | 260 | 37 | 44 | 33 |
- Determined-row agreement: 34,066 / 46,150 (73.8%)

### Sense reliability
- Both def and ans have nonzero counts: 20,532 (42.8%)
- At least one arbitrary: 27,401 (57.2%)

### Three-way reliability (pair-level)
- Both roles unambiguous or frequency-confirmed: 31,452 (65.6%)
- At least one role arbitrary: 16,481 (34.4%)

#### Per-role reliability (fraction of pairs)
| category | definition_wn | answer_wn |
| --- | --- | --- |
| unambiguous | 14.9% | 32.9% |
| frequency-confirmed | 68.8% | 41.5% |
| arbitrary | 16.4% | 25.6% |

## Runtimes

- Vocab census: 3.8s (53,930 words)
- spaCy POS tagging: 54.3s (115,258 unique surfaces)
