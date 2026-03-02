# Failure Analysis — Step 12

**Model:** Random Forest (Exp 2A, 32 features)
**Evaluation set:** Fold 0 test set (4,000 examples)
**Generated:** Sample mode = True

## Error Summary

- Total misclassified: **1,040** (26.00% error rate)
- False positives (distractor predicted as real): 378
- False negatives (real pair predicted as distractor): 662

## Failure Categories

### Polysemy Confusion
- Count: 247 / 1,040 (23.8% of errors)
- False positives: 131 | False negatives: 116

### Semantic Near-Miss
- Count: 623 / 1,040 (59.9% of errors)
- False positives: 234 | False negatives: 389

### Surface Feature Artifact
- Count: 516 / 1,040 (49.6% of errors)
- False positives: 171 | False negatives: 345

## Suggested Improvements

1. **Polysemy Confusion → Word Sense Disambiguation (WSD):** Use a WSD model to select the contextually appropriate synset before computing embeddings, replacing the allsense average.

2. **Semantic Near-Miss → Cross-Encoder Reranking:** Add a cross-encoder feature that jointly encodes definition and answer, capturing nuanced differences between near-synonyms.

3. **Surface Feature Artifact → Feature Regularization:** Apply L1 regularization or feature elimination to reduce reliance on orthographic features when semantic features suffice.

---

*See `misclassified_examples.csv` for the full misclassified DataFrame.*
*See `08_results_and_evaluation.ipynb` for detailed analysis and examples.*