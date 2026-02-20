# PROJECT_OVERVIEW.md — CCC Indicator Clustering: Project Overview

Last updated: February 19, 2026

---

## What This Project Is

This is a MADS capstone project at the University of Michigan focused on unsupervised clustering of cryptic crossword clue (CCC) indicator words/phrases. The central question: does the semantic space of indicator language naturally reflect the structure of CCC wordplay, and if so, at what level of granularity?

---

## Background: Why CCCs Are Interesting for NLP

Cryptic crossword clues (CCCs) are short phrases with a strict grammar involving letterplay, word substitution, and multiple simultaneous semantic readings of the same text. Srivastava et al. (2022) identify CCC solving as a task currently beyond LLM capabilities. CCCs deliberately misdirect the solver — the surface reading of the clue points away from the actual answer.

Every CCC contains three components:
- A **definition** (a synonym or near-synonym of the answer, found at the beginning or end of the clue)
- **Fodder** (the letters to be manipulated by wordplay)
- An **indicator** (a word or phrase signaling what type of wordplay is required)

The indicator is the focus of our clustering task. See DOMAIN_KNOWLEDGE.md for a full explanation of how indicators work and how they relate to wordplay types.

---

## Dataset

**Source:** George Ho's CCC dataset (Ho, 2022)
**URL:** https://cryptics.georgeho.org/
**License:** Open Database License
**Total clues:** 660,613

The indicators and wordplay labels were parsed from blog post commentary, not from the clues directly. George Ho confirmed via email: "those tables are constructed from English explanations that I scraped from various blogs." This means wordplay labels can be used as ground truth without data leakage concerns.

### Indicator Clustering Dataset (as of Feb 8, 2026 meeting)

| Wordplay Type | All Instances | Unique | Verified | Single-Word |
|---------------|--------------|--------|----------|-------------|
| alternation   | 769          | 244    | 216      | 30          |
| anagram       | 45,648       | 7,121  | 6,610    | 3,429       |
| container     | 14,144       | 1,909  | 1,728    | 774         |
| deletion      | 2,093        | 873    | 695      | 294         |
| hidden        | 3,381        | 1,110  | 972      | 441         |
| homophone     | 4,672        | 663    | 565      | 154         |
| insertion     | 11,171       | 2,155  | 1,915    | 613         |
| reversal      | 11,989       | 1,660  | 1,495    | 357         |
| **Total**     | **93,867**   | **15,735** | **14,196** | **6,092** |

**Key dataset facts:**
- Severe class imbalance: anagram dominates (~47% of all instances)
- Most verified indicators are 1-3 words long; very few exceed 4 words
- The same indicator string can appear under multiple wordplay types (e.g., "about" appears as container, reversal, and anagram) — this is a known challenge for clustering
- **Verified indicators** are those where the intact phrase appears in both the blog commentary AND the clue text. This is the primary quality filter (Victoria's checksum breakthrough). Use verified_indicators.csv as the canonical dataset.

---

## The Eight Wordplay Types in Our Dataset

See DOMAIN_KNOWLEDGE.md for detailed descriptions. Brief summary:

| Type | What the indicator signals |
|------|---------------------------|
| anagram | Letters of fodder are rearranged |
| reversal | Letters of a chunk are reversed (subtype of anagram) |
| hidden | Answer appears consecutively inside the fodder |
| container | One chunk is placed around another |
| insertion | One chunk is placed inside another (inverse of container) |
| deletion | A chunk is subtracted from another, OR the complement of a selection |
| homophone | Answer sounds like the fodder |
| alternation | Every other letter of the fodder is selected |

**Important ambiguities in the data:**
- Reversal is technically a subtype of anagram; some data may be mislabeled
- Container and insertion are inverse operations sharing many indicator phrases
- Deletion plays two distinct grammatical roles that may not cluster together

---

## Research Questions and Planned Experiments

**Primary question:** Can unsupervised clustering of CCC indicators produce groups interpretable as wordplay types?

**Planned experiments (in approximate priority order):**

1. **Full verified indicator set** — cluster all 14,196 indicators; compare to known wordplay types at multiple levels of granularity (k=3, k=4, k=8, k=14+)
2. **Anagram indicators only** — anagram has the most diverse vocabulary; clustering within this type may reveal conceptual metaphor subgroups
3. **Hidden / Insertion / Container together** — hypothesis: these share placement-oriented conceptual metaphors and will not separate cleanly
4. **Easy separation test (e.g., Reversal + Homophone)** — if we cannot separate theoretically well-distinguished types, we cannot expect success on the harder full problem
5. **Definitions as control** — cluster definitions using the same pipeline; if definitions cluster as well as indicators, the clustering is not detecting wordplay-specific structure
6. **Seed clustering** — cluster only seed words first to validate that seeds fall into their intended groups before expanding

**Primary hypothesis:** Indicators will not cluster cleanly into 8 wordplay types, but may cluster into higher-level conceptual groups. The degree to which clean clusters emerge is itself a meaningful research finding.

---

## Computational Resource Guide

| Task | Resource |
|------|----------|
| Data cleaning, EDA, loading embeddings | Local or Colab |
| BGE-M3 embedding generation | **Great Lakes (GPU required)** |
| UMAP dimensionality reduction | Great Lakes or Colab (GPU helps) |
| Clustering single runs | Colab |
| Clustering parameter sweeps | **Great Lakes** |
| Visualization, evaluation, report writing | Local or Colab |

---

## Key References

- Ho, G. (2022). A Dataset of Cryptic Crossword Clues. https://cryptics.georgeho.org/
- Srivastava et al. (2022). Beyond the Imitation Game. https://doi.org/10.48550/arXiv.2206.04615
- Efrat et al. (2021). Cryptonite. EMNLP 2021. https://doi.org/10.18653/v1/2021.emnlp-main.344
- Rozner et al. (2021). Decrypting Cryptic Crosswords. NeurIPS 2021.
- Cleary, J. (1996). Misleading contexts. Edinburgh Working Papers in Applied Linguistics.
- Tiernan, A., & Runnalls, L. (2025). Minute Cryptic. St. Martin's Griffin.
