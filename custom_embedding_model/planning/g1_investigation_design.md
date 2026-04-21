# g1 Investigation Design

**Authors:** Victoria Winters
**Advisor:** Dr. Kevyn Collins-Thompson (University of Michigan)
**Date:** April 2026
**Status:** Draft

---

## 1. Purpose

This document guides the hypothesis testing phase (Stage 6) for g1, our
first fine-tuned CALE model. g1 was trained using triplet margin loss on
T_1 triplets (see §2) and evaluated in NB 05 (Stage 5: Model Evaluation)
and the wordplay ATE breakdown exploration notebook. The evaluation revealed
that g1 learned something — but what it learned is not straightforwardly
useful, and several aspects of its behavior are surprising.

We have identified four design issues in how g1 was constructed: problems
with phrase format, sense selection, and distractor selection that we can
reason about independently of the results. We have also documented nine
empirical findings characterizing g1's behavior on the validation set.

The goal of this document is to connect the two: for each empirical finding,
which design issues could plausibly account for it, and what investigations
would test those connections? The investigations will be implemented in
`notebooks/06_g1_hypothesis_testing.ipynb`.

---

## 2. How g1 Was Built

g1 was fine-tuned from g_stock (unmodified CALE, `gabrielloiseau/CALE-MBERT-en`)
using triplet margin loss (alpha = 1.0, 3 epochs, lr = 2e-5, batch size 32)
on 69,921 training triplets. The triplet design (T_1) was:

| Component | Construction | Example |
|-----------|-------------|---------|
| **Anchor** | f_clue(definition) | `"Parties broken for <t>sea-faring group</t>"` |
| **Positive** | f_common_wndef(answer) | `"<t>fleet</t>: a group of ships"` |
| **Negative** | f_common_wndef(distractor) | `"<t>crew</t>: the people on a ship"` |

Where:
- f_clue wraps the definition word(s) in `<t></t>` delimiters within the
  clue's surface text
- f_common_wndef constructs `"<t>word</t>: <WordNet definition>"` using
  `synsets(word)[0].definition()`
- Distractors were drawn from `dataset_harder.parquet` (Milestone II),
  selected by cosine similarity to the answer embedding

Training used mean pooling (Decision 20). Full training details are in
FINDINGS.md (Stage 3: g1 Training).

---

## 3. Design Issues

These are problems we can identify by reasoning about the construction of
g1's training data, independent of any results.

### DI-1: Unnatural phrase format

f_common_wndef constructs phrases in the format `"<t>word</t>: WordNet
definition"`. This is not naturalistic text — it resembles a dictionary
entry rather than a sentence. CALE was trained on natural text with
`<t></t>` delimiters marking a target word within a passage. The model may
respond to the dictionary-entry format as a surface pattern, learning
something about the format rather than (or in addition to) the semantic
content.

### DI-2: POS-biased sense selection

`synsets(word)[0]` always returns a noun sense when one exists, regardless
of whether the word is more commonly used as another part of speech.
WordNet's `synsets()` groups senses by POS (nouns first, then verbs, then
adjectives, then adverbs). For words like "low" where the dominant usage is
adjectival, we are embedding a noun sense that may be rare or irrelevant.

### DI-3: Unreliable frequency ordering within POS

Even within a given POS, WordNet's sense ordering does not reliably reflect
how common a meaning is. Many senses have zero corpus frequency counts,
making their ordering effectively arbitrary. For "low" as a noun, "an air
mass of lower pressure" (count=0) ranks above "a low level or position or
degree" (count=0) — an ordering few humans would agree with. When the
ordering doesn't reflect actual frequency, we can't be confident that
sense[0] captures the word's most intuitive meaning — so even when we 
intentionally select POS, our assumptions about specific sense may be wrong.

### DI-4: Opaque distractor selection

Unlike FaceNet, where negatives are naturally occurring (other people's
faces), there is no naturally occurring set of wrong answers to a cryptic
clue. Distractors must be constructed or selected by some procedure, and
that procedure embeds assumptions about what makes a good negative. The
negatives in T_1 came from `dataset_harder.parquet`, selected by cosine
similarity to the answer embedding. The implications of this selection
method for what the model learned have not been examined.

---

## 4. Empirical Findings

These are observed behaviors of g1 from the Stage 5 model evaluation
(NB 05) and the wordplay ATE breakdown exploration notebook. All findings
are on the validation split. ATE is reported as a decomposable diagnostic:
changes in ATE confirm the model learned something, but the T=0 
(decontextualized definition-answer similarity) and T=1 (clue-contextualized 
definition-answer similarity) components reveal what it learned.

### EF-1: T=0 compression

Decontextualized similarities (cos_sim of g(f(def)) vs g(f(ans))) jumped
dramatically from g_stock to g1.

| Phrase type | g_stock T=0 mean | g1 T=0 mean | N |
|-------------|-----------------|-------------|---|
| wndef | 0.576 | 0.715 | 47,933 |
| wnex | 0.495 | 0.590 | 4,825 |

### EF-2: T=1 modest rise

Clue-contextualized similarities (cos_sim of g(f_clue(def)) vs g(f(ans)))
rose less than T=0.

| Phrase type | g_stock T=1 mean | g1 T=1 mean | N |
|-------------|-----------------|-------------|---|
| wndef | 0.513 | 0.591 | 47,933 |
| wnex | 0.486 | 0.547 | 4,825 |

### EF-3: Cross-f compression

wnex embeddings compressed even though g1 never trained on wnex phrases.
Mean pairwise cosine among random word pairs: 0.299 (g_stock) → 0.506 (g1).
N = 50,000 random pairs from 3,008 validation-only vocabulary words.

### EF-4: Global discriminability loss

Pairwise cosines increased ~0.17 on wndef (0.398 → 0.571, N = 50,000
random pairs from 26,152 validation vocabulary words). Total variance
dropped ~35% on both phrase types. However, effective dimensionality
slightly increased (wndef: 43.6 → 48.7; wnex: 47.7 → 77.5) — indicating
uniform contraction rather than dimensional collapse.

### EF-5: Fundamental reorganization of similarity structure

RSA between g_stock and g1 pairwise cosine matrices shows near-zero
Spearman rho (wndef: 0.112, wnex: 0.075, both p < 0.001, N = 1,000
words sampled per phrase type). Fine-tuning didn't just contract the
space — it rearranged which words are near which.

### EF-6: High validation triplet accuracy

90.0% on wndef validation triplets (N = 46,506) vs g_stock 38.8%. For each
triplet, the metric computes whether cos(g(f_clue(def)), g(f_wndef(ans))) >
cos(g(f_clue(def)), g(f_wndef(distractor))). g1 learned to place the
clue-contextualized definition embedding closer to the answer's wndef
embedding than to the distractor's wndef embedding in 90% of held-out
triplets. Entangled with distractor selection (DI-4).

### EF-7: Partial cross-f transfer

wnex triplet accuracy 67.2% vs g_stock baseline 40.3% (N = 2,985 matched
triplets), indicating g1 learned some discriminative structure that
transfers beyond the training phrase format. Entangled with distractor
selection (DI-4) and noun-dominant sense selection (DI-2).

### EF-8: Double-def clues show less misdirection than standard clues

Under g_stock, double-def clues show less negative ATE than standard clues
(−0.021 [−0.025, −0.018], N = 4,530 vs −0.068 [−0.069, −0.066],
N = 43,403; 95% bootstrap CIs). This is consistent with the structural
difference: in a double-def clue, the context around the tagged definition
is one or more other definitions of the answer rather than wordplay.

Component decomposition (double_def − standard, 95% Welch CIs):

| Component | g_stock diff | g1 diff |
|-----------|-------------|---------|
| T=0 | −0.120 [−0.125, −0.114] | −0.018 [−0.020, −0.016] |
| T=1 | −0.073 [−0.078, −0.068] | +0.014 [+0.012, +0.017] |

Under g_stock, double-def clues have significantly lower T=0 and lower T=1.
Under g1, the T=0 gap nearly vanishes (compression), and the T=1 difference
flips: double-def clues have significantly higher clue-contextualized
similarity than standard clues.

### EF-9: Letterplay clues show higher T=1 than no-letterplay clues under g1

Under g_stock, most individual (detectable) letterplay types show no significant T=1
difference from no_letterplay (the exception is anagram_consec). Under g1,
every well-powered letterplay type shows significantly higher T=1 than
no_letterplay (95% Welch CIs exclude zero). T=0 differences mostly converge
under g1 due to compression, so this pattern is driven by the clue-
contextualized component.

T=1 differences (letterplay type − no_letterplay, 95% Welch CIs):

| Category | N | g_stock T=1 diff | excludes 0? | g1 T=1 diff | excludes 0? |
|----------|---|-----------------|-------------|------------|-------------|
| anagram_consec | 2,101 | +0.035 [+0.028, +0.042] | yes | +0.035 [+0.032, +0.038] | yes |
| hidden_fwd | 1,494 | −0.003 [−0.012, +0.005] | no | +0.032 [+0.029, +0.035] | yes |
| anagram_single | 557 | +0.007 [−0.006, +0.021] | no | +0.036 [+0.030, +0.042] | yes |
| hidden_rev | 412 | +0.003 [−0.014, +0.019] | no | +0.015 [+0.009, +0.022] | yes |
| selection_firsts | 237 | +0.011 [−0.010, +0.032] | no | +0.019 [+0.010, +0.029] | yes |
| selection_alt | 189 | +0.001 [−0.024, +0.027] | no | +0.027 [+0.017, +0.036] | yes |

One observation about the clues in these categories: algorithmically
detectable letterplay uses whole-word mechanisms (single-word anagram,
answer hidden in consecutive words, etc.), which constrains the setter's
word choices. The setter may have to accept a surface that reads awkwardly
and/or a definition-answer connection that is more tenuous. Clues where we
didn't detect letterplay may be using charades (assembling the answer from
multiple fragments), which gives the setter more freedom in word choice.
Whether and how this connects to the T=1 pattern is an open question.

---

## 5. Proposed Investigations

*To be developed. Each investigation should identify:*

1. *Which design issue(s) it tests*
2. *Which empirical finding(s) it aims to explain*
3. *The specific analysis or computation*
4. *What the result would look like if the design issue does / does not
   account for the finding*

---

## 6. Open Questions

- To what extent do design issues DI-1 through DI-4 account for the
  empirical findings, individually or in combination?
- Are there additional design issues we have not yet identified?
- Which findings reflect problems with g1 specifically, and which reflect
  properties of the data or the evaluation framework that would persist
  across any model?
- What generalizations can we make about g1 that would inform the design
  of g2?
