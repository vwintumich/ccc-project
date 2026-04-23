# g1 Investigation Design

**Authors:** Victoria Winters<br>
**Advisor:** Dr. Kevyn Collins-Thompson (University of Michigan)<br>
**Date:** April 2026<br>
**Status:** Draft (v2)

---

## 1. Purpose

This document guides the hypothesis testing phase (Stage 6) for $g_1$, our
first fine-tuned CALE model. $g_1$ was trained using triplet margin loss on
$\tau_1$ triplets (see §2) and evaluated in NB 05 (Stage 5: Model Evaluation)
and the wordplay ATE breakdown exploration notebook. The evaluation revealed
that $g_1$ learned something — but what it learned is not straightforwardly
useful, and several aspects of its behavior are surprising.

We have identified five methodological issues in how $g_1$ was constructed — problems
with phrase format, training data composition, sense selection, and
distractor selection — that we can reason about independently of the
results. We have also documented nine empirical findings characterizing
$g_1$'s behavior on the validation set.

The goal of this document is to connect the two: for each empirical finding,
which methodological issues could plausibly account for it, and what investigations
would test those connections? The investigations will be implemented in
`notebooks/06_g1_hypothesis_testing.ipynb`.

---

## 2. How $g_1$ Was Built

$g_1$ was fine-tuned from $g_\text{stock}$ (unmodified CALE, `gabrielloiseau/CALE-MBERT-en`)
using triplet margin loss ($\alpha = 1.0$, 3 epochs, lr = 2e-5, batch size 32, mean pooling)
on 69,921 training triplets. The triplet design ($\tau_1$) was:

| Component | Construction | Example |
|-----------|-------------|---------|
| **Anchor** | $f_\text{clue}$(definition) | `"Parties broken for <t>sea-faring group</t>"` |
| **Positive** | $f_\text{wndef}$(answer) | `"<t>fleet</t>: a group of ships"` |
| **Negative** | $f_\text{wndef}$(distractor) | `"<t>crew</t>: the people on a ship"` |

Where:
- $f_\text{clue}$ wraps the definition word(s) in `<t></t>` delimiters within the
  clue's surface text
- $f_\text{wndef}$ constructs `"<t>word</t>: <WordNet definition>"` using
  `synsets(word)[0].definition()`
- Distractors were drawn from `dataset_harder.parquet` (Milestone II),
  selected by cosine similarity to the definition's allsense embedding

This training setup was meant to ask the model:

*Please learn to associate the definition word with the clue's true answer, despite
the misdirection introduced by the clue context.*

If the model learns this, we expect T=1 to increase (clue-contextualized definitions will be
closer to answer embeddings because $g_1$ overcame misdirection) and the ATE to no longer be
negative (taking clue context into account does not push the definition farther from the answer).

---

## 3. Model Performance

Before interpreting what $g_1$ learned about misdirection, we assess its
basic health as a fine-tuned model: did training converge? Is there evidence
of overfitting? What happened to the embedding space?

### 3.1 Training dynamics and overfitting

Training loss decreased rapidly over three epochs while validation loss
plateaued and then rose — a classic overfitting pattern:

| Epoch | Train Loss | Val Loss | Val Accuracy | Val Mean Margin |
|-------|-----------|----------|--------------|-----------------|
| 1 | 0.470 | 0.305 | 87.2% | 0.053 |
| 2 | 0.111 | 0.252 | 89.7% | 0.088 |
| 3 | 0.014 | 0.264 | 90.0% | 0.125 |


By epoch 3, training loss (0.014) was 19× lower than validation loss
(0.264) — the model had nearly memorized the training triplets. Validation
loss improved from epoch 1 to epoch 2 (0.305 → 0.252) but increased at
epoch 3 (0.252 → 0.264), indicating that the best-generalizing checkpoint
was epoch 2, not the final epoch we deployed.

Validation accuracy continued to rise slightly through epoch 3 (87.2% →
89.7% → 90.0%), and mean margin increased steadily (0.053 → 0.088 →
0.125). The model became more *confident* in its predictions even as
validation loss started rising — the correctly-classified triplets were
pushed further apart while the incorrectly-classified ones incurred
larger penalties.

### 3.2 Task performance

$g_1$ achieves 90.0% validation triplet accuracy on 46,506 held-out
$\tau_1$ triplets (wndef, full-vocabulary), compared to $g_\text{stock}$'s
38.8%. The model learned to place the clue-contextualized definition
embedding closer to the answer's wndef embedding than to the distractor's
wndef embedding in the large majority of held-out triplets.

### 3.3 Embedding space compression

Fine-tuning contracted the embedding space without collapsing it
dimensionally:

| Metric | $g_\text{stock}$ | $g_1$ |
|--------|-------------------|-------|
| Mean pairwise cosine (wndef, 50K random pairs) | 0.398 | 0.571 |
| Mean pairwise cosine (wnex, 50K random pairs) | 0.299 | 0.506 |
| Total variance (wndef) | 13,454,020 | 8,720,421 |
| Total variance (wnex) | 1,943,473 | 1,228,944 |
| Effective dimensionality (wndef) | 43.6 | 48.7 |
| Effective dimensionality (wnex) | 47.7 | 77.5 |

Total variance dropped ~35% on both phrase types, while effective
dimensionality (participation ratio of singular values) slightly
increased. This indicates uniform contraction — embeddings moved closer
together across all dimensions — rather than dimensional collapse, where
variance would concentrate into fewer dimensions.

In a well-trained metric learning model, related items are pulled together
while unrelated items are kept apart. The increase in random-pair cosines
indicates $g_1$ failed the second requirement: unrelated words became more
similar, not less.

### 3.4 Structural reorganization

RSA (Representational Similarity Analysis) between $g_\text{stock}$ and
$g_1$ pairwise cosine matrices shows near-zero Spearman correlation:

| Phrase type | Spearman ρ | p-value |
|-------------|-----------|---------|
| wndef | 0.112 | < 0.001 |
| wnex | 0.075 | < 0.001 |

(N = 1,000 words sampled per phrase type, random_state=42.)

Fine-tuning didn't just contract the space — it rearranged which words are
near which. Combined with §3.3, the picture is that $g_1$ built a new, more
compressed similarity structure rather than refining the pretrained one.

---

## 4. WordNet Idiosyncrasies

The phrase construction strategy $f_\text{wndef}$ assumes that the first
synset of a word in WordNet (`synsets(word)[0]`) corresponds to the
most common meaning of that word. However, this assumption holds for
only 13% of the wndef vocabulary, due to three properties of WordNet:

### WN-1: Half of words have only one meaning in WordNet

In the wndef vocabulary, 47% of words have only one synset. For these
words there is no "common" versus "uncommon" meaning — the sole synset
is the only meaning WordNet records.

### WN-2: WordNet sorts synsets by POS

`synsets(word)[0]` always returns a noun sense when one exists, regardless
of whether the word is more commonly used as another part of speech.
WordNet's `synsets()` groups senses by POS (nouns first, then verbs, then
adjectives, then adverbs). For words like "low" where the dominant usage is
adjectival, we are embedding a noun sense that may be rare or irrelevant.

### WN-3: Without lemma counts, synset order is arbitrary

Even within a given POS, WordNet's sense ordering does not reliably
reflect how common a meaning is. In the wndef vocabulary, only 19% of all
synsets have nonzero lemma counts. Lemma counts can disambiguate between
senses for only 18% of vocabulary words; 47% have only one synset (making
the question moot), and 35% have multiple synsets but lack the frequency
evidence to determine whether sense[0] is the most common meaning. For
"low" as a
noun, "an air mass of lower pressure" (count=0) ranks above "a low
level or position or degree" (count=0) — an ordering few humans would
agree with. When the ordering lacks frequency evidence, we cannot
determine whether `synsets(word)[0]` captures the word's most common
meaning within that POS.

---

## 5. Methodological Issues

These are problems we can identify by reasoning about the construction of
$g_1$'s training data, independent of any results.

### MI-1: Unnatural phrase format

$f_\text{wndef}$ constructs phrases in the format `"<t>word</t>: WordNet
definition"`. This is not naturalistic text — it resembles a dictionary
entry rather than a sentence. CALE was trained on natural text with
`<t></t>` delimiters marking a target word within a passage. The model may
respond to the dictionary-entry format as a surface pattern, learning
something about the format rather than (or in addition to) the semantic
content.

### MI-2: Nouns were overrepresented in the training data

Nouns were the majority POS class in the training triplets, with 73.5% of
positive (answer) roles and 70.6% of negative (distractor) roles having
noun as their sense[0] POS. The model received disproportionately strong
gradient signal for noun-sense embeddings, which may have driven it to
learn noun-specific structure rather than POS-general structure. This may
interact with WN-2: because WordNet sorts synsets by POS, $f_\text{wndef}$
would have selected a noun sense over a more common non-noun sense.

### MI-3: $f_\text{wndef}$ did not always capture the most common meaning

Due to WN-2 and WN-3, `synsets(word)[0]` is confirmed as the most
common synset for only 13% of the wndef vocabulary (the "Most Common"
category in the sense-level census). For 47% of words there is only one
synset (WN-1), making sense selection trivially correct but
uninformative about frequency. For 33% of words, all synsets have zero
lemma counts (WN-3), so we cannot determine whether sense[0] is the
most common meaning. And for 5.5% of words, sense[0] is demonstrably
not the most common synset — some other synset has a strictly higher
lemma count. The phrases for these words embed a less common meaning
than intended.

### MI-4: $f_\text{wndef}$ preferentially selected noun senses

Due to WN-2, $f_\text{wndef}$ selects a noun sense for every word that
has at least one noun synset, regardless of whether noun is the word's
most commonly used part of speech. For words like "low" (predominantly
adjectival) or "plant" (often verbal in clue contexts), the phrase
embeds a noun meaning that may not match how the word is used in
practice — and in particular, may not match the POS of the same word
when it appears as a definition in a clue surface (the $f_\text{clue}$ anchor).
The POS census shows that 22.3% of training anchors have a contextual
POS that disagrees with their WordNet sense[0] POS.

### MI-5: Opaque distractor selection

Unlike FaceNet, where negatives are naturally occurring (other people's
faces), there is no naturally occurring set of wrong answers to a cryptic
clue. Distractors must be constructed or selected by some procedure, and
that procedure embeds assumptions about what makes a good negative.

The negatives in the $\tau_1$ triplets came from `dataset_harder.parquet` (Milestone II NB 05).
For each definition, all candidate answer words were ranked by cosine
similarity between the definition's allsense embedding and the candidate's
allsense embedding, where the allsense embedding is the average of stock
CALE embeddings across all of a word's WordNet synsets (each synset embedded
using a per-synset phrase construction cascade that used a mix of wnex and wndef).
One distractor was sampled from the top-100 most similar candidates, excluding the true answer.

This procedure produced distractors that are, on average, *more* similar to
the definition than the true answer is (using the allsense embeddings): the mean definition-distractor
cosine similarity is 0.77, compared to 0.65 for real definition-answer
pairs. When we trained $g_1$ to move the anchor toward the positive and away
from the negative, we may have been effectively asking the model to reconfigure the
embedding space so that slightly *less* similar words are actually *closer*,
rather than asking it to learn something about clue misdirection.

---

## 6. What Did $g_1$ Learn? Competing Interpretations

The central question for Stage 6 is not just *what happened* — the empirical
findings document that — but *what did $g_1$ actually learn?* Several
interpretations are consistent with some subset of the evidence. The purpose
of the investigations (§6) is to determine which interpretations are
supported and which can be ruled out.

### Story A: $g_1$ learned something about misdirection

This is the optimistic interpretation — that $g_1$ learned something
genuinely useful about the relationship between cryptic clue context and the
answer. There are two variants:

**Story A1: "See past misdirection."** The clue's surface reading misleads
the embedding; $g_1$ learned to resist this misleading context and embed the
definition closer to the answer despite the surface reading pulling it away.
This predicts that T=1 (clue-contextualized similarity) should increase more
than T=0 (decontextualized similarity), because the model is specifically
counteracting the harmful effect of clue context.

**Story A2: "Discover deeper structure."** The clue's surface text contains
a signal — unique to cryptic crosswords — that actually points toward the
answer, but $g_\text{stock}$ cannot exploit it. $g_1$ learned to extract
this signal, making clue context *helpful* rather than harmful. This also
predicts T=1 increasing more than T=0, because the model is learning to
exploit useful information in the clue context that is invisible to
$g_\text{stock}$.

**Evidence against both variants:** The ATE went *more* negative under $g_1$
(−0.063 → −0.124). T=0 rose by +0.139 while T=1 rose by only +0.078. Both
A1 and A2 predict the opposite pattern — T=1 gaining more than T=0. The
model became *more* misled by clue context, not less.

**Why we cannot rule them out entirely:** A1 or A2 could be operating as a
partial effect that is overwhelmed by other changes to the embedding space.
The investigations below test whether alternative explanations account for
the observed behavior; whatever remains unexplained is candidate evidence
for Story A.

### Story B: $g_1$ compressed everything uniformly

Under this interpretation, fine-tuning simply contracted the embedding space
without learning any discriminative structure — a form of representational
collapse where all embeddings drift toward the mean.

**Evidence against:** This story is ruled out by EF-2 (90% validation
triplet accuracy — the model learned *which* words to place closer to which
anchors) and EF-4 (RSA ρ ≈ 0.1 — the similarity structure was rearranged,
not merely contracted). Uniform compression would produce near-chance
triplet accuracy and high RSA correlation.

### Story C: $g_1$ learned to invert the similarity ordering

The $\tau_1$ distractors were selected to be *more* similar to the definition
than the true answer (mean allsense cosine 0.77 vs 0.65). If this ordering
persists under $g_\text{stock}$ wndef embeddings, then the triplet loss was
receiving gradient signal to pull in a farther word (the answer) and push
away a nearer word (the distractor). $g_1$ may have learned to reconfigure
the embedding space so that these less-similar words end up closer — not
because it learned about misdirection, but because that is literally what
the triplets rewarded.

This story is distinct from Story B: the model didn't compress blindly —
it actively learned to place specific words closer together. But it did so
based on the distractor selection criterion rather than any property of
misdirection.

**What would confirm it:** INV-4 tests whether distractors are indeed closer
than answers under $g_\text{stock}$ wndef embeddings. If so, the training
signal was systematically asking the model to invert the similarity ordering,
and the T=0 compression (EF-1) and discriminability loss (EF-3) are
predictable consequences.

### Story D: $g_1$ learned our phrase construction strategy

The positive and negative in every $\tau_1$ triplet share the $f_\text{wndef}$
format (`"<t>word</t>: WordNet definition"`), while the anchor uses a
different format ($f_\text{clue}$, naturalistic text with delimiters). The
model may have learned to recognize and compress phrases sharing the
dictionary-entry format, without learning anything about the semantic content
of those phrases.

**Connection to MIs:** MI-1 (unnatural phrase format). EF-5 (full cross-f
compression) provides partial evidence against a pure format story — wnex
phrases also compressed — but the transfer could reflect shared structural
features between wndef and wnex formats rather than semantic generalization.

### Story E: $g_1$ learned WordNet's noun preference

Due to WN-2, $f_\text{wndef}$ selects noun senses whenever available,
regardless of a word's typical usage. 73.5% of positive roles and 70.6% of
negative roles were nouns. The model received disproportionately strong
gradient signal for noun-sense embeddings and may have learned noun-specific
structure that does not generalize to other parts of speech.

**Connection to MIs:** MI-2 (noun overrepresentation), MI-4 (noun sense
preference). INV-2 tests whether T=0 compression is concentrated in
noun-noun pairs.

### These stories are not mutually exclusive

Multiple stories could be operating simultaneously. The investigations in §6
are designed to assess the *relative contribution* of each — expressed as
comparative ratios (e.g., "noun-noun compression was 1.5× as large as
other-other compression") rather than decomposed percentages, since the
stories may interact and their effects need not be additive.
