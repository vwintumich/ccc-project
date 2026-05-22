# g1 Investigation Design

**Authors:** Victoria Winters<br>
**Advisor:** Dr. Kevyn Collins-Thompson (University of Michigan)<br>
**Date:** April 2026<br>
**Status:** Draft

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

## 3. WordNet Idiosyncrasies

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

## 4. Methodological Issues

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

## 5. Empirical Findings

These are observed behaviors of $g_1$ from the Stage 5 model evaluation
(NB 05) and the wordplay ATE breakdown exploration notebook. All findings
are on the validation split. ATE is reported as a decomposable diagnostic:
changes in ATE confirm the model learned something, but the T=0
(decontextualized definition-answer similarity) and T=1 (clue-contextualized
definition-answer similarity) components reveal what it learned.

**Notation:** $\text{sim}(A, B)$ denotes cosine similarity between
embedding vectors $A$ and $B$.

Findings are organized by importance, from the headline result through
diagnostic red flags, generalization checks, and subgroup analyses.

## The headline story

### EF-1: T=0 compression outpaced T=1, making ATE more negative

$g_1$'s ATE on the validation set is more negative than $g_\text{stock}$'s: the model
increased misdirection rather than reducing it. The decomposition reveals
why: decontextualized similarities (T=0) jumped dramatically, while
clue-contextualized similarities (T=1) rose much less.

| | $g_\text{stock}$  | $g_1$  | Shift |
|-----------|-------------|---------|-------|
| T=0: Decontextualized def-ans similarity | 0.576 | 0.715 | +0.139 |
| T=1: Clue Context def-ans similarity | 0.513 | 0.591 | +0.078 |
| ATE | −0.063 | −0.124 | −0.061 |

Note: This table uses $f_\text{wndef}$ for decontextualized embeddings
(N = 47,933 validation pairs). Under $f_\text{wnex}$, both shifts are
smaller and the asymmetry is attenuated — see EF-7. 

If a fine-tuned model were to overcome the misdirection introduced by clue context, we 
would expect T=1 to increase more than T=0 (reducing misdirection), but the opposite 
happened: $g_1$ is more misled by clue context than $g_\text{stock}$ was. The T=0 compression
is the central problem to explain: why do decontextualized embeddings look more alike under $g_1$?

### EF-2: High validation triplet accuracy

90.0% on wndef validation triplets (N = 46,506) vs $g_\text{stock}$ 38.8%. For each
triplet, the metric computes whether $\text{sim}(g(f_\text{clue}(\text{def})),\; g(f_\text{wndef}(\text{ans}))) > \text{sim}(g(f_\text{clue}(\text{def})),\; g(f_\text{wndef}(\text{dist})))$. $g_1$ learned to place the
clue-contextualized definition embedding closer to the answer's wndef
embedding than to the distractor's wndef embedding in 90% of held-out
triplets. Entangled with distractor selection (MI-5).

## Diagnostic red flags

### EF-3: Global discriminability loss

Pairwise cosines among random wndef word pairs increased +0.173
(0.398 → 0.571, N = 50,000 random pairs from 26,152 validation vocabulary
words). Total variance dropped ~35% on both phrase types. However, effective
dimensionality slightly increased (wndef: 43.6 → 48.7; wnex: 47.7 → 77.5),
indicating uniform contraction rather than dimensional collapse.

In metric learning, a well-trained model pulls related items together while
keeping unrelated items apart. The increase in random-pair cosines indicates
$g_1$ failed the second requirement.

### EF-4: Fundamental reorganization of similarity structure

RSA between $g_\text{stock}$ and $g_1$ pairwise cosine matrices shows near-zero
Spearman rho (wndef: 0.112, wnex: 0.075, both p < 0.001, N = 1,000
words sampled per phrase type). Fine-tuning didn't just contract the
space — it rearranged which words are near which. Combined with EF-3, the
picture is that $g_1$ built a new, more compressed representation rather than
refining the pretrained one.

## Generalization checks

### EF-5: Full cross-f compression

wnex embeddings compressed even though $g_1$ never trained on wnex phrases.
Mean pairwise cosine among random wnex word pairs shifted +0.207
(0.299 → 0.506, N = 50,000 random pairs from 3,008 validation-only
vocabulary words) — comparable to the wndef shift of +0.173 (EF-3). The
compression transferred fully across phrase formats; whatever caused the
embedding space to contract, it was not specific to the wndef dictionary-
entry format.

### EF-6: Partial cross-f discriminative transfer

wnex triplet accuracy 67.2% vs $g_\text{stock}$ baseline 40.3% (N = 2,985 matched
triplets), indicating $g_1$ learned some discriminative structure that
transfers beyond the training phrase format. However, the drop from 90.0%
(wndef, EF-2) to 67.2% (wnex) shows that a substantial portion of the
discriminative learning is format-specific. Contrast with EF-5: the
compression transferred fully, but the discrimination transferred only
partially. Entangled with distractor selection (MI-5) and noun
overrepresentation (MI-2).

### EF-7: T=0/T=1 asymmetry is attenuated under wnex

On the 4,825 validation pairs resolvable under both $f_\text{wndef}$ and
$f_\text{wnex}$, wnex shows the same qualitative pattern as EF-1 — T=0
compresses more than T=1 — but the shifts are smaller and the ATE
shows that while $g_1$ is still slightly more misled by clue context 
than $g_\text{stock}$ was, this is less pronounced for $f_\text{wnex}$.

| | $g_\text{stock}$ | $g_1$ | Shift |
|--|--|--|--|
| T=0 | 0.495 | 0.590 | +0.095 |
| T=1 | 0.486 | 0.547 | +0.061 |
| ATE | −0.009 | −0.043 | −0.034 |

Compare with EF-1's wndef results (T=0 shift +0.139, T=1 shift +0.078,
ATE shift −0.061). Both components are smaller under wnex, and the ATE
deterioration is roughly half as large. However, this comparison is
confounded: the 4,825 wnex-resolvable pairs are a different subset of
clues than the 47,933 wndef pairs, so the difference could reflect word
composition rather than phrase format. INV-3 isolates the format effect by
computing wndef T=0 and T=1 on these same 4,825 pairs.

## Subgroup analyses

### EF-8: Double-def clues show less misdirection than standard clues

Under $g_\text{stock}$, double-def clues show less negative ATE than standard clues
(−0.021, N = 4,530 vs −0.068, N = 43,403). This is consistent with the structural
difference: in a double-def clue, the context around the tagged definition
is one or more other definitions of the answer rather than wordplay.

Component decomposition (double_def − standard):

| Component | $g_\text{stock}$ diff | $g_1$ diff |
|-----------|-------------|---------|
| T=0 | −0.120 | −0.018 |
| T=1 | −0.073 | +0.014 |

Under $g_\text{stock}$, double-def clues have significantly lower T=0 and lower T=1.
Under $g_1$, the T=0 gap nearly vanishes (compression), and the T=1 difference
flips: double-def clues have significantly higher clue-contextualized
similarity than standard clues.

### EF-9: Letterplay clues show higher T=1 than no-letterplay clues under $g_1$

Under $g_\text{stock}$, most individual (detectable) letterplay types show no
significant T=1 difference from no_letterplay (the exception is
anagram_consec). Under $g_1$, every well-powered letterplay type shows
significantly higher T=1 than no_letterplay.
T=0 differences mostly converge under $g_1$ due to compression, so this
pattern is driven by the clue-contextualized component.

T=1 differences (letterplay type − no_letterplay):

| Category | N | $g_\text{stock}$ T=1 diff | sig? | $g_1$ T=1 diff | sig? |
|----------|---|-----------------|------|------------|------|
| anagram_consec | 2,101 | +0.035 | yes | +0.035 | yes |
| hidden_fwd | 1,494 | −0.003 | no | +0.032 | yes |
| anagram_single | 557 | +0.007 | no | +0.036 | yes |
| hidden_rev | 412 | +0.003 | no | +0.015 | yes |
| selection_firsts | 237 | +0.011 | no | +0.019 | yes |
| selection_alt | 189 | +0.001 | no | +0.027 | yes |

One observation about the clues in these categories: algorithmically
detectable letterplay uses whole-word mechanisms (single-word anagram,
answer hidden in consecutive words, etc.), which constrains the setter's
word choices. The setter may have to accept a surface that reads awkwardly
and/or a definition-answer connection that is more tenuous. Clues where we
didn't detect letterplay may be using charades (assembling the answer from
multiple fragments), which gives the setter more freedom in word choice.
Whether and how this connects to the T=1 pattern is an open question.

---

## 6. Proposed Investigations

Each investigation identifies which methodological issue(s) it tests, which
empirical finding(s) it aims to explain, the specific analysis or
computation, and what the result would look like if the methodological issue
does or does not account for the finding.

All investigations use existing validation-split embeddings and run
locally on CPU. No new GPU work is needed.

### INV-1: Generic vs. semantic compression

**Tests:** MI-1 (format compression)
**Explains:** EF-1 (ATE decomposition), EF-3 (global discriminability loss)

**Logic:** If T=0 compression is format-driven, then the cosine increase
should be roughly the same for *random* word pairs as for *actual*
(definition, answer) pairs — the model just pushed the whole wndef cloud
together. If the compression is semantically targeted, actual pairs
should compress more than random pairs.

**Computation:** Using validation-split wndef embeddings for both
$g_\text{stock}$ and $g_1$:

- For each of the 47,933 actual (definition, answer) pairs, compute the
  T=0 shift:

$$\Delta_{\text{T=0}} = \text{sim}\bigl(g_1(f_\text{wndef}(\text{def})),\; g_1(f_\text{wndef}(\text{ans}))\bigr) - \text{sim}\bigl(g_\text{stock}(f_\text{wndef}(\text{def})),\; g_\text{stock}(f_\text{wndef}(\text{ans}))\bigr)$$

- Compute the same cosine shift for 47,933 random word pairs (same
  count, random_state=42)
- Compare distributions: mean, median, histograms overlaid

**What to expect:**
- If shifts are similar → compression is generic (supports MI-1)
- If actual pairs compress more than random pairs → the model learned
  something about the semantic relationship, not just the format
- If actual pairs compress *less* than random pairs → the model may
  have learned to spread out semantically related words (unlikely but
  worth checking)

### INV-2: T=0 compression stratified by POS composition

**Tests:** MI-2 (noun overrepresentation)
**Explains:** EF-1 (ATE decomposition)

**Logic:** If noun overrepresentation in training causes T=0 compression,
then noun-noun validation pairs should show larger T=0 shifts than
non-noun pairs, because the model received much stronger gradient signal
for noun embeddings.

**Computation:** Using the POS census (§5b of `pos_wordnet_census.ipynb`
already gives us the 2×2), stratify validation pairs into four groups by
(def sense[0] POS, ans sense[0] POS): noun-noun (67.8%), noun-other,
other-noun, other-other (15.8%). For each group:
- Mean $g_\text{stock}$ T=0, mean $g_1$ T=0, mean $\Delta_{\text{T=0}}$
- Distribution of per-pair shifts (box plot or violin, one per group)
- Welch CIs on the difference in mean $\Delta_{\text{T=0}}$ between noun-noun
  and other-other

**What to expect:**
- If noun-noun shifts $\gg$ other-other shifts → MI-2 is a major
  contributor
- If shifts are comparable across groups → compression is POS-agnostic
  (points to MI-1 or something else)
- The noun-other and other-noun groups are informative too: if one noun
  role is enough to drive compression, that's a different story than
  requiring both

### INV-3: Format isolation on matched pairs

**Tests:** MI-1 (format compression) in isolation from word composition
**Explains:** EF-1 (ATE decomposition), EF-5 (cross-f compression)

**Logic:** NB 05 already showed that wnex T=0 compressed less than wndef
T=0, but the comparison was confounded by word composition: the 4,825
wnex-resolvable pairs are a different (overlapping) subset than the
47,933 wndef pairs. This investigation isolates the format effect by
computing T=0 under *both* phrase formats on the *exact same* word
pairs.

NB 05 §6 already reports the wnex-side T=0 and T=1 for the 4,825
matched pairs, and §6d reports the matched wndef ATE. The missing piece
is the wndef T=0 and T=1 means on those same 4,825 pairs (§5 reports
wndef T=0/T=1 on the full 47,933 pairs, not the matched subset). One
additional computation completes the picture.

**Computation:** Take the 4,825 validation pairs where both
`definition_wn` and `answer_wn` are in the wnex vocabulary. For each
pair, compute four T=0 values:

$$\text{T=0}_\text{wndef}^{g_\text{stock}},\quad \text{T=0}_\text{wndef}^{g_1},\quad \text{T=0}_\text{wnex}^{g_\text{stock}},\quad \text{T=0}_\text{wnex}^{g_1}$$

The wnex values are already in NB 05 §6. Then compare the T=0 shift
$\Delta_{\text{T=0}} = \text{T=0}^{g_1} - \text{T=0}^{g_\text{stock}}$ between the two
formats on identical pairs. The difference in shift is purely
attributable to format, since the words are held constant.

**What to expect:**
- If $\Delta_{\text{T=0}}^\text{wndef} \gg \Delta_{\text{T=0}}^\text{wnex}$ on
  matched pairs → the wndef format itself is being compressed
  (supports MI-1)
- If shifts are similar → the compression is happening at the semantic
  level, not the format level, and wnex is just along for the ride
- Compare the wnex shift magnitude here against the random-pair
  baseline from INV-1 to see if wnex compression is generic or
  targeted

### INV-4: What were these triplets trying to teach the model?

**Tests:** MI-5 (distractor selection)
**Explains:** EF-1 (ATE decomposition), EF-2 (triplet accuracy)

**Logic:** Distractors were selected in Milestone II using allsense
embeddings, where the mean definition-distractor cosine similarity (0.77)
exceeds the mean definition-answer similarity (0.65). But $g_1$ was
trained on $f_\text{wndef}$ embeddings, not allsense embeddings. Before
interpreting what $g_1$ learned, we need to characterize the training
signal it actually received: under $g_\text{stock}$ wndef embeddings, is
the distractor still closer to the definition than the true answer? If
so, the triplet loss was receiving gradient signal to pull in a farther
word (the answer) and push away a nearer word (the distractor).

**Computation:** Using $g_\text{stock}$ full-vocabulary wndef embeddings
and `g1_train.csv` (69,921 training triplets), compute for each triplet:

$$\text{sim}\bigl(g_\text{stock}(f_\text{wndef}(\text{def})),\; g_\text{stock}(f_\text{wndef}(\text{ans}))\bigr) \quad\text{vs.}\quad \text{sim}\bigl(g_\text{stock}(f_\text{wndef}(\text{def})),\; g_\text{stock}(f_\text{wndef}(\text{dist}))\bigr)$$

- Report: mean, median, and distribution of both similarities
- Report: fraction of triplets where distractor is closer to definition
  than answer (the "uphill" fraction)
- Repeat on validation triplets (`g1_val.csv`, 46,506 triplets) for
  comparison

Also compute the same quantities using $f_\text{clue}$ anchors (which is
what the model actually trains on):

$$\text{sim}\bigl(g_\text{stock}(f_\text{clue}(\text{def})),\; g_\text{stock}(f_\text{wndef}(\text{ans}))\bigr) \quad\text{vs.}\quad \text{sim}\bigl(g_\text{stock}(f_\text{clue}(\text{def})),\; g_\text{stock}(f_\text{wndef}(\text{dist}))\bigr)$$

NB 05 §2 already tells us $g_\text{stock}$ triplet accuracy is 38.8% on
validation, meaning in 61.2% of triplets the distractor is closer to the
$f_\text{clue}$ anchor than the answer. The decontextualized comparison
above reveals how much of this is due to the clue context shifting the
anchor versus the distractor being inherently closer.

**What to expect:**
- If distractor is closer than answer under $g_\text{stock}$ wndef →
  the model was being asked to invert the similarity ordering, which is
  hard and may explain the aggressive compression (EF-1) as a side
  effect
- If answer is closer than distractor under $g_\text{stock}$ wndef →
  the distractor selection's allsense-based ranking doesn't transfer to
  wndef, and the training task was easier than it appeared
- The gap between the decontextualized and $f_\text{clue}$ versions
  quantifies how much the clue context contributes to the difficulty of
  the triplet task

---

## 7. Open Questions

- To what extent do methodological issues MI-1 through MI-5 account for the
  empirical findings, individually or in combination?
- Are there additional methodological issues we have not yet identified?
- Which findings reflect problems with $g_1$ specifically, and which reflect
  properties of the data or the evaluation framework that would persist
  across any model?
- What have we learned that should inform the design of $g_2$?
