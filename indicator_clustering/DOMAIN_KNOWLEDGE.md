# DOMAIN_KNOWLEDGE.md — CCC Wordplay Structure and Indicator Taxonomy

Last updated: February 19, 2026
Primary author: Victoria (with input from faculty meetings)

This document contains everything Claude needs to understand about cryptic crossword clues (CCCs) and their indicators. Read this before making any decisions about clustering structure, seed words, or evaluation criteria.

---

## Glossary of Key Terms

**CCC (Cryptic Crossword Clue):** A short phrase that simultaneously functions as a surface-reading sentence and as a set of instructions for wordplay. Every CCC has exactly one correct solution.

**Fodder:** The word(s) in the clue text that provide the raw letters for wordplay.

**Chunk:** A series of letters produced by applying an operation to fodder. A chunk may be an intermediate result or the final answer.

**Indicator:** A word or phrase in the clue text that signals what type of wordplay operation to apply to the fodder. Indicators are the subject of our clustering task.

**Definition:** A word or phrase at the beginning or end of the clue that is a synonym or near-synonym of the answer.

**Wordplay type:** The category of operation indicated by an indicator word. The Ho dataset includes 8 labeled types: anagram, reversal, hidden, container, insertion, deletion, homophone, alternation.

**Conceptual metaphor:** The underlying semantic concept that makes an indicator word appropriate for a given wordplay type. For example, "scrambled", "mixed up", and "broken" all invoke the conceptual metaphor of DISORDER, which is why they work as anagram indicators.

**Surface reading:** The literal interpretation of the clue as a normal English sentence, which is deliberately designed to mislead the solver away from the actual wordplay meaning.

---

## How CCC Solving Works

### Simple CCC (no charades)

1. From the clue text, identify the word(s) serving as **fodder** for wordplay
2. Apply an extraction, substitution, or other operation to the fodder, guided by the **indicator**, to produce a **chunk** of letters
3. Optionally rearrange letters within the chunk (guided by an indicator)
4. If the chunk matches the clue requirements (letter count, fits the definition), it is the solution

### CCC with Charades (multiple fodder components)

When a clue has multiple charades, there are multiple fodder/chunk sequences that must be combined:
1. Identify multiple fodder sets
2. Each fodder produces a chunk via its own operation
3. Optionally rearrange letters within each chunk
4. Combine chunks using a positioning indicator (e.g., "inside", "before", "after")
5. Verify against the definition and letter count

### Worked Example

*"Car somersaults during distance event, defying belief. (7)"*
Answer: MIRACLE

- Fodder #1: "Car" → reversed by "somersaults" → RAC
- Fodder #2: "distance" → hyponym of "distance" = MILE → MILE
- "during" is the container/insertion indicator → RAC inside MILE = M-RAC-ILE = MIRACLE? No: MILE contains RAC → MI(RAC)LE = MIRACLE ✓
- Definition: "event, defying belief"

**Key insight:** "somersaults" is a reversal indicator (invoking movement/inversion). "during" is an insertion/container indicator (invoking placement/containment). The surface reading ("car somersaults during a distance event") is entirely misleading.

---

## The Eight Wordplay Types in Detail

### Operations on Fodder (Fodder → Chunk)

**HIDDEN**
The answer appears as consecutive letters within the fodder, spanning one or more words.
Indicator concepts: concealment (buried, hidden, contains), revelation (found in, displays), segment (part of, piece of, sample of)
Example: "Some *SCANDAL*ous behavior" → SCANDAL hidden in "scandalous"

**ALTERNATION**
Every other letter of the fodder is selected.
Indicator concepts: regular intervals (regularly, oddly, evenly, intermittently), alternating pattern (alternating, even letters, odd letters)
Example: "EvErY oThEr" → EYOR? (alternation indicators signal the pattern)

**DELETION (as selection modifier)**
Takes the complement of a selection — the letters that a selector would *not* pick. Acts as "NOT" on a selection indicator.
Example: "headless" = delete the first letter; "gutted" = delete middle letters; "curtailed" = delete last letter

**HOMOPHONE**
The answer sounds like the fodder word(s) when spoken aloud.
Indicator concepts: hearing (I hear, sounds like, reportedly), speaking (said, declared, broadcast), audibility (vocal, aloud, audibly, outspoken)
Example: "reportedly brave" → the answer sounds like "brave" = BRAVED? No — sounds like = BRAID? Context determines which homophone applies.

### Operations on a Chunk (Chunk → New Chunk)

**ANAGRAM**
All letters of the fodder/chunk are rearranged.
Indicator concepts: disorder/incorrectness (jumbled, broken, wrong, confused, mixed up), movement (mixing, stirring, dancing), cooking/making (cooked, designed, engineered), damage (smashed, upset, damaged)
Note: Anagram has by far the most diverse and numerous indicator vocabulary in the dataset.

**REVERSAL**
Letters of the chunk are reversed. This is technically a subtype of anagram (a specific rearrangement).
Indicator concepts: backward direction (backwards, back, reversed, returned), reflection (reflected, mirrored, flipped), upward movement for down clues (rising, upwards, lifted, raised, skyward), geographic direction (west, left, north)
Note: Some data labeled as "anagram" in the Ho dataset may have been intended as "reversal" by the puzzle creator.

### Combination Operations (Chunk + Chunk → Chunk)

**CONTAINER / INSERTION**
These are inverse operations describing the same physical relationship from different perspectives.
- INSERTION(C1, C2) = put C1 *inside* C2
- CONTAINER(C1, C2) = put C2 *inside* C1

Because they describe the same spatial relationship from opposite perspectives, they share many indicator phrases. The same word (e.g., "about", "in", "around") frequently appears as both container and insertion indicators.

Indicator concepts: containment (inside, within, in, containing, holding, caged), surrounding (about, around, outside of), consumption/absorption (eat, consume, absorb, devour), acquisition (capturing, catching, getting into)

**DELETION (as subtraction)**
Removes one chunk from within another chunk. Acts as arithmetic subtraction on two letter sequences.
Indicator concepts: removal (removing, excluding, dropping, cutting), absence (absent, missing, without), reduction (less, short, curtailed), leaking (leaking out of)

Note: Deletion plays TWO distinct roles (selection modifier and subtraction operator). These may not cluster together because they invoke different conceptual metaphors.

---

## The Conceptual Hierarchy: How Many Clusters?

This is one of the key open questions, but the theoretical framework is well-developed. There are multiple valid levels of granularity:

### Level 1: Highest Abstraction (k=3)
Indicators cluster according to what they operate on:
- **Fodder → Chunk** (hidden, alternation, homophone, deletion-as-selector)
- **Chunk → Chunk** (anagram, reversal)
- **Chunk + Chunk → Chunk** (container, insertion, deletion-as-subtraction)

### Level 2: Operation Type (k=4)
- Selecting and Extracting Letters (hidden, alternation, deletion-as-selector)
- Substituting Words (homophone)
- Rearranging Letters (anagram, reversal)
- Combining Chunks (container, insertion, deletion-as-subtraction)

Note: The placement of deletion at k=4 is unresolved — it spans both "extracting" and "combining" categories.

### Level 3: Wordplay Type as CCC Solvers Know Them (k=8 or k=14)
- k=8: The 8 types labeled in the Ho dataset (counting deletion once)
- k=14: All distinct wordplay types including Minute Cryptic types (alternation, first/last/inner/border selectors, etc.), counting deletion twice

### Level 4: Conceptual Metaphor (k=?)
At this level, the dominant concept of **placement** overrides the extraction vs. combination distinction. Indicators for HIDDEN, BORDERS, INNERS, MIDDLES, INSERTION, and CONTAINER may all cluster together because they invoke spatial placement metaphors regardless of the direction of the operation.

Additionally, some wordplay types invoke **opposing conceptual poles**: HIDDEN indicators include both "concealment" words (buried, hiding, concealing) and "revelation" words (found in, displays, exhibits). These opposing concepts within a single wordplay type suggest that k > 8 at this level.

---

## Implications for Clustering

- **There is no single "correct" number of clusters.** The right k depends on which level of the hierarchy you are targeting, and this is a research question, not a preprocessing parameter.
- **Indicators can legitimately belong to multiple wordplay types.** The word "about" is a valid indicator for container, reversal, and anagram. Hard cluster assignment may not capture this.
- **The most distinct wordplay types** (likely to separate well): anagram, reversal, homophone
- **The most entangled types** (likely to overlap): container/insertion, hidden/insertion/container (placement metaphor), deletion (dual role)
- **The hardest separation problem**: alternation, which is very small and conceptually narrow

---

## Seed Words by Clustering Scheme

Seed words are indicator examples compiled from expert sources (Minute Cryptic book and Cryptic Crosswords for Dummies) plus Victoria's conceptual metaphor analysis. There are five seed sets corresponding to different clustering philosophies. These are available in Wordplay_Seeds.xlsx.

### cc_for_dummies_ho_6: Seeds for 6 Ho Types (from Cryptic Crosswords for Dummies)
Maps to: anagram, container, hidden, reversal, deletion, homophone
(Insertion and alternation are not covered by this source)

| Type | Example seeds |
|------|--------------|
| anagram | jumbled up, broken, damaged, cooked, confused, upset, edited, ugly, insane, invented, engineered |
| container | around, inside, acquiring, keeping, possessing, devouring, hugging, amidst, occupying, getting into, set in |
| hidden | found, found inside, inside, a bit of, buried in, essentially, fragment, held in, part of, sample of |
| reversal | backwards, around, backslide, brought about, come back, flipped over, going west, knocked over, reflected, rising, held up, lifted, skyward, up, upwards |
| deletion | remove, absent, excluding, losing, not, dropped, cut, without, short |
| homophone | listened to, said aloud, on the air, broadcast, I hear, said, declared, audibly, outspoken, reportedly, sounds like, vocal |

### cc_for_dummies_ALL: Extended seeds including positioning/charade words
Adds: deletion_positioning (first, head, opener, tail, end, conclusion, half, middle, centre), charade_positioning (following, added to, leading from, and, but, in, which, makes, provides, next to, joining, on, with)

### minute_cryptic_ho_7: Seeds for 7 Ho Types (from Minute Cryptic book)
Includes alternation seeds, and breaks hidden/container categories into subcategories (hiding vs. revealing; within vs. around vs. active containment). More granular than the Dummies source.

Key addition — **alternation seeds**: oddly, evenly, occasionally, intermittently, regularly

### minute_cryptic_ALL: Most granular seed set (26 columns)
Breaks down each wordplay type into subcategories based on conceptual metaphor. For example, hidden is split into: hiding (conceals, secretly, hides), revealing (displays, exhibit in, found), contained (trapping, caged, inside), segment (piece of, sector, sample, members of, sequence, some).

### conceptual_groups: Seeds organized by conceptual metaphor (not by wordplay type)
This is the most theoretically motivated seed set. Each row contains: the conceptual category, the seed word/phrase, and the list of wordplay types that indicator belongs to.

Key insight from this tab: Many conceptual categories (containing, hiding, segment, surrounding, consuming, holding, revealing) are associated with MULTIPLE wordplay types — container, hidden, AND insertion all share these conceptual groups. This is the empirical basis for the hypothesis that these three types will not cluster separately.

**What makes a good seed word (per KCT, Feb 8 meeting):**
1. Common / high frequency (skew toward words that appear frequently in the data)
2. Explicit and conceptually aligned with the wordplay type
3. Tricky / unintuitive / easy to miss (drawn from CCC tutorials — these are the words novice solvers overlook)

Seeds do not need to be unique across wordplay types. A word like "about" can be a valid seed for multiple types, because the goal is to anchor the semantic space near known examples, not to create exclusive membership.
