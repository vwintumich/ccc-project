# Spec: Wordplay Metadata

**Location:** `ccc-project/notebooks/wordplay_metadata.ipynb` (root-level shared notebook)
**Output:** `ccc-project/data/wordplay_metadata.csv`
**Date:** 2026-04-19
**Status:** Approved

## Purpose

Produce a shared metadata file identifying algorithmically verifiable wordplay
types for each clue, analogous to `puzzle_metadata.csv`. The output is one row
per `clue_id` with boolean columns indicating which wordplay types are
consistent with the clue's surface text and answer. This file is joinable to
any downstream file on `clue_id`.

**Motivation:** The advisor wants to investigate whether g1 performs differently
on different types of clues. This file enables that analysis by providing
wordplay type labels that can be joined to the custom_embedding_model
evaluation data.

**Important framing:** These are *algorithmic verifications*, not
classifications. A `True` value means the structural pattern is present — it
does not guarantee that the setter intended that wordplay type. A clue may
match multiple types (e.g., an answer that happens to be both an anagram of a
surface word and hidden across a word boundary). A `False` value means the
pattern was not detected, not that the clue definitely uses a different
mechanism.

## Inputs

- `ccc-project/data/clues_filtered.csv` — columns: `clue_id`, `surface`,
  `definition`, `answer`

No other inputs required. All checks are purely algorithmic using the surface
text and answer.

## Outputs

- `ccc-project/data/wordplay_metadata.csv` — one row per unique `clue_id`,
  with boolean columns for each wordplay type (schema below)
- `ccc-project/notebooks/outputs/wordplay_metadata-results.md` — coverage
  statistics, type frequencies, co-occurrence counts

## Output Schema

| Column | Type | Description |
|--------|------|-------------|
| `clue_id` | int | Unique clue identifier; join key to `clues_filtered.csv` |
| `anagram_single_word` | bool | Answer is an anagram (not reversed) of exactly one word in the surface |
| `anagram_consec_words` | bool | Answer is an anagram (not reversed) of 2+ consecutive intact words in the surface |
| `hidden_fwd` | bool | Answer appears as a forward substring in the concatenated surface, spanning at least one word boundary |
| `hidden_rev` | bool | Answer reversed appears as a substring in the concatenated surface, spanning at least one word boundary |
| `selection_alt` | bool | Answer is spelled by alternating letters of some span in the surface |
| `selection_alt_rev` | bool | Answer reversed is spelled by alternating letters of some span in the surface |
| `selection_firsts` | bool | Answer is spelled by the first letters of consecutive surface words |
| `selection_firsts_rev` | bool | Answer reversed is spelled by the first letters of consecutive surface words |
| `selection_lasts` | bool | Answer is spelled by the last letters of consecutive surface words |
| `selection_lasts_rev` | bool | Answer reversed is spelled by the last letters of consecutive surface words |
| `double_def` | bool | `clue_id` appears more than once in `clues_filtered.csv` (multiple valid definitions) |

All boolean columns default to `False`. Multiple columns may be `True` for the
same clue.

## Implementation Details

### §0 — Setup and environment detection

Standard imports, pathlib paths, environment auto-detection, version reporting.

### §1 — Load and deduplicate

1. Load `clues_filtered.csv` with `keep_default_na=False, na_values=[""]`.
2. Count rows per `clue_id` to identify double-definition clues (count > 1).
   Store as a Series or dict: `double_def_flags`.
3. Select columns `clue_id`, `surface`, `answer`. Drop duplicates on all three.
4. Assert `clue_id` is now unique.
5. Report: total rows in clues_filtered, unique clue_ids, number of
   double-def clue_ids.

### §2 — Answer normalization

For all wordplay checks, normalize the answer:
- Lowercase
- Strip spaces and hyphens (multi-word and hyphenated answers become a single
  letter sequence)

Store as a new column `answer_norm`.

**Minimum length filter:** All wordplay checks (§3–§6) apply only to clues
where `len(answer_norm) >= 4`. Clues with shorter answers receive `False` for
all wordplay type columns (they may still be `True` for `double_def`, which
has no length requirement). Report how many clues are skipped by this filter.

Also normalize the surface for each check type (details below). Comparisons
are always case-insensitive.

### §3 — Anagram checks

**Surface words:** Tokenize the surface by splitting on whitespace. Lowercase
each token. Strip any non-alphabetic characters from each token (punctuation
at word boundaries). Drop empty tokens.

**Reversal test:** A sequence of letters is a "simple reversal" of the answer
if it equals `answer_norm[::-1]`. This must be excluded from all anagram
results.

**`anagram_single_word`:** For each surface word token, check if
`sorted(token) == sorted(answer_norm)` AND `token != answer_norm` (not
identical) AND `token != answer_norm[::-1]` (not a simple reversal).
True if any single token passes.

**`anagram_consec_words`:** For each consecutive subsequence of 2+ surface
word tokens, concatenate their letters and check if
`sorted(concat) == sorted(answer_norm)` AND `concat != answer_norm[::-1]`.
True if any consecutive subsequence passes. Limit subsequence length to
`len(answer_norm)` characters max to avoid unnecessary computation (once the
concatenated letters exceed the answer length, stop extending).

**Performance note:** The consecutive-subsequence check is O(n²) in the number
of surface words per clue, but surfaces are short (typically 5–15 words), so
this is fast. No need for special optimization.

### §4 — Hidden word checks

**Surface concatenation:** Lowercase the surface, split on whitespace, strip
non-alphabetic characters from each token, then concatenate all tokens into a
single letter string. Call this `surface_concat`. Also record the lengths of
each token to determine word boundary positions.

**`hidden_fwd`:** Search for `answer_norm` as a substring of `surface_concat`.
A match is valid if it is NOT an exact standalone surface word — i.e., the
matched span does not cover exactly one complete token's character range from
start to end. Matches within a single longer word (e.g., "plant" in
"supplanted") are valid. Matches spanning word boundaries are valid. Only
matches where the answer IS a complete word are rejected. True if any valid
match exists.

**`hidden_rev`:** Search for `answer_norm[::-1]` as a substring of
`surface_concat`. Any match is valid — no exclusions, even if the reversed
answer happens to be an exact standalone surface word. True if any match
exists.

### §5 — Selection checks (alternating letters)

**Surface letter sequence:** Same `surface_concat` as §4 (lowercase, alpha
only, no spaces).

**`selection_alt`:** For each starting position `i` in `surface_concat`, take
every other character: `surface_concat[i::2]`. If `answer_norm` appears as a
substring of this alternating-letter sequence, the check passes. Also try
offset starts: `surface_concat[i+1::2]`. More precisely: for each start
position `s` and step 2, extract `len(answer_norm)` characters by taking
`surface_concat[s], surface_concat[s+2], surface_concat[s+4], ...`. True if
any such extraction equals `answer_norm`.

Actually, the cleaner formulation: for each starting index `s` in
`surface_concat` where `s + 2*(len(answer_norm)-1) < len(surface_concat)`,
extract the subsequence `[surface_concat[s + 2*k] for k in range(len(answer_norm))]`.
True if this subsequence equals `answer_norm` for any `s`.

**`selection_alt_rev`:** Same check but compare against `answer_norm[::-1]`.

### §6 — Selection checks (first/last letters)

**Surface words:** Same tokenization as §3 — split on whitespace, lowercase,
strip non-alpha, drop empties.

**`selection_firsts`:** Take the first letter of each surface word to form a
string `firsts`. Check if `answer_norm` appears as a substring of `firsts`.
(The answer may be spelled by a contiguous run of words, not necessarily all
words.) True if found.

**`selection_firsts_rev`:** Check if `answer_norm[::-1]` appears as a
substring of `firsts`.

**`selection_lasts`:** Same but using the last letter of each word.

**`selection_lasts_rev`:** Check if `answer_norm[::-1]` appears as a
substring of `lasts`.

### §7 — Double definition

Assign from `double_def_flags` computed in §1. No additional logic needed.

### §8 — Assemble and save

1. Combine all boolean columns into a single DataFrame keyed on `clue_id`.
2. Assert `clue_id` is unique and no nulls exist in any column.
3. Save to `data/wordplay_metadata.csv` with `index=False`.
4. Report total clue count and type frequencies.

### §9 — Summary statistics and results file

Write `outputs/wordplay_metadata-results.md` containing:

- Total clues checked
- Frequency of each wordplay type (count and % of total)
- Number and % of clues matching at least one type
- Number and % of clues matching zero types
- Co-occurrence matrix: for each pair of types, count how many clues have
  both True (as a table)
- Top 5 most common type combinations (e.g., "anagram + anagram_single_word")

Display a few example clues (surface + answer) for each wordplay type as a
sanity check — pick 3 random examples per type using `random_state=42`.

## Logical relationships between types

The implementation must enforce these assertions before saving:
- `anagram_single_word` and `anagram_consec_words` are NOT mutually exclusive.
  A clue can match both if the answer is an anagram of one surface word AND
  also an anagram of a different group of consecutive words (e.g., a short
  word like "a" adjacent to the fodder word). Observed in 6 rows.
- The `_rev` selection variants are not mutually exclusive with their forward
  counterparts (a palindromic answer would match both). This is expected and
  fine — do not assert exclusivity.
- All wordplay type columns (except `double_def`) must be `False` for clues
  where `len(answer_norm) < 4`. Assert this.

## Environment

Local (CPU). No GPU required. Expected runtime: a few minutes for ~500K clues
(all checks are string operations, no embeddings involved).

## Notebook structure

- Use §-numbered markdown sections before each logical block
- Include environment auto-detection for local/Great Lakes/Colab compatibility
- Standard notebook header (primary author: Victoria; builds on:
  structural_filtering.ipynb)
- Create `notebooks/outputs/` directory if it does not exist (use `mkdir -p`
  or `Path.mkdir(parents=True, exist_ok=True)`)
- Write results file to `notebooks/outputs/wordplay_metadata-results.md`
- Standard summary cell at the end

## Post-implementation documentation

After the notebook is complete and committed, update:
- `ccc-project/WORKFLOW.md` — add a section for `wordplay_metadata.ipynb`
- `ccc-project/DATA_RAW.md` — add schema for `wordplay_metadata.csv` and
  join pattern documentation
