# DATA_RAW.md — Raw Data Schema and Puzzle Metadata Extraction

This document describes the raw input data for the clue misdirection pipeline,
the data architecture connecting all derived files, and the extraction logic
implemented in `puzzle_metadata.ipynb`. It is intended as a reference for
Claude Code when building or editing any notebook or script in this project.

---

## 1. Source File

**File:** `../data/clues_raw.csv`
**Origin:** Extracted from George Ho's CCC dataset (`data.sqlite3`) by
`indicator_clustering/NB00`. Do not go back to the sqlite directly.
**Size:** 660,613 rows
**Loading:** `clue_id` is kept as a regular column, not set as the pandas
index. It is unique in this file and serves as the primary key.

---

## 2. Columns in `clues_raw.csv`

| Column | Type | Description |
|--------|------|-------------|
| `clue_id` | int | Unique row identifier; primary key of `clues_raw.csv`; kept as a column, not used as pandas index |
| `clue` | str | Full clue text including answer format in parentheses, e.g. `"Plant in a garden party (5)"` |
| `answer` | str | The crossword answer, uppercase, e.g. `"PARTY"` |
| `definition` | str | The definition substring within the clue. For double-definition clues, contains `/`-separated alternatives. Frequently NaN for some sources (see §8). |
| `clue_number` | str | Clue position in the grid, e.g. `"23a"`, `"18d"`. Parseable into `clue_no` (int) and `clue_direction` (`"across"` or `"down"`). Currently not used in the pipeline. |
| `puzzle_date` | str | Publication date of the puzzle. Present for most sources; NaN for `thebrowser`. |
| `puzzle_name` | str | Blog entry title or puzzle name. Format varies significantly by source (see §6). |
| `source_url` | str | URL or file path of the source blog entry or puzzle file. Used as the puzzle grouping key by `assign_ids.py`. No null values. |
| `source` | str | Source identifier. One of 10 values (see §3). Used directly as the `blog` column in `puzzle_metadata.csv`. |

---

## 3. Sources

Ten distinct values of `source`, with row counts and puzzle counts:

| source | rows | puzzles | notes |
|--------|------|---------|-------|
| `bigdave44` | 232,884 | 7,942 | UK cryptic blog; Daily/Sunday Telegraph + community series |
| `fifteensquared` | 225,725 | 7,730 | UK cryptic blog; Guardian, Independent, FT, Observer, etc. |
| `times_xwd_times` | 101,240 | 3,484 | UK cryptic blog; The Times and Sunday Times stable |
| `thehinducrosswordcorner` | 71,719 | 2,484 | Indian cryptic blog; The Hindu newspaper |
| `natpostcryptic` | 11,014 | 357 | Canadian cryptic blog; National Post |
| `cru_cryptics` | 7,287 | 240 | Community forum; **no `definition` parsed** |
| `nytimes` | 4,687 | 152 | New York Times; **no `definition` parsed** |
| `newyorker` | 3,242 | 148 | The New Yorker |
| `thebrowser` | 2,601 | 84 | US cryptic publication (weekly, 2021–2025) |
| `leoedit` | 214 | 7 | Amuselabs Leoedit; **no `definition` parsed** |

**Total:** 660,613 clues across 22,628 puzzles (~29–31 clues per puzzle).

Sources with no `definition`: `cru_cryptics`, `nytimes`, `leoedit`. These are
excluded from `structural_filtering.ipynb`. Verify with:
```python
assert df[df['source'].isin(['cru_cryptics', 'nytimes', 'leoedit'])]['definition'].isna().all()
```

---

## 4. Data Architecture

### 4.1 Files and primary keys

| file | primary key | unique? | assigned in | notes |
|------|-------------|---------|-------------|-------|
| `clues_raw.csv` | `clue_id` | ✅ | George Ho (pre-existing) | one row per clue |
| `clues_filtered.csv` | `row_id` | ✅ | `structural_filtering.ipynb` (last step before write) | one row per (clue, definition) pair; `clue_id` retained as FK |
| `id_map.csv` | `clue_id` | ✅ | `assign_ids.py` | maps `clue_id` → `puzzle_id`; generated file, in `.gitignore` |
| `puzzle_metadata.csv` | `puzzle_id` | ✅ | `puzzle_metadata.ipynb` | one row per puzzle |
| `clue_metadata.csv` | `clue_id` | ✅ | future notebook | one row per original clue |

### 4.2 Why `clue_id` is not unique in `clues_filtered.csv`

Double-definition clues have a `definition` field containing `/`-separated
alternatives. `structural_filtering.ipynb` expands these into multiple rows —
one per valid definition — so the same `clue_id` can appear multiple times.
The composite `(clue_id, definition)` is unique. `row_id` is assigned as a
simple sequential integer to give each expanded row a single unambiguous
primary key.

### 4.3 Join patterns

- **Puzzle metadata → clues_filtered:** join `clues_filtered.csv` on `clue_id`
  to `id_map.csv` to get `puzzle_id`, then join `puzzle_metadata.csv` on
  `puzzle_id`. Puzzle metadata will repeat across double-definition rows —
  this is correct and expected.
- **Clue metadata → clues_filtered:** join directly on `clue_id`. Clue
  metadata will also repeat across double-definition rows — also correct.

### 4.4 Independence requirement

No notebook at the root level may depend on another notebook having run first.
`assign_ids.py` is a script (not a notebook) and must be run before
`puzzle_metadata.ipynb`. This dependency is documented in `README.md`.
`structural_filtering.ipynb` and `puzzle_metadata.ipynb` are fully independent
of each other.

---

## 5. `assign_ids.py`

**Location:** `notebooks/assign_ids.py`
**Run:** `cd notebooks && python assign_ids.py`
**Input:** `../data/clues_raw.csv` — loads `clue_id`, `source`, `puzzle_name`,
`source_url`.
**Output:** `../data/id_map.csv` — columns `clue_id` (int), `puzzle_id` (int);
sorted by `(puzzle_id, clue_id)`; `index=False`.

**How `puzzle_id` is assigned:**

Puzzles are identified by grouping clues on `source_url` after normalization:

1. **Trailing slash strip:** `str.rstrip("/")` applied to all sources.
   Collapses one known `fifteensquared` anomaly identified in investigation.
2. **`bigdave44` numeric-ID normalization:** Some puzzles have two URLs — a
   human-readable slug (e.g. `.../toughie-2766/`) and a numeric WordPress post
   ID (e.g. `.../141686/`). Numeric-ID URLs are detected via `r"/\d+/$"`. For
   any `puzzle_name` that maps to both forms, the numeric-ID URL is replaced by
   the slug URL. Numeric-ID URLs with no slug sibling are kept as-is. 32
   replacements were made in the current dataset.
3. **`puzzle_id` assignment:** Unique normalized `source_url` values are sorted
   by `(source, normalized_source_url)` and assigned sequential integer IDs via
   `pd.factorize`. Sort order ensures stability across reruns.

**Validation:** Asserts that every `clue_id` from the input appears exactly
once in the output and that no `puzzle_id` is null.

---

## 6. `puzzle_metadata.csv` Schema

Produced by `puzzle_metadata.ipynb`. One row per puzzle.

| column | source |
|--------|--------|
| `puzzle_id` | from `id_map.csv` |
| `blog` | directly from `source` column in `clues_raw.csv` |
| `publisher` | hardcoded or extracted via `publisher_lookup.csv` |
| `series` | extracted via `publisher_lookup.csv` |
| `setter` | hardcoded, extracted via lookup, or from `source_url`; pipe-separated for collaborations |
| `puzzle_date` | from `puzzle_date` column or extracted from `puzzle_name`/`source_url` |

`puzzle_no` is not included — `puzzle_id` serves as the unique identifier.
`clue_no` and `clue_direction` are clue-level metadata and belong in the
future `clue_metadata.csv`, not here.

### 6.1 Hardcoded publisher assignments

| source | publisher | setter | notes |
|--------|-----------|--------|-------|
| `natpostcryptic` | `"National Post"` | `"Hex"` | Hex = pseudonym of Emily Cox & Henry Rathvon |
| `cru_cryptics` | `"Cru Cryptics Forum"` | — | |
| `nytimes` | `"New York Times"` | — | |
| `leoedit` | `"Amuselabs Leoedit"` | — | |
| `thebrowser` | `"The Browser"` | from `source_url` | setters use real surnames |
| `newyorker` | `"The New Yorker"` | — | |
| `thehinducrosswordcorner` | `"The Hindu"` | from `puzzle_name` | |

### 6.2 Extraction fields

The lookup table `data/publisher_lookup.csv` maps extracted substrings to
canonical `publisher`, `series`, and `setter` values via the `field` column:

| field value | extraction location |
|-------------|---------------------|
| `puzzle_name_leading` | Text before the first run of digits in `puzzle_name` |
| `puzzle_name_trailing` | Text after the puzzle number (and optional date) in `puzzle_name` |
| `source_url_trailing` | Last word before the file extension in `source_url` |

Which sources use which fields:

| source | puzzle_name_leading | puzzle_name_trailing | source_url_trailing |
|--------|--------------------|--------------------|-------------------|
| `bigdave44` | ✅ publisher + series | | |
| `fifteensquared` | ✅ publisher + series | ✅ setter | |
| `times_xwd_times` | ✅ series | | |
| `thehinducrosswordcorner` | ✅ series prefix | ✅ setter | |
| `thebrowser` | | | ✅ setter |

### 6.3 `puzzle_name` formats by source

#### `bigdave44`
```
[Publisher/Series] [Number]
[Publisher/Series] – [Number]
```
Examples: `"Daily Telegraph 27164"`, `"NTSPP – 488"`, `"Toughie 2766"`

- Leading token (normalized lowercase, "Cryptic" stripped) → lookup in CSV
- Number: 3–5 digits, optionally preceded by `–` or `No`
- No setter present in `puzzle_name`
- `puzzle_date` already present; no date extraction needed

#### `fifteensquared`
```
[Publisher] [Number] by [Setter]
[Publisher] [Number] [Setter]
[Series] [Number]
```
Examples: `"Guardian Cryptic 27509 by Qaos"`, `"Financial Times 14450 Neo"`, `"Azed 2446"`

- Leading token → allowlist lookup (publisher/series); strip `"Cryptic"` noise before lookup
- Trailing token (after number, optional `by`) → setter lookup
- Collaborative puzzles stored as pipe-separated list: `"Enigmatist|Soup"`
- Azed special types (Misprints, Playfair, Overlaps, Across) → `setter = "Azed"`, `series = [type name]`
- Sloggers & Betters entries: `publisher = "Sloggers & Betters"`, city extracted as series, setter(s) after `"by"` keyword
- `puzzle_date` already present; no date extraction needed
- Known missing lookup entries: `"quiptic"` → Guardian/Quiptic; `"guardian prize puzzle"` → Guardian/Prize; `"guardian n"` → Guardian/—; `"times quick crossword"` → The Times/Quick Cryptic
- Known trailing pollution handled in extraction regex: `"with picture quiz"` → strip; `"plain competition puzzle"` → `setter = "Azed"`, `series = "Plain"`; bare number suffix (e.g. `"2"`) → strip (blog disambiguation suffix, not a different setter or series)

#### `times_xwd_times`
```
[Series] [Number]
[Series] No [Number]
[Series] [Number,]
[Series] [Number with comma]
```
Examples: `"Times Cryptic No 27948"`, `"QC 1425"`, `"Times 26,495"`, `"Jumbo 1433"`

- Publisher hardcoded: `"The Times"` for Times series, `"Sunday Times"` for Sunday series, `"Times Literary Supplement"` for TLS
- Leading token (normalized) → series name via lookup
- Strip `"Cryptic"` and `"No"` noise; strip commas from puzzle number
- Significant malformed data: month-only tokens, puzzle titles leaking in, blank entries (185 rows) — map to NaN and log
- `puzzle_date` already present

#### `thehinducrosswordcorner`
```
Format 1 (main series):
No NNNNN, Weekday DD Mon YYYY, Setter
No NNNNN, Weekday DD Mon YYYY

Format 2 (Sunday Crossword):
The Sunday Crossword No NNNNN, Weekday DD Mon YYYY
The Sunday Crossword (NNNNN), Weekday DD Mon YYYY
```
Examples: `"No 11542, Wednesday 04 Nov 2015, Gridman"`, `"The Sunday Crossword No 2957, Sunday 20 Aug 2017"`

- Publisher hardcoded: `"The Hindu"`
- Leading prefix (before number): empty → main series; `"The Sunday Crossword"` (and misspellings) → series lookup
- Setter: trailing field after the date comma; may contain `.` (e.g. `"Dr. X"`); absent for Sunday Crossword entries
- Setter normalization lookup handles misspellings and punctuation variants
- `"Anon"` → `setter = NaN` (pre-2008 anonymous puzzles)
- `puzzle_date` already present

#### `natpostcryptic`
```
[Day of week], [Month] [D], [Year] — [Puzzle Title]
```
Example: `"Saturday, December 27, 2014 — Holiday Film Fest"`

- Publisher hardcoded: `"National Post"`; setter hardcoded: `"Hex"`
- No publisher, setter, or puzzle number extractable from `puzzle_name`
- `puzzle_date` already present

#### `cru_cryptics`
```
source_url: cru-cryptics/CrypticNNN.puz
```
- Publisher hardcoded: `"Cru Cryptics Forum"`
- `puzzle_name` is uninformative (variations of `"Cru Cryptic"` with redacted content)
- No setter available

#### `nytimes`
```
source_url: nytimes/NY Times Variety - YYYYMMDD - Cryptic Crossword.puz
```
- Publisher hardcoded: `"New York Times"`
- Date extractable from `source_url` via `r"\d{8}"`, parsed as `%Y%m%d`
- No puzzle number or setter available

#### `thebrowser`
```
source_url:   thebrowser/Cryptic NN - MonDDYY SetterSurname.puz
puzzle_name:  CRYPTIC #13 (March 27, 2021)
```
- Publisher hardcoded: `"The Browser"`
- Puzzle number: extracted from `puzzle_name` via `r"#(\d+)"`
- Date: extracted from `puzzle_name` parenthesized date; verify against `source_url` compressed date (`%b%d%y`)
- Setter: extracted from `source_url` (last word before `.puz`); lookup handles collaborative entries (`"Jacobs Goodchild"` → `"Jacobs|Goodchild"`) and stray dashes (`"- Ries"` → `"Ries"`)
- Setters use real surnames, not pseudonyms (North American convention)
- `puzzle_date` mostly NaN in raw data — use extracted date

#### `newyorker`
```
puzzle_name:  "Cryptic Crossword No. 94"  or  "The Cryptic Crossword: Sunday, January 30, 2022"
source_url:   https://www.newyorker.com/.../cryptic-crossword/no-21  or  .../2022/01/30
```
- Publisher hardcoded: `"The New Yorker"`
- Puzzle number: extract from `puzzle_name` via `r"No\.\s*(\d+)"` or from `source_url` via `r"no-(\d+)"`; NaN for date-only entries
- No setter available
- `puzzle_date` already present

#### `leoedit`
- Publisher hardcoded: `"Amuselabs Leoedit"`
- Nothing extractable from `puzzle_name` or `source_url`
- `puzzle_date` already present

---

## 7. Lookup Table

**File:** `data/publisher_lookup.csv` (committed to repo)
**Columns:** `source`, `field`, `raw`, `publisher`, `series`, `setter`, `confidence`, `notes`
**Rows:** 353
**Key:** `(source, field, raw)` — `raw` is the normalized (lowercased, stripped) extracted substring

The `raw` column uses `QUOTE_MINIMAL` quoting — only fields containing commas
are quoted (e.g. `"dr. x,"`, `"dr, x,"`, `"spinner,"`).

The `setter` column uses pipe-separated values for collaborative puzzles.
The `confidence` column takes values `High`, `Medium`, or `Low`.

When loading, apply `.drop_duplicates(subset=["source", "field", "raw"])`
defensively before joining. A known duplicate exists for
`(thehinducrosswordcorner, puzzle_name_trailing, kriskross)` and should be
removed from the CSV directly as a follow-up.

Any extracted substring not found in the lookup table is assigned NaN for
all derived fields and logged to the unmatched-tokens summary at the end of
`puzzle_metadata.ipynb`.

---

## 8. Edge Cases and Surprises

### `definition` null rates
- `cru_cryptics`, `nytimes`, `leoedit`: definition is NaN for essentially all rows — excluded from `structural_filtering.ipynb`.
- Other sources: `definition` may be NaN for individual rows; `structural_filtering.ipynb` drops these.

### `puzzle_date` mostly NaN for `thebrowser`
- Extract date from `puzzle_name` parenthesized format instead.
- Cross-verify against `source_url` compressed date (`"Mar2721"` → `%b%d%y`).

### `times_xwd_times` `puzzle_name` noise
- Month-only tokens, puzzle titles leaking in, single-letter entries, blank entries (185 rows).
- Stray trailing commas on puzzle numbers; commas embedded in puzzle numbers.
- All unresolvable tokens → NaN; logged in notebook summary cell.

### `fifteensquared` setter extraction complexity
- ~199 plausible setters after regex refinement.
- Month names bleed through as fake setter names from S&B date-labelled entries; excluded via blocklist.
- Number suffixes (e.g. `"Armonie 2"`) are blog disambiguation suffixes for same-day same-setter puzzles — strip to base name.
- `"with picture quiz"` is a blog post feature label, not setter information — strip.
- `"plain competition puzzle"` → `setter = "Azed"`, `series = "Plain"`.
- `"Guardian"` as trailing substring → publisher leaked into setter field; NaN.
- `"Anon"` → NaN.

### `thehinducrosswordcorner` setter normalization
- Pre-2008 puzzles anonymous; `"Anon"` → NaN.
- `"Dr. X"` variants: `"Dr. X,"`, `"Dr X"`, `"Dr, X,"` — all normalize to `"Dr. X"`.
- `"Phantom"` → `"The Phantom"`.
- `"Skulldugger"` / `"Skuldugger"` → `"Skulldugger"`.
- `"Spinner,"` → `"Spinner"`.
- `"KrisKross"` / `"Kriskross"` → `"KrisKross"`.

### `source_url` quality (from `source_url_investigation.ipynb`)
- No null values across any source.
- `bigdave44` uses `http` (not `https`); all other URL-based sources use `https`.
- `cru_cryptics`, `nytimes`, `thebrowser` use file paths, not URLs; uppercase present in paths — consistent within source.
- One `fifteensquared` row missing trailing slash — handled by `str.rstrip("/")` in `assign_ids.py`.
- 18 `source_url` → `puzzle_name` collisions are spurious (missing space before setter name in `puzzle_name`); `source_url` is correct.
- Some `bigdave44` puzzles have two URLs (slug + numeric WordPress ID); 32 resolved by `assign_ids.py`; puzzles with two numeric-ID URLs and no slug cannot be resolved and are treated as separate puzzles.

### `natpostcryptic` syndication
- Weekday puzzles may have originally appeared in the Daily Telegraph UK. Publisher hardcoded as `"National Post"` for all rows without distinguishing syndicated vs. original content.

### Setter field is pipe-separated
- Stored as a pipe-separated string (e.g. `"Enigmatist|Soup"`) for collaborative puzzles. Parse with `.str.split("|")` downstream.

### `puzzle_name` setter name embedded in leading token
- `fifteensquared` examples: `"Guardian Prize Picaroon"`, `"Independent Punk"`.
- Handled by dedicated rows in `publisher_lookup.csv`, not by generic trailing-token extraction.
