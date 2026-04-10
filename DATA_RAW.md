# DATA_RAW.md — Raw Data Schema and Puzzle Metadata Extraction

This document describes the raw input data for the CCC project and the puzzle
metadata extraction logic used across all pipeline components. It is intended
as a reference for Claude Code when building or editing data cleaning and
metadata extraction notebooks.

**Shared resource:** This file lives at the `ccc-project/` root and applies
to all components (`clue_misdirection/`, `custom_embedding_model/`, etc.).

---

## 1. Source File

**File:** `data/clues_raw.csv`
**Origin:** Extracted from George Ho's CCC dataset (`data.sqlite3`) by
`indicator_clustering/NB00`. Do not go back to the sqlite directly.
**Size:** 660,613 rows
**Index:** `clue_id` (integer, unique per row)

---

## 2. Columns in `clues_raw.csv`

These are the actual column names as loaded. The index column is `clue_id`.

| Column | Type | Description |
|--------|------|-------------|
| `clue_id` | int | Unique row identifier (used as index) |
| `clue` | str | Full clue text including answer format in parentheses, e.g. `"Plant in a garden party (5)"` |
| `answer` | str | The crossword answer, uppercase, e.g. `"PARTY"` |
| `definition` | str | The definition substring within the clue. For double-definition clues, contains `/`-separated alternatives. Frequently NaN (see §5). |
| `clue_number` | str | Clue position in the grid, e.g. `"23a"`, `"18d"`. Should be parsed into `clue_no` (int) and `clue_direction` (`"across"` or `"down"`) and passed through to `clues_filtered.csv`. |
| `puzzle_date` | str | Publication date of the puzzle. Present for most sources; NaN for `thebrowser`. |
| `puzzle_name` | str | Blog entry title or puzzle name. Format varies significantly by source (see §4). |
| `source_url` | str | URL or file path of the source blog entry or puzzle file. Format varies by source (see §4). |
| `source` | str | Source identifier. One of 10 values (see §3). |

**Column loading by notebook:**
- Structural filtering notebook (`01_structural_filtering.ipynb`) loads:
  `clue_id`, `clue`, `answer`, `definition`, `clue_number`
- Puzzle metadata notebook (`02_puzzle_metadata.ipynb`) loads:
  `clue_id`, `source`, `puzzle_name`, `source_url`, `puzzle_date`

---

## 3. Sources

Ten distinct values of `source`, with row counts:

| source | rows | notes |
|--------|------|-------|
| `bigdave44` | 232,884 | UK cryptic blog; Daily/Sunday Telegraph + community series |
| `fifteensquared` | 225,725 | UK cryptic blog; Guardian, Independent, FT, Observer, etc. |
| `times_xwd_times` | 101,240 | UK cryptic blog; The Times and Sunday Times stable |
| `thehinducrosswordcorner` | 71,719 | Indian cryptic blog; The Hindu newspaper |
| `natpostcryptic` | 11,014 | Canadian cryptic blog; National Post (some weekday puzzles syndicated from Daily Telegraph UK) |
| `cru_cryptics` | 7,287 | Community forum; **no `definition` parsed** |
| `nytimes` | 4,687 | New York Times; **no `definition` parsed** |
| `newyorker` | 3,242 | The New Yorker |
| `thebrowser` | 2,601 | US cryptic publication (weekly, 2021–2025) |
| `leoedit` | 214 | Amuselabs Leoedit; **no `definition` parsed** |

Sources with no `definition`: `cru_cryptics`, `nytimes`, `leoedit`. These are
excluded from the main filtering pipeline. Verify with:
```python
assert df[df['source'].isin(['cru_cryptics', 'nytimes', 'leoedit'])]['definition'].isna().all()
```

---

## 4. Derived Columns in `puzzle_metadata.csv`

The puzzle metadata notebook (`02_puzzle_metadata.ipynb`) produces
`data/puzzle_metadata.csv` with the following derived columns. It is a
standalone file — not joined into `clues_filtered.csv` — and can be run
independently of the structural filtering notebook.

### 4.1 Hardcoded publisher assignments

For these sources, `publisher` is assigned directly in code without any lookup:

| source | publisher | setter | notes |
|--------|-----------|--------|-------|
| `natpostcryptic` | `"National Post"` | `"Hex"` | Hex = pseudonym of Emily Cox & Henry Rathvon. Weekday puzzles may be syndicated from the Daily Telegraph UK, but this is not flagged in the output. |
| `cru_cryptics` | `"Cru Cryptics Forum"` | — | Excluded from main pipeline. |
| `nytimes` | `"New York Times"` | — | Excluded from main pipeline. |
| `leoedit` | `"Amuselabs Leoedit"` | — | Excluded from main pipeline. |
| `thebrowser` | `"The Browser"` | from `source_url` | US publication; setters use real surnames, not pseudonyms. |
| `newyorker` | `"The New Yorker"` | — | |
| `thehinducrosswordcorner` | `"The Hindu"` | from `puzzle_name` | |

### 4.2 Extraction from `puzzle_name` and `source_url`

The lookup table `data/publisher_lookup.csv` maps extracted substrings to
canonical `publisher`, `series`, and `setter` values. The `field` column
identifies where the substring was extracted from:

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
| `times_xwd_times` | ✅ series (publisher hardcoded to The Times/Sunday Times) | | |
| `thehinducrosswordcorner` | ✅ series prefix | ✅ setter | |
| `thebrowser` | | | ✅ setter |

### 4.3 `puzzle_name` formats by source

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
[Series] [Number]              ← setter-only entries (Azed, Everyman, etc.)
```
Examples: `"Guardian Cryptic 27509 by Qaos"`, `"Financial Times 14450 Neo"`, `"Azed 2446"`

- Leading token → allowlist lookup (publisher/series). Strip `"Cryptic"` noise before lookup.
- Trailing token (after number, optional `by`) → setter lookup
- Collaborative puzzles stored as pipe-separated list: `"Enigmatist|Soup"`
- Azed special types (Misprints, Playfair, Overlaps, Across) → `setter = "Azed"`, `series = [type name]`
- Sloggers & Betters entries: `publisher = "Sloggers & Betters"`, city extracted as series, setter(s) after `"by"` keyword
- `puzzle_date` already present; no date extraction needed

#### `times_xwd_times`
```
[Series] [Number]
[Series] No [Number]
[Series] [Number,]             ← trailing comma (stray); strip before parsing
[Series] [Number with comma]   ← e.g. "26,495"; strip comma from number
```
Examples: `"Times Cryptic No 27948"`, `"QC 1425"`, `"Times 26,495"`, `"Jumbo 1433"`

- Publisher hardcoded: `"The Times"` for Times series, `"Sunday Times"` for Sunday series, `"Times Literary Supplement"` for TLS
- Leading token (normalized) → series name via lookup
- Strip `"Cryptic"` and `"No"` noise; strip commas from puzzle number
- Significant malformed data: month-only tokens, puzzle titles leaking in, stray commas — map to NaN and log
- `puzzle_date` already present

#### `thehinducrosswordcorner`
Two formats:
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
- Setter normalization lookup handles misspellings and punctuation variants (e.g. `"Dr. x,"` → `"Dr. X"`)
- `"Anon"` → `setter = NaN` (pre-2008 anonymous puzzles before bylines were introduced)
- `puzzle_date` already present

#### `natpostcryptic`
```
[Day of week], [Month] [D], [Year] — [Puzzle Title]
```
Examples: `"Saturday, December 27, 2014 — Holiday Film Fest"`

- Publisher hardcoded: `"National Post"`; setter hardcoded: `"Hex"`
- No publisher, setter, or puzzle number extractable from `puzzle_name`
- `puzzle_date` already present; puzzle title not extracted (not needed)

#### `cru_cryptics`
```
cru-cryptics/CrypticNNN.puz
```
- Publisher hardcoded: `"Cru Cryptics Forum"`
- Puzzle number extracted directly from `source_url` via `r"Cryptic(\d+)\.puz"`
- No setter available

#### `nytimes`
```
nytimes/NY Times Variety - YYYYMMDD - Cryptic Crossword.puz
```
- Publisher hardcoded: `"New York Times"`
- Date extracted from `source_url` via `r"(\d{8})"`, parsed as `%Y%m%d`
- No puzzle number or setter available

#### `thebrowser`
```
thebrowser/Cryptic NN - MonDDYY SetterSurname.puz
```
Examples: `"thebrowser/Cryptic 13 - Mar2721 Ries.puz"`, `"thebrowser/Cryptic 32 - Aug0821 Zawistowski.puz"`

- Publisher hardcoded: `"The Browser"`
- Puzzle number: extracted from `puzzle_name` via `r"#(\d+)"` (format: `"CRYPTIC #13 (March 27, 2021)"`)
- Date: extracted from `puzzle_name` parenthesized date; verify against `source_url` compressed date (`%b%d%y`)
- Setter: extracted from `source_url` (last word before `.puz`); lookup table handles collaborative entries (`"Jacobs Goodchild"` → `["Jacobs", "Goodchild"]`) and stray dashes (`"- Ries"` → `"Ries"`)
- Setters use real surnames, not pseudonyms (North American convention)
- `puzzle_date` mostly NaN in raw data — use extracted date

#### `newyorker`
```
puzzle_name examples:
  "Cryptic Crossword No. 94"
  "The Cryptic Crossword: Sunday, January 30, 2022"

source_url examples:
  https://www.newyorker.com/puzzles-and-games-dept/cryptic-crossword/no-21
  https://www.newyorker.com/puzzles-and-games-dept/cryptic-crossword/2022/01/30
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

## 5. Edge Cases and Surprises

### `definition` null rates
- `cru_cryptics`, `nytimes`, `leoedit`: definition is NaN for essentially all
  rows — these sources could not be parsed for definitions and are excluded
  from the main pipeline.
- Other sources: `definition` is present but may be NaN for individual rows;
  the structural filtering notebook drops these.

### `puzzle_date` mostly NaN for `thebrowser`
- Extract date from `puzzle_name` (parenthesized format: `"March 27, 2021"`)
  instead.
- Cross-verify against `source_url` compressed date (`"Mar2721"` → `%b%d%y`).

### `times_xwd_times` has significant `puzzle_name` noise
- Month-only tokens (`"April"`, `"October"`), puzzle titles leaking in
  (`"Red Scare,"`, `"Keats And Yeats Are On Your Side,"`), single-letter
  entries (`"T"`), and blank entries (185 rows).
- Stray trailing commas on puzzle numbers (e.g. `"4835,"`); strip before
  parsing.
- Commas embedded in puzzle numbers (e.g. `"26,495"`); strip before casting
  to int.
- All unresolvable tokens → NaN; logged in notebook summary cell.

### `fifteensquared` setter extraction complexity
- 539 unique raw trailing substrings; refined regex reduces to ~199 plausible
  setters.
- Month names bleed through as fake setter names (from Sloggers & Betters
  date-labelled entries); excluded via blocklist.
- `"Enigmatist And Soup"` → collaborative puzzle; split to
  `["Enigmatist", "Soup"]`.
- Azed special types (`"Misprints"`, `"Playfair"`, `"Overlaps"`, `"Across"`)
  → not setter names; `setter = "Azed"`, `series = [type]`.
- `"Guardian"` appearing as trailing substring → publisher leaked into setter
  field; NaN.
- `"Anon"` → NaN (no setter recorded).
- S&B entries: setter name sometimes followed by date or event title (e.g.
  `"Hob Saturday Prize Puzzle 080815"`); strip everything from `"Saturday"`
  onwards.

### `thehinducrosswordcorner` setter normalization
- Pre-2008 puzzles published anonymously; `"Anon"` → NaN.
- `"Dr. X"` has multiple punctuation variants in the data: `"Dr. X,"`,
  `"Dr X"`, `"Dr, X,"` — all normalize to `"Dr. X"`.
- `"Phantom"` → `"The Phantom"` (canonical form used on the blog).
- `"Skulldugger"` / `"Skuldugger"` → `"Skulldugger"`.
- `"Spinner,"` (stray comma) → `"Spinner"`.
- `"KrisKross"` / `"Kriskross"` → `"KrisKross"`.

### `natpostcryptic` syndication
- Weekday puzzles may have originally appeared in the Daily Telegraph UK
  several months earlier. Publisher is hardcoded as `"National Post"` for all
  rows without distinguishing syndicated vs. original content.

### Setter field is a list
- Stored as a pipe-separated string (e.g. `"Enigmatist|Soup"`,
  `"Jacobs|Goodchild"`) to accommodate collaborative puzzles. Parse with
  `.str.split("|")` downstream.

### `puzzle_name` sometimes contains the setter name within the leading token
- `fifteensquared` examples: `"Guardian Prize Picaroon"`, `"Independent Punk"`,
  `"Guardian Prize Enigmatist"`.
- These are handled by dedicated rows in `publisher_lookup.csv` rather than
  by the generic trailing-token extraction.

---

## 6. Lookup Table

**File:** `data/publisher_lookup.csv`
**Columns:** `source`, `field`, `raw`, `publisher`, `series`, `setter`,
  `confidence`, `notes`
**Rows:** 349
**Key:** `(source, field, raw)` — `raw` is the normalized (lowercased,
  stripped) extracted substring

The `raw` column uses `QUOTE_MINIMAL` quoting — only fields containing commas
are quoted (e.g. `"dr. x,"`, `"dr, x,"`, `"spinner,"`).

The `setter` column uses pipe-separated values for collaborative puzzles.
The `confidence` column takes values `High`, `Medium`, or `Low`.

Any extracted substring not found in the lookup table is assigned NaN for
all derived fields and logged to the unmatched-tokens summary at the end of
the puzzle metadata notebook.
