# Wordplay Metadata — Results

Generated from `data/clues_filtered.csv` (434,465 unique `clue_id`s after dedup).

Output file: `data/wordplay_metadata.csv`.

## Coverage

- Clues matching **at least one** wordplay type: 75,939 (17.48%)
- Clues matching **zero** wordplay types: 358,526 (82.52%)

## Per-type frequency

| Type | Count | % of total |
|------|------:|-----------:|
| `anagram_single_word` | 6,191 | 1.42% |
| `anagram_consec_words` | 26,946 | 6.20% |
| `hidden_fwd` | 13,702 | 3.15% |
| `hidden_rev` | 3,496 | 0.80% |
| `selection_alt` | 1,592 | 0.37% |
| `selection_alt_rev` | 174 | 0.04% |
| `selection_firsts` | 2,042 | 0.47% |
| `selection_firsts_rev` | 60 | 0.01% |
| `selection_lasts` | 275 | 0.06% |
| `selection_lasts_rev` | 20 | 0.00% |
| `double_def` | 21,941 | 5.05% |

## Co-occurrence (clues with both types True)

| | `anagram_single_word` | `anagram_consec_words` | `hidden_fwd` | `hidden_rev` | `selection_alt` | `selection_alt_rev` | `selection_firsts` | `selection_firsts_rev` | `selection_lasts` | `selection_lasts_rev` | `double_def` |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `anagram_single_word` | 6,191 | 6 | 41 | 20 | 0 | 3 | 4 | 0 | 1 | 0 | 36 |
| `anagram_consec_words` | 6 | 26,946 | 24 | 9 | 1 | 0 | 0 | 0 | 0 | 0 | 143 |
| `hidden_fwd` | 41 | 24 | 13,702 | 71 | 8 | 6 | 6 | 0 | 1 | 0 | 59 |
| `hidden_rev` | 20 | 9 | 71 | 3,496 | 3 | 1 | 1 | 0 | 0 | 0 | 11 |
| `selection_alt` | 0 | 1 | 8 | 3 | 1,592 | 12 | 1 | 0 | 1 | 0 | 3 |
| `selection_alt_rev` | 3 | 0 | 6 | 1 | 12 | 174 | 0 | 1 | 0 | 0 | 0 |
| `selection_firsts` | 4 | 0 | 6 | 1 | 1 | 0 | 2,042 | 15 | 0 | 0 | 16 |
| `selection_firsts_rev` | 0 | 0 | 0 | 0 | 0 | 1 | 15 | 60 | 0 | 0 | 0 |
| `selection_lasts` | 1 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 275 | 3 | 0 |
| `selection_lasts_rev` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 20 | 0 |
| `double_def` | 36 | 143 | 59 | 11 | 3 | 0 | 16 | 0 | 0 | 0 | 21,941 |

## Top 5 type combinations

| Count | Combination |
|------:|-------------|
| 26,764 | `anagram_consec_words` |
| 21,673 | `double_def` |
| 13,492 | `hidden_fwd` |
| 6,083 | `anagram_single_word` |
| 3,386 | `hidden_rev` |

## Example clues (3 random per type, random_state=42)

### `anagram_single_word` (6,191 matches)

| clue_id | surface | answer |
|--------:|---------|--------|
| 103114 | Fauns dancing all over the place | SNAFU |
| 48808 | Ravel preludes caused dislike | REPULSED |
| 644905 | Privately, American misbehaving | IN CAMERA |

### `anagram_consec_words` (26,946 matches)

| clue_id | surface | answer |
|--------:|---------|--------|
| 3738 | A pact goes wrong: this one takes the blame | SCAPEGOAT |
| 137372 | Fly-by-night, showing more mph, tore off | EMPEROR MOTH |
| 78634 | Description of safety measure wound up reactionary | PRECAUTIONARY |

### `hidden_fwd` (13,702 matches)

| clue_id | surface | answer |
|--------:|---------|--------|
| 365079 | Dishes out some appetiser vessels | SERVES |
| 228677 | Man in the orchestra playing Beethoven’s Choral Symphony | NINTH |
| 185241 | Country’s merit realised to some extent | ERITREA |

### `hidden_rev` (3,496 matches)

| clue_id | surface | answer |
|--------:|---------|--------|
| 555676 | Urge to get radically snogged without restraint | EGG ON |
| 590230 | Spirit in Canada, I anticipate, going west | NAIAD |
| 372835 | Bones in ocean luridly reflected | ULNAE |

### `selection_alt` (1,592 matches)

| clue_id | surface | answer |
|--------:|---------|--------|
| 344338 | Regularly scrummy, Iona’s seasoning | CUMIN |
| 489391 | Even for rugby it’s a riotous event | ORGY |
| 617602 | Imitative play every other playroom day | PARODY |

### `selection_alt_rev` (174 matches)

| clue_id | surface | answer |
|--------:|---------|--------|
| 606745 | Country, near ruin at intervals, that’s on the rise | IRAN |
| 566312 | Far off and regularly war-weary | AWAY |
| 319624 | Paste menu’s logo evenly over displays | GLUE |

### `selection_firsts` (2,042 matches)

| clue_id | surface | answer |
|--------:|---------|--------|
| 405094 | Former PM he elected at the historic primaries | HEATH |
| 317252 | Originally South American music, Brazilian actually | SAMBA |
| 332391 | Government officials taking to authoritarianism, all leaders have to | GOTTA |

### `selection_firsts_rev` (60 matches)

| clue_id | surface | answer |
|--------:|---------|--------|
| 8621 | Mountain in Japan, utterly fabulous, initially, to climb | FUJI |
| 44162 | Bones heads to Enterprise after nicely leading Uhura back | ULNAE |
| 349697 | Perform with female fellow or don? Just the opposite | DOFF |

### `selection_lasts` (275 matches)

| clue_id | surface | answer |
|--------:|---------|--------|
| 38059 | If confidence low, the Exchequer ultimately produces not so many | FEWER |
| 274166 | Quest, the conclusions of which you can reject | HUNT |
| 393034 | Brave conclusions reached by good people of jury | DEFY |

### `selection_lasts_rev` (20 matches)

| clue_id | surface | answer |
|--------:|---------|--------|
| 1210 | Sole aplenty, reel in two, all tails up | ONLY |
| 335434 | Backs in Cambridge extra parlous with westerly — that’s a breeze | EASY |
| 244702 | 15 22 across ends up in Latin America, superb archipelago | OBAN |

### `double_def` (21,941 matches)

| clue_id | surface | answer |
|--------:|---------|--------|
| 72259 | Friendly    drink | CORDIAL |
| 542974 | Spot description of a kind of trading H | COMMERCIAL |
| 594405 | Relative value of speaking without extremes | RATIO |
