# Decisions — ccc-project (Shared Pipeline)

These decisions are **locked in** and should not be revisited or second-guessed
without explicit team discussion. They govern the two shared upstream notebooks
(`puzzle_metadata.ipynb`, `structural_filtering.ipynb`) and the shared
`clue_utils.py` module. Component-specific decisions live in each component's
own `DECISIONS.md`.

---

## Shared Pipeline

### `structural_filtering.ipynb` — null filter scope

**Choice:** A single `dropna(subset=["clue", "definition", "answer"])` call is
used to drop rows where any of `clue`, `definition`, or `answer` is null.

**Rationale:** The original spec named only `definition` and `answer`, but a
`TypeError` during development revealed that non-null but float-valued `clue`
entries survive a narrower `dropna` and then break downstream string
operations (e.g. the surface regex and the answer-format extraction). Adding
`clue` to the subset list eliminates the entire class of stray non-string
`clue` values in one step, so no defensive `.astype(str)` logic is needed
further down.

---

### `structural_filtering.ipynb` — square brackets in the clue surface

**Choice:** Brackets in the clue surface are handled as follows:

- Rows where brackets surround or partially surround an all-caps sequence
  (e.g., `[VERVE]`, `[ELGIAN]`) are **excluded**.
- Brackets are **stripped** from all remaining rows.
- A diagnostic cell displaying surviving rows containing `[` is retained in
  the notebook for reference.

**Rationale:** Brackets do not appear in original published cryptic clues —
they were inserted by bloggers transcribing the puzzles. Where a bracketed
all-caps sequence appears, the clue comes from an "extra-word" puzzle variant
that violates standard cryptic crossword rules and is out of scope for this
project. The remaining bracketed rows are ordinary clues with blogger
annotations that can safely be removed by stripping the bracket characters.

---

### `structural_filtering.ipynb` — asterisks in the clue surface

**Choice:** Asterisks in the clue surface are handled as follows:

- Rows where `*` is the very first character of the clue **and** is
  immediately followed by an uppercase letter have the **leading `*`
  stripped** from the surface (the rest of the clue is preserved). The
  row is **not dropped**.
- All other occurrences of `*` are **kept** as part of the surface.
- No rows are ever excluded on the basis of `*`.
- A diagnostic cell displaying surviving rows containing `*` is retained in
  the notebook for reference.

**Rationale:** A leading `*` followed by a capital letter is a thematic
marker (e.g. a Nina or specialty puzzle variant) indicating the clue
belongs to a theme set, but the text that follows is itself a standalone
cryptic clue — for example, `*A despicable person (6)` → TOERAG reduces
cleanly to `A despicable person` once the marker is removed. Stripping
just the leading `*` preserves these clues while removing the non-surface
annotation. Other uses of `*` — typically censorship (`b*** hell`,
`M*A*S*H`) or blogger annotations (`(Real ner[d])*`) — occur inside
genuine clue text and must be preserved so the surface reading stays
intact.

**History:** An earlier version of this filter excluded the leading-`*` +
capital rows outright and stripped `*` unconditionally from all remaining
surfaces. That behavior was wrong on both counts: it lost recoverable
thematic clues, and it corrupted censorship/annotation surfaces like
`b*** hell` by silently removing the `*`. The current rule replaces it.

---

### `structural_filtering.ipynb` — forward slashes in the clue surface

**Choice:** Forward slashes are stripped unconditionally from the clue
surface. A diagnostic cell displaying surviving rows containing `/` is
retained in the notebook for reference.

**Rationale:** Slashes in the clue surface were inserted by bloggers to mark
the boundary between the two definitions in double-definition clues — they
are not part of the original published surface text. Stripping them
unconditionally keeps the surface faithful to how the clue was actually read.

One known exception exists: `clue_id` 9773 contains a slash inside a quoted
phrase that is genuinely part of the surface. Losing this single case is
accepted as collateral loss — a targeted exception would add complexity that
is not justified by a single affected row.

---

### `data/id_map.csv` and `data/publisher_lookup.csv` — committed to repo

**Choice:** `data/id_map.csv` and `data/publisher_lookup.csv` are committed
to the repo via `.gitignore` exceptions (`!data/id_map.csv` and
`!data/publisher_lookup.csv` following the `data/*` ignore line). Neither
file should be manually edited; any regeneration requires a documented
reason recorded in this file first.

**Rationale:** `id_map.csv` is small, fully deterministic, and derived from
fixed raw data (`clues_raw.csv`) by `data_preparation/assign_ids.py`. Committing it
eliminates the run-order dependency between `assign_ids.py` and
`puzzle_metadata.ipynb`: contributors can open and run
`puzzle_metadata.ipynb` directly without first executing a separate script.
`publisher_lookup.csv` is the authoritative, hand-curated lookup table for
puzzle metadata extraction and was already committed — the exception line
makes this explicit alongside the `id_map.csv` exception so the
gitignore/exception pair documents the policy for both files in one place.
