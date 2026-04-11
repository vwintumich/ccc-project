"""assign_ids.py — Assign globally unique puzzle_ids by grouping clues on source_url.

Reads:  data/clues_raw.csv
Writes: data/id_map.csv  (columns: clue_id, puzzle_id)

Must be run before puzzle_metadata.ipynb. Has no dependency on any other
notebook or script.

Normalization rules applied before grouping:

1. bigdave44 only — some puzzles have two distinct URLs pointing to the
   same blog post: a human-readable slug URL
   (e.g. http://bigdave44.com/2021/12/15/toughie-2766/) and a numeric
   WordPress post-ID URL (http://bigdave44.com/2021/12/15/141686/). For any
   puzzle_name whose rows contain both a slug URL and a numeric-ID URL, the
   numeric-ID URL is rewritten to the slug URL before grouping. Numeric-ID
   URLs with no slug sibling are left as-is.

2. All sources — strip trailing slash from source_url. This collapses the
   single fifteensquared trailing-slash anomaly identified in
   source_url_investigation.ipynb.

Run from the notebooks/ directory:

    python assign_ids.py
"""

import sys
import time
from pathlib import Path

import pandas as pd

# ===
# Paths
# ===
NOTEBOOK_DIR = Path(__file__).resolve().parent
DATA_DIR = NOTEBOOK_DIR.parent / "data"
INPUT_CSV = DATA_DIR / "clues_raw.csv"
OUTPUT_CSV = DATA_DIR / "id_map.csv"

# Regex for bigdave44 numeric-ID (WordPress post) URLs: a trailing
# /<digits>/ path component anchored to the end of the URL. Slug URLs like
# .../toughie-2766/ do not match because the final component contains
# non-digit characters.
NUMERIC_ID_URL_PATTERN = r"/\d+/$"


def rewrite_bigdave44_numeric_urls(df):
    """Rewrite bigdave44 numeric-ID URLs to slug URLs where a slug sibling
    exists under the same puzzle_name. Mutates ``df`` in place and returns
    the number of rows whose source_url was rewritten.
    """
    mask_bd = df["source"] == "bigdave44"
    if not mask_bd.any():
        return 0

    bd = df.loc[mask_bd, ["puzzle_name", "source_url"]].copy()
    bd["is_numeric"] = bd["source_url"].str.contains(
        NUMERIC_ID_URL_PATTERN, na=False, regex=True
    )

    # For each puzzle_name, pick the first slug URL after a deterministic
    # lexicographic sort. Rare puzzle_names with multiple slug variants
    # therefore map to a reproducible choice across reruns.
    slug_rows = (
        bd.loc[~bd["is_numeric"]]
          .sort_values("source_url", kind="stable")
          .drop_duplicates("puzzle_name", keep="first")
    )
    slug_by_name = pd.Series(
        slug_rows["source_url"].values,
        index=slug_rows["puzzle_name"].values,
    )

    numeric_idx = bd.index[bd["is_numeric"]]
    replacements = bd.loc[numeric_idx, "puzzle_name"].map(slug_by_name)
    has_replacement = replacements.notna()
    target_idx = numeric_idx[has_replacement.values]

    df.loc[target_idx, "source_url"] = replacements[has_replacement].values
    return int(has_replacement.sum())


def main():
    t0 = time.time()

    if not INPUT_CSV.exists():
        sys.exit(f"Input file not found: {INPUT_CSV}")

    # puzzle_name is required by Step 1 (bigdave44 slug-vs-numeric URL
    # reconciliation) even though the rest of the script only touches
    # clue_id, source, and source_url.
    t_load = time.time()
    df = pd.read_csv(
        INPUT_CSV,
        usecols=["clue_id", "source", "source_url", "puzzle_name"],
        keep_default_na=False,
        na_values=[""],
    )
    n_input = len(df)
    print(f"Loaded {n_input:,} rows from {INPUT_CSV}  ({time.time() - t_load:.1f}s)")

    # ---------------------------------------------------------------
    # Step 1 — Normalize source_url
    # ---------------------------------------------------------------
    n_replaced = rewrite_bigdave44_numeric_urls(df)
    print(f"bigdave44: replaced {n_replaced:,} numeric-ID URLs with slug URLs")

    df["normalized_source_url"] = df["source_url"].str.rstrip("/")

    # ---------------------------------------------------------------
    # Step 2 — Deterministic global puzzle_id assignment
    # ---------------------------------------------------------------
    # Grouping is on normalized_source_url alone. Sorting by
    # (source, normalized_source_url) before deduplicating gives a stable
    # ordering of the unique URL list across reruns on the same input.
    url_order = (
        df[["source", "normalized_source_url"]]
          .sort_values(["source", "normalized_source_url"], kind="stable")
          .drop_duplicates("normalized_source_url", keep="first")
          .reset_index(drop=True)
    )
    url_to_id = dict(zip(url_order["normalized_source_url"], range(len(url_order))))
    df["puzzle_id"] = df["normalized_source_url"].map(url_to_id)

    # ---------------------------------------------------------------
    # Step 3 — Validate
    # ---------------------------------------------------------------
    assert len(df) == n_input, "row count changed during assignment"
    assert df["clue_id"].is_unique, "clue_id duplicated in output"
    assert df["puzzle_id"].notna().all(), "null puzzle_id in output"
    n_puzzles = int(df["puzzle_id"].nunique())
    assert n_puzzles < len(df), (
        f"no grouping occurred: {n_puzzles} puzzles for {len(df)} rows"
    )

    print("\nUnique puzzles by source:")
    per_source = (
        df.groupby("source")["puzzle_id"]
          .nunique()
          .sort_values(ascending=False)
    )
    for src, n in per_source.items():
        print(f"  {src:<26} {n:>7,}")
        

    # ---------------------------------------------------------------
    # Step 4 — Write output
    # ---------------------------------------------------------------
    out = (
        df[["clue_id", "puzzle_id"]]
          .astype({"clue_id": int, "puzzle_id": int})
          .sort_values(["puzzle_id", "clue_id"], kind="stable")
          .reset_index(drop=True)
    )
    out.to_csv(OUTPUT_CSV, index=False)

    print(f"\nTotal clues processed: {n_input:,}")
    print(f"Total unique puzzles:  {n_puzzles:,}")
    print(f"Output written to:     {OUTPUT_CSV}")
    print(f"Total runtime:         {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
