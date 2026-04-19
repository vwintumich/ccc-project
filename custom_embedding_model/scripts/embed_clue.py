"""
Embed clue-contextualized (f_clue) phrases for a given model and split.

Reads ``data/filtered_split/wn_synset/clue_phrases/f_clue.csv`` (one row per
(clue_id, definition) pair, all splits), filters by the requested split,
and writes an ``.npy`` of shape ``(N, 1024)`` alongside a row-to-key index
CSV.

Output filenames are determined by ``--split``:

  ``--split train``     → ``f_clue_train.npy`` + ``f_clue_train_index.csv``
  ``--split validate``  → ``f_clue_val.npy``   + ``f_clue_val_index.csv``
  ``--split all``       → ``f_clue.npy``       + ``f_clue_index.csv``

Supports an optional ``--verify-against`` flag that compares the new
output against an existing ``.npy`` file rowwise by cosine similarity (keys
matched by ``(clue_id, definition)`` via the reference's index CSV). This
is how the seven verification runs in ``verify_embedding_scripts.sh``
confirm the refactored scripts reproduce the committed artifacts.

Typical Great Lakes usage (via SLURM):

    # g_stock full-dataset f_clue embeddings (replaces embed_f_clue_gstock.py)
    python scripts/embed_clue.py \\
        --model-path gabrielloiseau/CALE-MBERT-en \\
        --pooling meanpool --split all \\
        --output-dir data/embeddings/g_stock

    # g1 validation-split f_clue embeddings (replaces embed_val.py f_clue portion)
    python scripts/embed_clue.py \\
        --model-path models/g1/model \\
        --pooling meanpool --split validate \\
        --output-dir data/embeddings/g1

Smoke test on a small sample (runs on CPU):

    python scripts/embed_clue.py \\
        --model-path gabrielloiseau/CALE-MBERT-en \\
        --pooling meanpool --split validate \\
        --output-dir /tmp/embed_clue_smoke \\
        --batch-size 16 --sample 50
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# embedding_utils lives next to this file; support both "python scripts/..."
# invocation (cwd = project root) and direct execution by adding the
# script's own directory to sys.path if needed.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import embedding_utils as eu  # noqa: E402


# =============================================================================
# CLI arguments
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed f_clue phrases for a given model and split."
    )
    parser.add_argument("--model-path", type=str, required=True,
                        help="HuggingFace ID or local model directory "
                             "(e.g., 'gabrielloiseau/CALE-MBERT-en' or 'models/g1/model')")
    parser.add_argument("--pooling", type=str, required=True,
                        choices=["meanpool", "tokenspan"],
                        help="Extraction method (Decision 20 canonical = meanpool). "
                             "Use 'tokenspan' only for g1_tokenspan and its matching "
                             "g_stock_tokenspan baseline.")
    parser.add_argument("--split", type=str, required=True,
                        choices=["train", "validate", "all"],
                        help="Split filter. 'all' uses every row including test — "
                             "only permitted for g_stock full-dataset embedding "
                             "(Decision 7).")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory for output files.")
    parser.add_argument("--f-clue-csv", type=Path,
                        default=Path("data/filtered_split/wn_synset/clue_phrases/f_clue.csv"),
                        help="Path to f_clue.csv "
                             "(default: data/filtered_split/wn_synset/clue_phrases/f_clue.csv)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Encoding batch size (default: 64)")
    parser.add_argument("--max-length", type=int, default=128,
                        help="Tokenizer max_length (default: 128, matches train_g1.py)")
    parser.add_argument("--sample", type=int, default=0,
                        help="If > 0, take first N rows after split filtering (smoke test)")
    parser.add_argument("--verify-against", type=Path, default=None,
                        help="Path to existing .npy to compare against rowwise by "
                             "cosine similarity. Reference index CSV is inferred "
                             "from the .npy stem.")
    return parser.parse_args()


# =============================================================================
# Output naming
# =============================================================================

def output_stems_for_split(split: str) -> tuple:
    """Return (npy_filename, index_filename) for the given split.

    Naming convention (from DATA.md and the existing committed artifacts):
      - ``validate`` → ``f_clue_val``   (matches existing g_stock/g1 artifacts)
      - ``train``    → ``f_clue_train`` (consistent with the ``_train`` triplet convention)
      - ``all``      → ``f_clue``       (matches existing g_stock/f_clue.npy)
    """
    if split == "validate":
        return "f_clue_val.npy", "f_clue_val_index.csv"
    if split == "train":
        return "f_clue_train.npy", "f_clue_train_index.csv"
    if split == "all":
        return "f_clue.npy", "f_clue_index.csv"
    raise ValueError(f"Unknown split: {split!r}")


def reference_index_path(ref_npy: Path) -> Path:
    """Derive the reference index CSV path from the reference .npy path.

    The two sit side by side with the same stem (e.g. ``f_clue_val.npy`` and
    ``f_clue_val_index.csv``), so we append ``_index.csv`` to the stem.
    """
    return ref_npy.with_name(ref_npy.stem + "_index.csv")


# =============================================================================
# Consistency check
# =============================================================================

def verify_against_reference(new_embeddings: np.ndarray,
                             new_index: pd.DataFrame,
                             ref_npy_path: Path) -> None:
    """Compare new embeddings against an existing ``.npy`` rowwise by cosine.

    Rows are matched by ``(clue_id, definition)`` between the new index
    DataFrame and the reference index CSV. If there are more than 500
    matches, a sample of 500 is used to keep the print output short (the
    full-dataset f_clue.npy has 239,406 rows). The assertion threshold is
    mean cosine > 0.999 per the spec.
    """
    ref_index_path = reference_index_path(ref_npy_path)
    print("-" * 72)
    print(f"Consistency check: {ref_npy_path} ...")
    print("-" * 72)

    # Reference arrays — keep_default_na=False because definition may
    # legitimately be the string "nan" (grandmother, a valid answer).
    ref_embeddings = np.load(ref_npy_path)
    ref_index = pd.read_csv(
        ref_index_path, keep_default_na=False, na_values=[""]
    )
    assert len(ref_index) == ref_embeddings.shape[0], (
        f"Reference index/embedding length mismatch: "
        f"{len(ref_index)} vs {ref_embeddings.shape[0]}"
    )

    # Inner join the two indexes on (clue_id, definition). suffixes makes
    # the two row columns easy to tell apart.
    matched = new_index.merge(
        ref_index, on=["clue_id", "definition"],
        how="inner", suffixes=("_new", "_ref"),
    )
    n_matched = len(matched)
    assert n_matched > 0, (
        f"No (clue_id, definition) keys matched between new output and "
        f"{ref_npy_path}. Is the reference from a different scope/split?"
    )
    print(f"Matched rows: {n_matched:,} "
          f"(new={len(new_index):,}, ref={len(ref_index):,})")

    # Subsample for reporting if large — every row still has to exist in
    # both indexes; we just compare 500 rather than 239K cosines.
    if n_matched > 500:
        matched = matched.sample(n=500, random_state=42).reset_index(drop=True)
        print(f"Comparing a sample of {len(matched):,} rows")

    new_slice = new_embeddings[matched["row_new"].to_numpy()].astype(np.float32)
    ref_slice = ref_embeddings[matched["row_ref"].to_numpy()].astype(np.float32)

    # Rowwise cosine similarity — normalize, then elementwise multiply and sum.
    new_norms = np.linalg.norm(new_slice, axis=1)
    ref_norms = np.linalg.norm(ref_slice, axis=1)
    cos_sims = (new_slice * ref_slice).sum(axis=1) / (new_norms * ref_norms)

    print(f"Cosine similarity: min={cos_sims.min():.6f}, "
          f"mean={cos_sims.mean():.6f}, max={cos_sims.max():.6f}")
    mean_cos = float(cos_sims.mean())
    if mean_cos > 0.999:
        print(f"Consistency check PASSED (mean cosine {mean_cos:.6f} > 0.999).")
    else:
        # Print FAIL first so it's visible in the log before the assertion
        # stack trace swallows the rest of the line.
        print(f"Consistency check FAILED (mean cosine {mean_cos:.6f} <= 0.999).")
        raise AssertionError(
            f"Consistency check FAILED: mean cosine {mean_cos:.6f} <= 0.999 "
            f"against {ref_npy_path}"
        )
    print()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()
    wall_start = time.time()

    eu.print_environment()

    # Reproducibility — encoding is deterministic on a fixed model, but we
    # seed defensively in case any library uses RNG internally.
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU:    {gpu_name} ({vram_gb:.1f} GB VRAM)")
    print()

    model, tokenizer = eu.load_model(args.model_path, device)

    # --- Load f_clue.csv ---
    # keep_default_na=False: protects "nan" (a valid crossword word).
    print(f"Loading f_clue phrases from: {args.f_clue_csv}")
    df = pd.read_csv(args.f_clue_csv, keep_default_na=False, na_values=[""])
    required_cols = {"clue_id", "definition", "split", "phrase"}
    missing = required_cols - set(df.columns)
    assert not missing, f"f_clue.csv missing columns: {missing}"
    assert df["phrase"].notna().all(), "Found null values in f_clue phrase column"
    print(f"  Total rows (all splits): {len(df):,}")

    # --- Filter by split ---
    if args.split == "all":
        # Matches the embed_f_clue_gstock.py output scope (Decision 7).
        df_out = df.reset_index(drop=True)
        print(f"  Using all rows: {len(df_out):,}")
    else:
        # 'train' or 'validate' — simple string filter on the split column.
        df_out = df[df["split"] == args.split].reset_index(drop=True)
        print(f"  Rows with split == {args.split!r}: {len(df_out):,}")

    if args.sample > 0:
        df_out = df_out.head(args.sample).reset_index(drop=True)
        print(f"  SAMPLE MODE: using first {len(df_out)} rows")
    print()

    # --- Encode ---
    phrases = df_out["phrase"].tolist()
    print(f"Encoding {len(phrases):,} f_clue phrases "
          f"(batch_size={args.batch_size}, pooling={args.pooling}) ...")
    t0 = time.time()
    embeddings = eu.encode_phrases(
        model, tokenizer, phrases, device,
        batch_size=args.batch_size, max_length=args.max_length,
        pooling=args.pooling,
    )
    encode_time = time.time() - t0
    print(f"  Encoded in {encode_time:.1f}s "
          f"({len(phrases) / max(encode_time, 1e-9):.0f} phrases/sec)")
    print()

    # --- Validate ---
    eu.validate_embeddings(embeddings, len(df_out), label=f"f_clue_{args.split}")
    print()

    # --- Build index ---
    # Row i of the .npy corresponds to row i of this index DataFrame.
    index_df = pd.DataFrame({
        "clue_id": df_out["clue_id"].values,
        "definition": df_out["definition"].values,
        "row": np.arange(len(df_out)),
    })

    # --- Save outputs ---
    args.output_dir.mkdir(parents=True, exist_ok=True)
    npy_name, idx_name = output_stems_for_split(args.split)
    npy_path = args.output_dir / npy_name
    idx_path = args.output_dir / idx_name
    eu.save_npy_atomic(embeddings, npy_path)
    eu.save_csv_atomic(index_df, idx_path)
    print(f"Saved: {npy_path}")
    print(f"Saved: {idx_path}")
    print()

    # --- Consistency check (optional) ---
    if args.verify_against is not None:
        verify_against_reference(embeddings, index_df, args.verify_against)

    # --- Summary ---
    total_runtime = time.time() - wall_start
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"Model path:        {args.model_path}")
    print(f"Pooling method:    {args.pooling}")
    print(f"Split:             {args.split}")
    print(f"Rows embedded:     {len(df_out):,}")
    print(f"Encoding time:     {encode_time:.1f}s")
    for path in (npy_path, idx_path):
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  {path}  ({size_mb:.1f} MB)")
    print(f"Total runtime:     {total_runtime:.1f}s "
          f"({total_runtime / 60:.1f} min)")
    if device.type == "cuda":
        print(f"GPU:               {torch.cuda.get_device_name(0)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
