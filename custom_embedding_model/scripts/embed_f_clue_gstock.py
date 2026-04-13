"""
Encode all f_clue phrases using g_stock (unmodified CALE) via SentenceTransformer.encode().

Produces:
    f_clue.npy        — dense embedding array, float32, shape (N, 1024)
    f_clue_index.csv  — row-to-key mapping with columns (clue_id, definition, row)

Usage:
    python scripts/embed_f_clue_gstock.py \
        --input data/filtered_split/wn_synset/clue_phrases/f_clue.csv \
        --output-dir data/embeddings/g_stock \
        --batch-size 64

    For quick smoke-testing on CPU:
        python scripts/embed_f_clue_gstock.py \
            --input data/filtered_split/wn_synset/clue_phrases/f_clue.csv \
            --output-dir data/embeddings/g_stock \
            --batch-size 32 --sample 100
"""

# =============================================================================
# §1 — Imports and configuration
# =============================================================================

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, __version__ as st_version

# Reproducibility seeds
np.random.seed(42)
torch.manual_seed(42)

# CALE model identifier — 1024-dim, uses <t></t> delimiters for concept-aligned extraction
MODEL_NAME = "gabrielloiseau/CALE-MBERT-en"

# Auto-detect device: prefer CUDA if available, fall back to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device:                   {device}")
if device == "cuda":
    print(f"GPU:                      {torch.cuda.get_device_name(0)}")
print(f"PyTorch version:          {torch.__version__}")
print(f"sentence-transformers:    {st_version}")
print()

# =============================================================================
# §2 — Parse arguments and load phrase data
# =============================================================================

parser = argparse.ArgumentParser(
    description="Embed f_clue phrases with g_stock (CALE)"
)
parser.add_argument(
    "--input", type=Path, required=True,
    help="Path to f_clue.csv"
)
parser.add_argument(
    "--output-dir", type=Path, required=True,
    help="Directory for output files (f_clue.npy, f_clue_index.csv)"
)
parser.add_argument(
    "--batch-size", type=int, default=64,
    help="Encoding batch size (default: 64)"
)
parser.add_argument(
    "--sample", type=int, default=0,
    help="If > 0, embed only the first N rows (for testing)"
)
args = parser.parse_args()

wall_start = time.time()

# Load f_clue.csv
# keep_default_na=False: "nan" is a valid crossword word; prevent pandas converting it to NaN
df = pd.read_csv(args.input, keep_default_na=False, na_values=[""])

# Validate expected columns
expected_cols = {"clue_id", "definition", "split", "phrase"}
assert expected_cols.issubset(df.columns), (
    f"Missing columns: {expected_cols - set(df.columns)}"
)
# Every row must have a phrase — null phrases would produce meaningless embeddings
assert df["phrase"].notna().all(), "Found null values in phrase column"

# Deterministic subset for smoke-testing: take the first N rows to keep it reproducible
if args.sample > 0:
    df = df.head(args.sample)
    print(f"SAMPLE MODE: using first {args.sample} rows")

print(f"Rows to embed:            {len(df):,}")
print(f"Input file:               {args.input}")
print()

# =============================================================================
# §3 — Load model
# =============================================================================

print(f"Loading model: {MODEL_NAME} ...")
t0 = time.time()
model = SentenceTransformer(MODEL_NAME)
print(f"Model loaded in {time.time() - t0:.1f}s")

# Verify CALE's 1024-dim embedding space
assert model.get_sentence_embedding_dimension() == 1024, (
    f"Expected 1024-dim embeddings, got {model.get_sentence_embedding_dimension()}"
)

print(f"Embedding dimension:      {model.get_sentence_embedding_dimension()}")
print(f"Device used by model:     {model.device}")
print()

# =============================================================================
# §4 — Encode phrases
# =============================================================================

phrases = df["phrase"].tolist()

print(f"Encoding {len(phrases):,} phrases (batch_size={args.batch_size}) ...")
t0 = time.time()

embeddings = model.encode(
    phrases,
    batch_size=args.batch_size,
    show_progress_bar=True,
    normalize_embeddings=False,  # Save raw embeddings; downstream code
)                                # normalizes as needed for cosine similarity
embeddings = np.array(embeddings, dtype=np.float32)

encode_time = time.time() - t0
print(f"Encoding completed in {encode_time:.1f}s "
      f"({len(phrases) / encode_time:.0f} phrases/sec)")
print()

# --- Validation checks ---

# Shape: rows must match input, columns must be 1024
assert embeddings.shape == (len(df), 1024), (
    f"Expected shape ({len(df)}, 1024), got {embeddings.shape}"
)

# No NaN values — would indicate a model or input problem
assert not np.isnan(embeddings).any(), "Found NaN values in embeddings"

# No all-zero rows — every phrase should produce a nonzero embedding
row_norms = np.linalg.norm(embeddings, axis=1)
assert (row_norms > 0).all(), (
    f"Found {(row_norms == 0).sum()} all-zero rows in embeddings"
)

print(f"Embedding shape:          {embeddings.shape}")
print(f"Embedding dtype:          {embeddings.dtype}")
print(f"L2 norm range:            [{row_norms.min():.4f}, {row_norms.max():.4f}]")
print()

# =============================================================================
# §5 — Build index file
# =============================================================================

# The index maps each row in the .npy array back to its (clue_id, definition) key
index_df = pd.DataFrame({
    "clue_id": df["clue_id"].values,
    "definition": df["definition"].values,
    "row": np.arange(len(df)),  # 0-indexed, contiguous
})

assert len(index_df) == embeddings.shape[0], (
    f"Index length {len(index_df)} != embedding rows {embeddings.shape[0]}"
)

# =============================================================================
# §6 — Save outputs atomically
# =============================================================================

output_dir = args.output_dir
output_dir.mkdir(parents=True, exist_ok=True)

# Write to .tmp first, then rename — prevents corrupt partial files if killed mid-write
tmp_npy = output_dir / "f_clue.npy.tmp"
tmp_csv = output_dir / "f_clue_index.csv.tmp"

final_npy = output_dir / "f_clue.npy"
final_csv = output_dir / "f_clue_index.csv"

np.save(tmp_npy, embeddings)
index_df.to_csv(tmp_csv, index=False)

tmp_npy.rename(final_npy)
tmp_csv.rename(final_csv)

# =============================================================================
# §7 — Summary
# =============================================================================

npy_size_mb = final_npy.stat().st_size / (1024 * 1024)
csv_size_mb = final_csv.stat().st_size / (1024 * 1024)
total_time = time.time() - wall_start

print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Total rows embedded:      {len(df):,}")
print(f"Embedding array shape:    {embeddings.shape}")
print(f"Embedding dtype:          {embeddings.dtype}")
print(f"Output .npy:              {final_npy}  ({npy_size_mb:.1f} MB)")
print(f"Output index .csv:        {final_csv}  ({csv_size_mb:.1f} MB)")
print(f"Encoding time:            {encode_time:.1f}s")
print(f"Total wall-clock time:    {total_time:.1f}s")
print("=" * 60)
