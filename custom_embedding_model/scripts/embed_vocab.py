"""
Embed decontextualized vocabulary phrases for a given model, vocabulary
file, and phrase file.

Unlike ``embed_clue.py``, this script is vocabulary-agnostic: you pass the
vocabulary CSV and the phrase CSV explicitly, so the same script handles
``f_common_wndef``, ``f_common_wnex``, any validation-subset variant, and
any future vocabulary-indexed phrase file.

The vocabulary file serves as the canonical index for the output ``.npy``
(Decision 6) — no separate index file is written. The script verifies
post-join that the phrase rows are in vocabulary order before encoding, so
row ``i`` of the output ``.npy`` corresponds exactly to row ``i`` (and
word at position ``i``) of the vocabulary file.

Typical Great Lakes usage (via SLURM):

    # g_stock full-vocab wnex embeddings (the immediate job this refactor enables)
    python scripts/embed_vocab.py \\
        --model-path gabrielloiseau/CALE-MBERT-en \\
        --pooling meanpool \\
        --vocab-file data/filtered_split/wn_synset/wnex/vocabulary_wnex.csv \\
        --phrase-file data/filtered_split/wn_synset/wnex/f_common_wnex.csv \\
        --output-file data/embeddings/g_stock/f_common_wnex.npy

    # g1 validation-subset wndef embeddings (replaces embed_val.py wndef portion)
    python scripts/embed_vocab.py \\
        --model-path models/g1/model \\
        --pooling meanpool \\
        --vocab-file data/filtered_split/wn_synset/wndef/vocabulary_wndef_val.csv \\
        --phrase-file data/filtered_split/wn_synset/wndef/f_common_wndef.csv \\
        --output-file data/embeddings/g1/f_common_wndef_val.npy
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
        description="Embed decontextualized vocabulary phrases."
    )
    parser.add_argument("--model-path", type=str, required=True,
                        help="HuggingFace ID or local model directory")
    parser.add_argument("--pooling", type=str, required=True,
                        choices=["meanpool", "tokenspan"],
                        help="Extraction method (Decision 20 canonical = meanpool)")
    parser.add_argument("--vocab-file", type=Path, required=True,
                        help="Vocabulary CSV (columns: word, row; row is "
                             "contiguous 0..N-1 and defines the canonical "
                             "embedding array index)")
    parser.add_argument("--phrase-file", type=Path, required=True,
                        help="Phrase CSV (columns include: word, phrase). "
                             "May contain more words than the vocabulary; "
                             "the script joins vocab onto phrases and asserts "
                             "every vocab word has a phrase.")
    parser.add_argument("--output-file", type=Path, required=True,
                        help="Output .npy path.")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Encoding batch size (default: 64)")
    parser.add_argument("--max-length", type=int, default=128,
                        help="Tokenizer max_length (default: 128)")
    parser.add_argument("--sample", type=int, default=0,
                        help="If > 0, take first N rows (smoke test)")
    parser.add_argument("--verify-against", type=Path, default=None,
                        help="Path to existing .npy to compare against rowwise "
                             "by cosine similarity. Reference must be indexed "
                             "by the same vocabulary file.")
    return parser.parse_args()


# =============================================================================
# Vocabulary/phrase loading and alignment
# =============================================================================

def load_vocab(vocab_path: Path) -> pd.DataFrame:
    """Load a vocabulary CSV and assert its canonical-ordering invariant.

    The ``row`` column must be contiguous ``0..N-1``; if it isn't, this
    vocabulary file has been reordered and any ``.npy`` produced against it
    would silently misalign words with embedding rows (Decision 6).
    """
    # keep_default_na=False: protects the word "nan" (grandmother).
    vocab = pd.read_csv(vocab_path, keep_default_na=False, na_values=[""])
    assert {"word", "row"}.issubset(vocab.columns), (
        f"{vocab_path} missing required columns {{'word', 'row'}}"
    )
    # The `row` column IS the canonical index — assert it is contiguous
    # 0..N-1 so list order can be treated as row order further down.
    assert (vocab["row"].to_numpy() == np.arange(len(vocab))).all(), (
        f"{vocab_path} 'row' column is not contiguous 0..N-1; "
        f"vocabulary ordering is the canonical embedding index and "
        f"must not be reordered."
    )
    print(f"  Vocabulary size: {len(vocab):,}")
    return vocab


def load_phrases(phrase_path: Path) -> pd.DataFrame:
    """Load a phrase CSV. keep_default_na=False protects word='nan'."""
    phrases_df = pd.read_csv(
        phrase_path, keep_default_na=False, na_values=[""]
    )
    assert {"word", "phrase"}.issubset(phrases_df.columns), (
        f"{phrase_path} missing required columns {{'word', 'phrase'}}"
    )
    assert phrases_df["phrase"].notna().all(), (
        f"Found null values in phrase column of {phrase_path}"
    )
    return phrases_df


def align_phrases_to_vocab(vocab: pd.DataFrame,
                           phrases_df: pd.DataFrame,
                           label: str) -> pd.DataFrame:
    """Join vocab onto phrases and assert alignment invariants.

    Three invariants must hold for the resulting ``.npy`` to be safely
    indexed by the vocabulary file:
      1. Lossless — every vocab word has a phrase (strict f, no fallbacks);
      2. Same length — exactly one phrase row per vocab row;
      3. Same ordering — post-join ``word`` order matches vocab order, so
         row ``i`` of the phrase list is the word at vocab row ``i``.
    """
    # Left join so the result mirrors vocab ordering and any missing match
    # surfaces as NaN in the phrase column.
    joined = vocab[["word"]].merge(
        phrases_df[["word", "phrase"]], on="word", how="left"
    )
    n_missing = int(joined["phrase"].isna().sum())
    assert n_missing == 0, (
        f"[{label}] {n_missing} vocabulary words have no phrase. "
        f"Each f is strict (no fallbacks), so this indicates the phrase "
        f"file is stale or the vocabulary was derived from a different "
        f"constraint."
    )
    assert (joined["word"].values == vocab["word"].values).all(), (
        f"[{label}] Post-merge word ordering does not match vocabulary "
        f"ordering. This would misalign the .npy against the vocabulary index."
    )
    assert len(joined) == len(vocab), (
        f"[{label}] Join row count {len(joined)} != vocab length {len(vocab)}"
    )
    return joined


# =============================================================================
# Consistency check
# =============================================================================

def verify_against_reference(new_embeddings: np.ndarray,
                             ref_npy_path: Path,
                             vocab_size: int) -> None:
    """Compare new embeddings against an existing ``.npy`` rowwise by cosine.

    The reference file must be indexed by the same vocabulary file as the
    new output — otherwise rows with the same index refer to different
    words. We assert ``ref.shape[0] == vocab_size`` as a structural guard.
    """
    print("-" * 72)
    print(f"Consistency check: {ref_npy_path} ...")
    print("-" * 72)
    ref_embeddings = np.load(ref_npy_path)
    assert ref_embeddings.shape[0] == vocab_size, (
        f"Reference shape[0]={ref_embeddings.shape[0]} does not match "
        f"vocabulary size {vocab_size}. The reference must be indexed by "
        f"the same vocabulary file passed via --vocab-file."
    )
    assert ref_embeddings.shape[1] == new_embeddings.shape[1] == 1024, (
        f"Embedding dim mismatch: new={new_embeddings.shape[1]}, "
        f"ref={ref_embeddings.shape[1]}, expected 1024"
    )

    # Subsample for reporting if large — rowwise cosine on 26K rows is
    # fast, but printing a sample keeps the SLURM log readable.
    if vocab_size > 500:
        rng = np.random.default_rng(seed=42)
        idx = rng.choice(vocab_size, size=500, replace=False)
        idx.sort()
        new_slice = new_embeddings[idx].astype(np.float32)
        ref_slice = ref_embeddings[idx].astype(np.float32)
        print(f"Comparing a sample of {len(idx):,} rows "
              f"(of {vocab_size:,} total)")
    else:
        new_slice = new_embeddings.astype(np.float32)
        ref_slice = ref_embeddings.astype(np.float32)
        print(f"Comparing all {vocab_size:,} rows")

    new_norms = np.linalg.norm(new_slice, axis=1)
    ref_norms = np.linalg.norm(ref_slice, axis=1)
    cos_sims = (new_slice * ref_slice).sum(axis=1) / (new_norms * ref_norms)

    print(f"Cosine similarity: min={cos_sims.min():.6f}, "
          f"mean={cos_sims.mean():.6f}, max={cos_sims.max():.6f}")
    mean_cos = float(cos_sims.mean())
    if mean_cos > 0.999:
        print(f"Consistency check PASSED (mean cosine {mean_cos:.6f} > 0.999).")
    else:
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

    # --- Load vocabulary and phrase file ---
    print(f"Loading vocabulary: {args.vocab_file}")
    vocab = load_vocab(args.vocab_file)
    print(f"Loading phrases:    {args.phrase_file}")
    phrases_df = load_phrases(args.phrase_file)

    label = args.output_file.stem
    joined = align_phrases_to_vocab(vocab, phrases_df, label=label)

    if args.sample > 0:
        joined = joined.head(args.sample).reset_index(drop=True)
        print(f"  SAMPLE MODE: using first {len(joined)} rows")
    print()

    # --- Encode ---
    phrases = joined["phrase"].tolist()
    print(f"Encoding {len(phrases):,} {label} phrases "
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
    eu.validate_embeddings(embeddings, len(joined), label=label)
    print()

    # --- Save ---
    # The vocabulary file IS the index (Decision 6), so no _index.csv sidecar.
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    eu.save_npy_atomic(embeddings, args.output_file)
    print(f"Saved: {args.output_file}")
    print()

    # --- Consistency check (optional) ---
    if args.verify_against is not None:
        verify_against_reference(
            embeddings, args.verify_against, vocab_size=len(vocab)
        )

    # --- Summary ---
    total_runtime = time.time() - wall_start
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"Model path:        {args.model_path}")
    print(f"Pooling method:    {args.pooling}")
    print(f"Vocab file:        {args.vocab_file}")
    print(f"Phrase file:       {args.phrase_file}")
    print(f"Vocabulary size:   {len(vocab):,}")
    print(f"Rows embedded:     {len(joined):,}")
    print(f"Encoding time:     {encode_time:.1f}s")
    size_mb = args.output_file.stat().st_size / (1024 * 1024)
    print(f"  {args.output_file}  ({size_mb:.1f} MB)")
    print(f"Total runtime:     {total_runtime:.1f}s "
          f"({total_runtime / 60:.1f} min)")
    if device.type == "cuda":
        print(f"GPU:               {torch.cuda.get_device_name(0)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
