"""
Generate validation-split embeddings for one g model across all three phrase
types: f_clue, f_common_wndef, and f_common_wnex.

This is a single reusable script run once per model. It produces the
embedding arrays Stage 5 needs for ATE computation and cross-f
generalization testing.

Model loading uses HuggingFace `AutoModel` plus the manual concept-aligned
extraction function ported from `train_g1.py`. This is used uniformly for
both g_stock and g_1 so that the pooling is guaranteed to match what g_1
learned during training. To confirm this does not introduce a discontinuity
with the existing `g_stock/f_clue.npy` (generated earlier via
`SentenceTransformer.encode()`), the script runs a small consistency
verification at startup: re-embed 200 phrases and assert cosine similarity
> 0.999 against the existing array.

Usage (typical Great Lakes submission, via SLURM):
    # For g_stock:
    python scripts/embed_val.py \
        --model-path gabrielloiseau/CALE-MBERT-en \
        --output-dir data/embeddings/g_stock \
        --batch-size 64

    # For g_1:
    python scripts/embed_val.py \
        --model-path models/g1/model \
        --output-dir data/embeddings/g1 \
        --batch-size 64

Smoke test on a small sample:
    python scripts/embed_val.py \
        --model-path gabrielloiseau/CALE-MBERT-en \
        --output-dir /tmp/embed_val_smoke \
        --batch-size 16 --sample 50

Outputs to --output-dir:
    f_clue_val.npy               — (N_val_clues, 1024), float32
    f_clue_val_index.csv         — (clue_id, definition, row)
    f_common_wndef_val.npy       — (26152, 1024), float32
    f_common_wnex_val.npy        — (3008, 1024), float32
"""

# =============================================================================
# §1 — Imports and environment reporting
# =============================================================================

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import transformers
from transformers import AutoModel, AutoTokenizer

# Per Decision 19: print environment versions at startup so the SLURM log
# permanently records the exact versions that produced the committed artifacts.
print("=" * 72)
print("Environment versions")
print("=" * 72)
print(f"Python:        {sys.version.split()[0]}")
print(f"torch:         {torch.__version__}")
print(f"transformers:  {transformers.__version__}")
print(f"numpy:         {np.__version__}")
print(f"pandas:        {pd.__version__}")
print(f"CUDA build:    {getattr(torch.version, 'cuda', 'n/a')}")
print(f"CUDA runtime:  {torch.cuda.is_available()}")
print("=" * 72)
print()

# Reproducibility seeds — encoding itself is deterministic on a fixed model,
# but we seed anyway in case any library uses RNG internally (e.g. dropout in
# train mode — we guard against this by calling model.eval()).
np.random.seed(42)
torch.manual_seed(42)


# =============================================================================
# §2 — CLI arguments
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed validation-split phrases with a given g model"
    )
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path or HuggingFace ID for the model "
                             "(e.g., 'gabrielloiseau/CALE-MBERT-en' or 'models/g1/model')")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory (e.g., data/embeddings/g_stock)")
    parser.add_argument("--data-dir", type=Path,
                        default=Path("data/filtered_split/wn_synset"),
                        help="Root of filtered_split data for this scope")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Encoding batch size (default: 64)")
    parser.add_argument("--max-length", type=int, default=128,
                        help="Tokenizer max_length (matches train_g1.py: 128)")
    parser.add_argument("--sample", type=int, default=0,
                        help="If > 0, take first N rows of each phrase set "
                             "(smoke test only)")
    parser.add_argument("--verify-gstock-npy", type=Path,
                        default=Path("data/embeddings/g_stock/f_clue.npy"),
                        help="Path to existing g_stock f_clue.npy for consistency check")
    parser.add_argument("--verify-gstock-index", type=Path,
                        default=Path("data/embeddings/g_stock/f_clue_index.csv"),
                        help="Path to existing g_stock f_clue_index.csv for consistency check")
    parser.add_argument("--verify-sample-size", type=int, default=200,
                        help="Number of rows to use for the consistency verification")
    parser.add_argument("--skip-verify", action="store_true",
                        help="Skip the consistency verification step (use only if the "
                             "reference g_stock f_clue.npy is not available on this host)")
    return parser.parse_args()


# =============================================================================
# §3 — Concept-aligned embedding extraction
# =============================================================================
#
# Ported verbatim from train_g1.py §5 (which in turn was ported from NB 09
# §4.2). Averages hidden states over tokens whose character span lies inside
# the <t>...</t> delimited region. This is the extraction g_1 was trained
# against, so it must be the extraction used at inference time too.


def find_delimiter_char_offsets(text: str) -> tuple:
    """Return (start, end) character offsets of the text between <t> and </t>.

    Offsets are into the ORIGINAL text (before tokenization strips the tags).
    Returns (None, None) if either tag is missing so the caller can fall back
    to mean pooling defensively.
    """
    start_tag, end_tag = "<t>", "</t>"
    start_pos = text.find(start_tag)
    if start_pos == -1:
        return None, None
    content_start = start_pos + len(start_tag)
    end_pos = text.find(end_tag, content_start)
    if end_pos == -1:
        return None, None
    return content_start, end_pos


def extract_concept_embedding(model, tokenizer, texts, device, max_length: int = 128):
    """Concept-aligned embedding for a batch of CALE-delimited texts.

    For each text: tokenize, forward pass, then average last_hidden_state
    over tokens whose character span falls inside the <t></t> region.
    Must be called inside a torch.no_grad() context at inference time.
    """
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)

    outputs = model(**encoded)
    hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)

    batch_size = hidden_states.shape[0]
    hidden_dim = hidden_states.shape[2]
    concept_vectors = torch.zeros(
        batch_size, hidden_dim, device=device, dtype=hidden_states.dtype
    )

    for i in range(batch_size):
        text = texts[i]
        start_char, end_char = find_delimiter_char_offsets(text)

        if start_char is None:
            # Defensive fallback — should never trigger on clean inputs, but
            # falling back to attention-masked mean pooling avoids emitting a
            # zero vector if one malformed row sneaks through.
            attention_mask = encoded["attention_mask"][i]
            concept_vectors[i] = (
                hidden_states[i] * attention_mask.unsqueeze(-1)
            ).sum(dim=0) / attention_mask.sum()
            continue

        # A token belongs to the concept span if its character span falls
        # entirely within [start_char, end_char). token_to_chars() returns
        # None for special tokens ([CLS], [SEP], [PAD]), which we skip.
        token_indices = []
        for tok_idx in range(encoded["input_ids"].shape[1]):
            span = encoded.token_to_chars(i, tok_idx)
            if span is None:
                continue
            if span.start >= start_char and span.end <= end_char:
                token_indices.append(tok_idx)

        if token_indices:
            concept_vectors[i] = hidden_states[i, token_indices, :].mean(dim=0)
        else:
            # Delimiters found but truncation cut off the span — fall back to
            # attention-masked mean pooling rather than emitting a zero vector.
            attention_mask = encoded["attention_mask"][i]
            concept_vectors[i] = (
                hidden_states[i] * attention_mask.unsqueeze(-1)
            ).sum(dim=0) / attention_mask.sum()

    return concept_vectors


def encode_phrases(model, tokenizer, phrases, device, batch_size: int,
                   max_length: int) -> np.ndarray:
    """Encode a list of phrases into an (N, 1024) float32 numpy array.

    Runs under torch.no_grad() with model.eval() so dropout is disabled and
    no gradient graph is built.
    """
    model.eval()
    n = len(phrases)
    all_vecs = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch_texts = phrases[start:start + batch_size]
            vecs = extract_concept_embedding(
                model, tokenizer, list(batch_texts), device, max_length
            )
            # Move to CPU and upcast to float32 immediately; accumulating on
            # GPU would blow past memory on large encoding runs.
            all_vecs.append(vecs.detach().to(torch.float32).cpu().numpy())
            if (start // batch_size) % 20 == 0:
                done = min(start + batch_size, n)
                print(f"    encoded {done:,} / {n:,}")
    return np.vstack(all_vecs)


# =============================================================================
# §4 — Atomic save helpers
# =============================================================================

def save_npy_atomic(array: np.ndarray, path: Path) -> None:
    """Save array to `path` atomically via a .tmp.npy intermediate.

    np.save() auto-appends '.npy' when the path doesn't already end in it, so
    the temp name must already end in '.npy' to avoid a double extension
    (see commit c3653b9).
    """
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, array)
    tmp.rename(path)


def save_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    """Save DataFrame to `path` atomically via a .tmp.csv intermediate."""
    tmp = path.with_suffix(".tmp.csv")
    df.to_csv(tmp, index=False)
    tmp.rename(path)


# =============================================================================
# §5 — Validation checks for each embedding array
# =============================================================================

def validate_embeddings(embeddings: np.ndarray, n_expected: int, label: str) -> None:
    """Assert shape, no NaN, no all-zero rows. Print L2 norm range."""
    assert embeddings.shape == (n_expected, 1024), (
        f"[{label}] Expected shape ({n_expected}, 1024), got {embeddings.shape}"
    )
    assert not np.isnan(embeddings).any(), f"[{label}] Found NaN values in embeddings"
    row_norms = np.linalg.norm(embeddings, axis=1)
    n_zero = int((row_norms == 0).sum())
    assert n_zero == 0, f"[{label}] Found {n_zero} all-zero rows in embeddings"
    print(f"  [{label}] shape={embeddings.shape}, dtype={embeddings.dtype}, "
          f"L2 norm range=[{row_norms.min():.4f}, {row_norms.max():.4f}]")


# =============================================================================
# §6 — Consistency verification against existing g_stock/f_clue.npy
# =============================================================================
#
# The existing data/embeddings/g_stock/f_clue.npy (239,406 rows) was produced
# by SentenceTransformer.encode(). This script uses AutoModel + manual
# concept-aligned extraction. Before producing new embeddings, verify the two
# methods agree: load N rows from the existing array, re-embed those same
# phrases with our extraction, and assert mean cosine similarity > 0.999.
# If they disagree, the mismatch must be resolved before proceeding.


def verify_consistency_with_gstock(
    model,
    tokenizer,
    device,
    args: argparse.Namespace,
) -> None:
    ref_npy_path = args.verify_gstock_npy
    ref_index_path = args.verify_gstock_index
    f_clue_csv = args.data_dir / "clue_phrases" / "f_clue.csv"

    if args.skip_verify:
        print("Consistency verification SKIPPED (--skip-verify set).")
        print()
        return

    missing = [p for p in (ref_npy_path, ref_index_path, f_clue_csv) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Cannot run consistency verification; missing files: "
            + ", ".join(str(p) for p in missing)
            + ". Pass --skip-verify only if you understand the risk."
        )

    print("-" * 72)
    print("Consistency check: AutoModel extraction vs existing g_stock/f_clue.npy")
    print("-" * 72)
    # Reference embeddings — row-indexed by f_clue_index.csv
    ref_index = pd.read_csv(ref_index_path, keep_default_na=False, na_values=[""])
    ref_embeddings = np.load(ref_npy_path)
    assert len(ref_index) == ref_embeddings.shape[0], (
        f"Reference index/embedding length mismatch: "
        f"{len(ref_index)} vs {ref_embeddings.shape[0]}"
    )

    # Phrase text — join by (clue_id, definition) to recover the exact inputs
    # that produced the reference embeddings.
    f_clue = pd.read_csv(f_clue_csv, keep_default_na=False, na_values=[""])
    n_check = min(args.verify_sample_size, len(ref_index))
    sample_index = ref_index.head(n_check)
    sample = sample_index.merge(
        f_clue[["clue_id", "definition", "phrase"]],
        on=["clue_id", "definition"],
        how="left",
    )
    assert sample["phrase"].notna().all(), (
        "Verification sample failed to merge — some (clue_id, definition) "
        "keys in f_clue_index.csv were not found in f_clue.csv."
    )
    # Preserve the reference row ordering so row i of ref == row i of sample.
    sample = sample.sort_values("row").reset_index(drop=True)
    phrases = sample["phrase"].tolist()

    print(f"Sampling {n_check} phrases from g_stock/f_clue.npy for verification ...")
    t0 = time.time()
    new_embeddings = encode_phrases(
        model, tokenizer, phrases, device,
        batch_size=args.batch_size, max_length=args.max_length,
    )
    print(f"Re-embedded {n_check} phrases in {time.time() - t0:.1f}s")

    # Cosine similarity row-wise
    ref_slice = ref_embeddings[sample["row"].to_numpy()].astype(np.float32)
    # Normalize and take dot product row-wise for cosine similarity
    ref_norms = np.linalg.norm(ref_slice, axis=1, keepdims=True)
    new_norms = np.linalg.norm(new_embeddings, axis=1, keepdims=True)
    cos_sims = (ref_slice * new_embeddings).sum(axis=1) / (
        ref_norms.squeeze() * new_norms.squeeze()
    )

    print(f"Cosine similarity: min={cos_sims.min():.6f}, "
          f"mean={cos_sims.mean():.6f}, max={cos_sims.max():.6f}")

    # Threshold from the spec: mean > 0.999. If a few rows fall slightly below
    # due to fp16 vs fp32 rounding, the mean will still be well above 0.999;
    # a systematic pooling mismatch would drive the mean far below that.
    mean_cos = float(cos_sims.mean())
    if mean_cos <= 0.999:
        raise AssertionError(
            f"Consistency check FAILED: mean cosine similarity {mean_cos:.6f} "
            f"<= 0.999. AutoModel extraction does not match "
            f"SentenceTransformer.encode() output for g_stock. "
            f"This is a blocking issue — do not proceed."
        )
    print(f"Consistency check PASSED (mean cosine {mean_cos:.6f} > 0.999).")
    print()


# =============================================================================
# §7 — Per-f embedding routines
# =============================================================================

def embed_f_clue_val(
    model, tokenizer, device, args: argparse.Namespace,
) -> tuple:
    """Encode validation-split f_clue phrases. Returns (embeddings, index_df, encode_time)."""
    f_clue_path = args.data_dir / "clue_phrases" / "f_clue.csv"
    print(f"Loading f_clue phrases from: {f_clue_path}")
    # keep_default_na=False: "nan" is a valid crossword word; without this
    # flag pandas silently converts it to NaN.
    df = pd.read_csv(f_clue_path, keep_default_na=False, na_values=[""])

    required = {"clue_id", "definition", "split", "phrase"}
    missing = required - set(df.columns)
    assert not missing, f"f_clue.csv missing columns: {missing}"
    assert df["phrase"].notna().all(), "Found null values in f_clue phrase column"

    # Keep only validation rows — Stage 4 computes embeddings for the val
    # split only; the test set is off-limits until a final model is chosen.
    df_val = df[df["split"] == "validate"].reset_index(drop=True)
    print(f"  Total rows: {len(df):,}; validation rows: {len(df_val):,}")

    if args.sample > 0:
        df_val = df_val.head(args.sample).reset_index(drop=True)
        print(f"  SAMPLE MODE: using first {len(df_val)} rows")

    phrases = df_val["phrase"].tolist()
    print(f"Encoding {len(phrases):,} f_clue_val phrases "
          f"(batch_size={args.batch_size}) ...")
    t0 = time.time()
    embeddings = encode_phrases(
        model, tokenizer, phrases, device,
        batch_size=args.batch_size, max_length=args.max_length,
    )
    encode_time = time.time() - t0
    print(f"  Encoded in {encode_time:.1f}s "
          f"({len(phrases) / max(encode_time, 1e-9):.0f} phrases/sec)")

    # Build the row-to-key index so downstream code can look up embeddings
    # by (clue_id, definition) without relying on DataFrame row order.
    index_df = pd.DataFrame({
        "clue_id": df_val["clue_id"].values,
        "definition": df_val["definition"].values,
        "row": np.arange(len(df_val)),
    })

    validate_embeddings(embeddings, len(df_val), "f_clue_val")
    return embeddings, index_df, encode_time


def embed_f_common_val(
    model, tokenizer, device, args: argparse.Namespace,
    vocab_path: Path, phrase_path: Path, label: str,
) -> tuple:
    """Encode a validation-vocab subset of a vocabulary-indexed phrase file.

    Returns (embeddings, encode_time). The output array is indexed by
    `vocab_path` — row i of the returned array corresponds to row i of
    vocab_path (which is also the word in vocab_path.iloc[i]["word"]).
    """
    print(f"Loading vocabulary: {vocab_path}")
    # keep_default_na=False: protects the word 'nan' and similar; without
    # this, pandas would silently drop rows like word='nan' (meaning
    # grandmother), breaking the canonical vocabulary ordering.
    vocab = pd.read_csv(vocab_path, keep_default_na=False, na_values=[""])
    assert {"word", "row"}.issubset(vocab.columns), (
        f"{vocab_path} missing required columns {{'word', 'row'}}"
    )
    # The `row` column in a vocabulary file IS the canonical index. Assert
    # it is contiguous 0..N-1 so we can safely treat list order as row order.
    assert (vocab["row"].to_numpy() == np.arange(len(vocab))).all(), (
        f"{vocab_path} 'row' column is not contiguous 0..N-1; "
        "vocabulary ordering is the canonical embedding index and must not be reordered."
    )
    print(f"  Vocabulary size: {len(vocab):,}")

    print(f"Loading phrases: {phrase_path}")
    phrases_df = pd.read_csv(phrase_path, keep_default_na=False, na_values=[""])
    assert {"word", "phrase"}.issubset(phrases_df.columns), (
        f"{phrase_path} missing required columns {{'word', 'phrase'}}"
    )
    assert phrases_df["phrase"].notna().all(), (
        f"Found null values in phrase column of {phrase_path}"
    )

    # Inner-join on `word`; this selects exactly the validation words and
    # drops everything else. We then re-assert the result matches the
    # vocabulary ordering so the resulting .npy is safely indexed by the
    # vocabulary file.
    joined = vocab[["word"]].merge(
        phrases_df[["word", "phrase"]], on="word", how="left"
    )
    # Assert the join is lossless — every validation word must have a phrase.
    n_missing = int(joined["phrase"].isna().sum())
    assert n_missing == 0, (
        f"[{label}] {n_missing} validation words have no phrase in {phrase_path.name}. "
        f"Each f is strict (no fallbacks), so this indicates the phrase file is "
        f"stale or the vocabulary was derived from a different constraint."
    )
    # Assert post-join ordering matches vocabulary ordering — the .npy array
    # will be indexed by the vocabulary file, so any reordering here would
    # misalign words with embedding rows.
    assert (joined["word"].values == vocab["word"].values).all(), (
        f"[{label}] Post-merge word ordering does not match vocabulary ordering. "
        f"This would misalign the .npy array against the vocabulary index."
    )
    assert len(joined) == len(vocab), (
        f"[{label}] Join row count {len(joined)} != vocab length {len(vocab)}"
    )

    if args.sample > 0:
        joined = joined.head(args.sample).reset_index(drop=True)
        print(f"  SAMPLE MODE: using first {len(joined)} rows")

    phrases = joined["phrase"].tolist()
    print(f"Encoding {len(phrases):,} {label} phrases "
          f"(batch_size={args.batch_size}) ...")
    t0 = time.time()
    embeddings = encode_phrases(
        model, tokenizer, phrases, device,
        batch_size=args.batch_size, max_length=args.max_length,
    )
    encode_time = time.time() - t0
    print(f"  Encoded in {encode_time:.1f}s "
          f"({len(phrases) / max(encode_time, 1e-9):.0f} phrases/sec)")

    validate_embeddings(embeddings, len(joined), label)
    return embeddings, encode_time


# =============================================================================
# §8 — Main
# =============================================================================

def main() -> None:
    args = parse_args()
    wall_start = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU:    {gpu_name} ({vram_gb:.1f} GB VRAM)")
    print()

    # --- Load model and tokenizer ---
    print(f"Loading model: {args.model_path} ...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModel.from_pretrained(args.model_path)
    model = model.to(device)
    model.eval()  # Disable dropout; inference mode for encoding.
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s")

    hidden_dim = model.config.hidden_size
    assert hidden_dim == 1024, f"Expected 1024-dim CALE; got {hidden_dim}"
    print(f"Hidden dim: {hidden_dim}")
    print()

    # --- Consistency check (skipped if --skip-verify) ---
    verify_consistency_with_gstock(model, tokenizer, device, args)

    # --- Prepare output directory ---
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- (1) f_clue_val ---
    print("=" * 72)
    print("(1/3) f_clue_val")
    print("=" * 72)
    f_clue_emb, f_clue_index, f_clue_time = embed_f_clue_val(
        model, tokenizer, device, args
    )
    f_clue_npy_path = args.output_dir / "f_clue_val.npy"
    f_clue_idx_path = args.output_dir / "f_clue_val_index.csv"
    save_npy_atomic(f_clue_emb, f_clue_npy_path)
    save_csv_atomic(f_clue_index, f_clue_idx_path)
    print(f"Saved: {f_clue_npy_path}")
    print(f"Saved: {f_clue_idx_path}")
    print()

    # --- (2) f_common_wndef_val ---
    print("=" * 72)
    print("(2/3) f_common_wndef_val")
    print("=" * 72)
    wndef_vocab = args.data_dir / "wndef" / "vocabulary_wndef_val.csv"
    wndef_phrases = args.data_dir / "wndef" / "f_common_wndef.csv"
    wndef_emb, wndef_time = embed_f_common_val(
        model, tokenizer, device, args,
        vocab_path=wndef_vocab, phrase_path=wndef_phrases,
        label="f_common_wndef_val",
    )
    wndef_npy_path = args.output_dir / "f_common_wndef_val.npy"
    save_npy_atomic(wndef_emb, wndef_npy_path)
    print(f"Saved: {wndef_npy_path}")
    print()

    # --- (3) f_common_wnex_val ---
    print("=" * 72)
    print("(3/3) f_common_wnex_val")
    print("=" * 72)
    wnex_vocab = args.data_dir / "wnex" / "vocabulary_wnex_val.csv"
    wnex_phrases = args.data_dir / "wnex" / "f_common_wnex.csv"
    wnex_emb, wnex_time = embed_f_common_val(
        model, tokenizer, device, args,
        vocab_path=wnex_vocab, phrase_path=wnex_phrases,
        label="f_common_wnex_val",
    )
    wnex_npy_path = args.output_dir / "f_common_wnex_val.npy"
    save_npy_atomic(wnex_emb, wnex_npy_path)
    print(f"Saved: {wnex_npy_path}")
    print()

    # --- Summary ---
    total_runtime = time.time() - wall_start
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"Model path:        {args.model_path}")
    print(f"Output directory:  {args.output_dir}")
    print(f"Model load time:   {load_time:.1f}s")
    print(f"f_clue_val:        {f_clue_emb.shape}, encode {f_clue_time:.1f}s")
    print(f"f_common_wndef_val:{wndef_emb.shape}, encode {wndef_time:.1f}s")
    print(f"f_common_wnex_val: {wnex_emb.shape}, encode {wnex_time:.1f}s")
    for path in (f_clue_npy_path, f_clue_idx_path, wndef_npy_path, wnex_npy_path):
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  {path.relative_to(args.output_dir)}  ({size_mb:.1f} MB)")
    print(f"Total runtime:     {total_runtime:.1f}s "
          f"({total_runtime / 60:.1f} min)")
    if device.type == "cuda":
        print(f"GPU:               {torch.cuda.get_device_name(0)}")
    print(f"Python:            {sys.version.split()[0]}")
    print(f"torch:             {torch.__version__}")
    print(f"transformers:      {transformers.__version__}")
    print("=" * 72)


if __name__ == "__main__":
    main()
