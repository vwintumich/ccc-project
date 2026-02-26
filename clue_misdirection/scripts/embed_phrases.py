"""
Step 2 (GPU): Embed CALE phrases for clue_misdirection pipeline.

Loads phrase CSVs from data/embeddings/ (produced by the CPU portion of
02_embedding_generation.ipynb) and generates embeddings using
gabrielloiseau/CALE-MBERT-en (1024-dim).

Input:
    data/embeddings/definition_phrases.csv
    data/embeddings/answer_phrases.csv
    data/embeddings/clue_context_phrases.csv

Output:
    data/embeddings/definition_embeddings.npy  — shape (N_def, 3, 1024)
    data/embeddings/definition_index.csv       — maps row position to definition_wn string
    data/embeddings/answer_embeddings.npy      — shape (N_ans, 3, 1024)
    data/embeddings/answer_index.csv           — maps row position to answer_wn string
    data/embeddings/clue_context_embeddings.npy — shape (N_rows, 1024)
    data/embeddings/clue_context_index.csv     — maps row position to clue_id

Usage:
    python scripts/embed_phrases.py
    python scripts/embed_phrases.py --data-dir /path/to/data
    python scripts/embed_phrases.py --sample 50    # quick test with 50 words

Environment:
    Requires GPU. On Great Lakes, submit via the companion SLURM script.
    On Colab, run after mounting Drive and running the notebook CPU cells.

Author: Victoria
AI assistance: Claude Code (Anthropic)
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CALE_NAME = "gabrielloiseau/CALE-MBERT-en"
EMBED_DIM = 1024
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _embed_word_phrases(phrases_df, model, batch_size, label):
    """Embed definition or answer phrases and return (embeddings, index_df).

    Parameters
    ----------
    phrases_df : pd.DataFrame
        Must contain columns: word, synset_idx, is_common, is_obscure, phrase.
    model : SentenceTransformer
    batch_size : int
    label : str
        Human-readable label for progress messages (e.g. "definition").

    Returns
    -------
    embeddings : np.ndarray, shape (N_unique_words, 3, 1024)
        Axis 1: [0] allsense_avg, [1] common, [2] obscure.
    index_df : pd.DataFrame
        Single column 'word', one row per unique word, matching embeddings
        row order.
    """
    # Sorted unique words — deterministic ordering
    unique_words = sorted(phrases_df["word"].unique())
    n_words = len(unique_words)
    print(f"\n{'='*60}")
    print(f"Embedding {label} phrases: {n_words} unique words")
    print(f"{'='*60}")

    # ----- Collect phrases for each embedding type -----
    common_phrases = []
    obscure_phrases = []
    all_phrases = []       # every synset phrase, for allsense average
    word_ranges = []       # (start, end) index into all_phrases per word

    word_to_group = phrases_df.groupby("word")

    for word in unique_words:
        group = word_to_group.get_group(word).sort_values("synset_idx")

        # Common = first synset (is_common == True, synset_idx == 0)
        common_row = group.loc[group["is_common"]]
        common_phrases.append(common_row["phrase"].iloc[0])

        # Obscure = last synset (is_obscure == True)
        obscure_row = group.loc[group["is_obscure"]]
        obscure_phrases.append(obscure_row["phrase"].iloc[0])

        # All phrases for allsense average
        start = len(all_phrases)
        all_phrases.extend(group["phrase"].tolist())
        end = len(all_phrases)
        word_ranges.append((start, end))

    print(f"  Common phrases to encode:  {len(common_phrases)}")
    print(f"  Obscure phrases to encode: {len(obscure_phrases)}")
    print(f"  All phrases to encode:     {len(all_phrases)}")

    # ----- Encode common -----
    t0 = time.time()
    print(f"\n  Encoding common {label} phrases ...")
    common_embs = model.encode(
        common_phrases, batch_size=batch_size, show_progress_bar=True
    )
    print(f"  Common done in {time.time() - t0:.1f}s")

    # ----- Encode obscure -----
    t0 = time.time()
    print(f"  Encoding obscure {label} phrases ...")
    obscure_embs = model.encode(
        obscure_phrases, batch_size=batch_size, show_progress_bar=True
    )
    print(f"  Obscure done in {time.time() - t0:.1f}s")

    # ----- Encode all (for allsense average) -----
    t0 = time.time()
    print(f"  Encoding all {label} phrases ({len(all_phrases)} total) ...")
    all_embs = model.encode(
        all_phrases, batch_size=batch_size, show_progress_bar=True
    )
    print(f"  All-synset encoding done in {time.time() - t0:.1f}s")

    # ----- Compute allsense averages -----
    allsense_embs = np.zeros((n_words, EMBED_DIM), dtype=np.float32)
    for word_idx, (start, end) in enumerate(word_ranges):
        allsense_embs[word_idx] = all_embs[start:end].mean(axis=0)

    # ----- Stack into (N_words, 3, 1024): [allsense, common, obscure] -----
    embeddings = np.stack(
        [allsense_embs, np.array(common_embs), np.array(obscure_embs)],
        axis=1,
    )
    assert embeddings.shape == (n_words, 3, EMBED_DIM), (
        f"Expected ({n_words}, 3, {EMBED_DIM}), got {embeddings.shape}"
    )

    # ----- Build index DataFrame -----
    index_df = pd.DataFrame({"word": unique_words})

    print(f"\n  {label.capitalize()} embeddings shape: {embeddings.shape}")
    print(f"  Memory: {embeddings.nbytes / 1024**2:.1f} MB")

    return embeddings, index_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Step 2 (GPU): Embed CALE phrases for clue_misdirection."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Path to data directory (default: data/)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for model.encode() (default: 64)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help="Run on a small subset for testing: first N unique words for "
             "definitions/answers, first N*5 clue-context rows. Output files "
             "are prefixed with test_ to avoid overwriting real outputs.",
    )
    args = parser.parse_args()

    script_start = time.time()
    emb_dir = args.data_dir / "embeddings"

    # ------------------------------------------------------------------
    # a. Setup — verify inputs, GPU, seeds
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 2 (GPU): Embed CALE phrases")
    print("=" * 60)
    print(f"Data directory:    {args.data_dir.resolve()}")
    print(f"Embeddings dir:    {emb_dir.resolve()}")
    print(f"Batch size:        {args.batch_size}")

    input_files = [
        emb_dir / "definition_phrases.csv",
        emb_dir / "answer_phrases.csv",
        emb_dir / "clue_context_phrases.csv",
    ]
    for f in input_files:
        assert f.exists(), f"Missing input file: {f}"
    print("All 3 input phrase CSVs found.")

    gpu_available = torch.cuda.is_available()
    if gpu_available:
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: No GPU detected — encoding will be slow on CPU.")

    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    print(f"Random seed: {RANDOM_SEED}")

    sample_n = args.sample
    prefix = "test_" if sample_n else ""
    if sample_n:
        print(f"\n{'*'*60}")
        print(f"SAMPLE MODE: Using first {sample_n} words/rows for testing.")
        print(f"Output files prefixed with test_.")
        print(f"{'*'*60}")

    # ------------------------------------------------------------------
    # b. Load model
    # ------------------------------------------------------------------
    print(f"\nLoading model: {CALE_NAME} ...")
    t0 = time.time()
    model = SentenceTransformer(CALE_NAME)
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s")
    print(f"Embedding dimension: {model.get_sentence_embedding_dimension()}")
    assert model.get_sentence_embedding_dimension() == EMBED_DIM, (
        f"Expected {EMBED_DIM}-dim, got {model.get_sentence_embedding_dimension()}"
    )

    # ------------------------------------------------------------------
    # c. Embed definition phrases
    # ------------------------------------------------------------------
    def_phrases = pd.read_csv(emb_dir / "definition_phrases.csv", keep_default_na=False)
    print(f"\nLoaded definition_phrases.csv: {len(def_phrases)} rows")
    if sample_n:
        keep_words = sorted(def_phrases["word"].unique())[:sample_n]
        def_phrases = def_phrases[def_phrases["word"].isin(keep_words)]
        print(f"  SAMPLE: kept first {len(keep_words)} unique words "
              f"→ {len(def_phrases)} phrase rows")
    definition_embeddings, definition_index = _embed_word_phrases(
        def_phrases, model, args.batch_size, label="definition"
    )
    n_def = len(definition_index)

    # ------------------------------------------------------------------
    # d. Embed answer phrases
    # ------------------------------------------------------------------
    ans_phrases = pd.read_csv(emb_dir / "answer_phrases.csv", keep_default_na=False)
    print(f"\nLoaded answer_phrases.csv: {len(ans_phrases)} rows")
    if sample_n:
        keep_words = sorted(ans_phrases["word"].unique())[:sample_n]
        ans_phrases = ans_phrases[ans_phrases["word"].isin(keep_words)]
        print(f"  SAMPLE: kept first {len(keep_words)} unique words "
              f"→ {len(ans_phrases)} phrase rows")
    answer_embeddings, answer_index = _embed_word_phrases(
        ans_phrases, model, args.batch_size, label="answer"
    )
    n_ans = len(answer_index)

    # ------------------------------------------------------------------
    # e. Embed clue-context phrases
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Embedding clue-context phrases")
    print(f"{'='*60}")
    cc_phrases = pd.read_csv(emb_dir / "clue_context_phrases.csv", keep_default_na=False)
    print(f"Loaded clue_context_phrases.csv: {len(cc_phrases)} rows")
    if sample_n:
        cc_phrases = cc_phrases.head(sample_n * 5)
        print(f"  SAMPLE: kept first {len(cc_phrases)} rows")
    n_rows = len(cc_phrases)

    phrases_list = cc_phrases["clue_context_phrase"].tolist()

    t0 = time.time()
    print("  Encoding clue-context phrases ...")
    clue_context_embeddings = model.encode(
        phrases_list, batch_size=args.batch_size, show_progress_bar=True
    )
    clue_context_embeddings = np.array(clue_context_embeddings)
    print(f"  Done in {time.time() - t0:.1f}s")

    clue_context_index = pd.DataFrame({"clue_id": cc_phrases["clue_id"]})

    print(f"  Clue-context embeddings shape: {clue_context_embeddings.shape}")
    print(f"  Memory: {clue_context_embeddings.nbytes / 1024**2:.1f} MB")

    # ------------------------------------------------------------------
    # f. Save all 6 files
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Saving output files")
    print(f"{'='*60}")

    np.save(emb_dir / f"{prefix}definition_embeddings.npy", definition_embeddings)
    definition_index.to_csv(emb_dir / f"{prefix}definition_index.csv", index=True)

    np.save(emb_dir / f"{prefix}answer_embeddings.npy", answer_embeddings)
    answer_index.to_csv(emb_dir / f"{prefix}answer_index.csv", index=True)

    np.save(emb_dir / f"{prefix}clue_context_embeddings.npy", clue_context_embeddings)
    clue_context_index.to_csv(emb_dir / f"{prefix}clue_context_index.csv", index=True)

    # Print file sizes
    output_files = [
        f"{prefix}definition_embeddings.npy", f"{prefix}definition_index.csv",
        f"{prefix}answer_embeddings.npy", f"{prefix}answer_index.csv",
        f"{prefix}clue_context_embeddings.npy", f"{prefix}clue_context_index.csv",
    ]
    total_mb = 0.0
    for fname in output_files:
        fpath = emb_dir / fname
        size_mb = fpath.stat().st_size / 1024**2
        total_mb += size_mb
        print(f"  {fname}: {size_mb:.1f} MB")
    print(f"  Total: {total_mb:.1f} MB")

    # ------------------------------------------------------------------
    # g. Verification
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Verification")
    print(f"{'='*60}")

    # Reload and check definition embeddings
    def_emb_check = np.load(emb_dir / f"{prefix}definition_embeddings.npy")
    def_idx_check = pd.read_csv(emb_dir / f"{prefix}definition_index.csv", index_col=0, keep_default_na=False)
    assert def_emb_check.shape == (n_def, 3, EMBED_DIM), (
        f"definition_embeddings shape mismatch: {def_emb_check.shape}"
    )
    assert len(def_idx_check) == n_def, (
        f"definition_index length mismatch: {len(def_idx_check)} vs {n_def}"
    )
    assert def_emb_check.shape[2] == EMBED_DIM
    assert not np.isnan(def_emb_check).any(), "NaN in definition_embeddings"
    print(f"  definition_embeddings: {def_emb_check.shape} — OK")

    # Reload and check answer embeddings
    ans_emb_check = np.load(emb_dir / f"{prefix}answer_embeddings.npy")
    ans_idx_check = pd.read_csv(emb_dir / f"{prefix}answer_index.csv", index_col=0, keep_default_na=False)
    assert ans_emb_check.shape == (n_ans, 3, EMBED_DIM), (
        f"answer_embeddings shape mismatch: {ans_emb_check.shape}"
    )
    assert len(ans_idx_check) == n_ans, (
        f"answer_index length mismatch: {len(ans_idx_check)} vs {n_ans}"
    )
    assert ans_emb_check.shape[2] == EMBED_DIM
    assert not np.isnan(ans_emb_check).any(), "NaN in answer_embeddings"
    print(f"  answer_embeddings:     {ans_emb_check.shape} — OK")

    # Reload and check clue-context embeddings
    cc_emb_check = np.load(emb_dir / f"{prefix}clue_context_embeddings.npy")
    cc_idx_check = pd.read_csv(emb_dir / f"{prefix}clue_context_index.csv", index_col=0, keep_default_na=False)
    assert cc_emb_check.shape == (n_rows, EMBED_DIM), (
        f"clue_context_embeddings shape mismatch: {cc_emb_check.shape}"
    )
    assert len(cc_idx_check) == n_rows, (
        f"clue_context_index length mismatch: {len(cc_idx_check)} vs {n_rows}"
    )
    assert cc_emb_check.shape[1] == EMBED_DIM
    assert not np.isnan(cc_emb_check).any(), "NaN in clue_context_embeddings"
    print(f"  clue_context_embeddings: {cc_emb_check.shape} — OK")

    print("\nAll verification checks passed.")

    # ------------------------------------------------------------------
    # h. Total timing
    # ------------------------------------------------------------------
    total_time = time.time() - script_start
    minutes = int(total_time // 60)
    seconds = total_time % 60
    print(f"\nTotal runtime: {minutes}m {seconds:.1f}s")


if __name__ == "__main__":
    main()
