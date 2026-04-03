"""
Phase 2 (GPU): One-time test-set evaluation of a finalized learned g.

This script is the companion to train_g_triplet.py, which fine-tunes CALE
on cryptic crossword data and evaluates the result on a validation set. Once
a final g has been selected based on validation-set performance, this script
opens the held-out test set exactly once, re-embeds it with the finalized
model, computes the test ATE, and records the results.

The embedding and ATE approach follows Nathan Cantwell's original notebook
(notebooks/archive/09_learned_g_misdirection.ipynb) as adapted in
train_g_triplet.py. Concept-aligned embeddings are extracted by averaging
hidden states within the <t></t> delimiter span (not [CLS]), matching
CALE's training objective.

WARNING: This script evaluates the held-out test set. Run this only once,
after a final g has been selected based on validation results. Running this
during experimentation compromises the validity of the causal inference.

Design:
    - Loads a fine-tuned model saved by train_g_triplet.py via
      AutoModel.from_pretrained() (never torch.load() — see CLAUDE.md).
    - Verifies that test_evaluated is false in the experiment's
      ate_results.json before proceeding. If true, aborts.
    - Requires interactive confirmation before opening the test set.
    - Saves learned test embeddings and updates ate_results.json with
      test ATE fields and test_evaluated: true.

Usage:
    python scripts/evaluate_final_g.py \\
        --dataset harder \\
        --model-tag model_triplet_e3_lr2e-05_m1.0

Author: Victoria Winters
Original approach: Nathan Cantwell
AI assistance: Claude Code (Anthropic)
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


# =========================================================================
# Constants
# =========================================================================

EMBED_DIM = 1024
RANDOM_SEED = 42


# =========================================================================
# Section 1 — Setup and argument parsing
# =========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "One-time test-set evaluation of a finalized learned g. "
            "Run only after selecting a final model on validation results."
        ),
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        help="Which data/learned_g/{dataset}/ folder to read from",
    )
    parser.add_argument(
        "--model-tag", type=str, required=True,
        help=(
            "Name of the model subdirectory to load, e.g. "
            "model_triplet_e3_lr2e-05_m1.0"
        ),
    )
    return parser.parse_args()


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================================================================
# Concept-aligned embedding extraction (from train_g_triplet.py)
# =========================================================================

def find_delimiter_char_offsets(text):
    """Find character offsets of content between <t> and </t> delimiters.

    Returns (start, end) character indices of the CONTENT (excluding tags).
    Adapted from Nathan Cantwell's implementation in
    notebooks/archive/09_learned_g_misdirection.ipynb.
    """
    start_tag = "<t>"
    end_tag = "</t>"

    start_pos = text.find(start_tag)
    if start_pos == -1:
        return None, None

    content_start = start_pos + len(start_tag)
    end_pos = text.find(end_tag, content_start)
    if end_pos == -1:
        return None, None

    return content_start, end_pos


def extract_concept_embedding(model, tokenizer, texts, device):
    """Extract concept-aligned embeddings for a batch of CALE-delimited texts.

    For each text, identifies the <t></t> span, maps character offsets to
    token indices, and averages the hidden states of those tokens. This is
    the correct extraction method for CALE — [CLS] does not carry the same
    semantic precision because CALE's training objective specifically
    optimizes the delimited-span representation.

    Adapted from Nathan Cantwell's implementation.

    Args:
        model: The CALE transformer model
        tokenizer: The CALE tokenizer
        texts: list of str — texts with <t></t> delimiters
        device: torch device

    Returns:
        torch.Tensor of shape (batch_size, hidden_dim)
    """
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    ).to(device)

    outputs = model(**encoded)
    hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)

    batch_size = hidden_states.shape[0]
    hidden_dim = hidden_states.shape[2]
    concept_vectors = torch.zeros(batch_size, hidden_dim, device=device)

    for i in range(batch_size):
        text = texts[i]
        start_char, end_char = find_delimiter_char_offsets(text)

        if start_char is None:
            # No delimiters — fall back to mean pooling over non-padding tokens
            attention_mask = encoded["attention_mask"][i]
            concept_vectors[i] = (
                hidden_states[i] * attention_mask.unsqueeze(-1)
            ).sum(dim=0) / attention_mask.sum()
            continue

        # Map character offsets to token indices
        token_indices = []
        for tok_idx in range(encoded["input_ids"].shape[1]):
            span = encoded.token_to_chars(i, tok_idx)
            if span is None:
                continue  # special tokens ([CLS], [SEP], [PAD])
            if span.start >= start_char and span.end <= end_char:
                token_indices.append(tok_idx)

        if token_indices:
            concept_vectors[i] = hidden_states[i, token_indices, :].mean(dim=0)
        else:
            # Fallback: mean pooling over all non-padding tokens
            attention_mask = encoded["attention_mask"][i]
            concept_vectors[i] = (
                hidden_states[i] * attention_mask.unsqueeze(-1)
            ).sum(dim=0) / attention_mask.sum()

    return concept_vectors


def batch_embed(model, tokenizer, texts, device, batch_size=64):
    """Generate concept-aligned embeddings for a list of CALE-delimited texts.

    Processes in batches to avoid OOM. Returns numpy array of shape
    (N, hidden_dim).
    """
    model.eval()
    all_embeddings = []

    for start in tqdm(range(0, len(texts), batch_size),
                      desc="Embedding", leave=False):
        batch_texts = texts[start : start + batch_size]

        with torch.no_grad():
            emb = extract_concept_embedding(model, tokenizer, batch_texts, device)

        all_embeddings.append(emb.cpu().numpy())

    return np.vstack(all_embeddings)


# =========================================================================
# Allsense embedding (from train_g_triplet.py)
# =========================================================================

def build_allsense_phrase_lookup(phrases_df):
    """Map each word to the list of all its synset phrases."""
    lookup = {}
    for word, group in phrases_df.groupby("word"):
        lookup[word] = group.sort_values("synset_idx")["phrase"].tolist()
    return lookup


def embed_allsense(model, tokenizer, words, phrase_lookup, device,
                   batch_size=64):
    """Embed words using allsense averaging: for each word, embed all
    synset phrases and average the resulting vectors.

    This replicates Phase 1's allsense embedding approach (slot 0 of
    definition_embeddings.npy / answer_embeddings.npy).
    """
    all_embeddings = []

    for word in tqdm(words, desc="Allsense embedding", leave=False):
        phrases = phrase_lookup.get(word, [f"<t>{word}</t>"])
        if len(phrases) == 0:
            phrases = [f"<t>{word}</t>"]

        # Embed all synset phrases for this word
        word_embs = batch_embed(model, tokenizer, phrases, device, batch_size)
        # Average across synsets
        all_embeddings.append(word_embs.mean(axis=0))

    return np.array(all_embeddings, dtype=np.float32)


# =========================================================================
# ATE estimation (from train_g_triplet.py)
# =========================================================================

def cosine_sim_rowwise(a, b):
    """Row-wise cosine similarity between two (N, D) arrays."""
    dot = np.sum(a * b, axis=1)
    norm_a = np.linalg.norm(a, axis=1)
    norm_b = np.linalg.norm(b, axis=1)
    return dot / (norm_a * norm_b + 1e-10)


def compute_ate(def_emb, ans_emb, clue_emb):
    """Compute the cosine-based misdirection ATE.

    ATE = mean(cos(clue_emb, ans_emb) - cos(def_emb, ans_emb))

    A negative ATE means clue context pushes the definition embedding
    AWAY from the true answer — this is the misdirection signal. If
    learned g is working, its ATE should be less negative than stock
    CALE's (i.e., the model partially resists misdirection).

    Returns dict with ate, ate_se, ate_ci_lower, ate_ci_upper.
    """
    cos_clue_ans = cosine_sim_rowwise(clue_emb, ans_emb)
    cos_def_ans = cosine_sim_rowwise(def_emb, ans_emb)
    delta = cos_clue_ans - cos_def_ans

    ate = float(delta.mean())
    ate_se = float(delta.std() / np.sqrt(len(delta)))
    return {
        "ate": ate,
        "ate_se": ate_se,
        "ate_ci_lower": ate - 1.96 * ate_se,
        "ate_ci_upper": ate + 1.96 * ate_se,
    }


# =========================================================================
# Main
# =========================================================================

def main():
    args = parse_args()
    script_start = time.time()

    # --- Print prominent warning ---
    print()
    print("!" * 65)
    print("!  WARNING: This script evaluates the held-out test set.       !")
    print("!  Run this only once, after a final g has been selected       !")
    print("!  based on validation results. Running this during            !")
    print("!  experimentation compromises the validity of the causal      !")
    print("!  inference.                                                  !")
    print("!" * 65)
    print()

    # --- Require explicit confirmation ---
    response = input(
        "Type 'yes' to confirm you intend to open the test set: "
    ).strip().lower()
    if response != "yes":
        print("Aborted. Test set was not opened.")
        sys.exit(0)
    print()

    # --- Seeds and device ---
    set_seeds(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Paths ---
    data_dir = Path("data")
    emb_dir = data_dir / "embeddings"
    output_dir = data_dir / "learned_g" / args.dataset
    model_dir = output_dir / args.model_tag

    # --- Validate experiment directory exists ---
    if not model_dir.exists():
        print(f"ERROR: Model directory not found: {model_dir}")
        print(f"Available directories in {output_dir}:")
        if output_dir.exists():
            for p in sorted(output_dir.iterdir()):
                if p.is_dir():
                    print(f"  {p.name}/")
        sys.exit(1)

    # --- Load and verify ate_results.json ---
    results_path = output_dir / "ate_results.json"
    if not results_path.exists():
        print(f"ERROR: ate_results.json not found in {output_dir}")
        print("Run train_g_triplet.py first to produce validation results.")
        sys.exit(1)

    with open(results_path) as f:
        results = json.load(f)

    if results.get("test_evaluated", False):
        print("ERROR: test_evaluated is already true in ate_results.json.")
        print("The test set has already been opened for this experiment.")
        print("Re-running this script would compromise the causal inference.")
        print(f"Results file: {results_path}")
        sys.exit(1)

    # --- Print configuration ---
    print("=" * 65)
    print("Phase 2: Final Test-Set Evaluation of Learned g")
    print("=" * 65)
    print(f"Dataset:       {args.dataset}")
    print(f"Model tag:     {args.model_tag}")
    print(f"Device:        {device}")
    print(f"Output dir:    {output_dir}")
    if torch.cuda.is_available():
        print(f"GPU:           {torch.cuda.get_device_name(0)}")
    print()

    # =====================================================================
    # Section 2 — Load test set files
    # =====================================================================
    print("=" * 65)
    print("Section 2: Loading test set stock embeddings")
    print("=" * 65)

    test_indices = np.load(output_dir / "test_indices.npy")
    test_stock_def = np.load(output_dir / "test_stock_def_emb.npy")
    test_stock_ans = np.load(output_dir / "test_stock_ans_emb.npy")
    test_stock_clue = np.load(output_dir / "test_stock_clue_emb.npy")

    print(f"Test set: {len(test_indices):,} rows")
    print(f"  test_stock_def_emb:  {test_stock_def.shape}")
    print(f"  test_stock_ans_emb:  {test_stock_ans.shape}")
    print(f"  test_stock_clue_emb: {test_stock_clue.shape}")
    print()

    # =====================================================================
    # Section 3 — Load finalized model
    # =====================================================================
    print("=" * 65)
    print("Section 3: Loading finalized model")
    print("=" * 65)

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir)
    model = model.to(device)
    load_time = time.time() - t0

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded from: {model_dir}")
    print(f"Parameters: {n_params:,}")
    print(f"Hidden dim: {model.config.hidden_size}")
    print(f"Loaded in {load_time:.1f}s")
    print()

    # =====================================================================
    # Section 4 — Re-embed test set with learned g
    # =====================================================================
    print("=" * 65)
    print("Section 4: Re-embedding test set with learned g")
    print("=" * 65)

    # Load phrase CSVs for text-based re-embedding
    def_phrases_df = pd.read_csv(
        emb_dir / "definition_phrases.csv", keep_default_na=False
    )
    ans_phrases_df = pd.read_csv(
        emb_dir / "answer_phrases.csv", keep_default_na=False
    )
    clue_phrases_df = pd.read_csv(
        emb_dir / "clue_context_phrases.csv", keep_default_na=False
    )

    def_phrase_lookup = build_allsense_phrase_lookup(def_phrases_df)
    ans_phrase_lookup = build_allsense_phrase_lookup(ans_phrases_df)

    # Build clue-context phrase lookup: (clue_id, definition_wn) -> phrase
    clue_phrase_lookup = {
        (row["clue_id"], row["definition_wn"]): row["clue_context_phrase"]
        for _, row in clue_phrases_df.iterrows()
    }

    # Load dataset to map test indices back to words/clue_ids
    df_full = pd.read_parquet(data_dir / f"dataset_{args.dataset}.parquet")
    test_df = df_full.loc[test_indices]

    print(f"Re-embedding test set ({len(test_df):,} rows)...")
    t0 = time.time()

    learned_def_emb = embed_allsense(
        model, tokenizer, test_df["definition_wn"].tolist(),
        def_phrase_lookup, device, batch_size=64
    )
    learned_ans_emb = embed_allsense(
        model, tokenizer, test_df["answer_wn"].tolist(),
        ans_phrase_lookup, device, batch_size=64
    )

    clue_phrases = [
        clue_phrase_lookup.get(
            (row["clue_id"], row["definition_wn"]),
            f"<t>{row['definition_wn']}</t>"
        )
        for _, row in test_df.iterrows()
    ]
    learned_clue_emb = batch_embed(
        model, tokenizer, clue_phrases, device, batch_size=64
    )

    embed_time = time.time() - t0
    print(f"Test re-embedding done in {embed_time:.0f}s")

    # Save learned test embeddings
    np.save(output_dir / "test_learned_def_emb.npy", learned_def_emb)
    np.save(output_dir / "test_learned_ans_emb.npy", learned_ans_emb)
    np.save(output_dir / "test_learned_clue_emb.npy", learned_clue_emb)

    print(f"\nLearned test embeddings saved:")
    print(f"  test_learned_def_emb:  {learned_def_emb.shape}")
    print(f"  test_learned_ans_emb:  {learned_ans_emb.shape}")
    print(f"  test_learned_clue_emb: {learned_clue_emb.shape}")
    print()

    # =====================================================================
    # Section 5 — Compute final ATE
    # =====================================================================
    print("=" * 65)
    print("Section 5: Final ATE estimation (test set)")
    print("=" * 65)

    test_stock_ate = compute_ate(test_stock_def, test_stock_ans, test_stock_clue)
    test_learned_ate = compute_ate(learned_def_emb, learned_ans_emb, learned_clue_emb)

    # Print results table
    print(f"\n{'':20s} {'ATE':>10s} {'SE':>10s} {'95% CI':>24s}")
    print("-" * 66)
    for label, result in [
        ("Test Stock CALE", test_stock_ate),
        ("Test Learned g", test_learned_ate),
    ]:
        print(f"{label:20s} {result['ate']:>10.4f} {result['ate_se']:>10.4f} "
              f"[{result['ate_ci_lower']:>10.4f}, {result['ate_ci_upper']:>10.4f}]")

    test_delta = test_learned_ate["ate"] - test_stock_ate["ate"]
    print(f"\n{'Delta(learned-stock)':20s} {test_delta:>10.4f}")

    # Include validation results from ate_results.json for comparison
    print(f"\nComparison with validation results:")
    print(f"  Val  Stock ATE:   {results['val_stock_ate']:>10.4f}")
    print(f"  Val  Learned ATE: {results['val_learned_ate']:>10.4f}")
    val_delta = results["val_learned_ate"] - results["val_stock_ate"]
    print(f"  Val  Delta:       {val_delta:>10.4f}")
    print(f"  Test Stock ATE:   {test_stock_ate['ate']:>10.4f}")
    print(f"  Test Learned ATE: {test_learned_ate['ate']:>10.4f}")
    print(f"  Test Delta:       {test_delta:>10.4f}")
    print()

    # =====================================================================
    # Section 6 — Update results and save
    # =====================================================================
    print("=" * 65)
    print("Section 6: Updating ate_results.json")
    print("=" * 65)

    import datetime

    results["test_stock_ate"] = test_stock_ate["ate"]
    results["test_stock_ate_se"] = test_stock_ate["ate_se"]
    results["test_stock_ate_ci"] = [
        test_stock_ate["ate_ci_lower"],
        test_stock_ate["ate_ci_upper"],
    ]
    results["test_learned_ate"] = test_learned_ate["ate"]
    results["test_learned_ate_se"] = test_learned_ate["ate_se"]
    results["test_learned_ate_ci"] = [
        test_learned_ate["ate_ci_lower"],
        test_learned_ate["ate_ci_upper"],
    ]
    results["test_evaluated"] = True
    results["test_evaluated_date"] = str(datetime.date.today())

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Updated: {results_path}")
    print(f"  test_evaluated: true")

    # Final summary
    total_time = time.time() - script_start
    minutes = int(total_time // 60)
    seconds = total_time % 60
    print(f"\nTotal runtime: {minutes}m {seconds:.1f}s")

    print(f"\nOutput files in {output_dir}:")
    for p in sorted(output_dir.iterdir()):
        if p.is_dir():
            print(f"  {p.name}/")
        else:
            size_mb = p.stat().st_size / (1024 * 1024)
            print(f"  {p.name:40s}  {size_mb:>8.1f} MB")

    print()
    print("=" * 65)
    print("IMPORTANT: Record these test results in FINDINGS.md before")
    print("drawing any conclusions. The test set is now spent for this")
    print("experiment — do not re-run this script.")
    print("=" * 65)
    print()


if __name__ == "__main__":
    main()
