"""
Phase 2 (GPU): Fine-tune CALE using triplet margin loss to learn a custom
codebook function g for misdirection analysis.

This script pioneers an approach originally developed by Nathan Cantwell in
notebooks/archive/09_learned_g_misdirection.ipynb. It adapts and extends
that work to run cleanly on Great Lakes using the prepared inputs from
09_learned_g_prep.ipynb.

Purpose:
    Loads pre-built triplets and stock embeddings, fine-tunes CALE using
    triplet margin loss, re-embeds the validation set with the learned g,
    estimates the misdirection ATE on the validation set, and saves all
    results. The test set is NOT touched by this script — test set
    evaluation is handled separately by evaluate_final_g.py, which is
    run exactly once after a final g has been selected on validation.

Inputs (all from data/learned_g/{dataset}/):
    train_triplets.npz — anchors, positives, negatives (N_triplets, 1024)
    val_indices.npy, val_stock_*.npy — validation set stock embeddings
    split_config.json — metadata from NB09

Outputs (saved to data/learned_g/{dataset}/):
    model_{method}_{tag}/ — fine-tuned model via save_pretrained()
    val_learned_*.npy — learned g embeddings for validation set
    ate_results.json — validation ATE estimates (stock vs learned)
    training_log.csv — per-step loss and learning rate

Key design decisions:
    - Pre-computed embeddings: NB09 extracted triplet embeddings from Phase 1
      .npy files on CPU. The DataLoader serves tensors directly — no
      tokenization during training. This is faster than Nathan's original
      approach (which re-tokenized phrases on every forward pass) and avoids
      inconsistency.
    - save_pretrained(): Model weights saved via HuggingFace's native method
      for portability across PyTorch versions (see CLAUDE.md Phase 2 Coding
      Conventions). Never use torch.save() for model weights.
    - Concept-aligned extraction: Following Nathan's approach and CALE's
      design, we extract embeddings by averaging hidden states within the
      <t></t> delimiter span, NOT using [CLS]. CALE's training objective
      specifically optimizes the delimited-span representation.
    - ATE estimation: Cosine-based ATE = mean(cos(clue_emb, ans_emb) -
      cos(def_emb, ans_emb)). A negative ATE means clue context pushes the
      definition embedding away from the true answer (misdirection). If
      learned g works, its ATE should be less negative than stock CALE's.

Usage:
    # Local testing with small sample
    python scripts/train_g_triplet.py --sample 500

    # Full run (submit via SLURM on Great Lakes)
    sbatch scripts/train_g_triplet.sh

    # Custom hyperparameters
    python scripts/train_g_triplet.py --epochs 5 --lr 1e-5 --margin 0.5

Author: Victoria Winters
Original approach: Nathan Cantwell
AI assistance: Claude Code (Anthropic)
"""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


# =========================================================================
# Constants
# =========================================================================

CALE_MODEL_NAME = "gabrielloiseau/CALE-MBERT-en"
EMBED_DIM = 1024
RANDOM_SEED = 42


# =========================================================================
# Section 1 — Setup and argument parsing
# =========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune CALE with triplet margin loss for learned g."
    )
    parser.add_argument(
        "--dataset", type=str, default="harder",
        help="Which data/learned_g/{dataset}/ folder to read from (default: harder)",
    )
    parser.add_argument(
        "--epochs", type=int, default=3,
        help="Number of fine-tuning epochs (default: 3)",
    )
    parser.add_argument(
        "--lr", type=float, default=2e-5,
        help="Learning rate (default: 2e-5)",
    )
    parser.add_argument(
        "--margin", type=float, default=1.0,
        help="Triplet loss margin (default: 1.0)",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.01,
        help="AdamW weight decay (default: 0.01)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Training batch size (default: 16)",
    )
    parser.add_argument(
        "--sample", type=int, default=0,
        help="If > 0, subsample this many triplets for fast testing (default: 0)",
    )
    return parser.parse_args()


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================================================================
# Section 4 — PyTorch Dataset
# =========================================================================

class TripletEmbeddingDataset(Dataset):
    """Wraps pre-computed embedding arrays for triplet training.

    Because NB09 pre-extracted the embeddings from the Phase 1 .npy files,
    the DataLoader serves float32 tensors directly — no tokenization happens
    during training. This is faster than Nathan's original approach (which
    re-tokenized phrases on every forward pass) and is possible because NB09
    already did the embedding extraction step.
    """

    def __init__(self, anchors, positives, negatives):
        self.anchors = torch.from_numpy(anchors).float()
        self.positives = torch.from_numpy(positives).float()
        self.negatives = torch.from_numpy(negatives).float()

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, idx):
        return {
            "anchor": self.anchors[idx],
            "positive": self.positives[idx],
            "negative": self.negatives[idx],
        }


# =========================================================================
# Section 7 — Concept-aligned embedding extraction
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
# Section 8 — ATE estimation
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

    # --- Seeds and device ---
    set_seeds(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Paths ---
    data_dir = Path("data")
    emb_dir = data_dir / "embeddings"
    output_dir = data_dir / "learned_g" / args.dataset

    # --- Model tag for output directory ---
    model_tag = (
        f"model_triplet_e{args.epochs}_lr{args.lr}_m{args.margin}"
    )

    # --- Print configuration ---
    print("=" * 65)
    print("Phase 2: Fine-tune CALE with Triplet Margin Loss")
    print("=" * 65)
    print(f"Dataset:       {args.dataset}")
    print(f"Epochs:        {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Margin:        {args.margin}")
    print(f"Weight decay:  {args.weight_decay}")
    print(f"Batch size:    {args.batch_size}")
    print(f"Sample:        {args.sample if args.sample > 0 else 'full dataset'}")
    print(f"Device:        {device}")
    print(f"Model tag:     {model_tag}")
    print(f"Output dir:    {output_dir}")
    if torch.cuda.is_available():
        print(f"GPU:           {torch.cuda.get_device_name(0)}")
    print()

    # =====================================================================
    # Section 2 — Load inputs
    # =====================================================================
    print("=" * 65)
    print("Section 2: Loading inputs")
    print("=" * 65)

    triplets = np.load(output_dir / "train_triplets.npz")
    anchors = triplets["anchors"]
    positives = triplets["positives"]
    negatives = triplets["negatives"]

    if args.sample > 0:
        n_sample = min(args.sample, len(anchors))
        rng = np.random.RandomState(RANDOM_SEED)
        idx = rng.choice(len(anchors), n_sample, replace=False)
        anchors = anchors[idx]
        positives = positives[idx]
        negatives = negatives[idx]
        print(f"SAMPLE MODE: using {n_sample} of {triplets['anchors'].shape[0]} triplets")

    print(f"Triplets: {len(anchors):,}")
    print(f"Shapes: anchors {anchors.shape}, positives {positives.shape}, "
          f"negatives {negatives.shape}")

    # Report triplet ordering — this is a known issue for the harder dataset
    # (see FINDINGS.md Phase 2 section). Distractors selected by cosine
    # similarity to definitions are often MORE similar to definitions than
    # true answers, inverting the expected triplet ordering.
    sim_pos = cosine_sim_rowwise(anchors, positives)
    sim_neg = cosine_sim_rowwise(anchors, negatives)
    frac_correct = float((sim_pos > sim_neg).mean())
    print(f"\nTriplet ordering check:")
    print(f"  Mean cos(anchor, positive): {sim_pos.mean():.4f}")
    print(f"  Mean cos(anchor, negative): {sim_neg.mean():.4f}")
    print(f"  Fraction correctly ordered: {frac_correct:.3f}")
    if frac_correct < 0.25:
        print(f"\n  ⚠ WARNING: Only {frac_correct:.1%} of triplets have correct ordering.")
        print(f"  This means the training signal is weak — most triplets contribute")
        print(f"  zero gradient because the negative is already farther from the anchor")
        print(f"  than the positive. This is a known consequence of using cosine-")
        print(f"  similarity distractors (see FINDINGS.md, Decision 20). The model")
        print(f"  may still learn from the {frac_correct:.1%} of informative triplets.")
    print()

    # Load stock embeddings for validation only.
    # The test set is kept completely locked during the experimental cycle.
    # Loading it here, even without looking at the results, creates
    # unnecessary risk of data leakage influencing experimental decisions.
    # Test set evaluation happens once only, in evaluate_final_g.py, after
    # a final g has been selected on validation performance.
    val_stock_def = np.load(output_dir / "val_stock_def_emb.npy")
    val_stock_ans = np.load(output_dir / "val_stock_ans_emb.npy")
    val_stock_clue = np.load(output_dir / "val_stock_clue_emb.npy")
    val_indices = np.load(output_dir / "val_indices.npy")
    print(f"Validation set: {len(val_indices):,} rows")

    with open(output_dir / "split_config.json") as f:
        split_config = json.load(f)
    print()

    # =====================================================================
    # Section 3 — Load CALE model
    # =====================================================================
    # We load with AutoModel/AutoTokenizer rather than sentence-transformers
    # because we need direct access to the model parameters for fine-tuning
    # with a custom loss function. sentence-transformers abstracts away the
    # forward pass and pooling, making it difficult to train with triplet
    # loss on pre-computed embeddings while also needing the model for
    # re-embedding later.
    print("=" * 65)
    print("Section 3: Loading CALE model")
    print("=" * 65)

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(CALE_MODEL_NAME)
    model = AutoModel.from_pretrained(CALE_MODEL_NAME)
    model = model.to(device)
    load_time = time.time() - t0

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {CALE_MODEL_NAME}")
    print(f"Parameters: {n_params:,} total, {n_trainable:,} trainable")
    print(f"Hidden dim: {model.config.hidden_size}")
    print(f"Loaded in {load_time:.1f}s")
    print()

    # =====================================================================
    # Section 4 — DataLoader
    # =====================================================================
    print("=" * 65)
    print("Section 4: Creating DataLoader")
    print("=" * 65)

    dataset = TripletEmbeddingDataset(anchors, positives, negatives)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,  # avoid partial batches that destabilize training
    )
    print(f"Dataset:          {len(dataset):,} triplets")
    print(f"Batches per epoch: {len(loader):,}")
    print(f"Batch size:       {args.batch_size}")
    print()

    # =====================================================================
    # Section 5 — Training configuration
    # =====================================================================
    # Following Nathan Cantwell's approach in
    # notebooks/archive/09_learned_g_misdirection.ipynb:
    #
    # - TripletMarginLoss: standard metric learning objective that pushes
    #   positives closer to anchors than negatives by at least `margin`.
    # - AdamW: weight decay prevents the model from drifting too far from
    #   CALE's pretrained distribution during fine-tuning.
    # - Linear warmup: stabilizes early training when gradients are large
    #   from the randomly-initialized loss landscape.
    # - Gradient clipping: prevents exploding gradients, which are common
    #   when fine-tuning large transformers with small learning rates.
    print("=" * 65)
    print("Section 5: Training configuration")
    print("=" * 65)

    loss_fn = nn.TripletMarginLoss(margin=args.margin, p=2)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    total_steps = len(loader) * args.epochs
    warmup_steps = int(0.1 * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, 1.0 - (step - warmup_steps) / (total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"Loss:          TripletMarginLoss(margin={args.margin}, p=2)")
    print(f"Optimizer:     AdamW(lr={args.lr}, weight_decay={args.weight_decay})")
    print(f"Total steps:   {total_steps:,}")
    print(f"Warmup steps:  {warmup_steps:,}")
    print(f"Grad clipping: max_norm=1.0")
    print()

    # =====================================================================
    # Section 6 — Training loop
    # =====================================================================
    # Note: Because we use pre-computed embeddings (not raw text), the
    # training loop operates directly on embedding tensors. The triplet
    # loss computes distances in the 1024-dim space without any forward
    # pass through the transformer — the model is only used later for
    # re-embedding. This means training is very fast, but the model
    # weights are NOT updated by the triplet loss on pre-computed
    # embeddings alone.
    #
    # To actually fine-tune the model, we need to pass raw text through
    # the transformer during training. However, NB09 pre-computed the
    # embeddings to avoid GPU tokenization overhead. We adopt a hybrid
    # approach: use pre-computed embeddings to compute the loss signal,
    # but we still need the model for re-embedding afterward.
    #
    # UPDATE: The correct approach (following Nathan's notebook) is to
    # pass the raw CALE phrases through the model during training so
    # gradients flow through the transformer weights. We load the phrase
    # CSVs and construct a text-based dataset for training.
    print("=" * 65)
    print("Section 6: Training loop")
    print("=" * 65)

    # Load phrase CSVs for text-based training
    # We need the original CALE-format phrases to pass through the model
    # so gradients update the transformer weights.
    def_phrases_df = pd.read_csv(
        emb_dir / "definition_phrases.csv", keep_default_na=False
    )
    ans_phrases_df = pd.read_csv(
        emb_dir / "answer_phrases.csv", keep_default_na=False
    )
    clue_phrases_df = pd.read_csv(
        emb_dir / "clue_context_phrases.csv", keep_default_na=False
    )

    # Build lookup: word → list of all synset phrases (for allsense averaging)
    def build_allsense_phrase_lookup(phrases_df):
        """Map each word to the list of all its synset phrases."""
        lookup = {}
        for word, group in phrases_df.groupby("word"):
            lookup[word] = group.sort_values("synset_idx")["phrase"].tolist()
        return lookup

    def_phrase_lookup = build_allsense_phrase_lookup(def_phrases_df)
    ans_phrase_lookup = build_allsense_phrase_lookup(ans_phrases_df)

    # Build clue-context phrase lookup: (clue_id, definition_wn) → phrase
    clue_phrase_lookup = {
        (row["clue_id"], row["definition_wn"]): row["clue_context_phrase"]
        for _, row in clue_phrases_df.iterrows()
    }

    # Load dataset to map indices back to words/clue_ids
    df_full = pd.read_parquet(data_dir / f"dataset_{args.dataset}.parquet")

    # Build text-based triplet dataset from the training split.
    # For each pre-computed triplet, we need to reconstruct the text phrases.
    # Training rows are real rows whose indices are NOT in the validation set
    # and NOT in the test set. We load test_indices only for this exclusion —
    # no test embeddings or results are computed.
    test_indices_for_exclusion = np.load(output_dir / "test_indices.npy")
    excluded = set(val_indices) | set(test_indices_for_exclusion)
    train_real = df_full[
        (~df_full.index.isin(excluded)) & (df_full["label"] == 1)
    ]

    # Build anchor/positive/negative text arrays
    print("Building text-based triplets from phrase CSVs...")
    anchor_texts = []
    positive_texts = []
    negative_texts = []

    # Distractor lookup: (clue_id, definition_wn) → distractor row
    dist_df = df_full[df_full["label"] == 0]
    dist_lookup = {}
    for _, row in dist_df.iterrows():
        dist_lookup[(row["clue_id"], row["definition_wn"])] = row

    skipped = 0
    for _, row in train_real.iterrows():
        defn = row["definition_wn"]
        ans = row["answer_wn"]
        key = (row["clue_id"], row["definition_wn"])

        # Find matched distractor
        dist_row = dist_lookup.get(key)
        if dist_row is None:
            skipped += 1
            continue

        distractor_word = dist_row["distractor_source"]

        # Look up phrases — need all three
        if defn not in def_phrase_lookup or ans not in ans_phrase_lookup:
            skipped += 1
            continue
        if distractor_word not in ans_phrase_lookup:
            skipped += 1
            continue

        # Anchor: use the first definition synset phrase (most common sense).
        # For allsense we would average across synsets, but for training we
        # use the common-sense phrase as a representative anchor.
        anchor_texts.append(def_phrase_lookup[defn][0])
        positive_texts.append(ans_phrase_lookup[ans][0])
        negative_texts.append(ans_phrase_lookup[distractor_word][0])

    print(f"Text triplets built: {len(anchor_texts):,} (skipped {skipped:,})")

    if args.sample > 0:
        n_sample = min(args.sample, len(anchor_texts))
        rng = np.random.RandomState(RANDOM_SEED)
        idx = rng.choice(len(anchor_texts), n_sample, replace=False)
        anchor_texts = [anchor_texts[i] for i in idx]
        positive_texts = [positive_texts[i] for i in idx]
        negative_texts = [negative_texts[i] for i in idx]
        print(f"SAMPLE MODE: using {n_sample} text triplets")

    # Text-based Dataset and DataLoader
    class TripletTextDataset(Dataset):
        """Dataset returning raw text triplets for training through the model."""

        def __init__(self, anchors, positives, negatives):
            self.anchors = anchors
            self.positives = positives
            self.negatives = negatives

        def __len__(self):
            return len(self.anchors)

        def __getitem__(self, idx):
            return {
                "anchor": self.anchors[idx],
                "positive": self.positives[idx],
                "negative": self.negatives[idx],
            }

    text_dataset = TripletTextDataset(anchor_texts, positive_texts, negative_texts)
    text_loader = DataLoader(
        text_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    print(f"Text DataLoader: {len(text_loader):,} batches per epoch")

    # Recompute total steps and scheduler for text-based training
    total_steps = len(text_loader) * args.epochs
    warmup_steps = int(0.1 * total_steps)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    def lr_lambda_text(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, 1.0 - (step - warmup_steps) / (total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_text)

    print(f"Total training steps: {total_steps:,}")
    print(f"Warmup steps: {warmup_steps:,}")

    # Training loop
    training_log = []
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []
        t0_epoch = time.time()

        for step, batch in enumerate(tqdm(text_loader,
                                          desc=f"Epoch {epoch+1}/{args.epochs}")):
            # Forward pass: extract concept-aligned embeddings for all three
            # components by passing raw text through the model. This is where
            # gradients flow through the transformer weights.
            z_anchor = extract_concept_embedding(
                model, tokenizer, batch["anchor"], device
            )
            z_positive = extract_concept_embedding(
                model, tokenizer, batch["positive"], device
            )
            z_negative = extract_concept_embedding(
                model, tokenizer, batch["negative"], device
            )

            loss = loss_fn(z_anchor, z_positive, z_negative)

            loss.backward()
            # Gradient clipping prevents exploding gradients during fine-tuning
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_losses.append(loss.item())
            global_step += 1

            # Log every 100 steps
            if (step + 1) % 100 == 0:
                avg_loss = np.mean(epoch_losses[-100:])
                lr_current = scheduler.get_last_lr()[0]
                training_log.append({
                    "epoch": epoch + 1,
                    "step": global_step,
                    "loss": avg_loss,
                    "lr": lr_current,
                })
                print(f"  Step {global_step:>5d} | "
                      f"Loss: {avg_loss:.4f} | LR: {lr_current:.2e}")

        epoch_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        epoch_time = time.time() - t0_epoch
        print(f"\nEpoch {epoch+1} complete: avg loss = {epoch_loss:.4f}, "
              f"time = {epoch_time:.0f}s")

        # Log final epoch stats
        training_log.append({
            "epoch": epoch + 1,
            "step": global_step,
            "loss": epoch_loss,
            "lr": scheduler.get_last_lr()[0],
        })

    # Save fine-tuned model using save_pretrained() — NEVER use torch.save()
    # for model weights. See CLAUDE.md Phase 2 Coding Conventions.
    model_path = output_dir / model_tag
    model_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"\nModel saved to: {model_path}")

    # Save training log
    log_path = output_dir / "training_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "step", "loss", "lr"])
        writer.writeheader()
        writer.writerows(training_log)
    print(f"Training log saved to: {log_path}")
    print()

    # =====================================================================
    # Section 7 — Re-embed validation set with learned g
    # =====================================================================
    # Only the validation set is re-embedded here. The test set is kept
    # locked — it will be re-embedded in evaluate_final_g.py after a
    # single g has been selected based on validation performance.
    print("=" * 65)
    print("Section 7: Re-embedding validation set with learned g")
    print("=" * 65)

    # For re-embedding, we need the original CALE-format phrases for each
    # val row. We look them up from the phrase CSVs using the word strings
    # and clue_ids from the dataset.

    def get_allsense_phrase(word, phrase_lookup):
        """Get the common-sense phrase for a word (first synset).

        For re-embedding we use the same phrase that was used to compute
        allsense embeddings in Phase 1: average across all synset phrases.
        But the model processes one phrase at a time, so we return all
        phrases and average the embeddings afterward.
        """
        return phrase_lookup.get(word, [f"<t>{word}</t>"])

    def embed_allsense(model, tokenizer, words, phrase_lookup, device,
                       batch_size=64):
        """Embed words using allsense averaging: for each word, embed all
        synset phrases and average the resulting vectors.

        This replicates Phase 1's allsense embedding approach (slot 0 of
        definition_embeddings.npy / answer_embeddings.npy).
        """
        all_embeddings = []

        for word in tqdm(words, desc="Allsense embedding", leave=False):
            phrases = get_allsense_phrase(word, phrase_lookup)
            if len(phrases) == 0:
                phrases = [f"<t>{word}</t>"]

            # Embed all synset phrases for this word
            word_embs = batch_embed(model, tokenizer, phrases, device, batch_size)
            # Average across synsets
            all_embeddings.append(word_embs.mean(axis=0))

        return np.array(all_embeddings, dtype=np.float32)

    # Get word lists and clue phrases for the validation set
    val_df = df_full.loc[val_indices]

    print(f"\nRe-embedding validation set ({len(val_df):,} rows)...")
    t0 = time.time()
    val_learned_def = embed_allsense(
        model, tokenizer, val_df["definition_wn"].tolist(),
        def_phrase_lookup, device, batch_size=64
    )
    val_learned_ans = embed_allsense(
        model, tokenizer, val_df["answer_wn"].tolist(),
        ans_phrase_lookup, device, batch_size=64
    )
    val_clue_phrases = [
        clue_phrase_lookup.get(
            (row["clue_id"], row["definition_wn"]),
            f"<t>{row['definition_wn']}</t>"
        )
        for _, row in val_df.iterrows()
    ]
    val_learned_clue = batch_embed(
        model, tokenizer, val_clue_phrases, device, batch_size=64
    )
    print(f"  Validation re-embedding done in {time.time() - t0:.0f}s")

    # Save learned validation embeddings
    np.save(output_dir / "val_learned_def_emb.npy", val_learned_def)
    np.save(output_dir / "val_learned_ans_emb.npy", val_learned_ans)
    np.save(output_dir / "val_learned_clue_emb.npy", val_learned_clue)

    print(f"\nLearned embeddings saved:")
    print(f"  val_learned_def_emb:  {val_learned_def.shape}")
    print(f"  val_learned_ans_emb:  {val_learned_ans.shape}")
    print(f"  val_learned_clue_emb: {val_learned_clue.shape}")
    print()

    # =====================================================================
    # Section 8 — ATE estimation
    # =====================================================================
    print("=" * 65)
    print("Section 8: ATE estimation")
    print("=" * 65)

    # Compute ATE on validation set only. Test ATE will be computed in
    # evaluate_final_g.py after a final g has been selected.
    val_stock_ate = compute_ate(val_stock_def, val_stock_ans, val_stock_clue)
    val_learned_ate = compute_ate(val_learned_def, val_learned_ans, val_learned_clue)

    # Print results table
    print(f"\n{'':20s} {'ATE':>10s} {'SE':>10s} {'95% CI':>24s}")
    print("-" * 66)
    for label, result in [
        ("Val Stock CALE", val_stock_ate),
        ("Val Learned g", val_learned_ate),
    ]:
        print(f"{label:20s} {result['ate']:>10.4f} {result['ate_se']:>10.4f} "
              f"[{result['ate_ci_lower']:>10.4f}, {result['ate_ci_upper']:>10.4f}]")

    val_delta = val_learned_ate["ate"] - val_stock_ate["ate"]
    print(f"\n{'Δ(learned-stock)':20s} {val_delta:>10.4f}")

    # Interpretation
    print(f"\nInterpretation:")
    print(f"  Negative ATE = misdirection (clue context hurts retrieval)")
    print(f"  Less negative = model partially resists misdirection")
    if val_delta > 0:
        print(f"  → Learned g shows LESS misdirection on validation "
              f"(Δ = {val_delta:+.4f})")
    else:
        print(f"  → Learned g shows MORE misdirection on validation "
              f"(Δ = {val_delta:+.4f})")
        print(f"    This may indicate the triplet ordering problem limited learning,")
        print(f"    or that fine-tuning amplified contextual sensitivity.")
    print(f"\n  Test ATE is NOT computed here — run evaluate_final_g.py after")
    print(f"  selecting a final g based on validation results.")
    print()

    # =====================================================================
    # Section 9 — Save results
    # =====================================================================
    print("=" * 65)
    print("Section 9: Saving results")
    print("=" * 65)

    import datetime

    results = {
        "dataset": args.dataset,
        "model_tag": model_tag,
        "epochs": args.epochs,
        "lr": args.lr,
        "margin": args.margin,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "sample": args.sample,
        "n_train_triplets": len(anchor_texts),
        "fraction_correct_ordering": frac_correct,
        "val_stock_ate": val_stock_ate["ate"],
        "val_stock_ate_se": val_stock_ate["ate_se"],
        "val_stock_ate_ci": [val_stock_ate["ate_ci_lower"],
                             val_stock_ate["ate_ci_upper"]],
        "val_learned_ate": val_learned_ate["ate"],
        "val_learned_ate_se": val_learned_ate["ate_se"],
        "val_learned_ate_ci": [val_learned_ate["ate_ci_lower"],
                               val_learned_ate["ate_ci_upper"]],
        "test_evaluated": False,
        "date_run": str(datetime.date.today()),
    }

    results_path = output_dir / "ate_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"ATE results saved to: {results_path}")

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

    print("\nDone.")


if __name__ == "__main__":
    main()
