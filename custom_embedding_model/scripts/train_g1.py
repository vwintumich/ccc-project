"""
Fine-tune g_stock (CALE-MBERT-en) with triplet margin loss to produce g_1.

Uses CALE's canonical mean pooling (Decision 20) — attention-masked mean over
all non-padding tokens of `last_hidden_state`, equivalent to
`SentenceTransformer.encode()`. This is the corrected version of
`train_g1_tokenspan.py`, which used non-standard token span extraction; the
two scripts are identical apart from the extraction method, so the Stage 5
comparison isolates the effect of pooling choice alone.

Reads one input: a committed triplet CSV (by default `data/triplets/g1.csv`)
with columns (clue_id, definition, answer_wn, distractor_wn, anchor, positive,
negative). The source phrase files and dataset_harder.parquet are NOT needed
on Great Lakes — all phrase text is already materialized in the CSV.

Usage (typical Great Lakes submission, via SLURM):
    python scripts/train_g1.py \
        --input data/triplets/g1.csv \
        --output-dir models/g1 \
        --epochs 3 \
        --batch-size 32 \
        --lr 2e-5 \
        --margin 1.0

Smoke test on a small sample (~5 min on GPU, ~20 min on CPU):
    python scripts/train_g1.py \
        --input data/triplets/g1.csv \
        --output-dir models/g1_smoke \
        --epochs 1 --batch-size 8 --sample 200

Outputs to --output-dir:
    model/                           — HuggingFace save_pretrained() format
                                        (model weights + tokenizer, loadable
                                        by AutoModel and SentenceTransformer)
    model_epoch{n}.pt                — per-epoch recovery checkpoints
                                        (state_dict + optimizer state)
    training_log.json                — per-step loss history and hyperparams
"""

# =============================================================================
# §1 — Imports and configuration
# =============================================================================

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.amp import GradScaler, autocast

import transformers
from transformers import AutoModel, AutoTokenizer

# Per Decision 19: print environment versions at startup so the SLURM log
# permanently records the exact versions that produced the committed weights.
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

# CALE model identifier — 1024-dim, uses <t></t> delimiters for concept-aligned extraction
MODEL_NAME = "gabrielloiseau/CALE-MBERT-en"

# Reduces GPU memory fragmentation during long training runs (NB 09 §5.2B)
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")


# =============================================================================
# §2 — CLI arguments
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune CALE with triplet margin loss (g_1, Step A)"
    )
    parser.add_argument("--input", type=Path, required=True,
                        help="Path to triplets/g1.csv")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory for model weights, checkpoints, training log")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--margin", type=float, default=1.0,
                        help="Triplet margin alpha (Decision 13: 1.0)")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-accum", type=int, default=1,
                        help="Gradient accumulation steps (increase for smaller batches)")
    parser.add_argument("--warmup-fraction", type=float, default=0.1,
                        help="Fraction of total steps spent on linear LR warmup")
    parser.add_argument("--max-length", type=int, default=128,
                        help="Tokenizer max_length (NB 09 used 128)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample", type=int, default=0,
                        help="If > 0, sample this many unique (definition, answer_wn) "
                             "pairs for a smoke test")
    return parser.parse_args()


# =============================================================================
# §3 — Load triplet data
# =============================================================================

def load_triplets(args: argparse.Namespace) -> pd.DataFrame:
    """Load g1.csv, validate schema, optionally sample for smoke testing."""

    # keep_default_na=False: "nan" (grandmother) is a valid crossword word;
    # without this flag pandas silently converts it to NaN and downstream
    # tokenization sees a NaN string.
    df = pd.read_csv(args.input, keep_default_na=False, na_values=[""])

    required = {"clue_id", "definition", "answer_wn", "distractor_wn",
                "anchor", "positive", "negative"}
    missing = required - set(df.columns)
    assert not missing, f"Missing columns in {args.input}: {missing}"

    # Null phrase text would produce meaningless embeddings; fail loudly here
    # rather than hours into training.
    for col in ("anchor", "positive", "negative"):
        n_null = df[col].isna().sum()
        assert n_null == 0, f"Found {n_null} null values in column '{col}'"

    print(f"Loaded triplet file: {args.input}")
    print(f"  Total rows: {len(df):,}")

    if args.sample > 0:
        # Sample at the (definition, answer_wn) pair level so a smoke run
        # covers a realistic variety of pair types, not just the first N rows.
        pair_keys = df[["definition", "answer_wn"]].drop_duplicates()
        n_pairs = min(args.sample, len(pair_keys))
        sampled = pair_keys.sample(n=n_pairs, random_state=args.seed)
        df = df.merge(sampled, on=["definition", "answer_wn"], how="inner").copy()
        print(f"  SAMPLE MODE: {n_pairs:,} unique pairs → {len(df):,} triplet rows")

    return df.reset_index(drop=True)


# =============================================================================
# §4 — PyTorch Dataset
# =============================================================================

class TripletDataset(Dataset):
    """Triplet dataset returning raw text strings.

    Tokenization is deferred to the training loop so that padding and
    truncation happen at the batch level (NB 09 §3.1). Each __getitem__
    returns a dict with keys 'anchor', 'positive', 'negative'.
    """

    def __init__(self, dataframe: pd.DataFrame):
        # Materialize as plain numpy arrays so indexing is cheap and no pandas
        # overhead is incurred per sample.
        self.anchors = dataframe["anchor"].values
        self.positives = dataframe["positive"].values
        self.negatives = dataframe["negative"].values

    def __len__(self) -> int:
        return len(self.anchors)

    def __getitem__(self, idx: int) -> dict:
        return {
            "anchor": self.anchors[idx],
            "positive": self.positives[idx],
            "negative": self.negatives[idx],
        }


# =============================================================================
# §5 — Mean-pooled embedding extraction (canonical)
# =============================================================================
#
# Per Decision 20, CALE's canonical pooling is an attention-masked mean over
# `last_hidden_state` for all non-padding tokens — the same operation that
# `SentenceTransformer.encode()` performs internally. The `<t></t>` delimiters
# remain in the tokenized input and guide attention during the forward pass;
# the pooling itself is just the average over the resulting hidden states.
# Gradients flow through this operation, which is why we use AutoModel rather
# than SentenceTransformer (the latter would wrap the model in an inference-
# only pipeline).


def extract_meanpool_embedding(model, tokenizer, texts, device, max_length: int = 128):
    """Mean-pooled embedding matching SentenceTransformer.encode() behavior.

    Averages last_hidden_state over all non-padding tokens using the
    attention mask — CALE's canonical pooling (Decision 20). The <t></t>
    delimiters are still present in the input and guide the attention
    patterns during the forward pass; the pooling itself just averages
    the resulting hidden states.

    Gradients flow through this operation, enabling fine-tuning.
    """
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)

    outputs = model(**encoded)
    hidden_states = outputs.last_hidden_state          # (batch, seq_len, dim)
    # unsqueeze(-1) broadcasts the mask across the hidden dim so padding
    # tokens contribute 0 to the sum and 0 to the count.
    mask = encoded["attention_mask"].unsqueeze(-1).to(hidden_states.dtype)
    summed = (hidden_states * mask).sum(dim=1)         # (batch, dim)
    counts = mask.sum(dim=1)                           # (batch, 1)
    return summed / counts                             # (batch, dim)


# =============================================================================
# §6 — Training loop
# =============================================================================

def train(args: argparse.Namespace) -> None:
    # --- Reproducibility seeds ---
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU:    {gpu_name} ({vram_gb:.1f} GB VRAM)")
    print()

    # --- Load triplets ---
    df = load_triplets(args)

    dataset = TripletDataset(df)
    # drop_last=True avoids an underfilled final batch, which destabilizes
    # gradient magnitudes (NB 09 §3.1). num_workers=0 keeps memory predictable
    # and avoids Colab/SLURM fork issues that arise with workers > 0.
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    print(f"Triplet dataset: {len(dataset):,} rows")
    print(f"Batches per epoch: {len(loader):,} (batch size = {args.batch_size})")
    print()

    # --- Load model and tokenizer ---
    print(f"Loading model: {MODEL_NAME} ...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    # Trade compute for memory: recompute activations during backward instead
    # of caching them. Critical for fitting batch_size=32 on a 16GB GPU.
    model.gradient_checkpointing_enable()
    model = model.to(device)
    model.train()
    print(f"Model loaded in {time.time() - t0:.1f}s")

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    hidden_dim = model.config.hidden_size
    print(f"Parameters:      {n_params:,} total, {n_trainable:,} trainable")
    print(f"Hidden dim:      {hidden_dim}")
    assert hidden_dim == 1024, f"Expected 1024-dim CALE; got {hidden_dim}"
    print()

    # --- Loss, optimizer, scheduler ---
    triplet_loss_fn = nn.TripletMarginLoss(margin=args.margin, p=2)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Linear warmup (first warmup_fraction of steps) then linear decay to 0.
    # Warmup stabilizes early training when pretrained weights are still
    # producing large gradients.
    total_steps = len(loader) * args.epochs
    warmup_steps = max(1, int(args.warmup_fraction * total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, 1.0 - (step - warmup_steps) / max(1, total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Suppress the known false-positive warning about lr_scheduler.step() being
    # called before optimizer.step(). The call ordering below is correct
    # (optimizer.step() then scheduler.step()), but PyTorch cannot distinguish
    # the first iteration from a genuine ordering bug and warns unconditionally.
    warnings.filterwarnings(
        "ignore",
        message=r"Detected call of `lr_scheduler\.step\(\)` before `optimizer\.step\(\)`.*",
    )

    print("Training configuration:")
    print(f"  Learning rate:   {args.lr}")
    print(f"  Weight decay:    {args.weight_decay}")
    print(f"  Margin:          {args.margin}")
    print(f"  Epochs:          {args.epochs}")
    print(f"  Grad accum:      {args.grad_accum}")
    print(f"  Total steps:     {total_steps:,}")
    print(f"  Warmup steps:    {warmup_steps:,}")
    print(f"  Max length:      {args.max_length}")
    print(f"  Seed:            {args.seed}")
    print()

    # --- Mixed precision ---
    scaler = GradScaler()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    training_log = []
    per_epoch_loss = []
    wall_start = time.time()

    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []
        t0_epoch = time.time()
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(loader):
            # Forward pass in fp16 to roughly halve activation memory while
            # keeping the loss/optimizer state in fp32 (handled by GradScaler).
            with autocast(device_type=device.type, dtype=torch.float16):
                z_anchor = extract_meanpool_embedding(
                    model, tokenizer, batch["anchor"], device, args.max_length
                )
                z_positive = extract_meanpool_embedding(
                    model, tokenizer, batch["positive"], device, args.max_length
                )
                z_negative = extract_meanpool_embedding(
                    model, tokenizer, batch["negative"], device, args.max_length
                )

                loss = triplet_loss_fn(z_anchor, z_positive, z_negative)
                if args.grad_accum > 1:
                    loss = loss / args.grad_accum

            scaler.scale(loss).backward()

            # Only step the optimizer after grad_accum micro-batches, to
            # simulate a larger effective batch without extra memory.
            if (step + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            # Detach to a Python float before recording — holding autograd-
            # tracked tensors in a Python list leaks memory across steps.
            detached_loss = float(loss.detach().item())
            epoch_losses.append(detached_loss * args.grad_accum)

            # Free the large intermediate tensors explicitly. Without this,
            # the next forward pass may OOM on memory-tight GPUs.
            del z_anchor, z_positive, z_negative, loss

            if (step + 1) % 100 == 0:
                avg_loss = float(np.mean(epoch_losses[-100:]))
                lr_current = scheduler.get_last_lr()[0]
                training_log.append({
                    "epoch": epoch + 1,
                    "step": step + 1,
                    "loss": avg_loss,
                    "lr": lr_current,
                })
                print(f"  Epoch {epoch+1} | Step {step+1:>5d} | "
                      f"Loss: {avg_loss:.4f} | LR: {lr_current:.2e}")

        epoch_loss = float(np.mean(epoch_losses))
        epoch_time = time.time() - t0_epoch
        per_epoch_loss.append(epoch_loss)
        print(f"\nEpoch {epoch+1} complete: avg loss = {epoch_loss:.4f}, "
              f"time = {epoch_time:.0f}s")

        # --- Per-epoch recovery checkpoint ---
        # Written atomically so a killed save doesn't corrupt the file.
        ckpt_path = args.output_dir / f"model_epoch{epoch+1}.pt"
        tmp_ckpt = ckpt_path.with_suffix(".pt.tmp")
        torch.save({
            "epoch": int(epoch + 1),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": epoch_loss,
        }, tmp_ckpt)
        tmp_ckpt.rename(ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

    # --- Save final model in HuggingFace format ---
    # save_pretrained() produces a directory that is loadable by both
    # AutoModel.from_pretrained() and SentenceTransformer() — the latter is
    # what the downstream embedding scripts (Stage 4) use.
    model_dir = args.output_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    print(f"\nFinal model saved to: {model_dir}")

    # --- Save training log + hyperparameters ---
    log_path = args.output_dir / "training_log.json"
    tmp_log = log_path.with_suffix(".json.tmp")
    log_payload = {
        "model_name": MODEL_NAME,
        "input": str(args.input),
        "output_dir": str(args.output_dir),
        "n_training_rows": len(df),
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "margin": args.margin,
            "weight_decay": args.weight_decay,
            "grad_accum": args.grad_accum,
            "warmup_fraction": args.warmup_fraction,
            "max_length": args.max_length,
            "seed": args.seed,
        },
        "per_epoch_loss": per_epoch_loss,
        "step_log": training_log,
        "total_runtime_seconds": time.time() - wall_start,
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "cuda": getattr(torch.version, "cuda", None),
        },
    }
    with open(tmp_log, "w") as f:
        json.dump(log_payload, f, indent=2)
    tmp_log.rename(log_path)
    print(f"Training log saved to: {log_path}")

    # --- Summary block for SLURM log → models/g1/README.md ---
    total_runtime = time.time() - wall_start
    print()
    print("=" * 72)
    print("SUMMARY (copy into models/g1/README.md and FINDINGS.md)")
    print("=" * 72)
    print(f"Base model:      {MODEL_NAME}")
    print(f"Pooling method:  meanpool")
    print(f"Triplet input:   {args.input}")
    print(f"Training rows:   {len(df):,}")
    print(f"Epochs:          {args.epochs}")
    print(f"Batch size:      {args.batch_size}")
    print(f"Learning rate:   {args.lr}")
    print(f"Margin:          {args.margin}")
    print(f"Weight decay:    {args.weight_decay}")
    print(f"Grad accum:      {args.grad_accum}")
    print(f"Warmup fraction: {args.warmup_fraction}")
    print(f"Seed:            {args.seed}")
    print(f"Per-epoch loss:  {per_epoch_loss}")
    print(f"Total runtime:   {total_runtime:.0f}s "
          f"({total_runtime / 60:.1f} min, {total_runtime / 3600:.2f} h)")
    if device.type == "cuda":
        print(f"GPU:             {torch.cuda.get_device_name(0)}")
    print(f"Python:          {sys.version.split()[0]}")
    print(f"torch:           {torch.__version__}")
    print(f"transformers:    {transformers.__version__}")
    print(f"Output dir:      {args.output_dir}")
    # List output directory contents with sizes so the SLURM log shows
    # exactly what was produced.
    for path in sorted(args.output_dir.rglob("*")):
        if path.is_file():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  {path.relative_to(args.output_dir)}  ({size_mb:.1f} MB)")
    print("=" * 72)


if __name__ == "__main__":
    args = parse_args()
    train(args)
