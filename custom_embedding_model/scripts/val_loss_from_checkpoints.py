"""
Compute validation triplet loss and accuracy at each g_1 training epoch.

The original `train_g1.py` tracked training loss per epoch but did not compute
validation loss. This script fills that gap post hoc by running a single
forward pass over `data/triplets/g1_val.csv` with each saved epoch checkpoint.
The results reveal whether g_1 was overfitting (training loss down, val loss
flat or up) and whether earlier epochs generalized better than the final one.

Inputs:
    --checkpoint-dir models/g1
        Must contain model_epoch{1,2,3}.pt and training_log.json.
    --val-triplets data/triplets/g1_val.csv
        Validation triplets (46,506 rows) — same schema as g1_train.csv.

Outputs (to --checkpoint-dir):
    val_loss_results.json     — per-epoch val loss / accuracy / mean+median
                                margin, plus the training losses for
                                side-by-side comparison
    Printed summary table on stdout (captured in the SLURM log).

The loss function, margin, max_length, and embedding extraction method are
read from training_log.json (or copied from train_g1.py verbatim) so the
validation loss is directly comparable to the per-epoch training loss.

Usage (typical Great Lakes submission, via SLURM):
    python scripts/val_loss_from_checkpoints.py \
        --checkpoint-dir models/g1 \
        --val-triplets data/triplets/g1_val.csv
"""

# =============================================================================
# §1 — Imports and configuration
# =============================================================================

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import transformers
from transformers import AutoModel, AutoTokenizer

# Per Decision 19: print environment versions so the SLURM log permanently
# records the exact versions that produced the validation numbers.
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

MODEL_NAME = "gabrielloiseau/CALE-MBERT-en"


# =============================================================================
# §2 — CLI arguments
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute validation loss/accuracy at each g_1 epoch checkpoint"
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True,
                        help="Directory containing model_epoch*.pt and training_log.json")
    parser.add_argument("--val-triplets", type=Path, required=True,
                        help="Path to g1_val.csv")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for the forward pass")
    return parser.parse_args()


# =============================================================================
# §3 — Triplet dataset (same as train_g1.py)
# =============================================================================

class TripletDataset(Dataset):
    """Returns raw anchor/positive/negative text; tokenization is batch-level."""

    def __init__(self, dataframe: pd.DataFrame):
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
# §4 — Mean-pooled embedding extraction (copied verbatim from train_g1.py)
# =============================================================================
#
# Decision 20: CALE's canonical pooling is an attention-masked mean over
# last_hidden_state for all non-padding tokens — the same op SentenceTransformer
# performs internally. The extraction must match training exactly so that the
# validation loss is directly comparable to the training loss.


def extract_meanpool_embedding(model, tokenizer, texts, device, max_length: int = 128):
    """Mean-pooled embedding matching SentenceTransformer.encode() behavior."""
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)

    outputs = model(**encoded)
    hidden_states = outputs.last_hidden_state
    mask = encoded["attention_mask"].unsqueeze(-1).to(hidden_states.dtype)
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1)
    return summed / counts


# =============================================================================
# §5 — Evaluate one checkpoint over the full validation set
# =============================================================================

def evaluate_checkpoint(
    ckpt_path: Path,
    loader: DataLoader,
    tokenizer,
    device: torch.device,
    margin: float,
    max_length: int,
) -> dict:
    """Load the checkpoint, run one forward pass over the val loader, return metrics."""
    print(f"Loading checkpoint: {ckpt_path.name}")
    t0 = time.time()

    model = AutoModel.from_pretrained(MODEL_NAME)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    triplet_loss_fn = nn.TripletMarginLoss(margin=margin, p=2)

    total_loss = 0.0
    n_triplets = 0
    n_correct = 0
    margins: list[float] = []

    with torch.no_grad():
        for batch in loader:
            z_anchor = extract_meanpool_embedding(
                model, tokenizer, batch["anchor"], device, max_length
            )
            z_positive = extract_meanpool_embedding(
                model, tokenizer, batch["positive"], device, max_length
            )
            z_negative = extract_meanpool_embedding(
                model, tokenizer, batch["negative"], device, max_length
            )

            batch_n = z_anchor.size(0)
            loss = triplet_loss_fn(z_anchor, z_positive, z_negative)
            # Multiply by batch size so a final partial batch is weighted
            # correctly when we divide by total triplets at the end.
            total_loss += float(loss.detach().item()) * batch_n

            cos_pos = F.cosine_similarity(z_anchor, z_positive, dim=-1)
            cos_neg = F.cosine_similarity(z_anchor, z_negative, dim=-1)
            batch_margins = (cos_pos - cos_neg).detach().cpu().numpy()
            margins.extend(batch_margins.tolist())
            n_correct += int((cos_pos > cos_neg).sum().item())
            n_triplets += batch_n

    margins_arr = np.asarray(margins, dtype=np.float64)
    metrics = {
        "checkpoint": ckpt_path.name,
        "n_triplets": n_triplets,
        "val_loss": total_loss / n_triplets,
        "val_accuracy": n_correct / n_triplets,
        "val_mean_margin": float(margins_arr.mean()),
        "val_median_margin": float(np.median(margins_arr)),
        "eval_seconds": time.time() - t0,
    }

    # Release GPU memory before the next checkpoint loads the full model again.
    del model, checkpoint
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(
        f"  n={metrics['n_triplets']:,}  "
        f"val_loss={metrics['val_loss']:.4f}  "
        f"acc={metrics['val_accuracy']*100:.1f}%  "
        f"mean_margin={metrics['val_mean_margin']:.4f}  "
        f"median_margin={metrics['val_median_margin']:.4f}  "
        f"time={metrics['eval_seconds']:.0f}s"
    )
    print()
    return metrics


# =============================================================================
# §6 — Main
# =============================================================================

def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU:    {gpu_name} ({vram_gb:.1f} GB VRAM)")
    print()

    # --- Load training log and read canonical hyperparameters ---
    log_path = args.checkpoint_dir / "training_log.json"
    assert log_path.exists(), f"training_log.json not found at {log_path}"
    with open(log_path) as f:
        training_log = json.load(f)

    hparams = training_log["hyperparameters"]
    margin = float(hparams["margin"])
    max_length = int(hparams["max_length"])
    per_epoch_train_loss = list(training_log["per_epoch_loss"])

    print("Hyperparameters read from training_log.json:")
    print(f"  margin:     {margin}")
    print(f"  max_length: {max_length}")
    print(f"  epochs logged: {len(per_epoch_train_loss)}")
    print()

    # --- Discover epoch checkpoints in numeric order ---
    ckpt_paths = sorted(
        args.checkpoint_dir.glob("model_epoch*.pt"),
        key=lambda p: int(p.stem.replace("model_epoch", "")),
    )
    assert len(ckpt_paths) == len(per_epoch_train_loss), (
        f"Found {len(ckpt_paths)} checkpoints but training_log.json has "
        f"{len(per_epoch_train_loss)} epochs of training loss — these must match."
    )
    print(f"Found {len(ckpt_paths)} epoch checkpoints:")
    for p in ckpt_paths:
        print(f"  {p.name}")
    print()

    # --- Load validation triplets ---
    # keep_default_na=False: "nan" (grandmother) is a valid crossword word.
    df = pd.read_csv(args.val_triplets, keep_default_na=False, na_values=[""])
    assert len(df) == 46506, f"Expected 46,506 validation triplets, got {len(df)}"
    for col in ("anchor", "positive", "negative"):
        n_null = df[col].isna().sum()
        assert n_null == 0, f"Found {n_null} null values in column '{col}'"
    print(f"Loaded {len(df):,} validation triplets from {args.val_triplets}")

    dataset = TripletDataset(df)
    # drop_last=False: evaluate every validation triplet, including the final
    # partial batch. No shuffling — deterministic pass over the full set.
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    print(f"Batches: {len(loader):,} (batch size = {args.batch_size}, drop_last=False)")
    print()

    # --- Load tokenizer once; it's shared across all three checkpoints ---
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print()

    # --- Evaluate each epoch checkpoint ---
    wall_start = time.time()
    results = []
    for ckpt_path in ckpt_paths:
        metrics = evaluate_checkpoint(
            ckpt_path, loader, tokenizer, device, margin, max_length
        )
        epoch_num = int(ckpt_path.stem.replace("model_epoch", ""))
        metrics["epoch"] = epoch_num
        metrics["train_loss"] = per_epoch_train_loss[epoch_num - 1]
        results.append(metrics)

    total_runtime = time.time() - wall_start

    # --- Summary table ---
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print("Epoch | Train Loss | Val Loss | Val Accuracy | Val Mean Margin")
    print("------+------------+----------+--------------+----------------")
    for r in results:
        print(
            f"  {r['epoch']:>3d} | "
            f"{r['train_loss']:>10.4f} | "
            f"{r['val_loss']:>8.4f} | "
            f"{r['val_accuracy']*100:>10.2f}%  | "
            f"{r['val_mean_margin']:>15.4f}"
        )
    print("=" * 72)
    print(f"Total runtime: {total_runtime:.0f}s ({total_runtime/60:.1f} min)")
    print()

    # --- Save structured results atomically ---
    payload = {
        "checkpoint_dir": str(args.checkpoint_dir),
        "val_triplets": str(args.val_triplets),
        "model_name": MODEL_NAME,
        "batch_size": args.batch_size,
        "margin": margin,
        "max_length": max_length,
        "n_val_triplets": int(len(df)),
        "per_epoch": results,
        "per_epoch_train_loss": per_epoch_train_loss,
        "total_runtime_seconds": total_runtime,
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "cuda": getattr(torch.version, "cuda", None),
        },
    }

    out_path = args.checkpoint_dir / "val_loss_results.json"
    tmp_path = out_path.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    tmp_path.rename(out_path)
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
