"""
Shared embedding utilities for ``embed_clue.py`` and ``embed_vocab.py``.

This module contains only format-agnostic embedding machinery: loading a
CALE-compatible ``AutoModel``/``AutoTokenizer`` pair, extracting token-span
or attention-masked mean-pooled embeddings from a batch of tagged texts,
validating the resulting arrays, and saving them atomically. It has no
knowledge of clue IDs, vocabulary files, splits, or phrase types — those
concerns live in the calling scripts.

Two extraction methods are supported (Decision 20):

- ``meanpool``: attention-masked mean over all non-padding tokens. Matches
  ``SentenceTransformer.encode()`` — CALE's canonical pooling, the method
  the model was trained, published, and evaluated with. Use this for the
  canonical g_stock baseline and for models trained with mean pooling
  (``g1``, future ``g_i``).

- ``tokenspan``: average hidden states only for tokens whose character
  span lies inside the ``<t></t>`` delimited region. This is the
  non-canonical NB 09 / ``train_g1_tokenspan.py`` method, retained for
  evaluating historical ``g1_tokenspan`` against a matching baseline.

Both methods assume the input text is CALE-delimited (contains ``<t>`` and
``</t>`` tags around the target word); the delimiter tokens themselves are
filtered out of the tokenspan average by construction (``<t>`` tokens map
to character spans outside ``[content_start, content_end)``), while
meanpool simply averages every non-padding token.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from transformers import AutoModel, AutoTokenizer


# =============================================================================
# Environment reporting (Decision 19)
# =============================================================================

def print_environment() -> None:
    """Print package versions and CUDA info to stdout.

    Called at the top of each script's ``main()`` so the SLURM log contains
    a permanent record of exactly which versions produced the committed
    embedding artifacts (Decision 19).
    """
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


# =============================================================================
# Model loading
# =============================================================================

def load_model(model_path: str, device: torch.device) -> tuple:
    """Load an ``AutoModel``/``AutoTokenizer`` pair, move to device, set eval.

    Accepts a HuggingFace model ID (``gabrielloiseau/CALE-MBERT-en``) or a
    local directory (``models/g1/model``). Asserts the model is 1024-dim
    (CALE's embedding width) as a sanity check against loading the wrong
    checkpoint. Returns ``(model, tokenizer)``.
    """
    print(f"Loading model: {model_path} ...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    model = model.to(device)
    model.eval()  # Disable dropout for deterministic inference.
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s")

    hidden_dim = model.config.hidden_size
    assert hidden_dim == 1024, (
        f"Expected 1024-dim CALE model; got hidden_size={hidden_dim}. "
        f"Check that --model-path points to a CALE checkpoint."
    )
    print(f"Hidden dim: {hidden_dim}")
    print()
    return model, tokenizer


# =============================================================================
# Extraction methods
# =============================================================================

def find_delimiter_char_offsets(text: str) -> tuple:
    """Return ``(start, end)`` character offsets of the text between ``<t>`` and ``</t>``.

    Offsets are into the ORIGINAL text (before tokenization). Returns
    ``(None, None)`` if either tag is missing so the caller can fall back
    to attention-masked mean pooling defensively rather than emitting a
    zero vector.
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


def extract_concept_embedding(model, tokenizer, texts, device,
                              max_length: int = 128) -> torch.Tensor:
    """Token-span-extracted embedding for a batch of CALE-delimited texts.

    For each text: tokenize, forward pass, then average ``last_hidden_state``
    over tokens whose character span lies inside the ``<t></t>`` region.
    Must be called inside a ``torch.no_grad()`` context at inference time.
    Returns a tensor of shape ``(batch, 1024)`` on ``device``.
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
            # Defensive fallback — clean inputs should never trigger this,
            # but falling back to attention-masked mean pooling avoids
            # emitting a zero vector if one malformed row sneaks through.
            attention_mask = encoded["attention_mask"][i]
            concept_vectors[i] = (
                hidden_states[i] * attention_mask.unsqueeze(-1)
            ).sum(dim=0) / attention_mask.sum()
            continue

        # A token belongs to the concept span iff its character span falls
        # entirely within [start_char, end_char). token_to_chars() returns
        # None for special tokens ([CLS], [SEP], [PAD]), which are skipped.
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
            # Delimiters found but truncation cut off the span — fall back
            # to attention-masked mean pooling rather than emit a zero vector.
            attention_mask = encoded["attention_mask"][i]
            concept_vectors[i] = (
                hidden_states[i] * attention_mask.unsqueeze(-1)
            ).sum(dim=0) / attention_mask.sum()

    return concept_vectors


def extract_meanpool_embedding(model, tokenizer, texts, device,
                               max_length: int = 128) -> torch.Tensor:
    """Attention-masked mean-pooled embedding for a batch of CALE-delimited texts.

    Averages ``last_hidden_state`` over all non-padding tokens using the
    attention mask — CALE's canonical pooling (Decision 20), equivalent to
    ``SentenceTransformer.encode()``. The ``<t></t>`` delimiters remain
    present in the input and guide attention during the forward pass; the
    pooling itself simply averages the resulting hidden states.

    Must be called inside a ``torch.no_grad()`` context at inference time.
    Returns a tensor of shape ``(batch, 1024)`` on ``device``.
    """
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)

    outputs = model(**encoded)
    hidden_states = outputs.last_hidden_state           # (batch, seq_len, dim)
    # unsqueeze(-1) broadcasts the mask across the hidden dim so padding
    # tokens contribute 0 to both the sum and the count.
    mask = encoded["attention_mask"].unsqueeze(-1).to(hidden_states.dtype)
    summed = (hidden_states * mask).sum(dim=1)          # (batch, dim)
    counts = mask.sum(dim=1)                            # (batch, 1)
    return summed / counts                              # (batch, dim)


# =============================================================================
# Batched encoding loop
# =============================================================================

def encode_phrases(model, tokenizer, phrases, device, batch_size: int,
                   max_length: int, pooling: str) -> np.ndarray:
    """Encode a list of phrases into an ``(N, 1024)`` float32 numpy array.

    Dispatches to ``extract_concept_embedding`` (``pooling='tokenspan'``) or
    ``extract_meanpool_embedding`` (``pooling='meanpool'``). Runs under
    ``torch.no_grad()`` with ``model.eval()`` so dropout is disabled and
    no gradient graph is built.

    Each batch result is moved to CPU as float32 immediately, because
    accumulating on the GPU would blow past memory on large encoding runs
    (e.g. the 239K-row full-dataset f_clue pass).
    """
    model.eval()
    n = len(phrases)
    all_vecs = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch_texts = phrases[start:start + batch_size]
            if pooling == "tokenspan":
                vecs = extract_concept_embedding(
                    model, tokenizer, list(batch_texts), device, max_length
                )
            elif pooling == "meanpool":
                vecs = extract_meanpool_embedding(
                    model, tokenizer, list(batch_texts), device, max_length
                )
            else:
                # Callers' argparse should restrict choices, so this is
                # belt-and-braces for direct library use.
                raise ValueError(f"Unknown pooling method: {pooling!r}")
            all_vecs.append(vecs.detach().to(torch.float32).cpu().numpy())
            # Progress print every 20 batches — matches embed_val.py density.
            if (start // batch_size) % 20 == 0:
                done = min(start + batch_size, n)
                print(f"    encoded {done:,} / {n:,}")
    return np.vstack(all_vecs)


# =============================================================================
# Output validation
# =============================================================================

def validate_embeddings(embeddings: np.ndarray, n_expected: int,
                        label: str) -> None:
    """Assert shape, no NaN, no all-zero rows. Print L2 norm range.

    The three failure modes this catches are the only ones that would
    silently corrupt downstream ATE computation:
      1. Shape mismatch — vocabulary/index misalignment;
      2. NaN values — forward-pass numerical failure;
      3. All-zero rows — malformed input that silently produced a zero
         vector (the delimiter fallback in ``extract_concept_embedding``
         mean-pools instead, so a zero row would indicate something worse).
    """
    assert embeddings.shape == (n_expected, 1024), (
        f"[{label}] Expected shape ({n_expected}, 1024), got {embeddings.shape}"
    )
    assert not np.isnan(embeddings).any(), (
        f"[{label}] Found NaN values in embeddings"
    )
    row_norms = np.linalg.norm(embeddings, axis=1)
    n_zero = int((row_norms == 0).sum())
    assert n_zero == 0, f"[{label}] Found {n_zero} all-zero rows in embeddings"
    print(f"  [{label}] shape={embeddings.shape}, dtype={embeddings.dtype}, "
          f"L2 norm range=[{row_norms.min():.4f}, {row_norms.max():.4f}]")


# =============================================================================
# Atomic save helpers
# =============================================================================

def save_npy_atomic(array: np.ndarray, path: Path) -> None:
    """Save ``array`` to ``path`` atomically via a ``.tmp.npy`` intermediate.

    ``np.save()`` auto-appends ``.npy`` when the path doesn't already end
    in it, so the temp name must already end in ``.npy`` to avoid a double
    extension (see commit c3653b9). The ``.rename()`` call is atomic on
    the same filesystem — either the final file exists in full or it
    doesn't exist at all, never a partial write.
    """
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, array)
    tmp.rename(path)


def save_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    """Save ``df`` to ``path`` atomically via a ``.tmp.csv`` intermediate."""
    tmp = path.with_suffix(".tmp.csv")
    df.to_csv(tmp, index=False)
    tmp.rename(path)


# Re-export the classes callers commonly need so they can `from
# embedding_utils import AutoModel` if convenient — strictly optional,
# but keeps the import surface in each script small.
__all__ = [
    "print_environment",
    "load_model",
    "find_delimiter_char_offsets",
    "extract_concept_embedding",
    "extract_meanpool_embedding",
    "encode_phrases",
    "validate_embeddings",
    "save_npy_atomic",
    "save_csv_atomic",
    "AutoModel",
    "AutoTokenizer",
]
