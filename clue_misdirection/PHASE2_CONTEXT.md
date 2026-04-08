# Phase 2 Research Context — Clue Misdirection

## Why Phase 2 Exists

Phase 1 established one solid finding: cryptic crossword clue context
measurably harms retrieval of the true answer. Clue context roughly doubles
the median retrieval rank of the true answer (1,015 → 2,160) compared to a
context-free baseline. This is the primary evidence for misdirection.

Phase 1 also included a binary classification experiment, but the classifier 
results likely reflect our methods for constructing synthetic distractors 
more than they reveal anything about misdirection in cryptic clues.

Phase 2 pursues two related goals, both of which our advisor (Dr.
Collins-Thompson) has identified as having genuine research value:

**1. Better synthetic datasets.** Because cryptic crossword clues have no
natural distractors, any causal analysis requires constructing them
synthetically. The challenge of building synthetic datasets that are
convincing enough to support valid causal inference is itself a research
problem — and the methods for doing so are of broader interest beyond this
specific application.

**2. Custom g via the Egami framework.** Egami et al. (2022) provide a
principled framework for causal inference with text, centered on the
codebook function g. We will experiment with learning a custom g by
fine-tuning CALE on cryptic crossword data, and attempt to estimate the
Average Treatment Effect (ATE) of clue context on answer retrieval.

We proceed with intellectual honesty about a potential fundamental flaw:
if our synthetic distractors are not convincing enough, the ATE estimates
may not validly measure misdirection, regardless of how carefully we apply
the Egami framework. Whether this flaw can be resolved — or whether it
constitutes a principled negative result — is itself an open question.
The process of working toward better distractors and better g's is the
research, not just a means to a predetermined conclusion.

## The Role of g

In the Egami framework, **g** is the codebook function that maps raw text
into a treatment variable suitable for causal inference. In our setting, g
is the embedding model itself: it maps a definition word (in or out of clue
context) into a vector, and the "treatment" is whether that vector is
computed with clue context (T=1) or without it (T=0).

Phase 1 used stock CALE as g — an off-the-shelf model with no knowledge of
cryptic crossword structure. Phase 2 learns a **custom g** by fine-tuning
CALE on cryptic crossword data using triplet loss, with the goal of
producing an embedding space better suited to measuring misdirection. The
trained g is then locked and applied to held-out test data to estimate the
ATE, following Egami's train/test split discipline.

## Current Experimental Structure

Phase 2 work is organized as follows:

- **`notebooks/09_learned_g_prep.ipynb`** (CPU, local): Loads a labeled
  dataset (real pairs + distractors), performs the train/test split, pulls
  stock CALE embeddings from existing Phase 1 `.npy` files, constructs
  triplets, and saves everything the GPU training script needs. Parameterized
  by dataset name so it can be rerun for different distractor sets without
  code changes.

- **`scripts/train_g_triplet.py`** (GPU, Great Lakes): Loads the prepared
  triplets, fine-tunes CALE using triplet loss, re-embeds the test set with
  both stock and learned g, estimates the ATE, and saves results. The
  fine-tuned model is saved using `model.save_pretrained()` for portability
  across PyTorch versions (see CLAUDE.md).

- **`notebooks/10_learned_g_results.ipynb`** (CPU, local): Loads saved ATE
  results and embeddings, renders comparison plots, and records findings.
  Generic — can be pointed at results from any completed experiment.

Output files for each experiment go into a named subfolder:
`data/learned_g/{dataset_name}/`. See DATA.md for the full output schema.

## Open Questions

These are the questions Phase 2 is designed to investigate. They are open
— do not treat them as settled.

- **Does learned g reduce the misdirection ATE?** The hypothesis is that
  fine-tuning on cryptic crossword triplets teaches the model to partially
  resist misdirection. The ATE comparison (stock vs. learned g) is the
  primary test.

- **How much does distractor quality matter for learning g?** The current
  harder dataset uses cosine-similarity-based distractors, which creates a
  circularity: distractors are selected using the very model being improved.
  Future experiments will explore distractor sets constructed using WordNet
  structure instead, which is independent of any embedding model.

- **What does the learned embedding space look like?** Neighborhood
  structure analysis (which words cluster together under learned g vs. stock
  CALE) may reveal whether fine-tuning produces cryptic-crossword-specific
  associations or simply overfits to the training triplets.

- **How should the misdirection score be formalized?** KCT's feedback
  suggests a unified log-odds misdirection score (change in log probability
  of correct retrieval between T=0 and T=1). This is a longer-term goal
  that depends on results from the ATE experiments.

## What Phase 2 Is Not

Phase 2 notebooks are **not** a linear pipeline in the way Phase 1 was.
Notebooks 09+ are experimental — the sequence and content of future
notebooks depends on results. Do not assume a notebook numbered 10 must
follow logically from 09 in the way that 02 follows from 01.