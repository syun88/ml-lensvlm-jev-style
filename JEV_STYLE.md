# Local Jev-style System-1 for LensVLM

This branch builds a **local decision model** inspired by the public Jev/System-One
idea. It does **not** call the Jev API and does not claim to reproduce Jev's
private architecture.

## Goal

Replace LensVLM's autoregressive page-routing step:

```
compressed pages -> 9B reasoning tokens -> read_page(N)
```

with a local parallel decision layer:

```
compressed pages -> frozen vision features -> small shared scorer
                 -> Choice probabilities (softmax)
                 -> Noul relevance probabilities (sigmoid)
                 -> read_page(N)
```

The original LensVLM remains the slower **System-2** reader/reasoner after page
selection.

## Architecture

### Frozen feature extractor

- **Page**: Qwen3.5/LensVLM `get_image_features()`
  - runs the vision tower
  - does not autoregressively decode with the 9B language model
  - each compressed page yields merged visual tokens
  - token statistics are pooled as `[mean || max]`
- **Question**: frozen LensVLM token embeddings
  - no 9B text forward
  - pooled as `[mean || max]`

### Trainable System-1 head

For N page candidates:

1. Project question and page features to a small hidden dimension.
2. Run set self-attention over all page candidates.
3. Score every page with one shared MLP using
   `[q, p, q*p, |q-p|]`.
4. Expose the same score as:
   - **Choice**: softmax over N pages
   - **Noul**: sigmoid independently for every page
5. Train with:
   - multi-positive Choice cross entropy
   - multi-label Noul BCE
   - Brier calibration loss
6. Learn a temperature parameter for probability scaling.

The head has no fixed maximum number of pages.

## Files

- `lensvlm/jev_style.py`: feature extraction + decision head
- `scripts/cache_jev_features.py`: frozen feature extraction
- `scripts/train_jev_head.py`: head-only training
- `scripts/eval_jev_head.py`: cached-feature evaluation

## Step 0: one-sample plumbing test

Use the official HotpotQA sample already prepared:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/cache_jev_features.py \
  --data_path ./data/hotpotqa_5x_test/eval.json \
  --output ./data/hotpotqa_5x_test/jev_features.pt \
  --device mps \
  --limit 1
```

This is the first important timing measurement: it shows how long the frozen
Vision-only System-1 feature extraction takes compared with the ~43 s original
autoregressive routing turn.

Then deliberately overfit the one sample only to verify the head and labels:

```bash
python scripts/train_jev_head.py \
  --features ./data/hotpotqa_5x_test/jev_features.pt \
  --output ./checkpoints/jev_head_smoke.pt \
  --device mps \
  --epochs 80 \
  --batch_size 1 \
  --lr 1e-3
```

Evaluate:

```bash
python scripts/eval_jev_head.py \
  --features ./data/hotpotqa_5x_test/jev_features.pt \
  --checkpoint ./checkpoints/jev_head_smoke.pt \
  --device mps
```

For the current sample, the smoke target is **GT page 5**.

## Step 1: real HotpotQA training

Prepare a larger official set with Apple's existing pipeline:

```bash
python scripts/prepare_data.py \
  --dataset hotpotqa \
  --output_dir ./data/hotpotqa_5x_train \
  --compression 5x \
  --max_samples 256 \
  --split train \
  --num_workers 4
```

Cache frozen features:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/cache_jev_features.py \
  --data_path ./data/hotpotqa_5x_train/eval.json \
  --output ./data/hotpotqa_5x_train/jev_features.pt \
  --device mps
```

Train:

```bash
python scripts/train_jev_head.py \
  --features ./data/hotpotqa_5x_train/jev_features.pt \
  --output ./checkpoints/jev_head_hotpotqa.pt \
  --device mps \
  --epochs 40
```

## Metrics to compare against original LensVLM

- top-1 evidence-page hit
- evidence recall@1
- evidence recall@3
- Noul Brier score
- page-routing latency
- total QA latency
- generated routing tokens (target: **0**)

## Next milestones

1. Add a separate held-out cache for honest validation.
2. Add runtime integration: local System-1 -> `read_page` -> LensVLM System-2.
3. Add an `enough_evidence` decision for iterative multi-hop routing.
4. Test HotpotQA then MuSiQue.
5. Compare 5x / 10x / 15x compression.
6. Distill/collapse the feature extractor so the System-1 path no longer needs
   the full LensVLM checkpoint resident in memory.
