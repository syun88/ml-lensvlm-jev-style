#!/usr/bin/env python3
"""Train the local Jev-style LensVLM page-decision head.

The LensVLM backbone is NOT loaded here.  Train only the small cached-feature
decision head, so this stage is cheap on MPS/CPU.

Example:
  python scripts/train_jev_head.py \
    --features ./data/hotpotqa_5x_train/jev_features.pt \
    --output ./checkpoints/jev_head_hotpotqa.pt \
    --device mps \
    --epochs 40
"""

from __future__ import annotations

import argparse
import os
import sys
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from lensvlm.jev_style import JevStyleDecisionHead, pad_feature_batch


class FeatureDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def _collate(items):
    return pad_feature_batch(items)


def _resolve_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    total = 0
    top1_hits = 0
    recall1_sum = 0.0
    recall3_sum = 0.0
    brier_sum = 0.0
    brier_count = 0

    for q, pages, mask, positive in loader:
        q = q.to(device)
        pages = pages.to(device)
        mask = mask.to(device)
        positive = positive.to(device)

        out = model(q, pages, mask)
        top1 = out.top1_index

        for i in range(q.shape[0]):
            gt = positive[i]
            gt_count = int(gt.sum().item())
            if gt_count == 0:
                continue

            total += 1
            if bool(gt[top1[i]].item()):
                top1_hits += 1

            k1 = torch.topk(out.choice_probs[i], k=1).indices
            k3 = torch.topk(
                out.choice_probs[i],
                k=min(3, int(mask[i].sum().item())),
            ).indices

            recall1_sum += float(gt[k1].sum().item()) / gt_count
            recall3_sum += float(gt[k3].sum().item()) / gt_count

        valid = mask
        targets = positive.float()
        brier_sum += float(
            ((out.noul_probs[valid] - targets[valid]) ** 2).sum().item()
        )
        brier_count += int(valid.sum().item())

    if total == 0:
        return {
            "top1_hit": 0.0,
            "recall@1": 0.0,
            "recall@3": 0.0,
            "brier": 0.0,
        }

    return {
        "top1_hit": top1_hits / total,
        "recall@1": recall1_sum / total,
        "recall@3": recall3_sum / total,
        "brier": brier_sum / max(1, brier_count),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="Training feature cache")
    parser.add_argument("--val_features", default=None, help="Optional held-out validation feature cache")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = _resolve_device(args.device)
    payload = torch.load(args.features, map_location="cpu", weights_only=False)
    samples = [
        s for s in payload["samples"]
        if s.get("gt_pages") and s["page_features"].shape[0] > 0
    ]

    if not samples:
        raise RuntimeError("Training feature cache contains no samples with gt_pages")

    if args.val_features:
        val_payload = torch.load(
            args.val_features,
            map_location="cpu",
            weights_only=False,
        )
        train_samples = samples
        val_samples = [
            s for s in val_payload["samples"]
            if s.get("gt_pages") and s["page_features"].shape[0] > 0
        ]
        if not val_samples:
            raise RuntimeError("Validation feature cache contains no samples with gt_pages")
        # Apple's prepare_data.py numbers IDs from zero independently for each
        # split, so train/validation IDs can collide even when samples differ.
        # Detect actual leakage by normalized question text instead.
        train_questions = {
            " ".join(s["question"].lower().split())
            for s in train_samples
        }
        val_questions = {
            " ".join(s["question"].lower().split())
            for s in val_samples
        }
        overlap = train_questions & val_questions
        if overlap:
            raise RuntimeError(
                f"Train/validation question overlap detected ({len(overlap)} samples)"
            )
        print(
            f"Using external held-out validation cache: {args.val_features}"
        )
    else:
        random.shuffle(samples)
        if len(samples) == 1:
            train_samples = samples
            val_samples = samples
            print("WARNING: one-sample smoke training; train and val are identical.")
        else:
            n_val = max(1, int(round(len(samples) * args.val_ratio)))
            n_val = min(n_val, len(samples) - 1)
            val_samples = samples[:n_val]
            train_samples = samples[n_val:]

    q_dim = int(train_samples[0]["question_feature"].numel())
    p_dim = int(train_samples[0]["page_features"].shape[-1])

    model = JevStyleDecisionHead(
        question_dim=q_dim,
        page_dim=p_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.heads,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    train_loader = DataLoader(
        FeatureDataset(train_samples),
        batch_size=min(args.batch_size, len(train_samples)),
        shuffle=True,
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        FeatureDataset(val_samples),
        batch_size=min(args.batch_size, len(val_samples)),
        shuffle=False,
        collate_fn=_collate,
    )

    print("=" * 72)
    print("LOCAL JEV-STYLE TRAINING")
    print("=" * 72)
    print("Device:", device)
    print("Train samples:", len(train_samples))
    print("Val samples:", len(val_samples))
    print("Question dim:", q_dim)
    print("Page dim:", p_dim)
    print("Head hidden dim:", args.hidden_dim)

    best_score = -1.0
    best_payload = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        steps = 0

        for q, pages, mask, positive in train_loader:
            q = q.to(device)
            pages = pages.to(device)
            mask = mask.to(device)
            positive = positive.to(device)

            optimizer.zero_grad(set_to_none=True)
            out = model(q, pages, mask)
            losses = model.loss(out, positive, mask)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += float(losses["loss"].detach().item())
            steps += 1

        metrics = evaluate(model, val_loader, device)
        avg_loss = epoch_loss / max(1, steps)

        # Prioritize routing accuracy, then multi-page recall, then calibration.
        score = (
            metrics["top1_hit"]
            + 0.05 * metrics["recall@3"]
            - 0.01 * metrics["brier"]
        )

        if score > best_score:
            best_score = score
            best_payload = {
                "format": "jev_style_head_v1",
                "head_config": model.config_dict(),
                "state_dict": {
                    k: v.detach().cpu()
                    for k, v in model.state_dict().items()
                },
                "source_features": str(Path(args.features).resolve()),
                "validation_features": (
                    str(Path(args.val_features).resolve())
                    if args.val_features
                    else None
                ),
                "source_model": payload.get("model"),
                "feature_pool": payload.get("feature_pool"),
                "epoch": epoch,
                "metrics": metrics,
            }

        print(
            f"epoch={epoch:03d} "
            f"loss={avg_loss:.4f} "
            f"top1={metrics['top1_hit']:.3f} "
            f"R@1={metrics['recall@1']:.3f} "
            f"R@3={metrics['recall@3']:.3f} "
            f"brier={metrics['brier']:.4f} "
            f"T={float(model.temperature.detach().cpu()):.3f}"
        )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_payload, args.output)

    print("\n" + "=" * 72)
    print("TRAINING COMPLETE")
    print("=" * 72)
    print("Checkpoint:", args.output)
    print("Best epoch:", best_payload["epoch"])
    print("Best metrics:", best_payload["metrics"])


if __name__ == "__main__":
    main()
