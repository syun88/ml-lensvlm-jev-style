#!/usr/bin/env python3
"""Train the token-level Jev-style V2 router on cached features."""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from lensvlm.jev_style_v2 import JevStyleV2Head, pad_v2_batch


class FeatureDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]


def collate(items):
    return pad_v2_batch(items)


def resolve_device(name):
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
    recall1 = 0.0
    recall3 = 0.0
    brier_sum = 0.0
    brier_count = 0

    for q, qmask, pages, pmask, positive in loader:
        q = q.to(device)
        qmask = qmask.to(device)
        pages = pages.to(device)
        pmask = pmask.to(device)
        positive = positive.to(device)

        out = model(q, qmask, pages, pmask)

        for i in range(q.shape[0]):
            gt = positive[i]
            n_gt = int(gt.sum().item())
            if n_gt == 0:
                continue
            total += 1

            pred = int(out.top1_index[i].item())
            top1_hits += int(bool(gt[pred].item()))

            valid_n = int(pmask[i].sum().item())
            probs = out.choice_probs[i, :valid_n]
            k1 = torch.topk(probs, 1).indices
            k3 = torch.topk(probs, min(3, valid_n)).indices
            recall1 += float(gt[k1].sum().item()) / n_gt
            recall3 += float(gt[k3].sum().item()) / n_gt

        valid = pmask
        target = positive.float()
        brier_sum += float(((out.noul_probs[valid] - target[valid]) ** 2).sum().item())
        brier_count += int(valid.sum().item())

    return {
        "top1_hit": top1_hits / max(1, total),
        "recall@1": recall1 / max(1, total),
        "recall@3": recall3 / max(1, total),
        "brier": brier_sum / max(1, brier_count),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--val_features", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    train_payload = torch.load(args.features, map_location="cpu", weights_only=False)
    val_payload = torch.load(args.val_features, map_location="cpu", weights_only=False)

    if train_payload.get("format") != "jev_style_v2_features_v1":
        raise RuntimeError("Training cache is not V2 format")
    if val_payload.get("format") != "jev_style_v2_features_v1":
        raise RuntimeError("Validation cache is not V2 format")

    for key in ("project_dim", "projection_seed", "spatial_h", "spatial_w"):
        if train_payload.get(key) != val_payload.get(key):
            raise RuntimeError(
                f"Train/val cache mismatch for {key}: "
                f"{train_payload.get(key)} != {val_payload.get(key)}"
            )

    train_samples = [
        s for s in train_payload["samples"]
        if s.get("gt_pages") and s["page_tokens"].shape[0] > 0
    ]
    val_samples = [
        s for s in val_payload["samples"]
        if s.get("gt_pages") and s["page_tokens"].shape[0] > 0
    ]

    train_q = {" ".join(s["question"].lower().split()) for s in train_samples}
    val_q = {" ".join(s["question"].lower().split()) for s in val_samples}
    overlap = train_q & val_q
    if overlap:
        raise RuntimeError(f"Train/validation question overlap: {len(overlap)}")

    input_dim = int(train_payload["project_dim"])
    model = JevStyleV2Head(
        input_dim=input_dim,
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
        collate_fn=collate,
    )
    val_loader = DataLoader(
        FeatureDataset(val_samples),
        batch_size=min(args.batch_size, len(val_samples)),
        shuffle=False,
        collate_fn=collate,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=" * 72)
    print("JEV-STYLE V2 TRAINING")
    print("=" * 72)
    print("Device:", device)
    print("Train samples:", len(train_samples))
    print("Val samples:", len(val_samples))
    print("Input dim:", input_dim)
    print("Hidden dim:", args.hidden_dim)
    print("Page tokens/page:", train_payload["page_tokens_per_page"])
    print(f"Trainable params: {n_params:,}")

    best_score = float("-inf")
    best_payload = None
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0

        for q, qmask, pages, pmask, positive in train_loader:
            q = q.to(device)
            qmask = qmask.to(device)
            pages = pages.to(device)
            pmask = pmask.to(device)
            positive = positive.to(device)

            optimizer.zero_grad(set_to_none=True)
            out = model(q, qmask, pages, pmask)
            losses = model.loss(out, positive, pmask)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += float(losses["loss"].detach().item())
            steps += 1

        metrics = evaluate(model, val_loader, device)
        avg_loss = total_loss / max(1, steps)
        score = (
            metrics["top1_hit"]
            + 0.05 * metrics["recall@3"]
            - 0.01 * metrics["brier"]
        )

        improved = score > best_score
        if improved:
            best_score = score
            epochs_without_improvement = 0
            best_payload = {
                "format": "jev_style_v2_head_v1",
                "head_config": model.config_dict(),
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "source_features": str(Path(args.features).resolve()),
                "validation_features": str(Path(args.val_features).resolve()),
                "feature_config": {
                    k: train_payload.get(k)
                    for k in (
                        "project_dim",
                        "projection_seed",
                        "spatial_h",
                        "spatial_w",
                        "page_tokens_per_page",
                        "max_question_tokens",
                    )
                },
                "epoch": epoch,
                "metrics": metrics,
            }
        else:
            epochs_without_improvement += 1

        print(
            f"epoch={epoch:03d} loss={avg_loss:.4f} "
            f"top1={metrics['top1_hit']:.3f} "
            f"R@1={metrics['recall@1']:.3f} "
            f"R@3={metrics['recall@3']:.3f} "
            f"brier={metrics['brier']:.4f} "
            f"T={float(model.temperature.detach().cpu()):.3f} "
            f"{'*' if improved else ''}"
        )

        if epochs_without_improvement >= args.patience:
            print(
                f"Early stopping: no validation improvement for "
                f"{args.patience} epochs."
            )
            break

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_payload, args.output)

    print("\n" + "=" * 72)
    print("V2 TRAINING COMPLETE")
    print("=" * 72)
    print("Checkpoint:", args.output)
    print("Best epoch:", best_payload["epoch"])
    print("Best metrics:", best_payload["metrics"])


if __name__ == "__main__":
    main()
