#!/usr/bin/env python3
"""Evaluate a trained local Jev-style head on cached official LensVLM features."""

from __future__ import annotations

import argparse
import time
import os
import sys

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--show", type=int, default=10)
    parser.add_argument(
        "--timing_repeats",
        type=int,
        default=20,
        help="Repeated warmed-up timing runs on the first batch",
    )
    args = parser.parse_args()

    device = _resolve_device(args.device)
    feature_payload = torch.load(
        args.features,
        map_location="cpu",
        weights_only=False,
    )
    ckpt = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    samples = [
        s for s in feature_payload["samples"]
        if s.get("gt_pages") and s["page_features"].shape[0] > 0
    ]

    model = JevStyleDecisionHead(**ckpt["head_config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    loader = DataLoader(
        FeatureDataset(samples),
        batch_size=min(args.batch_size, max(1, len(samples))),
        shuffle=False,
        collate_fn=_collate,
    )

    total = 0
    top1_hits = 0
    head_times = []
    warmed_timing = None
    recall1_sum = 0.0
    recall3_sum = 0.0
    brier_sum = 0.0
    brier_count = 0
    shown = 0

    with torch.inference_mode():
        offset = 0
        for q, pages, mask, positive in loader:
            batch_n = q.shape[0]
            q = q.to(device)
            pages = pages.to(device)
            mask = mask.to(device)
            positive = positive.to(device)

            # First untimed call warms up MPS/CUDA kernels so the published
            # steady-state number is not dominated by one-time compilation.
            if warmed_timing is None:
                _ = model(q, pages, mask)
                if device == "mps":
                    try:
                        torch.mps.synchronize()
                    except Exception:
                        pass
                elif device == "cuda":
                    torch.cuda.synchronize()

                repeats = max(1, args.timing_repeats)
                t0 = time.perf_counter()
                for _ in range(repeats):
                    _ = model(q, pages, mask)
                if device == "mps":
                    try:
                        torch.mps.synchronize()
                    except Exception:
                        pass
                elif device == "cuda":
                    torch.cuda.synchronize()
                warmed_timing = (time.perf_counter() - t0) / repeats

            if device == "mps":
                try:
                    torch.mps.synchronize()
                except Exception:
                    pass
            elif device == "cuda":
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            out = model(q, pages, mask)

            if device == "mps":
                try:
                    torch.mps.synchronize()
                except Exception:
                    pass
            elif device == "cuda":
                torch.cuda.synchronize()

            head_times.append(time.perf_counter() - t0)

            for i in range(batch_n):
                sample = samples[offset + i]
                valid_n = int(mask[i].sum().item())
                gt = positive[i, :valid_n]
                gt_count = int(gt.sum().item())
                if gt_count == 0:
                    continue

                probs = out.choice_probs[i, :valid_n]
                top1_idx = int(probs.argmax().item())
                top1_page = top1_idx + 1
                topk = torch.topk(probs, k=min(3, valid_n)).indices

                total += 1
                hit = bool(gt[top1_idx].item())
                top1_hits += int(hit)
                recall1_sum += float(gt[top1_idx].item()) / gt_count
                recall3_sum += float(gt[topk].sum().item()) / gt_count

                if shown < args.show:
                    ranked = torch.topk(probs, k=min(5, valid_n))
                    pairs = [
                        (int(idx.item()) + 1, float(p.item()))
                        for idx, p in zip(ranked.indices, ranked.values)
                    ]
                    print(
                        f"{sample['id']}: GT={sample['gt_pages']} "
                        f"pred={top1_page} hit={hit} "
                        f"top5={pairs}"
                    )
                    shown += 1

            valid = mask
            targets = positive.float()
            brier_sum += float(
                ((out.noul_probs[valid] - targets[valid]) ** 2).sum().item()
            )
            brier_count += int(valid.sum().item())
            offset += batch_n

    print("\n" + "=" * 72)
    print("LOCAL JEV-STYLE EVALUATION")
    print("=" * 72)
    print("Samples:", total)
    print(f"Top-1 evidence hit: {top1_hits / max(1,total):.3%}")
    print(f"Evidence recall@1: {recall1_sum / max(1,total):.3%}")
    print(f"Evidence recall@3: {recall3_sum / max(1,total):.3%}")
    print(f"Noul Brier score: {brier_sum / max(1,brier_count):.6f}")
    if head_times:
        total_head = sum(head_times)
        print(f"Decision-head total time: {total_head:.6f} sec")
        print(f"Decision-head mean batch time: {total_head / len(head_times):.6f} sec")
        print(f"Decision-head mean per sample: {total_head / max(1,total):.6f} sec")
        if warmed_timing is not None:
            print(f"Decision-head warmed batch time: {warmed_timing:.6f} sec")
    print(
        "Learned temperature:",
        f"{float(model.temperature.detach().cpu()):.4f}",
    )


if __name__ == "__main__":
    main()
