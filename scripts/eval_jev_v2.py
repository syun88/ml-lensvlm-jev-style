#!/usr/bin/env python3
"""Evaluate Jev-style V2 on a held-out compact token cache."""

from __future__ import annotations

import argparse
import os
import sys
import time

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


def sync(device):
    if device == "mps":
        try:
            torch.mps.synchronize()
        except Exception:
            pass
    elif device == "cuda":
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--show", type=int, default=20)
    ap.add_argument("--timing_repeats", type=int, default=50)
    args = ap.parse_args()

    device = resolve_device(args.device)
    payload = torch.load(args.features, map_location="cpu", weights_only=False)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    if payload.get("format") != "jev_style_v2_features_v1":
        raise RuntimeError("Feature cache is not V2 format")

    for key, expected in ckpt["feature_config"].items():
        if expected is not None and payload.get(key) != expected:
            raise RuntimeError(
                f"Checkpoint/cache mismatch for {key}: "
                f"{expected} != {payload.get(key)}"
            )

    samples = [
        s for s in payload["samples"]
        if s.get("gt_pages") and s["page_tokens"].shape[0] > 0
    ]

    model = JevStyleV2Head(**ckpt["head_config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    loader = DataLoader(
        FeatureDataset(samples),
        batch_size=min(args.batch_size, max(1, len(samples))),
        shuffle=False,
        collate_fn=collate,
    )

    total = 0
    top1_hits = 0
    recall1 = 0.0
    recall2 = 0.0
    recall3 = 0.0
    recall5 = 0.0
    all1 = 0
    all2 = 0
    all3 = 0
    all5 = 0
    mrr_sum = 0.0
    brier_sum = 0.0
    brier_count = 0
    shown = 0
    warmed_batch = None
    measured = []

    with torch.inference_mode():
        offset = 0
        for q, qmask, pages, pmask, positive in loader:
            batch_n = q.shape[0]
            q = q.to(device)
            qmask = qmask.to(device)
            pages = pages.to(device)
            pmask = pmask.to(device)
            positive = positive.to(device)

            if warmed_batch is None:
                _ = model(q, qmask, pages, pmask)
                sync(device)
                repeats = max(1, args.timing_repeats)
                t0 = time.perf_counter()
                for _ in range(repeats):
                    _ = model(q, qmask, pages, pmask)
                sync(device)
                warmed_batch = (time.perf_counter() - t0) / repeats

            sync(device)
            t0 = time.perf_counter()
            out = model(q, qmask, pages, pmask)
            sync(device)
            measured.append(time.perf_counter() - t0)

            for i in range(batch_n):
                sample = samples[offset + i]
                valid_n = int(pmask[i].sum().item())
                gt = positive[i, :valid_n]
                n_gt = int(gt.sum().item())
                if n_gt == 0:
                    continue

                probs = out.choice_probs[i, :valid_n]
                pred_idx = int(probs.argmax().item())
                pred_page = pred_idx + 1
                hit = bool(gt[pred_idx].item())
                ranked = torch.argsort(probs, descending=True)
                k1_idx = ranked[: min(1, valid_n)]
                k2_idx = ranked[: min(2, valid_n)]
                k3_idx = ranked[: min(3, valid_n)]
                k5_idx = ranked[: min(5, valid_n)]

                found1 = int(gt[k1_idx].sum().item())
                found2 = int(gt[k2_idx].sum().item())
                found3 = int(gt[k3_idx].sum().item())
                found5 = int(gt[k5_idx].sum().item())

                total += 1
                top1_hits += int(hit)
                recall1 += found1 / n_gt
                recall2 += found2 / n_gt
                recall3 += found3 / n_gt
                recall5 += found5 / n_gt
                all1 += int(found1 == n_gt)
                all2 += int(found2 == n_gt)
                all3 += int(found3 == n_gt)
                all5 += int(found5 == n_gt)

                gt_positions = torch.nonzero(gt[ranked], as_tuple=False)
                if gt_positions.numel() > 0:
                    mrr_sum += 1.0 / (int(gt_positions[0].item()) + 1)

                if shown < args.show:
                    top5 = torch.topk(probs, min(5, valid_n))
                    pairs = [
                        (int(idx.item()) + 1, float(prob.item()))
                        for idx, prob in zip(top5.indices, top5.values)
                    ]
                    print(
                        f"{sample['id']}: GT={sample['gt_pages']} "
                        f"pred={pred_page} hit={hit} top5={pairs}"
                    )
                    shown += 1

            valid = pmask
            target = positive.float()
            brier_sum += float(
                ((out.noul_probs[valid] - target[valid]) ** 2).sum().item()
            )
            brier_count += int(valid.sum().item())
            offset += batch_n

    head_total = sum(measured)
    print("\n" + "=" * 72)
    print("JEV-STYLE V2 EVALUATION")
    print("=" * 72)
    print("Samples:", total)
    print(f"Top-1 evidence hit: {top1_hits / max(1,total):.3%}")
    print(f"Evidence recall@1: {recall1 / max(1,total):.3%}")
    print(f"Evidence recall@2: {recall2 / max(1,total):.3%}")
    print(f"Evidence recall@3: {recall3 / max(1,total):.3%}")
    print(f"Evidence recall@5: {recall5 / max(1,total):.3%}")
    print(f"All-evidence@1: {all1 / max(1,total):.3%}")
    print(f"All-evidence@2: {all2 / max(1,total):.3%}")
    print(f"All-evidence@3: {all3 / max(1,total):.3%}")
    print(f"All-evidence@5: {all5 / max(1,total):.3%}")
    print(f"MRR (first evidence): {mrr_sum / max(1,total):.4f}")
    print(f"Noul Brier score: {brier_sum / max(1,brier_count):.6f}")
    print(f"Decision-head total time: {head_total:.6f} sec")
    print(f"Decision-head mean per sample: {head_total / max(1,total):.6f} sec")
    if warmed_batch is not None:
        print(f"Decision-head warmed batch time: {warmed_batch:.6f} sec")
        print(
            f"Warmed per-sample at batch={args.batch_size}: "
            f"{warmed_batch / min(args.batch_size, max(1,total)):.6f} sec"
        )
    print("Best checkpoint epoch:", ckpt["epoch"])
    print("Checkpoint validation metrics:", ckpt["metrics"])


if __name__ == "__main__":
    main()
