#!/usr/bin/env python3
"""Cache frozen LensVLM features for the local Jev-style System-1 head.

Example:
  PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/cache_jev_features.py \
    --data_path ./data/hotpotqa_5x_train/eval.json \
    --output ./data/hotpotqa_5x_train/jev_features.pt \
    --model apple/LensVLM-9B \
    --device mps \
    --limit 1

The expensive 9B autoregressive decoder is never run.  We only use:
- LensVLM/Qwen3.5 vision tower -> compressed-page features
- LensVLM token embedding table -> cheap question feature
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForMultimodalLM, AutoProcessor

from lensvlm.jev_style import extract_page_features, extract_question_feature


def _resolve_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _mps_memory() -> tuple[float | None, float | None]:
    if not torch.backends.mps.is_available():
        return None, None
    try:
        return (
            torch.mps.current_allocated_memory() / 1024**3,
            torch.mps.driver_allocated_memory() / 1024**3,
        )
    except Exception:
        return None, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="apple/LensVLM-9B")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=0, help="0 = all samples")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument(
        "--store_dtype",
        choices=["float16", "float32"],
        default="float16",
    )
    args = parser.parse_args()

    device = _resolve_device(args.device)

    with open(args.data_path) as f:
        samples = json.load(f)

    end = len(samples)
    if args.limit > 0:
        end = min(end, args.start + args.limit)
    samples = samples[args.start:end]

    if not samples:
        raise RuntimeError("No samples selected")

    print(f"Samples: {len(samples)}")
    print(f"Device: {device}")
    print(f"Model: {args.model}")

    print("\nLoading processor...")
    processor = AutoProcessor.from_pretrained(
        args.model,
        trust_remote_code=True,
    )

    print("Loading LensVLM feature backbone...")
    backbone = AutoModelForMultimodalLM.from_pretrained(
        args.model,
        dtype=torch.float16 if device != "cpu" else torch.float32,
        trust_remote_code=True,
    ).to(device)
    backbone.eval()

    # Absolutely no training gradients through the 9B backbone.
    for param in backbone.parameters():
        param.requires_grad_(False)

    alloc, driver = _mps_memory()
    if alloc is not None:
        print(f"MPS memory after load: {alloc:.2f} / {driver:.2f} GB")

    data_dir = os.path.dirname(os.path.abspath(args.data_path))
    cached = []
    page_times = []
    question_times = []

    out_dtype = torch.float16 if args.store_dtype == "float16" else torch.float32

    for local_idx, sample in enumerate(samples, start=1):
        paths = [
            p if os.path.isabs(p) else os.path.join(data_dir, p)
            for p in sample["images"]
        ]

        images = []
        try:
            for path in paths:
                images.append(Image.open(path).convert("RGB"))

            tq = time.perf_counter()
            q_feature = extract_question_feature(
                backbone,
                processor,
                sample["question"],
                device=device,
            )
            question_time = time.perf_counter() - tq

            tp = time.perf_counter()
            p_features = extract_page_features(
                backbone,
                processor,
                images,
                device=device,
            )
            page_time = time.perf_counter() - tp

        finally:
            for image in images:
                image.close()

        question_times.append(question_time)
        page_times.append(page_time)

        cached.append(
            {
                "id": sample["id"],
                "dataset": sample.get("dataset"),
                "question": sample["question"],
                "answer": sample.get("answer"),
                "gt_pages": list(sample.get("gt_pages", [])),
                "num_pages": int(sample["num_pages"]),
                "question_feature": q_feature.to(out_dtype),
                "page_features": p_features.to(out_dtype),
                "feature_seconds": page_time + question_time,
            }
        )

        hit = ",".join(str(x) for x in sample.get("gt_pages", [])) or "-"
        print(
            f"[{local_idx:4d}/{len(samples)}] "
            f"{sample['id']} pages={sample['num_pages']:3d} gt={hit:>6s} "
            f"vision={page_time:6.2f}s q={question_time:5.3f}s"
        )

        # MPS retains allocator caches aggressively.  Emptying between samples
        # avoids an artificial memory climb during long feature-cache jobs.
        if device == "mps":
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

    payload = {
        "format": "jev_style_features_v1",
        "model": args.model,
        "source_data": os.path.abspath(args.data_path),
        "feature_pool": "mean_max",
        "samples": cached,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)

    q_dim = cached[0]["question_feature"].numel()
    p_dim = cached[0]["page_features"].shape[-1]

    print("\n" + "=" * 72)
    print("FEATURE CACHE COMPLETE")
    print("=" * 72)
    print("Output:", args.output)
    print("Samples:", len(cached))
    print("Question feature dim:", q_dim)
    print("Page feature dim:", p_dim)
    print(f"Mean vision extraction: {sum(page_times) / len(page_times):.3f} sec")
    print(
        f"Mean question extraction: "
        f"{sum(question_times) / len(question_times):.4f} sec"
    )


if __name__ == "__main__":
    main()
