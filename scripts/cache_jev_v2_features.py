#!/usr/bin/env python3
"""Cache compact token-level features for Jev-style V2.

V2 keeps local visual structure and contextual question tokens while avoiding a
huge raw-token cache.

For each sample:
- page: LensVLM visual tokens (72 at 5x in the current setup)
        -> restore merged spatial grid
        -> adaptive pool to 6x4 = 24 tokens/page
        -> SAME fixed random projection 4096 -> 256
- question: one frozen contextual Qwen text forward (no generation)
        -> SAME fixed random projection 4096 -> 256

The fixed shared projection preserves cross-modal geometry approximately and
keeps the trainable V2 router small.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForMultimodalLM, AutoProcessor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def resolve_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def sync(device: str) -> None:
    if device == "mps":
        try:
            torch.mps.synchronize()
        except Exception:
            pass
    elif device == "cuda":
        torch.cuda.synchronize()


def feature_model(backbone):
    if hasattr(backbone, "get_image_features"):
        return backbone
    if hasattr(backbone, "model") and hasattr(backbone.model, "get_image_features"):
        return backbone.model
    raise AttributeError("Cannot find get_image_features()")


def base_model(backbone):
    if hasattr(backbone, "model"):
        return backbone.model
    return backbone


def make_projection(
    input_dim: int,
    output_dim: int,
    seed: int,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    # Generate on CPU for deterministic behavior across MPS/CUDA, then move.
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    proj = torch.randn(input_dim, output_dim, generator=gen, dtype=torch.float32)
    proj /= math.sqrt(output_dim)
    return proj.to(device=device, dtype=dtype)


def infer_merge_size(grid_h: int, grid_w: int, n_tokens: int) -> int:
    ratio = (grid_h * grid_w) / n_tokens
    candidate = int(round(math.sqrt(ratio)))
    if (
        candidate > 0
        and grid_h % candidate == 0
        and grid_w % candidate == 0
        and (grid_h // candidate) * (grid_w // candidate) == n_tokens
    ):
        return candidate
    raise RuntimeError(
        f"Cannot infer spatial merge size from grid=({grid_h},{grid_w}) "
        f"and n_tokens={n_tokens}"
    )


_POOL_WEIGHT_CACHE: dict[tuple, torch.Tensor] = {}


def _adaptive_axis_weights(
    in_size: int,
    out_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return exact adaptive-average pooling weights for one spatial axis.

    PyTorch MPS currently rejects adaptive_avg_pool2d when input dimensions are
    not divisible by output dimensions (e.g. 9 -> 6). Adaptive average pooling
    is separable, so we can express it exactly as two tiny matrix multiplies and
    keep all visual features on MPS.
    """
    key = (in_size, out_size, str(device), dtype)
    cached = _POOL_WEIGHT_CACHE.get(key)
    if cached is not None:
        return cached

    # Build deterministically on CPU; the matrix is tiny and cached thereafter.
    weights = torch.zeros(out_size, in_size, dtype=torch.float32)
    for i in range(out_size):
        start = math.floor(i * in_size / out_size)
        end = math.ceil((i + 1) * in_size / out_size)
        weights[i, start:end] = 1.0 / float(end - start)

    weights = weights.to(device=device, dtype=dtype)
    _POOL_WEIGHT_CACHE[key] = weights
    return weights


def spatial_pool(
    tokens: torch.Tensor,
    grid_thw: torch.Tensor,
    out_h: int,
    out_w: int,
) -> torch.Tensor:
    # tokens are returned flattened in merged spatial order.
    _, gh, gw = [int(x) for x in grid_thw.tolist()]
    merge = infer_merge_size(gh, gw, int(tokens.shape[0]))
    mh, mw = gh // merge, gw // merge

    # [H,W,D], retaining the merged visual-token layout.
    x = tokens.reshape(mh, mw, tokens.shape[-1])

    # Exact equivalent of adaptive_avg_pool2d for arbitrary H/W ratios, using
    # only matmul/einsum operations supported by MPS.
    h_weights = _adaptive_axis_weights(mh, out_h, x.device, x.dtype)
    w_weights = _adaptive_axis_weights(mw, out_w, x.device, x.dtype)

    # H pooling: [OH,H] x [H,W,D] -> [OH,W,D]
    x = torch.einsum("ah,hwd->awd", h_weights, x)
    # W pooling: [OH,W,D] x [OW,W] -> [OH,OW,D]
    x = torch.einsum("awd,bw->abd", x, w_weights)

    return x.reshape(out_h * out_w, -1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="apple/LensVLM-9B")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--project_dim", type=int, default=256)
    ap.add_argument("--spatial_h", type=int, default=6)
    ap.add_argument("--spatial_w", type=int, default=4)
    ap.add_argument("--projection_seed", type=int, default=20260923)
    ap.add_argument("--max_question_tokens", type=int, default=96)
    args = ap.parse_args()

    device = resolve_device(args.device)

    with open(args.data_path) as f:
        all_samples = json.load(f)

    end = len(all_samples)
    if args.limit > 0:
        end = min(end, args.start + args.limit)
    samples = all_samples[args.start:end]
    if not samples:
        raise RuntimeError("No samples selected")

    print("=" * 72)
    print("JEV-STYLE V2 FEATURE CACHE")
    print("=" * 72)
    print("Samples:", len(samples))
    print("Device:", device)
    print("Model:", args.model)
    print(
        "Compact page tokens:",
        f"{args.spatial_h}x{args.spatial_w}={args.spatial_h * args.spatial_w}",
    )
    print("Projected dim:", args.project_dim)
    print("Projection seed:", args.projection_seed)

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    backbone = AutoModelForMultimodalLM.from_pretrained(
        args.model,
        dtype=torch.float16 if device != "cpu" else torch.float32,
        trust_remote_code=True,
    ).to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    vision_model = feature_model(backbone)
    text_model = base_model(backbone)
    data_dir = os.path.dirname(os.path.abspath(args.data_path))

    projection = None
    cached = []
    vision_times = []
    question_times = []

    for sample_i, sample in enumerate(samples, start=1):
        paths = [
            p if os.path.isabs(p) else os.path.join(data_dir, p)
            for p in sample["images"]
        ]
        images = []
        try:
            for path in paths:
                images.append(Image.open(path).convert("RGB"))

            # ---------------------- vision ----------------------
            image_inputs = processor.image_processor(
                images=images,
                return_tensors="pt",
            )
            pixel_values = image_inputs["pixel_values"].to(device)
            image_grid_thw = image_inputs["image_grid_thw"].to(device)

            sync(device)
            tv = time.perf_counter()
            with torch.inference_mode():
                vision = vision_model.get_image_features(
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    return_dict=True,
                )
            sync(device)
            vision_seconds = time.perf_counter() - tv

            groups = list(vision.pooler_output)
            hidden_dim = int(groups[0].shape[-1])

            if projection is None:
                projection = make_projection(
                    hidden_dim,
                    args.project_dim,
                    args.projection_seed,
                    device,
                    groups[0].dtype,
                )

            page_tokens = []
            for tokens, grid in zip(groups, image_grid_thw):
                pooled = spatial_pool(
                    tokens,
                    grid,
                    args.spatial_h,
                    args.spatial_w,
                )
                projected = pooled @ projection
                page_tokens.append(projected)

            page_tokens = torch.stack(page_tokens, dim=0)

            # ---------------------- contextual question ----------------------
            retrieval_prompt = (
                "Find the document page containing evidence needed to answer "
                "this question:\n" + sample["question"]
            )
            encoded = processor.tokenizer(
                retrieval_prompt,
                return_tensors="pt",
                add_special_tokens=True,
                truncation=True,
                max_length=args.max_question_tokens,
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}

            sync(device)
            tq = time.perf_counter()
            with torch.inference_mode():
                text_out = text_model(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded.get("attention_mask"),
                    use_cache=False,
                    return_dict=True,
                )
            sync(device)
            question_seconds = time.perf_counter() - tq

            question_tokens = text_out.last_hidden_state.squeeze(0) @ projection

            vision_times.append(vision_seconds)
            question_times.append(question_seconds)

            cached.append(
                {
                    "id": sample["id"],
                    "dataset": sample.get("dataset"),
                    "question": sample["question"],
                    "answer": sample.get("answer"),
                    "gt_pages": list(sample.get("gt_pages", [])),
                    "num_pages": int(sample["num_pages"]),
                    "question_tokens": question_tokens.detach().to(
                        dtype=torch.float16,
                        device="cpu",
                    ),
                    "page_tokens": page_tokens.detach().to(
                        dtype=torch.float16,
                        device="cpu",
                    ),
                    "vision_seconds": vision_seconds,
                    "question_seconds": question_seconds,
                }
            )

            gt = ",".join(str(x) for x in sample.get("gt_pages", [])) or "-"
            print(
                f"[{sample_i:4d}/{len(samples)}] {sample['id']} "
                f"pages={sample['num_pages']:3d} gt={gt:>8s} "
                f"qT={question_tokens.shape[0]:2d} "
                f"vision={vision_seconds:5.2f}s q={question_seconds:5.2f}s"
            )

        finally:
            for image in images:
                image.close()

        if device == "mps":
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

    payload = {
        "format": "jev_style_v2_features_v1",
        "model": args.model,
        "source_data": os.path.abspath(args.data_path),
        "project_dim": args.project_dim,
        "projection_seed": args.projection_seed,
        "spatial_h": args.spatial_h,
        "spatial_w": args.spatial_w,
        "page_tokens_per_page": args.spatial_h * args.spatial_w,
        "max_question_tokens": args.max_question_tokens,
        "samples": cached,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)

    size_mb = os.path.getsize(args.output) / 1024**2
    print("\n" + "=" * 72)
    print("V2 FEATURE CACHE COMPLETE")
    print("=" * 72)
    print("Output:", args.output)
    print("Samples:", len(cached))
    print(f"File size: {size_mb:.2f} MiB")
    print("Projected dim:", args.project_dim)
    print("Page tokens/page:", args.spatial_h * args.spatial_w)
    print(
        f"Mean vision extraction: "
        f"{sum(vision_times) / len(vision_times):.3f} sec"
    )
    print(
        f"Mean contextual question: "
        f"{sum(question_times) / len(question_times):.3f} sec"
    )


if __name__ == "__main__":
    main()
