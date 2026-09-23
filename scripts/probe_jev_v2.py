#!/usr/bin/env python3
"""Probe LensVLM features needed by Jev-style V2.

This is a one-sample diagnostic. It measures:
- merged visual-token count for every compressed page
- visual hidden dimension
- full contextual question-forward latency (no generation)
- rough cache size for several token-budget / projection choices

No training and no model.generate() are used.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
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
    # For Qwen3.5ForConditionalGeneration this is Qwen3_5Model.
    if hasattr(backbone, "model"):
        return backbone.model
    return backbone


def estimate_gib(samples: int, avg_pages: float, page_tokens: int, dim: int) -> float:
    # fp16 page tokens only.
    nbytes = samples * avg_pages * page_tokens * dim * 2
    return nbytes / 1024**3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--model", default="apple/LensVLM-9B")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--sample_index", type=int, default=0)
    args = ap.parse_args()

    device = resolve_device(args.device)

    with open(args.data_path) as f:
        samples = json.load(f)
    sample = samples[args.sample_index]
    data_dir = os.path.dirname(os.path.abspath(args.data_path))

    images = [
        Image.open(p if os.path.isabs(p) else os.path.join(data_dir, p)).convert("RGB")
        for p in sample["images"]
    ]

    print("=" * 72)
    print("JEV-STYLE V2 FEATURE PROBE")
    print("=" * 72)
    print("ID:", sample["id"])
    print("Question:", sample["question"])
    print("Pages:", sample["num_pages"])
    print("GT pages:", sample.get("gt_pages"))
    print("Device:", device)

    print("\nLoading processor/model...")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    backbone = AutoModelForMultimodalLM.from_pretrained(
        args.model,
        dtype=torch.float16 if device != "cpu" else torch.float32,
        trust_remote_code=True,
    ).to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    # ------------------------------------------------------------------
    # Vision token probe
    # ------------------------------------------------------------------
    image_inputs = processor.image_processor(images=images, return_tensors="pt")
    pixel_values = image_inputs["pixel_values"].to(device)
    image_grid_thw = image_inputs["image_grid_thw"].to(device)

    sync(device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        vision = feature_model(backbone).get_image_features(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            return_dict=True,
        )
    sync(device)
    vision_seconds = time.perf_counter() - t0

    groups = list(vision.pooler_output)
    token_counts = [int(x.shape[0]) for x in groups]
    visual_dim = int(groups[0].shape[-1])

    print("\nVISION")
    print(f"Time: {vision_seconds:.4f} sec")
    print("image_grid_thw:")
    print(image_grid_thw.detach().cpu().tolist())
    print("Merged visual tokens/page:", token_counts)
    print(
        "Token count min/mean/max:",
        min(token_counts),
        f"{sum(token_counts) / len(token_counts):.1f}",
        max(token_counts),
    )
    print("Visual token dim:", visual_dim)

    # ------------------------------------------------------------------
    # Contextual question probe -- full frozen text backbone, no generation
    # ------------------------------------------------------------------
    retrieval_prompt = (
        "Find the document page containing evidence needed to answer this question:\n"
        + sample["question"]
    )
    encoded = processor.tokenizer(
        retrieval_prompt,
        return_tensors="pt",
        add_special_tokens=True,
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    base = base_model(backbone)

    # Warm-up once because MPS first-call compilation can dominate.
    with torch.inference_mode():
        warm = base(
            input_ids=encoded["input_ids"],
            attention_mask=encoded.get("attention_mask"),
            use_cache=False,
            return_dict=True,
        )
    sync(device)

    repeats = 5
    sync(device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(repeats):
            out = base(
                input_ids=encoded["input_ids"],
                attention_mask=encoded.get("attention_mask"),
                use_cache=False,
                return_dict=True,
            )
    sync(device)
    question_seconds = (time.perf_counter() - t0) / repeats

    hidden = out.last_hidden_state
    text_dim = int(hidden.shape[-1])

    print("\nCONTEXTUAL QUESTION")
    print("Input tokens:", int(encoded["input_ids"].shape[1]))
    print("Hidden shape:", tuple(hidden.shape))
    print(f"Warmed full text-forward time: {question_seconds:.4f} sec")
    print("Text hidden dim:", text_dim)
    print("Same visual/text space dim:", visual_dim == text_dim)

    # ------------------------------------------------------------------
    # Cache-size estimates. 4096 is the current train scale; 90k indicates
    # whether a representation is viable for later scaling.
    # ------------------------------------------------------------------
    print("\nCACHE SIZE ESTIMATES (page token tensors only, fp16)")
    avg_pages = float(sample["num_pages"])
    for n_tokens in (8, 16, 32, 64):
        for dim in (256, 512):
            gib_4k = estimate_gib(4096, avg_pages, n_tokens, dim)
            gib_90k = estimate_gib(90000, avg_pages, n_tokens, dim)
            print(
                f"{n_tokens:2d} tokens/page x {dim:3d} dim: "
                f"4096≈{gib_4k:5.2f} GiB, 90k≈{gib_90k:6.1f} GiB"
            )

    print("\nV2 runtime lower-bound estimate:")
    print(
        f"vision {vision_seconds:.3f}s + contextual-question "
        f"{question_seconds:.3f}s + small decision head"
    )

    for image in images:
        image.close()


if __name__ == "__main__":
    main()
