#!/usr/bin/env python3
"""Create a deterministic held-out slice from a rendered LensVLM eval.json.

Keep the output JSON in the SAME directory as the source eval.json so relative
image paths remain valid.

Example:
  python scripts/slice_eval.py \
    --input ./data/hotpotqa_5x_val512/eval.json \
    --output ./data/hotpotqa_5x_val512/eval_test256.json \
    --start 256 \
    --count 256
"""

from __future__ import annotations

import argparse
import json
import os


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--count", type=int, required=True)
    args = ap.parse_args()

    if args.start < 0 or args.count <= 0:
        raise ValueError("--start must be >= 0 and --count must be > 0")

    input_dir = os.path.dirname(os.path.abspath(args.input))
    output_dir = os.path.dirname(os.path.abspath(args.output))
    if input_dir != output_dir:
        raise RuntimeError(
            "Output must be in the same directory as input so relative image "
            "paths keep resolving correctly."
        )

    with open(args.input) as f:
        data = json.load(f)

    end = args.start + args.count
    if end > len(data):
        raise RuntimeError(
            f"Requested [{args.start}:{end}] but dataset has only {len(data)} samples"
        )

    subset = data[args.start:end]

    questions = [" ".join(x["question"].lower().split()) for x in subset]
    if len(set(questions)) != len(questions):
        raise RuntimeError("Duplicate normalized questions found inside requested slice")

    with open(args.output, "w") as f:
        json.dump(subset, f, indent=2, ensure_ascii=False)

    print("Input samples:", len(data))
    print("Slice:", f"[{args.start}:{end}]")
    print("Output samples:", len(subset))
    print("Output:", args.output)
    print("First ID:", subset[0]["id"])
    print("Last ID:", subset[-1]["id"])


if __name__ == "__main__":
    main()
