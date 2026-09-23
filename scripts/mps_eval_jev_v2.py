#!/usr/bin/env python3
"""End-to-end JevLens V2 + LensVLM evaluation on Apple-style eval.json.

This replaces LensVLM's autoregressive page-search turn with the local
non-generative V2 router:

  compressed page images + question
      -> cached V2 features
      -> JevStyleV2Head top-k pages
      -> prefilled read_page tool history for those pages
      -> ONE LensVLM answer-generation turn

The final answer turn still sees the original compressed page images, matching
LensVLM's normal selective-context setup as closely as possible. The routing
model, not LensVLM generation, decides which full page texts are expanded.

The V2 cache stores measured online feature-extraction times for each sample.
This script adds freshly measured V2-head latency to those values to report the
routing latency estimate without recomputing the expensive frozen backbone.

Example:
  python scripts/mps_eval_jev_v2.py \
    --data_path ./data/hotpotqa_5x_val512/eval_test256.json \
    --features ./data/hotpotqa_5x_val512/jev_v2_test256.pt \
    --checkpoint ./checkpoints/jev_v2_hotpotqa_4096.pt \
    --output ./results/jev_v2_e2e_test256.json \
    --device mps \
    --top_k 3 \
    --limit 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForMultimodalLM, AutoProcessor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from lensvlm.evaluator import exact_match, f1_score
from lensvlm.evaluate import _clean_response, extract_answer, _has_tool_call
from lensvlm.jev_style_v2 import JevStyleV2Head
from lensvlm.prompts import SYSTEM_PROMPT


MODEL_DEFAULT = "apple/LensVLM-9B"

# Keep Apple's original tool-use prompt, but make it explicit that page routing
# has already been performed by the external System-1 router. This prevents the
# answer stage from silently doing a second autoregressive routing pass.
ROUTED_SYSTEM_PROMPT = (
    SYSTEM_PROMPT
    + "\n\nFor this evaluation, an external page-routing system has already "
      "selected and expanded the pages to read. The corresponding read_page "
      "tool calls and tool responses are already present in the conversation. "
      "Do not call read_page again. Use the supplied page text together with "
      "the compressed page images and answer the question directly."
)


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


def mps_memory():
    try:
        return (
            torch.mps.current_allocated_memory() / 1024**3,
            torch.mps.driver_allocated_memory() / 1024**3,
        )
    except Exception:
        return None, None


def normalize_answers(sample: dict) -> list[str]:
    answers = sample.get("answers")
    if answers:
        return [str(x) for x in answers if str(x).strip()]
    answer = str(sample.get("answer", ""))
    return [answer] if answer.strip() else []


def answer_metrics(prediction: str, golds: list[str]) -> tuple[bool, float]:
    if not golds:
        return False, 0.0
    em = any(exact_match(prediction, gold) for gold in golds)
    f1 = max(f1_score(prediction, gold) for gold in golds)
    return em, f1


def make_tool_call(page_num: int) -> str:
    # Do not add synthetic reasoning; only reproduce the protocol event.
    return (
        '<tool_call>{"name": "read_page", "arguments": '
        f'{{"page": {page_num}}}'
        "</tool_call>"
    )


def make_tool_response(page_num: int, page_texts: list[str]) -> str:
    return (
        "<tool_response>\n"
        f"Text content of Page {page_num}:\n"
        f"{page_texts[page_num - 1]}\n"
        "</tool_response>"
    )


def build_conversation(
    pages: list[Image.Image],
    question: str,
    selected_pages: list[int],
    page_texts: list[str],
) -> list[dict]:
    content = [{"type": "image", "image": page} for page in pages]
    content.append(
        {
            "type": "text",
            "text": (
                f"There are {len(pages)} document pages.\n\n"
                f"Question: {question}"
            ),
        }
    )

    conversation: list[dict] = [
        {"role": "system", "content": ROUTED_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]

    # Reproduce the same read_page protocol LensVLM normally creates itself,
    # except page selection comes from JevLens V2.
    for page_num in selected_pages:
        conversation.append(
            {"role": "assistant", "content": make_tool_call(page_num)}
        )
        conversation.append(
            {
                "role": "user",
                "content": make_tool_response(page_num, page_texts),
            }
        )

    return conversation


@torch.inference_mode()
def route_one(
    router: JevStyleV2Head,
    cached: dict,
    device: str,
    top_k: int,
) -> tuple[list[int], list[tuple[int, float]], float]:
    q = cached["question_tokens"].float().unsqueeze(0).to(device)
    qmask = torch.ones(1, q.shape[1], dtype=torch.bool, device=device)
    pages = cached["page_tokens"].float().unsqueeze(0).to(device)
    pmask = torch.ones(
        1,
        pages.shape[1],
        dtype=torch.bool,
        device=device,
    )

    # Warm-up on the first caller is intentionally not hidden from the caller's
    # measured head time. Across a real stream, later calls show steady state.
    sync(device)
    t0 = time.perf_counter()
    out = router(q, qmask, pages, pmask)
    sync(device)
    head_seconds = time.perf_counter() - t0

    probs = out.choice_probs[0]
    k = min(top_k, int(pages.shape[1]))
    top = torch.topk(probs, k=k)
    selected = [int(i.item()) + 1 for i in top.indices]
    ranked = [
        (int(i.item()) + 1, float(p.item()))
        for i, p in zip(top.indices, top.values)
    ]
    return selected, ranked, head_seconds


@torch.inference_mode()
def generate_answer(
    model,
    processor,
    conversation: list[dict],
    pages: list[Image.Image],
    device: str,
    max_new_tokens: int,
) -> tuple[str, str, int, int, float]:
    prompt = processor.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        text=[prompt],
        images=pages,
        return_tensors="pt",
    )
    inputs = {
        k: v.to(device) if hasattr(v, "to") else v
        for k, v in inputs.items()
    }

    input_tokens = int(inputs["input_ids"].shape[1])

    sync(device)
    t0 = time.perf_counter()
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    sync(device)
    elapsed = time.perf_counter() - t0

    generated = outputs[:, input_tokens:]
    generated_tokens = int(generated.shape[1])
    raw = processor.batch_decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    raw = _clean_response(raw)
    answer = extract_answer(raw)

    return answer, raw, input_tokens, generated_tokens, elapsed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--top_k", type=int, default=3)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="0 = all remaining")
    ap.add_argument("--show", type=int, default=20)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    args = ap.parse_args()

    if args.top_k <= 0:
        raise ValueError("--top_k must be > 0")

    device = resolve_device(args.device)

    with open(args.data_path) as f:
        data = json.load(f)

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

    if feature_payload.get("format") != "jev_style_v2_features_v1":
        raise RuntimeError("Feature cache is not Jev V2 format")
    if ckpt.get("format") != "jev_style_v2_head_v1":
        raise RuntimeError("Checkpoint is not Jev V2 format")

    for key, expected in ckpt["feature_config"].items():
        if expected is not None and feature_payload.get(key) != expected:
            raise RuntimeError(
                f"Checkpoint/cache mismatch for {key}: "
                f"{expected} != {feature_payload.get(key)}"
            )

    feature_by_id = {x["id"]: x for x in feature_payload["samples"]}

    end = len(data)
    if args.limit > 0:
        end = min(end, args.start + args.limit)
    selected_data = data[args.start:end]
    if not selected_data:
        raise RuntimeError("No eval samples selected")

    missing = [s["id"] for s in selected_data if s["id"] not in feature_by_id]
    if missing:
        raise RuntimeError(
            f"{len(missing)} selected eval samples are missing from V2 cache; "
            f"first missing: {missing[0]}"
        )

    router = JevStyleV2Head(**ckpt["head_config"]).to(device)
    router.load_state_dict(ckpt["state_dict"])
    router.eval()

    print("=" * 78)
    print("JEVLENS V2 END-TO-END EVALUATION")
    print("=" * 78)
    print("Samples:", len(selected_data))
    print("Device:", device)
    print("Top-k expanded pages:", args.top_k)
    print("Router checkpoint epoch:", ckpt["epoch"])
    print("Loading LensVLM answer model:", args.model)

    processor = AutoProcessor.from_pretrained(
        args.model,
        trust_remote_code=True,
    )
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model,
        dtype=torch.float16 if device != "cpu" else torch.float32,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    mem, driver = mps_memory()
    if mem is not None:
        print(f"MPS memory after load: {mem:.2f} / {driver:.2f} GB")

    data_dir = os.path.dirname(os.path.abspath(args.data_path))
    records = []

    for idx, sample in enumerate(selected_data, start=1):
        cached = feature_by_id[sample["id"]]
        gt_pages = [int(x) for x in sample.get("gt_pages", [])]
        page_texts = sample.get("page_texts", [])

        if not page_texts:
            raise RuntimeError(f"{sample['id']} has no page_texts")

        image_paths = [
            p if os.path.isabs(p) else os.path.join(data_dir, p)
            for p in sample["images"]
        ]
        pages = []
        try:
            pages = [Image.open(p).convert("RGB") for p in image_paths]

            routed_pages, top_probs, head_seconds = route_one(
                router,
                cached,
                device,
                args.top_k,
            )

            invalid = [
                p for p in routed_pages
                if p < 1 or p > len(page_texts)
            ]
            if invalid:
                raise RuntimeError(
                    f"{sample['id']}: invalid routed pages {invalid}"
                )

            conversation = build_conversation(
                pages,
                sample["question"],
                routed_pages,
                page_texts,
            )

            (
                prediction,
                raw_response,
                answer_input_tokens,
                answer_generated_tokens,
                answer_seconds,
            ) = generate_answer(
                model,
                processor,
                conversation,
                pages,
                device,
                args.max_new_tokens,
            )

            golds = normalize_answers(sample)
            em, f1 = answer_metrics(prediction, golds)

            gt_set = set(gt_pages)
            routed_set = set(routed_pages)
            evidence_hit = bool(gt_set & routed_set)
            evidence_recall = (
                len(gt_set & routed_set) / len(gt_set)
                if gt_set
                else 0.0
            )
            all_evidence = bool(gt_set) and gt_set.issubset(routed_set)

            cached_vision = float(cached.get("vision_seconds", 0.0))
            cached_question = float(cached.get("question_seconds", 0.0))
            routing_seconds = (
                cached_vision + cached_question + head_seconds
            )
            estimated_total = routing_seconds + answer_seconds

            record = {
                "sample_id": sample["id"],
                "question": sample["question"],
                "gold_answers": golds,
                "prediction": prediction,
                "raw_response": raw_response,
                "gt_pages": gt_pages,
                "routed_pages": routed_pages,
                "top_probs": top_probs,
                "evidence_hit": evidence_hit,
                "evidence_recall": evidence_recall,
                "all_evidence_retrieved": all_evidence,
                "exact_match": em,
                "f1": f1,
                "routing": {
                    "vision_seconds_cached_measurement": cached_vision,
                    "question_seconds_cached_measurement": cached_question,
                    "head_seconds": head_seconds,
                    "estimated_online_seconds": routing_seconds,
                    "generated_tokens": 0,
                },
                "answer": {
                    "input_tokens": answer_input_tokens,
                    "generated_tokens": answer_generated_tokens,
                    "seconds": answer_seconds,
                },
                "estimated_total_seconds": estimated_total,
            }
            records.append(record)

            if idx <= args.show:
                print("\n" + "-" * 78)
                print(f"[{idx}/{len(selected_data)}] {sample['id']}")
                print("Question:", sample["question"])
                print("GT pages:", gt_pages)
                print("V2 pages:", routed_pages)
                print("Top probs:", top_probs)
                print(
                    f"Evidence: hit={evidence_hit} "
                    f"recall={evidence_recall:.3f} "
                    f"all={all_evidence}"
                )
                print("Gold:", golds)
                print("Answer:", prediction)
                print(f"EM={em} F1={f1:.3f}")
                print(
                    f"Routing≈{routing_seconds:.3f}s "
                    f"(vision={cached_vision:.3f}, "
                    f"q={cached_question:.3f}, "
                    f"head={head_seconds:.4f})"
                )
                print(
                    f"Answer={answer_seconds:.3f}s "
                    f"tokens={answer_generated_tokens}, "
                    f"Total≈{estimated_total:.3f}s"
                )
                if _has_tool_call(raw_response):
                    print(
                        "WARNING: answer stage attempted a read_page call "
                        "despite routing lock."
                    )

        finally:
            for page in pages:
                page.close()

        # Reduce long-run MPS fragmentation.
        if device == "mps":
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

        # Persist incrementally so a long MPS run is resumable at least by data.
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

    n = len(records)
    em_mean = sum(float(x["exact_match"]) for x in records) / max(1, n)
    f1_mean = sum(x["f1"] for x in records) / max(1, n)
    hit_mean = sum(float(x["evidence_hit"]) for x in records) / max(1, n)
    recall_mean = sum(x["evidence_recall"] for x in records) / max(1, n)
    all_mean = sum(
        float(x["all_evidence_retrieved"]) for x in records
    ) / max(1, n)
    routing_mean = sum(
        x["routing"]["estimated_online_seconds"] for x in records
    ) / max(1, n)
    answer_mean = sum(x["answer"]["seconds"] for x in records) / max(1, n)
    total_mean = sum(x["estimated_total_seconds"] for x in records) / max(1, n)
    answer_tokens_mean = sum(
        x["answer"]["generated_tokens"] for x in records
    ) / max(1, n)

    attempted_tool_calls = sum(
        int(_has_tool_call(x["raw_response"])) for x in records
    )

    summary = {
        "samples": n,
        "top_k": args.top_k,
        "exact_match": em_mean,
        "f1": f1_mean,
        "evidence_hit": hit_mean,
        "evidence_recall": recall_mean,
        "all_evidence_retrieved": all_mean,
        "routing_generated_tokens": 0,
        "mean_routing_seconds_estimated": routing_mean,
        "mean_answer_seconds": answer_mean,
        "mean_estimated_total_seconds": total_mean,
        "mean_answer_generated_tokens": answer_tokens_mean,
        "answer_stage_tool_call_attempts": attempted_tool_calls,
        "checkpoint_epoch": ckpt["epoch"],
    }

    summary_path = str(Path(args.output).with_suffix(".summary.json"))
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 78)
    print("JEVLENS V2 END-TO-END RESULT")
    print("=" * 78)
    print("Samples:", n)
    print(f"Answer EM: {em_mean:.3%}")
    print(f"Answer F1: {f1_mean:.3%}")
    print(f"Evidence hit@{args.top_k}: {hit_mean:.3%}")
    print(f"Evidence recall@{args.top_k}: {recall_mean:.3%}")
    print(f"All evidence@{args.top_k}: {all_mean:.3%}")
    print("Routing generated tokens: 0")
    print(f"Mean routing time (estimated online): {routing_mean:.3f} sec")
    print(f"Mean answer generation time: {answer_mean:.3f} sec")
    print(f"Mean estimated total time: {total_mean:.3f} sec")
    print(f"Mean answer generated tokens: {answer_tokens_mean:.1f}")
    print(
        "Answer-stage unexpected tool-call attempts:",
        attempted_tool_calls,
    )
    print("Predictions:", args.output)
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()
