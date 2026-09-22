import os
import json
import time
import torch
from PIL import Image
from transformers import AutoModelForMultimodalLM, AutoProcessor

from lensvlm.prompts import SYSTEM_PROMPT
from lensvlm.evaluate import (
    parse_tool_call,
    extract_answer,
    _clean_response,
    _has_tool_call,
)

MODEL = "apple/LensVLM-9B"
DEVICE = "mps"

DATA_PATH = "./data/hotpotqa_5x_test/eval.json"
MAX_TURNS = 6
MAX_NEW_TOKENS = 512

# ------------------------------------------------------------
# 1. Load Apple's prepared eval sample
# ------------------------------------------------------------

with open(DATA_PATH) as f:
    data = json.load(f)

sample = data[0]

question = sample["question"]
gt_answer = sample["answer"]
gt_pages = sample["gt_pages"]
page_texts = sample["page_texts"]

data_dir = os.path.dirname(DATA_PATH)

image_paths = [
    os.path.join(data_dir, p)
    for p in sample["images"]
]

pages = [
    Image.open(p).convert("RGB")
    for p in image_paths
]

print("=" * 70)
print("OFFICIAL SAMPLE")
print("=" * 70)
print("ID:", sample["id"])
print("Question:", question)
print("GT answer:", gt_answer)
print("Pages:", sample["num_pages"])
print("GT evidence pages:", gt_pages)

# ------------------------------------------------------------
# 2. Model
# ------------------------------------------------------------

print("\nLoading processor...")

processor = AutoProcessor.from_pretrained(
    MODEL,
    trust_remote_code=True,
)

print("Loading LensVLM-9B...")

model = AutoModelForMultimodalLM.from_pretrained(
    MODEL,
    dtype=torch.float16,
    trust_remote_code=True,
).to(DEVICE)

model.eval()

print("Model ready.")


def mps_memory():
    try:
        allocated = torch.mps.current_allocated_memory() / 1024**3
        driver = torch.mps.driver_allocated_memory() / 1024**3
        return allocated, driver
    except Exception:
        return None, None


allocated, driver = mps_memory()

if allocated is not None:
    print(
        f"MPS memory: {allocated:.2f} GB tensors / "
        f"{driver:.2f} GB driver"
    )


# ------------------------------------------------------------
# 3. Original LensVLM multimodal input
# ------------------------------------------------------------

content = []

for page in pages:
    content.append({
        "type": "image",
        "image": page,
    })

content.append({
    "type": "text",
    "text": (
        f"There are {len(pages)} document pages.\n\n"
        f"Question: {question}"
    ),
})

conversation = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT,
    },
    {
        "role": "user",
        "content": content,
    },
]


# ------------------------------------------------------------
# 4. Single generation turn
# ------------------------------------------------------------

def generate_turn():

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
        k: v.to(DEVICE) if hasattr(v, "to") else v
        for k, v in inputs.items()
    }

    input_tokens = inputs["input_ids"].shape[1]

    start = time.perf_counter()

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    elapsed = time.perf_counter() - start

    generated = outputs[:, input_tokens:]

    response = processor.batch_decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    response = _clean_response(response)

    return (
        response,
        input_tokens,
        generated.shape[1],
        elapsed,
    )


# ------------------------------------------------------------
# 5. Official read_page interaction
# ------------------------------------------------------------

expanded_pages = []
total_generated_tokens = 0
total_time_start = time.perf_counter()

final_answer = ""

for turn in range(1, MAX_TURNS + 1):

    print("\n" + "=" * 70)
    print(f"TURN {turn}")
    print("=" * 70)

    response, input_tokens, generated_tokens, elapsed = generate_turn()

    total_generated_tokens += generated_tokens

    print(response)

    print("\n--- Turn stats ---")
    print("Input tokens:", input_tokens)
    print("Generated tokens:", generated_tokens)
    print(f"Time: {elapsed:.2f} sec")

    if elapsed > 0:
        print(
            f"Generation speed: "
            f"{generated_tokens / elapsed:.2f} tok/s"
        )

    allocated, driver = mps_memory()

    if allocated is not None:
        print(
            f"MPS memory: {allocated:.2f} GB tensors / "
            f"{driver:.2f} GB driver"
        )

    conversation.append({
        "role": "assistant",
        "content": response,
    })

    # read_page
    if _has_tool_call(response):

        page_num = parse_tool_call(response)

        if page_num is None:
            print("\nERROR: Could not parse read_page call.")
            break

        print(f"\n>>> LensVLM selected Page {page_num}")

        expanded_pages.append(page_num)

        if not 1 <= page_num <= len(page_texts):
            print("ERROR: Invalid page number.")
            break

        tool_response = (
            "<tool_response>\n"
            f"Text content of Page {page_num}:\n"
            f"{page_texts[page_num - 1]}\n"
            "</tool_response>"
        )

        conversation.append({
            "role": "user",
            "content": tool_response,
        })

        continue

    # final answer
    final_answer = extract_answer(response)

    print("\n" + "=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)
    print(final_answer)

    break


total_elapsed = time.perf_counter() - total_time_start


# ------------------------------------------------------------
# 6. Benchmark
# ------------------------------------------------------------

def normalize(x):
    return (
        x.lower()
        .strip()
        .replace(".", "")
        .replace(",", "")
    )


answer_match = normalize(final_answer) == normalize(gt_answer)

page_hit = any(
    p in gt_pages
    for p in expanded_pages
)

page_recall = (
    len(set(expanded_pages) & set(gt_pages)) / len(gt_pages)
    if gt_pages else 0
)

print("\n" + "=" * 70)
print("OFFICIAL HOTPOTQA RESULT")
print("=" * 70)

print("Question:")
print(question)

print("\nGround truth answer:")
print(gt_answer)

print("\nLensVLM answer:")
print(final_answer)

print("\nGT pages:", gt_pages)
print("Expanded pages:", expanded_pages)

print("\nExact answer match:", answer_match)
print("Evidence page hit:", page_hit)
print(f"Evidence page recall: {page_recall:.2%}")

print("\nGenerated tokens:", total_generated_tokens)
print(f"Total inference time: {total_elapsed:.2f} sec")
