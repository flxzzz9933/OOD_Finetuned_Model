# scripts/eval_fim_gemma2_gpu.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any, Literal

import torch
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

FIM_PATH = "../data/processed/fim_dataset.jsonl"
BASE_MODEL_NAME = "google/gemma-2-2b-it"
LORA_DIR = "checkpoints/fim_gemma2_2b_lora"

# shorter for eval -> faster & safer
MAX_LENGTH = 2048
MAX_NEW_TOKENS = 64

TRAIN_TOPICS = ["controller", "keycommands", "turtles", "duration"]
VAL_TOPICS = ["reuse"]
TEST_TOPICS = ["adapters", "observer", "observers", "tttview"]


# ---------- dataset split ----------

def infer_topic_group(source_path: str) -> str:
    parts = source_path.split("code\\")
    if len(parts) < 2:
        return "unknown"
    after_code = parts[1]
    folder = after_code.split("\\")[0]
    prefix = folder.split("-")[0].lower()
    return prefix


def add_split(example: Dict[str, Any]) -> Dict[str, Any]:
    source = example["meta"]["source"]
    topic_group = infer_topic_group(source)

    if topic_group in TRAIN_TOPICS:
        split = "train"
    elif topic_group in VAL_TOPICS:
        split = "validation"
    elif topic_group in TEST_TOPICS:
        split = "test"
    else:
        split = "train"

    example["topic_group"] = topic_group
    example["split"] = split
    return example


def load_fim_splits(path: str | Path) -> DatasetDict:
    ds_all: Dataset = load_dataset("json", data_files=str(path), split="train")
    ds_all = ds_all.map(add_split)

    train = ds_all.filter(lambda e: e["split"] == "train")
    val = ds_all.filter(lambda e: e["split"] == "validation")
    test = ds_all.filter(lambda e: e["split"] == "test")

    return DatasetDict(train=train, validation=val, test=test)


# ---------- prompt builder ----------

def build_prompt(ex: Dict[str, Any], use_context: bool) -> str:
    system = ex["system"].strip()
    code_prefix = ex["code_prefix"]
    code_suffix = ex["code_suffix"]

    parts = [system]

    if use_context and ex.get("context", "").strip():
        parts.append(ex["context"].strip())

    parts.append("<CodePrefix>\n" + code_prefix)
    parts.append("<CodeSuffix>\n" + code_suffix)

    prompt = "\n\n".join(parts) + "\n\n<FillIn>\n"
    return prompt


# ---------- simple edit distance ----------

def edit_distance(a: str, b: str) -> int:
    la, lb = len(a), len(b)
    dp = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        dp[i][0] = i
    for j in range(lb + 1):
        dp[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[la][lb]


# ---------- generation (GPU) ----------

@torch.no_grad()
def generate_span_gpu(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    device: str = "cuda",
) -> str:
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc["attention_mask"].to(device)

    out = model.generate(
        input_ids=input_ids,
        attention_mask=attn_mask,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,   # greedy
        # no temperature arg (it warned before)
    )

    gen_ids = out[0, input_ids.shape[1]:]
    tokenizer.truncation_side = "left"
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text.strip()


def load_model(mode: str, device: str = "cuda"):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map={"": device},         # put everything on GPU 0
        attn_implementation="eager",      # gemma2 wants eager, avoids sdpa issues
    )

    if mode in ("baseline", "rag"):
        model = base
    else:
        # RAG + LoRA
        model = PeftModel.from_pretrained(base, LORA_DIR)

    model.eval()
    return tokenizer, model


def evaluate_split(
    ds: Dataset,
    mode: Literal["baseline", "rag", "rag_lora"],
    device: str = "cuda",
    max_examples: int = 30,
) -> None:
    print(f"\n=== Evaluating mode: {mode} on device {device} ===")

    tokenizer, model = load_model(mode, device=device)

    if max_examples is not None and len(ds) > max_examples:
        ds = ds.select(range(max_examples))

    exact = 0
    total = 0
    edit_sum = 0.0

    for ex in ds:
        use_context = mode in ("rag", "rag_lora")
        prompt = build_prompt(ex, use_context=use_context)
        pred = generate_span_gpu(model, tokenizer, prompt, device=device)
        gold = ex["target"].strip()

        if pred == gold:
            exact += 1
        edit_sum += edit_distance(pred, gold)
        total += 1

    print(f"  N = {total}")
    print(f"  Exact match = {exact / total:.3f}")
    print(f"  Avg edit distance = {edit_sum / total:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["baseline", "rag", "rag_lora"],
        required=True,
        help="Which setup to evaluate",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=30,
        help="Max number of test examples to evaluate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (e.g. 'cuda' or 'cuda:0')",
    )
    args = parser.parse_args()

    ds = load_fim_splits(FIM_PATH)
    test_ds = ds["test"]

    evaluate_split(
        test_ds,
        mode=args.mode,
        device=args.device,
        max_examples=args.max_examples,
    )


if __name__ == "__main__":
    main()
