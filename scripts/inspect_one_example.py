# scripts/inspect_one_example.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import torch
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

FIM_PATH = "../data/processed/fim_dataset.jsonl"
BASE_MODEL_NAME = "google/gemma-2-2b-it"
LORA_DIR = "checkpoints/fim_gemma2_2b_lora"

MAX_LENGTH = 512
MAX_NEW_TOKENS = 64

TRAIN_TOPICS = ["controller", "keycommands", "turtles", "duration"]
VAL_TOPICS = ["reuse"]
TEST_TOPICS = ["adapters", "observer", "observers", "tttview"]


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


def load_splits(path: str | Path) -> DatasetDict:
    ds_all: Dataset = load_dataset("json", data_files=str(path), split="train")
    ds_all = ds_all.map(add_split)
    train = ds_all.filter(lambda e: e["split"] == "train")
    val = ds_all.filter(lambda e: e["split"] == "validation")
    test = ds_all.filter(lambda e: e["split"] == "test")
    return DatasetDict(train=train, validation=val, test=test)


def build_prompt(ex: Dict[str, Any], use_context: bool) -> str:
    system = ex["system"].strip()
    code_prefix = ex["code_prefix"]
    code_suffix = ex["code_suffix"]
    parts = [system]
    if use_context and ex.get("context", "").strip():
        parts.append(ex["context"].strip())
    parts.append("<CodePrefix>\n" + code_prefix)
    parts.append("<CodeSuffix>\n" + code_suffix)
    return "\n\n".join(parts) + "\n\n<FillIn>\n"


@torch.no_grad()
def generate(model, tokenizer, prompt: str, device: str = "cuda") -> str:
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
        do_sample=False,
    )
    gen_ids = out[0, input_ids.shape[1]:]
    tokenizer.truncation_side = "left"
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def main():
    ds = load_splits(FIM_PATH)
    test_ds = ds["test"]

    # pick one test example (change index if you want another)
    ex = test_ds[0]

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda"},
        attn_implementation="eager",
    )
    base.eval()

    lora_model = PeftModel.from_pretrained(base, LORA_DIR)
    lora_model.eval()

    gold = ex["target"].strip()

    prompt_base = build_prompt(ex, use_context=False)
    prompt_rag = build_prompt(ex, use_context=True)

    pred_base = generate(base, tokenizer, prompt_base, device="cuda")
    pred_rag = generate(base, tokenizer, prompt_rag, device="cuda")
    pred_rag_lora = generate(lora_model, tokenizer, prompt_rag, device="cuda")

    print("=== GOLD TARGET ===")
    print(gold)
    print("\n=== BASELINE (no context) ===")
    print(pred_base)
    print("\n=== RAG (context, no LoRA) ===")
    print(pred_rag)
    print("\n=== RAG + LoRA ===")
    print(pred_rag_lora)


if __name__ == "__main__":
    main()
