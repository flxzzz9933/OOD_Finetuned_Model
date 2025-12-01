# scripts/finetune_fim_lora.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import torch
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model



# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Switched from Llama-3.1-8B to Gemma 2 9B-it (similar size, open access)
MODEL_NAME = "google/gemma-2-2b-it"

FIM_PATH = "../data/processed/fim_dataset.jsonl"
OUTPUT_DIR = "checkpoints/fim_gemma2_2b_lora"

MAX_LENGTH = 1024

# Topic-wise split (no leakage)
TRAIN_TOPICS = ["controller", "keycommands", "turtles", "duration"]
VAL_TOPICS = ["reuse"]
TEST_TOPICS = ["adapters", "observer", "observers", "tttview"]


# ---------------------------------------------------------------------------
# Dataset loading & splitting
# ---------------------------------------------------------------------------

def infer_topic_group(source_path: str) -> str:
    """
    Extract the topic group from meta['source'].

    Example source:
      'data\\raw\\code\\adapters-1030\\src\\cs3500\\lec09\\IntSet1.java'
    -> topic_group = 'adapters'
    """
    # Handle Windows-style backslashes
    parts = source_path.split("code\\")
    if len(parts) < 2:
        return "unknown"
    after_code = parts[1]
    folder = after_code.split("\\")[0]  # e.g., 'adapters-1030'
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
        # Default: treat as train
        split = "train"

    example["topic_group"] = topic_group
    example["split"] = split
    return example


def load_fim_splits(path: str | Path) -> DatasetDict:
    ds_all: Dataset = load_dataset(
        "json", data_files=str(path), split="train"
    )  # 'train' = whole file

    ds_all = ds_all.map(add_split)

    train = ds_all.filter(lambda e: e["split"] == "train")
    val = ds_all.filter(lambda e: e["split"] == "validation")
    test = ds_all.filter(lambda e: e["split"] == "test")

    return DatasetDict(train=train, validation=val, test=test)


# ---------------------------------------------------------------------------
# Prompt building & tokenization
# ---------------------------------------------------------------------------

def build_prompt_and_target(
    ex: Dict[str, Any],
    use_context: bool = True,
) -> tuple[str, str]:
    """
    Build the textual prompt and target (masked span) for one example.

    Input = system + (optional) context + code_prefix + code_suffix + '<FillIn>'
    Target = masked span only (ex['target']).
    """
    system = ex["system"].strip()
    code_prefix = ex["code_prefix"]
    code_suffix = ex["code_suffix"]
    target = ex["target"]

    parts = [system]

    if use_context and ex.get("context", "").strip():
        parts.append(ex["context"].strip())

    parts.append("<CodePrefix>\n" + code_prefix)
    parts.append("<CodeSuffix>\n" + code_suffix)
    # Marker right before where the model should start predicting
    prompt = "\n\n".join(parts) + "\n\n<FillIn>\n"

    return prompt, target


def make_tokenizer(model_name: str) -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(model_name)
    # Gemma tokenizer usually has pad_token set, but keep this safety.
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    tok.model_max_length = MAX_LENGTH
    return tok


def tokenize_fim_example(ex: Dict[str, Any], tokenizer: AutoTokenizer) -> Dict[str, Any]:
    """
    Turn one FIM JSON example into (input_ids, attention_mask, labels).

    We concatenate prompt + target:
        [prompt tokens][target tokens]

    labels:
      - For prompt tokens: -100 (ignored by loss)
      - For target tokens: same as input_ids

    The loss is *only* over the target span.
    """
    prompt, target = build_prompt_and_target(ex, use_context=True)

    # Encode prompt and target separately to know the boundary
    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )["input_ids"]

    target_ids = tokenizer(
        target,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_LENGTH,
    )["input_ids"]

    # Concatenate and truncate to MAX_LENGTH from the left if needed
    input_ids = prompt_ids + target_ids
    if len(input_ids) > MAX_LENGTH:
        # Keep the last MAX_LENGTH tokens (usually most useful)
        input_ids = input_ids[-MAX_LENGTH:]

    # Build labels aligned with input_ids
    labels = [-100] * len(input_ids)

    # Compute where the target starts *after* any truncation
    if len(prompt_ids) + len(target_ids) <= MAX_LENGTH:
        target_start = len(prompt_ids)
    else:
        total = len(prompt_ids) + len(target_ids)
        dropped = total - MAX_LENGTH
        prompt_survived = max(0, len(prompt_ids) - dropped)
        target_start = prompt_survived

    labels[target_start:] = input_ids[target_start:]

    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# ---------------------------------------------------------------------------
# Model, LoRA, Trainer
# ---------------------------------------------------------------------------

def make_model(model_name: str) -> AutoModelForCausalLM:
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",   # single GPU
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    return model




def main() -> None:
    fim_path = Path(FIM_PATH)
    ds = load_fim_splits(fim_path)

    tokenizer = make_tokenizer(MODEL_NAME)
    model = make_model(MODEL_NAME)

    # Tokenize datasets
    def _tok_fn(ex):
        return tokenize_fim_example(ex, tokenizer)

    cols_to_remove = list(ds["train"].features.keys())

    tokenized = DatasetDict(
        train=ds["train"].map(_tok_fn, remove_columns=cols_to_remove),
        validation=ds["validation"].map(_tok_fn, remove_columns=cols_to_remove),
        test=ds["test"].map(_tok_fn, remove_columns=cols_to_remove),
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=2,
        num_train_epochs=1,
        learning_rate=2e-4,
        weight_decay=0.01,
        logging_steps=1,
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    # Save LoRA adapter + tokenizer
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()
