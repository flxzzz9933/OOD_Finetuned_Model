#!/usr/bin/env python
"""
metrics.py

Unified evaluation + visualization script for your CS4120 project.

It does:

1. FIM edit-distance eval (baseline vs RAG vs RAG+LoRA)
   for any models you configure (e.g., TinyLlama, Gemma2).

2. Quiz eval (True/False, Short Answer, FIM Easy, FIM Hard, Essay)
   using the JSON files under:
      ../data/questionnaire/
        - true_false.json
        - short_answer.json
        - fim_easy.json
        - fim_hard.json
        - essay.json

3. Visualizations:
   - Edit distance bar chart for each model.
   - Question-type & topic accuracy chart combining:
       * TinyLlama existing numbers from teammate
       * Your model’s new quiz results.

Example usage (from the scripts/ directory):

  py -3.11 metrics.py --device cuda --output_dir results

"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Any, Tuple
import random

import numpy as np
import matplotlib.pyplot as plt

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from peft import AutoPeftModelForCausalLM
except ImportError:
    AutoPeftModelForCausalLM = None


# =============================================================================
# CONFIG
# =============================================================================

# Adjust these to match your setup.
# - base_model_name: HF model id (or local path) for baseline / RAG
# - lora_dir: directory where the PEFT fine-tuned model is saved
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "tinyllama": {
        "display_name": "TinyLlama 1.1B (Fine-tuned)",
        "base_model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "lora_dir": "checkpoints/fim_llama_1b_lora",
    },
    "gemma2": {
        "display_name": "Gemma 2 2B-it (Fine-tuned)",
        "base_model_name": "google/gemma-2-2b-it",
        "lora_dir": "checkpoints/fim_gemma2_2b_lora",
    },
}

# Question type keys in JSON vs pretty names
QUESTION_TYPE_ORDER = ["true_false", "short_answer", "fim_easy", "fim_hard", "essay"]
QUESTION_TYPE_LABELS = ["True/False", "Short Answer", "FIM Easy", "FIM Hard", "Essay"]

# Thresholds for marking answers "correct enough"
FIM_SIM_THRESHOLD = 0.4           # normalized edit similarity for FIM
SHORT_ANSWER_HIT_THRESHOLD = 0.5  # fraction of key_concepts present
ESSAY_HIT_THRESHOLD = 0.5         # fraction of expected_topics present

# Existing TinyLlama chart data from visualization_charts.py
TINYLLAMA_TYPE_LABELS = ["True/False", "Short Answer", "FIM Easy", "FIM Hard", "Essay"]
TINYLLAMA_TYPE_SCORES = [56.7, 14.0, 5.0, 13.0, 22.3]
TINYLLAMA_TYPE_COUNTS = [30, 30, 20, 10, 10]
TINYLLAMA_OVERALL_AVG = 25.7

TINYLLAMA_TOPIC_LABELS = [
    "Composition", "Inheritance", "Observer", "Decorator", "GUI",
    "Testing", "Adapter", "Strategy", "MVC", "Command", "Factory"
]
TINYLLAMA_TOPIC_SCORES = [66.7, 42.9, 33.3, 25.0, 20.0, 16.7, 12.5, 12.5, 11.8, 11.1, 0.0]
TINYLLAMA_TOPIC_COUNTS = [3, 7, 3, 8, 5, 12, 16, 8, 17, 9, 2]

# Topic normalization (for quiz topic accuracy)
TOPIC_CANONICAL_MAP = {
    "adapter": "Adapter",
    "observer": "Observer",
    "observer_pattern": "Observer",
    "mvc": "MVC",
    "mvc_architecture": "MVC",
    "decorator": "Decorator",
    "decorator_pattern": "Decorator",
    "strategy": "Strategy",
    "strategy_pattern": "Strategy",
    "command": "Command",
    "command_pattern": "Command",
    "inheritance": "Inheritance",
    "design_for_inheritance": "Inheritance",
    "inheritance_vs_composition": "Inheritance",
    "composition": "Composition",
    "testing": "Testing",
    "testing_design": "Testing",
    "gui": "GUI",
    "factory": "Factory",
}


# =============================================================================
# UTILITIES
# =============================================================================

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def canonical_topic(raw: str) -> str:
    key = raw.lower()
    if key in TOPIC_CANONICAL_MAP:
        return TOPIC_CANONICAL_MAP[key]

    # Strip common suffixes
    for suffix in ["_pattern", "_architecture", "_design"]:
        if key.endswith(suffix):
            base = key[:-len(suffix)]
            if base in TOPIC_CANONICAL_MAP:
                return TOPIC_CANONICAL_MAP[base]

    # Fallback: title case
    return key.replace("_", " ").title()


def levenshtein(a: str, b: str) -> int:
    """Standard Levenshtein edit distance."""
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur.append(
                min(
                    prev[j] + 1,      # deletion
                    cur[j - 1] + 1,   # insertion
                    prev[j - 1] + cost,  # substitution
                )
            )
        prev = cur
    return prev[-1]


def normalized_similarity(a: str, b: str) -> float:
    dist = levenshtein(a, b)
    max_len = max(len(a), len(b), 1)
    return 1.0 - (dist / max_len)


def parse_true_false(text: str) -> Any:
    """Heuristic to parse model output into True/False/None."""
    t = text.strip().lower()
    if not t:
        return None

    # Prefer explicit words
    has_true = "true" in t
    has_false = "false" in t
    if has_true and not has_false:
        return True
    if has_false and not has_true:
        return False

    first = t.split()[0]
    if first.startswith("t"):
        return True
    if first.startswith("f"):
        return False
    return None


# =============================================================================
# MODEL LOADING & GENERATION
# =============================================================================

def load_base_model(model_name: str, device: str):
    """Load plain base model + tokenizer."""
    print(f"Loading base model: {model_name}")
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_available() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()
    return tokenizer, model


def load_lora_model(lora_dir: str, device: str):
    """Load PEFT LoRA model + tokenizer."""
    if AutoPeftModelForCausalLM is None:
        raise RuntimeError("peft.AutoPeftModelForCausalLM is not available. Install `peft` first.")

    print(f"Loading LoRA model from: {lora_dir}")
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_available() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(lora_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoPeftModelForCausalLM.from_pretrained(
        lora_dir,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()
    return tokenizer, model


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
) -> str:
    """Greedy generation (no sampling)."""
    device = next(model.parameters()).device
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    gen_ids = out_ids[0, inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text.strip()


# =============================================================================
# PROMPT BUILDERS
# =============================================================================

def build_fim_prompt(q: Dict[str, Any], use_context: bool) -> str:
    system = q.get(
        "system",
        "You are an expert Java teaching assistant. "
        "Fill in the missing Java code between the prefix and suffix. "
        "Output ONLY the missing code. Do not repeat the prefix or suffix, "
        "and do not add comments or explanation."
    ).strip()

    code_prefix = q.get("code_prefix", q.get("prefix", ""))
    code_suffix = q.get("code_suffix", q.get("suffix", ""))

    context = ""
    if use_context:
        if q.get("context"):
            context = q["context"]
        else:
            chunks = q.get("context_chunks") or []
            if chunks:
                context = "\n".join(chunks)

    parts = [system]
    if context.strip():
        parts.append(context.strip())

    parts.append("<CodePrefix>\n" + code_prefix)
    parts.append("<CodeSuffix>\n" + code_suffix)

    prompt = "\n\n".join(parts) + "\n\n<FillIn>\n"

    return prompt



def build_true_false_prompt(q: Dict[str, Any]) -> str:
    system = (
        "You are a CS 3500 teaching assistant. "
        "Answer the following question with exactly one word: True or False."
    )
    return f"{system}\n\nQuestion: {q['question']}\n\nAnswer:"


def build_short_answer_prompt(q: Dict[str, Any]) -> str:
    system = (
        "You are a CS 3500 teaching assistant. "
        "Answer concisely in 2-4 sentences using clear, direct language."
    )
    key_concepts = q.get("key_concepts") or []
    concepts_text = ""
    if key_concepts:
        concepts_text = (
            "Key ideas you should try to mention: "
            + ", ".join(key_concepts)
            + ".\n\n"
        )

    return f"{system}\n\n{concepts_text}Question: {q['question']}\n\nAnswer:"


def build_essay_prompt(q: Dict[str, Any]) -> str:
    system = (
        "You are a CS 3500 teaching assistant. "
        "Write a clear, well-organized answer in 2-4 paragraphs."
    )
    topics = q.get("expected_topics") or []
    topics_text = ""
    if topics:
        topics_text = (
            "Important ideas to touch on if relevant: "
            + ", ".join(topics)
            + ".\n\n"
        )
    return f"{system}\n\n{topics_text}Question: {q['question']}\n\nAnswer:"


# =============================================================================
# GRADERS FOR QUIZ TYPES
# =============================================================================

def grade_fim_answer(predicted: str, expected: str) -> Tuple[bool, float]:
    sim = normalized_similarity(predicted, expected)
    return sim >= FIM_SIM_THRESHOLD, sim


def grade_short_answer(predicted: str, key_concepts: List[str]) -> Tuple[bool, float]:
    if not key_concepts:
        return False, 0.0
    text = predicted.lower()
    hits = 0
    for concept in key_concepts:
        if concept.lower() in text:
            hits += 1
    score = hits / len(key_concepts)
    return score >= SHORT_ANSWER_HIT_THRESHOLD, score


def grade_essay(predicted: str, expected_topics: List[str]) -> Tuple[bool, float]:
    if not expected_topics:
        return False, 0.0
    text = predicted.lower()
    hits = 0
    for topic in expected_topics:
        if topic.lower() in text:
            hits += 1
    score = hits / len(expected_topics)
    return score >= ESSAY_HIT_THRESHOLD, score


# =============================================================================
# DATA LOADING
# =============================================================================

def load_json_questions(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_fim_questions(questionnaire_dir: Path) -> List[Dict[str, Any]]:
    fim_easy_path = questionnaire_dir / "fim_easy.json"
    fim_hard_path = questionnaire_dir / "fim_hard.json"
    easy = load_json_questions(fim_easy_path)["questions"]
    hard = load_json_questions(fim_hard_path)["questions"]
    return easy + hard


# =============================================================================
# FIM EDIT DISTANCE EVAL (baseline / RAG / RAG+LoRA)
# =============================================================================

def evaluate_fim_mode(
    questions: List[Dict[str, Any]],
    tokenizer,
    model,
    mode: str,
    max_examples: int = None,
) -> Dict[str, Any]:
    """Run one mode: baseline / rag / rag_lora on FIM questions."""
    if max_examples is not None and max_examples < len(questions):
        qs = questions[:max_examples]
    else:
        qs = questions

    use_context = mode in ("rag", "rag_lora")
    results = []
    for q in qs:
        prompt = build_fim_prompt(q, use_context=use_context)
        pred = generate_text(model, tokenizer, prompt, max_new_tokens=256)
        expected = q["expected_answer"].strip()
        dist = levenshtein(pred, expected)
        results.append(
            {
                "id": q["id"],
                "topic": q["topic"],
                "edit_distance": dist,
                "expected_len": len(expected),
                "pred_len": len(pred),
                "expected": expected,
                "prediction": pred,
            }
        )

    N = len(results)
    avg_dist = sum(r["edit_distance"] for r in results) / N if N > 0 else math.nan
    print(f"  Mode={mode}, N={N}, Avg edit distance={avg_dist:.2f}")
    return {
        "mode": mode,
        "N": N,
        "avg_edit_distance": avg_dist,
        "examples": results,
    }


def run_fim_edit_distance_suite_for_model(
    model_key: str,
    cfg: Dict[str, Any],
    device: str,
    base_dir: Path,
    output_dir: Path,
    max_examples: int = None,
) -> Dict[str, Any]:
    print(f"\n=== FIM edit-distance eval for {model_key} on {device} ===")

    questions = load_fim_questions(base_dir)
    print(f"Total FIM questions (easy+hard): {len(questions)}")

    # Baseline & RAG with base model
    tokenizer_base, model_base = load_base_model(cfg["base_model_name"], device)
    edit_baseline = evaluate_fim_mode(
        questions, tokenizer_base, model_base,
        mode="baseline", max_examples=max_examples
    )
    edit_rag = evaluate_fim_mode(
        questions, tokenizer_base, model_base,
        mode="rag", max_examples=max_examples
    )
    del model_base
    if device == "cuda":
        torch.cuda.empty_cache()

    # RAG + LoRA
    edit_rag_lora = None
    if cfg.get("lora_dir"):
        tokenizer_lora, model_lora = load_lora_model(cfg["lora_dir"], device)
        edit_rag_lora = evaluate_fim_mode(
            questions, tokenizer_lora, model_lora,
            mode="rag_lora", max_examples=max_examples
        )
        del model_lora
        if device == "cuda":
            torch.cuda.empty_cache()
    else:
        print("  No lora_dir configured; skipping rag_lora.")

    all_modes = {
        "baseline": edit_baseline,
        "rag": edit_rag,
    }
    if edit_rag_lora is not None:
        all_modes["rag_lora"] = edit_rag_lora

    out_path = output_dir / f"edit_distance_{model_key}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(all_modes, f, indent=2)
    print(f"Saved FIM edit-distance metrics -> {out_path}")

    return all_modes


# =============================================================================
# QUIZ EVAL (True/False, Short Answer, FIM Easy, FIM Hard, Essay)
# =============================================================================

def evaluate_quiz_for_model(
    model_key: str,
    cfg: Dict[str, Any],
    device: str,
    base_dir: Path,
    output_dir: Path,
    max_per_type: int = None,
) -> Dict[str, Any]:
    """
    Run quiz evaluation using the LoRA model (mirroring teammate's setup).
    Types: TF, SA, FIM easy, FIM hard, Essay.
    """
    print(f"\n=== Quiz eval for {model_key} (LoRA) on {device} ===")

    if not cfg.get("lora_dir"):
        raise ValueError(f"Model {model_key} has no lora_dir configured but quiz eval uses LoRA.")

    tokenizer, model = load_lora_model(cfg["lora_dir"], device)

    json_paths = {
        "true_false": base_dir / "true_false.json",
        "short_answer": base_dir / "short_answer.json",
        "fim_easy": base_dir / "fim_easy.json",
        "fim_hard": base_dir / "fim_hard.json",
        "essay": base_dir / "essay.json",
    }

    type_stats: Dict[str, Any] = {}
    topic_raw_stats: Dict[str, Dict[str, int]] = {}

    for qtype in QUESTION_TYPE_ORDER:
        data = load_json_questions(json_paths[qtype])
        questions = data["questions"]
        if max_per_type is not None and len(questions) > max_per_type:
            questions = questions[:max_per_type]

        n = 0
        n_correct = 0
        per_q = []

        print(f"  Evaluating qtype={qtype}, count={len(questions)}")

        for q in questions:
            if qtype == "true_false":
                prompt = build_true_false_prompt(q)
                ans = generate_text(model, tokenizer, prompt, max_new_tokens=16)
                pred_bool = parse_true_false(ans)
                correct = (pred_bool is not None) and (pred_bool == q["answer"])
                raw_score = 1.0 if correct else 0.0

            elif qtype == "short_answer":
                prompt = build_short_answer_prompt(q)
                ans = generate_text(model, tokenizer, prompt, max_new_tokens=128)
                correct, raw_score = grade_short_answer(ans, q.get("key_concepts", []))

            elif qtype in ("fim_easy", "fim_hard"):
                prompt = build_fim_prompt(q, use_context=True)
                ans = generate_text(model, tokenizer, prompt, max_new_tokens=256)
                correct, raw_score = grade_fim_answer(ans, q["expected_answer"])

            elif qtype == "essay":
                prompt = build_essay_prompt(q)
                ans = generate_text(model, tokenizer, prompt, max_new_tokens=512)
                correct, raw_score = grade_essay(ans, q.get("expected_topics", []))
            else:
                continue

            n += 1
            if correct:
                n_correct += 1

            topic = canonical_topic(q["topic"])
            tstat = topic_raw_stats.setdefault(topic, {"n": 0, "n_correct": 0})
            tstat["n"] += 1
            if correct:
                tstat["n_correct"] += 1

            per_q.append(
                {
                    "id": q["id"],
                    "topic": topic,
                    "correct": bool(correct),
                    "score": float(raw_score),
                }
            )

        acc = 100.0 * n_correct / n if n > 0 else 0.0
        print(f"    -> Accuracy {acc:.1f}%  ({n_correct}/{n})")

        type_stats[qtype] = {
            "n": n,
            "n_correct": n_correct,
            "accuracy": acc,
            "questions": per_q,
        }

    topic_stats = {
        topic: {
            "n": stat["n"],
            "n_correct": stat["n_correct"],
            "accuracy": 100.0 * stat["n_correct"] / stat["n"] if stat["n"] > 0 else 0.0,
        }
        for topic, stat in topic_raw_stats.items()
    }

    result = {
        "model_key": model_key,
        "question_types": type_stats,
        "topics": topic_stats,
    }

    out_path = output_dir / f"quiz_results_{model_key}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Saved quiz metrics -> {out_path}")

    return result


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_edit_distance_bar(
    model_key: str,
    display_name: str,
    edit_results: Dict[str, Any],
    output_dir: Path,
) -> None:
    modes = []
    values = []
    for m in ["baseline", "rag", "rag_lora"]:
        if m in edit_results:
            modes.append(m)
            values.append(edit_results[m]["avg_edit_distance"])

    pretty_modes = {
        "baseline": "Baseline",
        "rag": "RAG",
        "rag_lora": "RAG + LoRA",
    }
    labels = [pretty_modes[m] for m in modes]

    x = np.arange(len(labels))
    width = 0.6

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(x, values, width, edgecolor="black", linewidth=1.2)

    for bar, v in zip(bars, values):
        h = bar.get_height()
        ax.annotate(
            f"{v:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_ylabel("Average edit distance (lower is better)")
    ax.set_xlabel("Evaluation mode")
    ax.set_title(f"FIM Edit Distance – {display_name}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(values) * 1.2 if values else 1)

    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()
    png_path = output_dir / f"edit_distance_{model_key}.png"
    pdf_path = output_dir / f"edit_distance_{model_key}.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved FIM edit-distance chart -> {png_path}")


def plot_type_accuracy_tinyllama_vs_model(
    model_key: str,
    model_display_name: str,
    quiz_result: Dict[str, Any],
    output_dir: Path,
) -> None:
    # TinyLlama existing scores
    tiny_scores = TINYLLAMA_TYPE_SCORES
    labels = TINYLLAMA_TYPE_LABELS

    # Our model's scores, in matching order
    model_scores = []
    for qtype in QUESTION_TYPE_ORDER:
        stats = quiz_result["question_types"].get(qtype, {})
        model_scores.append(stats.get("accuracy", 0.0))

    x = np.arange(len(labels))
    width = 0.35

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(x - width / 2, tiny_scores, width, label="TinyLlama (existing)", edgecolor="black")
    ax.bar(x + width / 2, model_scores, width, label=model_display_name, edgecolor="black")

    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Question Type")
    ax.set_title("Accuracy by Question Type (TinyLlama vs Your Model)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 80)

    ax.legend(loc="upper right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()
    png_path = output_dir / f"quiz_type_accuracy_tinyllama_vs_{model_key}.png"
    pdf_path = output_dir / f"quiz_type_accuracy_tinyllama_vs_{model_key}.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved question-type accuracy chart -> {png_path}")


def plot_topic_accuracy_tinyllama_vs_model(
    model_key: str,
    model_display_name: str,
    quiz_result: Dict[str, Any],
    output_dir: Path,
) -> None:
    labels = TINYLLAMA_TOPIC_LABELS
    tiny_scores = TINYLLAMA_TOPIC_SCORES

    # Our model's topic accuracies, aligned to TinyLlama topic list
    model_topic_stats = quiz_result["topics"]
    model_scores = []
    for topic_label in labels:
        stats = model_topic_stats.get(topic_label, {})
        model_scores.append(stats.get("accuracy", 0.0))

    x = np.arange(len(labels))
    width = 0.35

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.bar(x - width / 2, tiny_scores, width, label="TinyLlama (existing)", edgecolor="black")
    ax.bar(x + width / 2, model_scores, width, label=model_display_name, edgecolor="black")

    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Topic")
    ax.set_title("Accuracy by Topic (TinyLlama vs Your Model)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(0, 80)

    ax.legend(loc="upper right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()
    png_path = output_dir / f"quiz_topic_accuracy_tinyllama_vs_{model_key}.png"
    pdf_path = output_dir / f"quiz_topic_accuracy_tinyllama_vs_{model_key}.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved topic accuracy chart -> {png_path}")


def save_teammate_existing_charts(output_dir: Path) -> None:
    """
    Reproduce teammate's original TinyLlama-only charts
    into the same output directory.
    """
    plt.style.use("seaborn-v0_8-whitegrid")

    # Chart 1: by question type
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    x = np.arange(len(TINYLLAMA_TYPE_LABELS))
    bars1 = ax1.bar(
        TINYLLAMA_TYPE_LABELS,
        TINYLLAMA_TYPE_SCORES,
        color=["#2ecc71", "#e74c3c", "#e74c3c", "#e74c3c", "#f39c12"],
        edgecolor="black",
        linewidth=1.2,
    )
    for bar, score, count in zip(bars1, TINYLLAMA_TYPE_SCORES, TINYLLAMA_TYPE_COUNTS):
        h = bar.get_height()
        ax1.annotate(
            f"{score:.1f}%\n(n={count})",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    ax1.axhline(
        y=TINYLLAMA_OVERALL_AVG,
        color="#3498db",
        linestyle="--",
        linewidth=2,
        label=f"Overall Average: {TINYLLAMA_OVERALL_AVG}%",
    )
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_xlabel("Question Type")
    ax1.set_title("TinyLlama 1.1B (Fine-tuned): Accuracy by Question Type")
    ax1.set_ylim(0, 75)
    ax1.legend(loc="upper right")
    ax1.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax1.set_axisbelow(True)
    plt.tight_layout()
    png1 = output_dir / "tinyllama_chart_accuracy_by_type.png"
    pdf1 = output_dir / "tinyllama_chart_accuracy_by_type.pdf"
    plt.savefig(png1, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(pdf1, bbox_inches="tight", facecolor="white")
    plt.close(fig1)

    # Chart 2: by topic
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    x2 = np.arange(len(TINYLLAMA_TOPIC_LABELS))
    bars2 = ax2.bar(
        x2,
        TINYLLAMA_TOPIC_SCORES,
        color="#3498db",
        edgecolor="black",
        linewidth=1.2,
    )
    for bar, score, count in zip(bars2, TINYLLAMA_TOPIC_SCORES, TINYLLAMA_TOPIC_COUNTS):
        h = bar.get_height()
        ax2.annotate(
            f"{score:.1f}%\n(n={count})",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_xlabel("Topic")
    ax2.set_title("TinyLlama 1.1B (Fine-tuned): Accuracy by Topic")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(TINYLLAMA_TOPIC_LABELS, rotation=45, ha="right")
    ax2.set_ylim(0, 80)
    ax2.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax2.set_axisbelow(True)
    plt.tight_layout()
    png2 = output_dir / "tinyllama_chart_accuracy_by_topic.png"
    pdf2 = output_dir / "tinyllama_chart_accuracy_by_topic.pdf"
    plt.savefig(png2, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(pdf2, bbox_inches="tight", facecolor="white")
    plt.close(fig2)

    print(f"Saved teammate TinyLlama charts -> {png1}, {png2}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--output_dir", default="results", help="Output directory for metrics and plots")
    parser.add_argument("--max_fim_examples", type=int, default=None, help="Limit FIM questions for edit-distance eval")
    parser.add_argument("--max_quiz_per_type", type=int, default=None, help="Limit questions per type for quiz eval")
    parser.add_argument("--skip_tinyllama", action="store_true", help="Skip running TinyLlama (use existing charts only)")
    parser.add_argument("--skip_gemma2", action="store_true", help="Skip running Gemma2")
    args = parser.parse_args()

    set_seed(42)

    # scripts/metrics.py -> project root = parent of scripts
    project_root = Path(__file__).resolve().parents[1]
    questionnaire_dir = project_root / "data" / "questionnaire"
    out_dir = (project_root / args.output_dir).resolve()

    ensure_dir(out_dir)

    all_results: Dict[str, Any] = {}

    # TinyLlama (teammate model)
    if not args.skip_tinyllama and "tinyllama" in MODEL_CONFIGS:
        cfg = MODEL_CONFIGS["tinyllama"]
        edit_res = run_fim_edit_distance_suite_for_model(
            model_key="tinyllama",
            cfg=cfg,
            device=args.device,
            base_dir=questionnaire_dir,
            output_dir=out_dir,
            max_examples=args.max_fim_examples,
        )
        quiz_res = evaluate_quiz_for_model(
            model_key="tinyllama",
            cfg=cfg,
            device=args.device,
            base_dir=questionnaire_dir,
            output_dir=out_dir,
            max_per_type=args.max_quiz_per_type,
        )
        all_results["tinyllama"] = {
            "edit_distance": edit_res,
            "quiz": quiz_res,
        }

    # Your Gemma2 model
    if not args.skip_gemma2 and "gemma2" in MODEL_CONFIGS:
        cfg = MODEL_CONFIGS["gemma2"]
        edit_res = run_fim_edit_distance_suite_for_model(
            model_key="gemma2",
            cfg=cfg,
            device=args.device,
            base_dir=questionnaire_dir,
            output_dir=out_dir,
            max_examples=args.max_fim_examples,
        )
        quiz_res = evaluate_quiz_for_model(
            model_key="gemma2",
            cfg=cfg,
            device=args.device,
            base_dir=questionnaire_dir,
            output_dir=out_dir,
            max_per_type=args.max_quiz_per_type,
        )
        all_results["gemma2"] = {
            "edit_distance": edit_res,
            "quiz": quiz_res,
        }

    # Save combined JSON summary including teammate's fixed TinyLlama chart data
    summary = {
        "models": all_results,
        "teammate_tinyllama_existing": {
            "question_types": {
                "labels": TINYLLAMA_TYPE_LABELS,
                "scores": TINYLLAMA_TYPE_SCORES,
                "counts": TINYLLAMA_TYPE_COUNTS,
                "overall_avg": TINYLLAMA_OVERALL_AVG,
            },
            "topics": {
                "labels": TINYLLAMA_TOPIC_LABELS,
                "scores": TINYLLAMA_TOPIC_SCORES,
                "counts": TINYLLAMA_TOPIC_COUNTS,
            },
        },
    }
    summary_path = out_dir / "metrics_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved combined metrics summary -> {summary_path}")

    # Plots: FIM edit-distance per model
    for key, res in all_results.items():
        cfg = MODEL_CONFIGS[key]
        plot_edit_distance_bar(
            model_key=key,
            display_name=cfg["display_name"],
            edit_results=res["edit_distance"],
            output_dir=out_dir,
        )

    # Plots: TinyLlama vs Gemma2 on quiz accuracy (by type & topic)
    if "gemma2" in all_results:
        quiz_gemma = all_results["gemma2"]["quiz"]
        plot_type_accuracy_tinyllama_vs_model(
            model_key="gemma2",
            model_display_name=MODEL_CONFIGS["gemma2"]["display_name"],
            quiz_result=quiz_gemma,
            output_dir=out_dir,
        )
        plot_topic_accuracy_tinyllama_vs_model(
            model_key="gemma2",
            model_display_name=MODEL_CONFIGS["gemma2"]["display_name"],
            quiz_result=quiz_gemma,
            output_dir=out_dir,
        )

    # Also dump teammate's original TinyLlama-only charts into same dir
    save_teammate_existing_charts(out_dir)


if __name__ == "__main__":
    main()
