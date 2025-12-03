import json
import os
import time
import argparse
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from anthropic import Anthropic, APIError
import matplotlib.pyplot as plt
import numpy as np

# Configuration
DEFAULT_DATA_DIR = "/Users/josephgeorge/NLP/final_project/OOD_Finetuned_Model/data/questionnaire"
DEFAULT_OUTPUT_DIR = "/Users/josephgeorge/NLP/final_project/OOD_Finetuned_Model/results"
# Keeping the model ID as requested, though it may need to be 'claude-3-5-sonnet-latest' if 404 persists
DEFAULT_MODEL = "claude-sonnet-4-5-20250929" 

# Scoring Thresholds
SHORT_ANSWER_THRESHOLD = 0.5  # fraction of key concepts required
ESSAY_THRESHOLD = 0.5         # fraction of expected topics required
FIM_SIMILARITY_THRESHOLD = 0.4 # similarity score threshold

def load_json(file_path: Path) -> Dict[str, Any]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def levenshtein(s1: str, s2: str) -> int:
    """Calculates edit distance."""
    if len(s1) < len(s2): return levenshtein(s2, s1)
    if len(s2) == 0: return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def normalized_similarity(a: str, b: str) -> float:
    dist = levenshtein(a, b)
    max_len = max(len(a), len(b), 1)
    return 1.0 - (dist / max_len)

def get_claude_response(client: Anthropic, prompt: str, model: str) -> str:
    try:
        message = client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text.strip()
    except APIError as e:
        print(f"API Error: {e}")
        return ""

def build_prompt(q: Dict[str, Any]) -> str:
    system = q.get("system", "You are an expert Java teaching assistant.")
    context = q.get("context", "")
    base = f"{system}\n\n{context}\n\n"
    
    # CASE A: Coding Question (FIM)
    if "code_prefix" in q:
        return (f"{base}Code Prefix:\n{q['code_prefix']}\n\n"
                f"Code Suffix:\n{q['code_suffix']}\n\n"
                "Provide ONLY the missing code that goes between the prefix and suffix. "
                "Do not include the prefix or suffix in your output. "
                "Do not include any markdown formatting or explanations.")
    # CASE B: Text Question
    else:
        return f"{base}Question: {q.get('question','')}\n\nAnswer:"

def grade_response(q_type: str, predicted: str, expected: Any, meta: Dict[str, Any]) -> Tuple[bool, float]:
    """Heuristic grading based on question type."""
    predicted = predicted.lower()
    
    if q_type == "true_false":
        # Check for explicit True/False
        pred_bool = None
        if "true" in predicted and "false" not in predicted: pred_bool = True
        elif "false" in predicted and "true" not in predicted: pred_bool = False
        
        # Normalize expected
        if isinstance(expected, str):
            exp_bool = expected.lower() == "true"
        else:
            exp_bool = bool(expected)
            
        correct = (pred_bool == exp_bool)
        return correct, 1.0 if correct else 0.0

    elif q_type == "short_answer":
        concepts = meta.get("key_concepts", [])
        if not concepts: return False, 0.0
        hits = sum(1 for c in concepts if c.lower() in predicted)
        score = hits / len(concepts)
        return score >= SHORT_ANSWER_THRESHOLD, score

    elif q_type == "essay":
        topics = meta.get("expected_topics", [])
        if not topics: return False, 0.0
        hits = sum(1 for t in topics if t.lower() in predicted)
        score = hits / len(topics)
        return score >= ESSAY_THRESHOLD, score

    elif q_type in ["fim_easy", "fim_hard"]:
        expected_str = str(expected).strip()
        sim = normalized_similarity(predicted, expected_str)
        return sim >= FIM_SIMILARITY_THRESHOLD, sim

    return False, 0.0

def process_files(client, data_dir: Path, model: str):
    files = ["fim_easy.json", "fim_hard.json", "short_answer.json", "true_false.json", "essay.json"]
    
    all_results = []
    
    for fname in files:
        fpath = data_dir / fname
        if not fpath.exists(): 
            print(f"Skipping {fname} (not found)")
            continue
        
        print(f"Processing {fname}...")
        data = load_json(fpath)
        q_type = fname.replace(".json", "")
        
        for q in tqdm(data.get("questions", [])):
            prompt = build_prompt(q)
            pred = get_claude_response(client, prompt, model)
            
            # Normalize expected answer
            expected = q.get("expected_answer", q.get("answer", ""))
            
            # Grade
            is_correct, score = grade_response(q_type, pred, expected, q)
            
            # Edit Distance (always calculate for data recording)
            edit_dist = levenshtein(pred, str(expected))
            
            res_entry = {
                "id": q.get("id"),
                "type": q_type,
                "topic": q.get("topic", "unknown"),
                "correct": is_correct,
                "score": score,
                "edit_distance": edit_dist,
                "prediction": pred,
                "expected": expected
            }
            all_results.append(res_entry)
            time.sleep(0.5) # Rate limit handling

    return all_results

def generate_charts(results: List[Dict], output_dir: Path):
    print("\nGenerating charts...")
    
    # 1. Edit Distance Chart (FIM only)
    fim_results = [r for r in results if "fim" in r["type"]]
    if fim_results:
        easy_avg = np.mean([r["edit_distance"] for r in fim_results if r["type"]=="fim_easy"])
        hard_avg = np.mean([r["edit_distance"] for r in fim_results if r["type"]=="fim_hard"])
        
        plt.figure(figsize=(8, 6))
        bars = plt.bar(["FIM Easy", "FIM Hard"], [easy_avg, hard_avg], color=['#3498db', '#e74c3c'])
        plt.title("Claude: Average Edit Distance (Lower is Better)")
        plt.ylabel("Levenshtein Distance")
        
        # Add labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.2f}', ha='center', va='bottom')
                     
        plt.savefig(output_dir / "claude_edit_distance_chart.png")
        plt.close()
        print(f"Saved claude_edit_distance_chart.png")

    # 2. Accuracy by Question Type
    types = list(set(r["type"] for r in results))
    type_acc = {t: np.mean([r["score"] for r in results if r["type"]==t]) * 100 for t in types}
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(type_acc.keys(), type_acc.values(), color='#2ecc71')
    plt.title("Claude: Accuracy by Question Type")
    plt.ylabel("Accuracy Score (%)")
    plt.ylim(0, 100)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1f}%', ha='center', va='bottom')

    plt.savefig(output_dir / "claude_type_accuracy.png")
    plt.close()
    print(f"Saved claude_type_accuracy.png")

    # 3. Accuracy by Topic
    topics = list(set(r["topic"] for r in results))
    topic_acc = {t: np.mean([r["score"] for r in results if r["topic"]==t]) * 100 for t in topics}
    
    # Sort for cleaner chart
    sorted_topics = sorted(topic_acc.items(), key=lambda x: x[1], reverse=True)
    if sorted_topics:
        labels, values = zip(*sorted_topics)
        plt.figure(figsize=(12, 8))
        bars = plt.bar(labels, values, color='#9b59b6')
        plt.title("Claude: Accuracy by Topic")
        plt.ylabel("Accuracy Score (%)")
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 100)
        plt.tight_layout()
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.1f}%', ha='center', va='bottom')

        plt.savefig(output_dir / "claude_topic_accuracy.png")
        plt.close()
        print(f"Saved claude_topic_accuracy.png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not found.")
        return

    client = Anthropic()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = process_files(client, Path(args.data_dir), args.model)
    
    # Save raw data
    with open(out_dir / "claude_results_full.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved raw results to claude_results_full.json")
        
    generate_charts(results, out_dir)

if __name__ == "__main__":
    main()