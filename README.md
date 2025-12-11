# NLP CS4120 Final Project – RAG + FIM Code Generation Using CS3500 Notes

This repo contains the code for our CS4120 NLP final project.

We study how to get **small LLMs** to **fill in missing Java code** while still following **object-oriented design patterns** (Adapter, MVC, Observer, etc.).  
We combine **retrieval-augmented generation (RAG)** over CS3500 lecture notes with **fill-in-the-middle (FIM)** fine-tuning using LoRA.

We run the full pipeline on two compact chat models:

- **TinyLlama 1.1B-Chat**
- **Gemma-2 2B-it**

and compare:

- **Baseline** (no retrieval, no fine-tuning)  
- **RAG** (retrieval only)  
- **RAG + LoRA** (retrieval + FIM fine-tuning)

(Plus, in the report we compare against a large general chat model such as Claude as an additional reference point.)

---

## 1. Problem & Motivation

**Task.** Given a Java file with a *missing span* between a prefix and suffix, the model must **fill in the middle** with the correct code.

**Why it matters.**

- In CS3500, students must not only “make the code compile” but also obey **design principles** and **patterns** (Adapter, Strategy, MVC, testing discipline, etc.).
- Generic code LLMs often produce plausible snippets but **ignore the course’s specific design style** taught in lecture.
- Our idea:  
  1. **Retrieve** the most relevant CS3500 lecture / notes / example code.  
  2. **Fine-tune** a small model on a **FIM dataset with that retrieved context**, so it learns to:
     - Complete code,
     - While matching the **design patterns and conventions** present in the course material.

---

## 2. Repository Structure

```text
data/
  raw/                # Original CS3500 notes + code repos
  context_index/      # Chunks + embeddings for retrieval
  questionnaire/      # Quiz JSONs (TF, SA, FIM easy/hard, essay)
  fim_dataset.jsonl   # Final FIM dataset (train/val/test splits)

checkpoints/
  fim_llama_1b_lora/  # TinyLlama LoRA checkpoin
  fim_gemma2_2b_lora/ # Gemma-2 2B LoRA checkpoint

results/
  edit_distance_*.json          # FIM edit-distance metrics
  quiz_results_*.json           # Quiz metrics by type/topic
  metrics_summary.json          # Combined summary
  edit_distance_*.png/pdf       # FIM edit-distance bar charts
  quiz_*_accuracy_*.png/pdf     # Quiz accuracy charts
  training_loss_*.png           # Training loss curves

scripts/
  build_context_index.py
  make_fim_dataset.py
  finetune_fim_lora.py
  eval_fim_gemma2.py
  metrics.py
  rag_notes.py
  inspect_one_example.py
```

---

## 3. Key Scripts

- **`build_context_index.py`**  
  - Splits CS3500 lecture notes & code into chunks.  
  - Computes sentence-transformer embeddings and builds a retrieval index.

- **`make_fim_dataset.py`**  
  - Samples Java files, masks out a contiguous region (the “middle”).  
  - Uses the retriever to attach top-k context chunks.  
  - Produces `fim_dataset.jsonl` with `train/val/test` splits.

- **`finetune_fim_lora.py`**  
  - Loads **Gemma-2 2B-it** and applies a LoRA adapter on top.  
  - Fine-tunes on the FIM dataset (with context) using HuggingFace `Trainer`.  
  - Logs step-wise training loss and saves the LoRA weights to `checkpoints/fim_gemma2_2b_lora/`.

- **`eval_fim_gemma2.py`**  
  - Evaluates FIM completion on the **test** split.  
  - Supports three modes: `baseline` (no context), `rag` (context, no LoRA), `rag_lora` (context + LoRA).  
  - Reports **exact-match rate** and **average Levenshtein edit distance**.

- **`metrics.py`**  
  - Unified metrics + plotting script.  
  - Runs **FIM edit-distance eval** for both models (TinyLlama & Gemma2).  
  - Runs **quiz evaluation** using JSONs in `data/questionnaire/`:
    - `true_false.json`
    - `short_answer.json`
    - `fim_easy.json`
    - `fim_hard.json`
    - `essay.json`
  - Produces:
    - Per-mode FIM edit-distance charts,
    - Accuracy by **question type** and **topic**,
    - Saves everything into `results/`.

- **`rag_notes.py`**  
  - Minimal RAG helper to retrieve top-k CS3500 notes + generate an answer.

- **`inspect_one_example.py`**  
  - Loads the **exact training prompt format** from `finetune_fim_lora.py`.  
  - Samples test examples (topics: adapters / observer / tttview).  
  - Prints:
    - Prompt as seen during training,  
    - Gold target span,  
    - LoRA model’s predicted fill-in.

---

## 4. Dependencies

- Python **3.11**
- CUDA + PyTorch (recommended)
- Core libraries:
  - `transformers`
  - `peft`
  - `accelerate`
  - `datasets`
  - `sentence-transformers`
  - `faiss-cpu` or `faiss-gpu`
  - `matplotlib`
  - `numpy`

Example (CPU-only, minimal):

```bash
pip install "torch>=2.3" \
            "transformers>=4.42" \
            "peft>=0.11" \
            "accelerate>=0.30" \
            "datasets>=2.19" \
            "sentence-transformers>=3.0" \
            faiss-cpu \
            matplotlib numpy
```

---

## 5. End-to-End Usage

From the repo root:

```bash
cd scripts

# 1) Build retrieval index for CS3500 notes/code
py -3.11 build_context_index.py

# 2) Construct FIM dataset with retrieved context
py -3.11 make_fim_dataset.py

# 3) Fine-tune Gemma-2 2B with LoRA on the FIM data
py -3.11 finetune_fim_lora.py

# 4) Sanity-check FIM performance per mode (Gemma2)
py -3.11 eval_fim_gemma2.py --mode baseline  --max_examples 30 --device cuda
py -3.11 eval_fim_gemma2.py --mode rag       --max_examples 30 --device cuda
py -3.11 eval_fim_gemma2.py --mode rag_lora  --max_examples 30 --device cuda

# 5) Run full metrics suite and generate all charts
py -3.11 metrics.py --device cuda --output_dir ../results
#   (use --skip_tinyllama / --skip_gemma2 to disable a model if needed)

# 6) Inspect a few test examples in full training format
py -3.11 inspect_one_example.py --num_examples 5 --random --seed 42 --device cuda
```

All metrics and plots will appear in `results/`.

---

## 6. Evaluation & Outputs

We evaluate along two axes:

1. **FIM Code Completion Quality**  
   - Metric: **Levenshtein edit distance** between gold and predicted span.  
   - Lower edit distance → better code completion.  
   - We compare:
     - Baseline vs RAG vs RAG + LoRA  
     - For both **TinyLlama 1.1B** and **Gemma-2 2B**.

2. **Conceptual Understanding (Quiz)**  
   - Custom quiz data in `data/questionnaire/`:
     - **True/False** (basic conceptual checks),
     - **Short Answer** (2–4 sentence explanations),
     - **FIM Easy / Hard** (small vs larger code spans),
     - **Essay** (longer design-pattern explanations).
   - We grade using simple heuristics:
     - TF: exact match to True/False.  
     - Short Answer / Essay: fraction of **expected key concepts** present.  
     - FIM Easy / Hard: **normalized edit similarity** of the filled span.
   - We plot:
     - Accuracy by **question type**,
     - Accuracy by **topic** (Adapter, MVC, Observer, etc.).

**All artifacts** (JSON + PNG/PDF) live in `results/`:

- `edit_distance_tinyllama.json`, `edit_distance_gemma2.json`
- `quiz_results_tinyllama.json`, `quiz_results_gemma2.json`
- `metrics_summary.json`
- `edit_distance_*.png/pdf`
- `quiz_type_accuracy_tinyllama_vs_gemma2.png`
- `quiz_topic_accuracy_tinyllama_vs_gemma2.png`
- `training_loss_tinyllama.png`, `training_loss_gemma2.png`

---

## 7. High-Level Takeaways

- **RAG + LoRA substantially improves FIM code completion**, reducing edit distance compared to baseline/RAG-only (especially on Gemma-2 2B).  
- However, **quiz performance shows that pattern understanding/generalization is still limited**:
  - Models are best on simple True/False checks;  
  - They struggle with open-ended short answers and essays about OO design principles;  
  - FIM scores on course-style “easy/hard” items remain low.
- This suggests that our fine-tuning setup is good at **mimicking code snippets in context**, but does **not yet fully internalize** the higher-level design rules emphasized in CS3500.  
- Comparing TinyLlama vs Gemma2 vs a large general chat model (filled in externally) helps highlight the trade-off between **model size** and **specialized fine-tuning** for course-specific design patterns.

(See the project report and slides for detailed numbers and qualitative analysis.)
