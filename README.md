# NLP Final Project – RAG + FIM Fine-Tuning

This repository contains the code for our CS4120 NLP final project.
We build a small retrieval-augmented system for CS3500 lecture notes and fine-tune Gemma-2 on a custom fill-in-the-middle (FIM) dataset.

## Project Overview

* Index CS3500 lecture notes and code using sentence-transformer embeddings.
* Build a FIM dataset that includes retrieved context.
* Fine-tune Gemma-2 2B with LoRA on the FIM dataset.
* Evaluate baseline, RAG, and RAG + LoRA models.
* Generate plots and metrics in the results/ directory.

## Structure

data/            # raw notes/code, embeddings, processed FIM dataset
results/         # charts + JSON metrics (edit distance, quiz accuracy, loss)
scripts/         # all pipeline scripts

## Key Scripts

* build\_context\_index.py — create chunks + embeddings for retrieval
* make\_fim\_dataset.py — generate FIM dataset with retrieved context
* finetune\_fim\_lora.py — LoRA fine-tuning for Gemma-2
* eval\_fim\_gemma2.py — evaluate baseline / RAG / RAG+LoRA
* metrics.py — run full evaluation and plot results
* rag\_notes.py — lightweight RAG module
* inspect\_one\_example.py — inspect predictions for one example

## Usage

cd scripts
python build\_context\_index.py
python make\_fim\_dataset.py
python finetune\_fim\_lora.py
python eval\_fim\_gemma2.py --mode rag\_lora
python metrics.py

## Output

All evaluation artifacts (PDF/PNG/JSON) are located in results/, including:

* FIM edit-distance plots
* Quiz accuracy by type and topic
* Training-loss curves
