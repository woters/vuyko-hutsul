# VuykoMistral: LLM Adaptation for Hutsul Dialect

This repository contains code and resources for adapting large language models to the Hutsul dialect of Ukrainian.

Repository contains:
- Fine-tuning scripts for Mistral, LLaMA using LoRA
- Synthetic data generation pipeline (RAG + GPT-4o)
- Evaluation scripts (BLEU, chrF++, TER, GPT-4o judge)
- Public datasets and fine-tuned models

-----------------------------------------------------

## Project Structure

- `finetune/` – LoRA fine-tuning scripts
- `RAG/` – Synthetic data generation from standard Ukrainian
- `eval/` – Evaluation with automatic and LLM-based metrics

-----------------------------------------------------

## Datasets and Models

All datasets and fine-tuned models are available at:

**https://huggingface.co/hutsul**

Includes:
- Parallel corpus (manual + synthetic)
- Hutsul-Ukrainian dictionary
- Fine-tuned checkpoints for Mistral and LLaMA