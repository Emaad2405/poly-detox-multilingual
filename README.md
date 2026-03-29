# Poly-Detox: Multilingual Text Detoxification with LoRA-Adapted mT5

Parameter-efficient multilingual text detoxification using **LoRA fine-tuning** on **mT5-base** (580M parameters) — trained on English and transferred zero-shot and few-shot to **Spanish** and **Hindi**.

---

## Overview

Text detoxification is the task of rewriting toxic text into a non-toxic equivalent while preserving the original meaning. This project investigates whether a LoRA adapter trained on English detoxification data can transfer to typologically diverse languages with minimal additional data.

**Key finding:** LoRA outperforms full fine-tuning in the low-resource regime and achieves meaningful zero-shot transfer across script families (Latin → Devanagari).

---

## Research Questions

| RQ | Question |
|---|---|
| **RQ1** | Can an English-trained LoRA model transfer zero-shot to Spanish/Hindi? |
| **RQ2** | How does performance scale with 50/100/200 few-shot examples? |
| **RQ3** | Does LoRA outperform full fine-tuning in the low-resource regime? |
| **RQ4** | Which LoRA hyperparameters (rank, alpha, target modules) matter most? |

---

## Key Findings

- **Zero-shot transfer (RQ1):** English-trained LoRA produces detoxified outputs in Spanish and Hindi without any target-language fine-tuning, demonstrating cross-lingual transfer through mT5's multilingual pre-training on 101 languages.
- **Few-shot scaling (RQ2):** Performance improves consistently with 50 → 100 → 200 examples. Diminishing returns beyond 100.
- **LoRA vs full fine-tuning (RQ3):** LoRA consistently outperforms full fine-tuning in the low-resource regime by avoiding catastrophic forgetting of multilingual pre-training.
- **Hyperparameter sensitivity (RQ4):** Rank (r=32) and target modules (Q+V projections) are most impactful. Alpha follows rank (α=2r).

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| **STA** | Style Transfer Accuracy — fraction of outputs classified as non-toxic |
| **SIM** | Semantic Similarity — cosine similarity between input and output embeddings |
| **FL** | Fluency — perplexity-based score from a language model |
| **J-score** | STA × SIM × FL — composite primary metric |

---

## Model and Data

**Foundation model:** `google/mt5-base` — multilingual T5 pre-trained on 101 languages

**LoRA configuration:** rank r=32, alpha α=64, target modules: Q + V attention projections (~8.4M trainable parameters, 1.4% of total)

**Training data:**

| Language | Dataset | Size |
|---|---|---|
| English | ParaDeHate + multilingual_paradetox EN | ~8,676 pairs |
| Spanish | multilingual_paradetox ES + es_paradetox | ~720 pairs |
| Hindi | multilingual_paradetox HI | ~300 pairs |

**Test:** `multilingual_paradetox_test` — 600 toxic inputs per language

---

## Repository Structure

```
poly-detox-multilingual/
├── README.md
├── requirements.txt
├── run_all.py                   # Run all experiments sequentially (main entry point)
├── run_notebook.sh              # SLURM job script for cluster execution
├── poly_detox_workflow.ipynb    # Full experiment workflow notebook
├── poly_detox_cluster.ipynb     # Cluster-optimised notebook variant
├── cell_1.py                    # GPU setup and dependency check
├── cell_2.py                    # WandB login and config
├── cell_3.py                    # Configuration dataclasses and experiment presets
├── cell_4.py                    # Data loading and preprocessing
├── cell_5.py                    # Baseline models (Delete / Identity)
├── cell_6.py                    # mT5 + LoRA model utilities
├── cell_7.py                    # Training pipeline
├── cell_8.py                    # Evaluator (toxicity / similarity / fluency / J-score)
├── cell_9.py                    # Experiment A — baselines
├── cell_10.py                   # Experiment B — English training
├── cell_11.py                   # Experiment C — zero-shot transfer (RQ1)
├── cell_12.py                   # Experiment D — few-shot adaptation (RQ2/RQ3)
├── cell_13.py                   # Experiment E — LoRA ablation (RQ4)
└── cell_14.py                   # Visualisation and results summary
```

---

## Running

**Local / notebook:**
```bash
pip install -r requirements.txt
python run_all.py
# or: jupyter notebook poly_detox_workflow.ipynb
```

**On SLURM cluster (Edinburgh Informatics):**
```bash
sbatch run_notebook.sh
```

Set your WandB key before running:
```bash
export WANDB_API_KEY=your_key_here
```

> Requires ~16 GB VRAM. Tested on NVIDIA H100/H200. Training mT5-base with LoRA takes ~2–4 hours per language on a single GPU.

---

## Technical Stack

- **Model:** `google/mt5-base` (580M parameters)
- **Adapter:** LoRA via Hugging Face `peft`
- **Framework:** PyTorch, Hugging Face Transformers, Datasets
- **Experiment tracking:** Weights & Biases (WandB)
- **Languages:** English, Spanish, Hindi

---

## References

- Hu et al. (2021). [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- Xue et al. (2021). [mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer](https://arxiv.org/abs/2010.11934)
- Dale et al. (2022). [Text Detoxification using Large Pre-trained Neural Models](https://arxiv.org/abs/2109.08914)
- Logacheva et al. (2022). [ParaDetox: Detoxification with Parallel Data](https://aclanthology.org/2022.acl-long.469/)
