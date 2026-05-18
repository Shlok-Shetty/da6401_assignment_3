# DA6401 Assignment 3 — Transformer for Machine Translation

**[W&B Report](https://api.wandb.ai/links/shlokrshetty-indian-institute-of-technology-madras/e51ekupb)**

Implementation of the Transformer architecture from ["Attention Is All You Need"](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) (Vaswani et al., 2017) for German→English Neural Machine Translation on the Multi30k dataset.

## Overview

Built entirely from scratch in PyTorch — scaled dot-product attention, multi-head attention, sinusoidal positional encoding, encoder/decoder stacks, label smoothing loss, and Noam learning rate scheduler — without using any high-level Transformer abstractions from PyTorch.

## Project Structure

```
├── model.py          # Transformer architecture (MHA, PE, Encoder, Decoder)
├── train.py          # Training loop, greedy decoding, BLEU evaluation
├── dataset.py        # Multi30k loading, spacy tokenization, vocabulary
├── lr_scheduler.py   # Noam learning rate scheduler
├── requirements.txt  # Dependencies
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
```

## Training

```bash
# Base run
python train.py --run_name base_noam --scheduler noam

# Experiments
python train.py --run_name fixed_lr --scheduler fixed --lr 1e-4
python train.py --run_name no_scale --scheduler noam --no_attn_scale
python train.py --run_name learned_pe --scheduler noam --pe_type learned
python train.py --run_name no_smoothing --scheduler noam --smoothing 0.0
```

## References

- Vaswani et al., "Attention Is All You Need", NeurIPS 2017
- Multi30k Dataset: [bentrevett/multi30k](https://huggingface.co/datasets/bentrevett/multi30k)
