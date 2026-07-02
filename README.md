# Direct Multi-Token Decoding (DMTD)

Official training code for the paper **[Direct Multi-Token Decoding](https://arxiv.org/abs/2510.11958)**

- 📄 Paper: [https://arxiv.org/abs/2510.11958](https://arxiv.org/abs/2510.11958)
- 🤗 Model & inference code: [https://huggingface.co/xuan-luo/DMTD-Qwen3-4B](https://huggingface.co/xuan-luo/DMTD-Qwen3-4B)

**Direct Multi-Token Decoding (DMTD)** reuses the *late* ("decoding") layers of a decoder-only LLM to directly emit multiple tokens per cycle. DMTD adds **no extra parameters, no draft model, and no post-generation verification** — it simply fine-tunes the original network to decode in fixed multi-token cycles,
achieving up to ~2× inference speedup with minor quality loss.

> This repository contains the training code only. The model live on the Hugging Face Hub: [xuan-luo/DMTD-Qwen3-4B](https://huggingface.co/xuan-luo/DMTD-Qwen3-4B).

---



## Repository contents


| File                                       | Purpose                                                          |
| ------------------------------------------ | ---------------------------------------------------------------- |
| `swift-training.sh`                        | End-to-end SFT training launcher (ms-swift, 8× GPU).             |
| `datasets/export_am_messages_for_swift.py` | Convert raw AM-Thinking JSONL into swift-ready `messages` JSONL. |
| `datasets/export_swift_cached.sh`          | Pre-tokenize the JSONL into an ms-swift `cached_dataset`.        |


---

## Get the model code and weights

The training scripts need the DMTD model implementation (`register_dmtdqwen3.py`, `modeling_*.py`,
`config.json`) and initial weights, all hosted on Hugging Face. Download the model repo and point
`MODEL_PATH` at it:

```bash
hf download xuan-luo/DMTD-Qwen3-4B --local-dir DMTD-Qwen3-4B
export MODEL_PATH="$PWD/DMTD-Qwen3-4B"
```

`MODEL_PATH` must contain a **real** `model.safetensors` and `tokenizer.json` (not symlinks) — the
training launcher checks this and prints a fix hint if they are missing.

---



## Training

Training uses [ms-swift](https://github.com/modelscope/ms-swift) SFT on the
[a-m-team/AM-Thinking-v1-Distilled](https://huggingface.co/datasets/a-m-team/AM-Thinking-v1-Distilled)
dataset. The pipeline has three steps: **download data → process data → train**.

### Step 1 — Download the dataset

```bash
hf download a-m-team/AM-Thinking-v1-Distilled \
    --repo-type dataset \
    --local-dir datasets/raw-am-distilled
```



### Step 2 — Process the data

The dataset ships with per-turn metadata (`info`, `more`, etc.) that breaks Arrow schema inference.
Two scripts turn it into a pre-tokenized, training-ready cache.

**2a. Normalize to** `messages`**-only JSONL** — `export_am_messages_for_swift.py` walks the raw folder,
maps roles (`human/gpt → user/assistant`), drops all extra metadata, and writes one clean
`{"messages": [...]}` per line:

```bash
python datasets/export_am_messages_for_swift.py \
    --data_dir   datasets/raw-am-distilled \
    --output_dir datasets/am-thinking-messages-jsonl \
    --skip_bad_rows
```

- `--data_dir`: folder of raw `.jsonl(.gz)` / `.json(.gz)` files (default: `datasets/raw-am-distilled`).
- `--output_dir`: destination for the cleaned JSONL (default: `datasets/am-thinking-messages-jsonl`).
- `--skip_bad_rows`: skip un-parseable rows instead of aborting.

**2b. Pre-tokenize into a swift** `cached_dataset` — `export_swift_cached.sh` runs `swift export`
once so that later training runs skip the tokenize stage. The cache is keyed only on
(tokenizer + chat template + max_length + loss_scale), so all DMTD variants share the same cache.

```bash
MODEL_PATH="$MODEL_PATH" \
DATASET_DIR=datasets/am-thinking-messages-jsonl \
MAX_LENGTH=4096 \
LOSS_SCALE=ignore_empty_think \
    bash datasets/export_swift_cached.sh
```

Key environment overrides (all have sensible defaults):

- `MODEL_PATH`: any DMTD-Qwen3 dir (only its tokenizer + chat template are used).
- `DATASET_DIR`: the cleaned JSONL folder from step 2a.
- `MAX_LENGTH`: max sequence length baked into the cache (default `4096`).
- `LOSS_SCALE`: loss weighting scheme (default `ignore_empty_think`).
- `OUTPUT_DIR`: where the cached dataset is written.

The result lands in `.../am-thinking-swift-cached-<MAX_LENGTH>-<LOSS_SCALE>/train`, which is what
training consumes via `--cached_dataset`.

### Step 3 — Train

`swift-training.sh` launches full-parameter SFT on 8 GPUs. It will auto-build the cache (step 2b)
if it is missing, then run `swift sft` with DeepSpeed ZeRO-3, packing, Liger kernels, and FlashAttention:

```bash
MODEL_PATH="$MODEL_PATH" bash swift-training.sh
```

Useful environment overrides:

- `MODEL_PATH`: starting checkpoint (the Hugging Face model dir downloaded above).
- `CACHED_DATASET_DIR`: pre-built swift cache to train on.
- `MAX_LENGTH` / `LOSS_SCALE`: must match the values used to build the cache.
- `OUTPUT_DIR`: where checkpoints are written.
- `CUDA_VISIBLE_DEVICES` / `NPROC_PER_NODE`: GPU selection (default: 8 GPUs).

---



## Citation

```bibtex
@article{luo2025dmtd,
  title   = {Direct Multi-Token Decoding},
  author  = {Luo, Xuan and Wang, Weizhi and Yan, Xifeng},
  journal = {arXiv preprint arXiv:2510.11958},
  year    = {2025},
  url     = {https://arxiv.org/abs/2510.11958}
}
```
