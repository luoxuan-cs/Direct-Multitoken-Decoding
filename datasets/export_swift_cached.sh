#!/usr/bin/env bash
# One-time: pre-tokenize the AM-Thinking messages JSONL into an ms-swift
# `cached_dataset` so subsequent `swift sft` runs skip the tokenize stage.
#
# The output is keyed only on (tokenizer + chat template + max_length +
# loss_scale). All Qwen3DMTD variants share an identical Qwen3 tokenizer and
# chat template, so the same cached dataset can be reused by Vanilla, E0D8C3,
# and any future EnDxCy model — no need to regenerate per model.
#
# Usage:
#   bash export_swift_cached.sh                       # uses defaults below
#   MAX_LENGTH=8192 bash export_swift_cached.sh       # override max length
#   MODEL_PATH=/path/to/some-dmtd-model bash export_swift_cached.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Any DMTD-Qwen3 model dir works — only its tokenizer + chat template are used.
# Default: the parent directory of datasets/ (i.e. the repo root that contains the model files).
MODEL_PATH="${MODEL_PATH:-${SCRIPT_DIR}/..}"
REGISTER_PLUGIN="${REGISTER_PLUGIN:-${MODEL_PATH}/register_dmtdqwen3.py}"

DATASET_DIR="${DATASET_DIR:-${SCRIPT_DIR}/am-thinking-messages-jsonl}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
LOSS_SCALE="${LOSS_SCALE:-ignore_empty_think}"

# Cached output: shared by all DMTD variants.
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/am-thinking-swift-cached-${MAX_LENGTH}-${LOSS_SCALE}}"

CONDA_ENV="${CONDA_ENV:-llm-swift}"
# Resolved from PATH by default; activate your conda env before running, or set SWIFT_BIN explicitly.
SWIFT_BIN="${SWIFT_BIN:-$(command -v swift || echo swift)}"

echo "Model (tokenizer source): ${MODEL_PATH}"
echo "Dataset:                  ${DATASET_DIR}"
echo "Output (cached_dataset):  ${OUTPUT_DIR}"
echo "max_length:               ${MAX_LENGTH}"
echo "loss_scale:               ${LOSS_SCALE}"

if [[ -d "${OUTPUT_DIR}/train" ]]; then
    echo "Cached dataset already exists at ${OUTPUT_DIR} — nothing to do."
    echo "Pass a different OUTPUT_DIR to re-export, or rm -rf the existing one."
    exit 0
fi

# Notes:
#  * --to_cached_dataset true requires --packing false (packing is deferred to training).
#  * No GPU is used here; the model is materialized on the meta device.
#  * --split_dataset_ratio 0 keeps everything in train (we don't need an eval split).
"${SWIFT_BIN}" export \
    --to_cached_dataset true \
    --model "${MODEL_PATH}" \
    --model_type dmtdqwen3 \
    --external_plugins "${REGISTER_PLUGIN}" \
    --dataset "${DATASET_DIR}" \
    --split_dataset_ratio 0 \
    --max_length "${MAX_LENGTH}" \
    --loss_scale "${LOSS_SCALE}" \
    --dataset_num_proc 16 \
    --output_dir "${OUTPUT_DIR}" \
    "$@"

echo
echo "Done. To reuse in training, pass:"
echo "    --cached_dataset ${OUTPUT_DIR}/train"
echo "(also drop --dataset / --loss_scale / --max_length flags — they're baked in.)"
