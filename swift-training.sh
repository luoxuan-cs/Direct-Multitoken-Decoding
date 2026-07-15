#!/usr/bin/env bash
# MS-SWIFT SFT for DMTD Qwen3 (8x A100 40GB)
# Reference: https://swift.readthedocs.io/en/v3.12/BestPractices/Qwen3-Best-Practice.html
#
# Uses pre-built `cached_dataset` under datasets/ (am-thinking, 4096, ignore_empty_think).
# Packing (packing_length=4096) runs during training, not in the cache.

set -euo pipefail

###################
# User-configurable paths
###################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${MODEL_PATH:-${SCRIPT_DIR}}"
REGISTER_PLUGIN="${REGISTER_PLUGIN:-${SCRIPT_DIR}/register_dmtdqwen3.py}"

MAX_LENGTH="${MAX_LENGTH:-4096}"
LOSS_SCALE="${LOSS_SCALE:-ignore_empty_think}"
DATASETS_ROOT="${DATASETS_ROOT:-${SCRIPT_DIR}/datasets}"
CACHED_DATASET_DIR="${CACHED_DATASET_DIR:-${DATASETS_ROOT}/am-thinking-swift-cached-${MAX_LENGTH}-${LOSS_SCALE}}"

OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/am-thinking-E0D8C4_ctx4096_lr1e-4_1epoch}"

CONDA_ENV="${CONDA_ENV:-llm-swift}"
# Resolved from PATH by default; activate your conda env before running, or set SWIFT_BIN explicitly.
SWIFT_BIN="${SWIFT_BIN:-$(command -v swift || echo swift)}"

###################
# Hardware: 8 GPUs on one node
###################
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

###################
# Memory / NCCL
###################
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

###################
# Training launch
###################
# Global batch size = NPROC * per_device_train_batch_size * gradient_accumulation_steps
#                   = 8 * 1 * 64 = 512

mkdir -p "${OUTPUT_DIR}"

# Weight file must be a real file, not a symlink (broken links break `cp` and swift load).
WEIGHT_FILE="${MODEL_PATH}/model.safetensors"
TOKENIZER_FILE="${MODEL_PATH}/tokenizer.json"
if [[ -L "${WEIGHT_FILE}" ]] || [[ ! -f "${WEIGHT_FILE}" ]]; then
    echo "ERROR: ${WEIGHT_FILE} is missing or a symlink (swift cannot load it)."
    echo "Fix (use cp -L to follow source symlinks and write a real file):"
    echo "  cd ${MODEL_PATH}"
    echo "  rm -f model.safetensors tokenizer.json"
    echo "  cp -L -f /path/to/source-model/model.safetensors ./model.safetensors"
    echo "  cp -f /path/to/source-model/tokenizer.json ./tokenizer.json"
    echo "Do NOT use: cp DMTD/models/DMTDQwen3-4B/model.safetensors ./  (that only copies the link)"
    exit 1
fi
if [[ -L "${TOKENIZER_FILE}" ]] || [[ ! -f "${TOKENIZER_FILE}" ]]; then
    echo "ERROR: ${TOKENIZER_FILE} is missing or a symlink."
    exit 1
fi

if [[ ! -d "${CACHED_DATASET_DIR}/train" ]]; then
    echo "ERROR: Swift cached_dataset not found: ${CACHED_DATASET_DIR}/train"
    echo "Expected pre-built cache at: datasets/am-thinking-swift-cached-${MAX_LENGTH}-${LOSS_SCALE}/train"
    exit 1
fi

echo "Model:           ${MODEL_PATH}"
echo "Cached dataset:  ${CACHED_DATASET_DIR}/train"
echo "Output:          ${OUTPUT_DIR}"
echo "GPUs:            ${CUDA_VISIBLE_DEVICES} (NPROC_PER_NODE=${NPROC_PER_NODE})"
echo "Epochs:          1"
echo "Swift:           ${SWIFT_BIN}"

export NPROC_PER_NODE="${NPROC_PER_NODE}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"

"${SWIFT_BIN}" sft \
    --model "${MODEL_PATH}" \
    --model_type dmtdqwen3 \
    --external_plugins "${REGISTER_PLUGIN}" \
    --tuner_type full \
    --cached_dataset "${CACHED_DATASET_DIR}/train" \
    --torch_dtype bfloat16 \
    --attn_impl flash_attn \
    --use_liger_kernel true \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 32 \
    --learning_rate 1e-4 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --max_grad_norm 1.0 \
    --packing true \
    --padding_free false \
    --packing_length "${MAX_LENGTH}" \
    --truncation_strategy left \
    --gradient_checkpointing true \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --dataloader_drop_last true \
    --logging_steps 10 \
    --save_steps 100 \
    --save_total_limit 1 \
    --seed 42 \
    --data_seed 42 \
    --dataset_num_proc 8 \
    --dataloader_num_workers 4 \
    --deepspeed zero3 \
    --output_dir "${OUTPUT_DIR}" \
    "$@"
