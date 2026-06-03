#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FOODR1_ROOT="${FOODR1_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

if [[ -n "${FOODR1_ENV:-}" ]]; then
  # shellcheck source=/dev/null
  source "${FOODR1_ENV}"
fi

SWIFT_ROOT="${SWIFT_ROOT:?Set SWIFT_ROOT to your ms-swift checkout.}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the base Qwen-VL model path or HF id.}"
SFT_DATASETS="${SFT_DATASETS:?Set SFT_DATASETS to ':'-separated training JSON/JSONL files.}"
SFT_OUTPUT_DIR="${SFT_OUTPUT_DIR:-outputs/sft_foodr1}"

cd "${SWIFT_ROOT}"

export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-torchvision}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONPATH="${FOODR1_ROOT}/foodr1:${FOODR1_ROOT}:${SWIFT_ROOT}:${PYTHONPATH:-}"

GPUS="${GPUS:-${MLP_WORKER_GPU:-${IDP_N_GPU:-8}}}"
NNODES="${NNODES:-${MLP_WORKER_NUM:-${IDP_N_NODES:-1}}}"
NODE_RANK="${NODE_RANK:-${MLP_ROLE_INDEX:-${IDP_N_RANK:-0}}}"
MASTER_ADDR="${MASTER_ADDR:-${MLP_WORKER_0_HOST:-${IDP_MASTER_ADDR:-127.0.0.1}}}"
MASTER_PORT="${MASTER_PORT:-${MLP_WORKER_0_PORT:-29500}}"

IFS=':' read -r -a DATASET_ARRAY <<< "${SFT_DATASETS}"
missing=0
for dataset_path in "${DATASET_ARRAY[@]}"; do
  if [[ ! -f "${dataset_path}" ]]; then
    echo "Missing SFT dataset: ${dataset_path}" >&2
    missing=1
  fi
done
if [[ "${missing}" -ne 0 ]]; then
  exit 1
fi

DISTRIBUTED_ARGS=(
  --nproc_per_node "${GPUS}"
  --nnodes "${NNODES}"
  --node_rank "${NODE_RANK}"
  --master_addr "${MASTER_ADDR}"
  --master_port "${MASTER_PORT}"
)

echo "Food-R1 SFT"
echo "SWIFT_ROOT=${SWIFT_ROOT}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "SFT_OUTPUT_DIR=${SFT_OUTPUT_DIR}"
printf 'SFT_DATASET=%s\n' "${DATASET_ARRAY[@]}"

EXTRA_ARGS=()
if [[ "${SFT_LOAD_SAFETENSORS:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--load_safetensors true)
fi
if [[ "${SFT_SAVE_SAFETENSORS:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--save_safetensors true)
fi
if [[ -n "${SFT_TRAIN_ITERS:-}" ]]; then
  EXTRA_ARGS+=(--train_iters "${SFT_TRAIN_ITERS}")
fi

MAX_PIXELS="${MAX_PIXELS:-1048576}" \
OMP_NUM_THREADS="${OMP_NUM_THREADS:-14}" \
NPROC_PER_NODE="${GPUS}" \
SWIFT_PATCH_CONV3D="${SWIFT_PATCH_CONV3D:-1}" \
torchrun "${DISTRIBUTED_ARGS[@]}" \
  swift/cli/_megatron/sft.py \
  --model "${MODEL_PATH}" \
  --model_type "${MODEL_TYPE:-qwen3_vl}" \
  --no_initialization "${NO_INITIALIZATION:-false}" \
  --dataset "${DATASET_ARRAY[@]}" \
  --load_from_cache_file true \
  --tensor_model_parallel_size "${TENSOR_MODEL_PARALLEL_SIZE:-1}" \
  --sequence_parallel true \
  --packing true \
  --freeze_llm false \
  --freeze_vit false \
  --freeze_aligner false \
  --split_dataset_ratio "${SPLIT_DATASET_RATIO:-0.01}" \
  --micro_batch_size "${MICRO_BATCH_SIZE:-1}" \
  --global_batch_size "${GLOBAL_BATCH_SIZE:-128}" \
  --recompute_granularity full \
  --recompute_method uniform \
  --recompute_num_layers 1 \
  --finetune true \
  --cross_entropy_loss_fusion false \
  --lr "${SFT_LR:-4e-5}" \
  --lr_warmup_fraction "${SFT_LR_WARMUP_FRACTION:-0.05}" \
  --min_lr "${SFT_MIN_LR:-1e-6}" \
  --max_epochs "${SFT_MAX_EPOCHS:-3}" \
  --save "${SFT_OUTPUT_DIR}" \
  --save_interval "${SFT_SAVE_INTERVAL:-500}" \
  --vit_gradient_checkpointing true \
  --max_length "${SFT_MAX_LENGTH:-8192}" \
  --num_workers "${SFT_NUM_WORKERS:-32}" \
  --no_save_optim true \
  --no_save_rng true \
  --dataset_num_proc "${SFT_DATASET_NUM_PROC:-32}" \
  --attention_backend "${SFT_ATTENTION_BACKEND:-flash}" \
  "${EXTRA_ARGS[@]}"

