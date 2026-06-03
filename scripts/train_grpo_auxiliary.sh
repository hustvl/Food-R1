#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FOODR1_ROOT="${FOODR1_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

if [[ -n "${FOODR1_ENV:-}" ]]; then
  # shellcheck source=/dev/null
  source "${FOODR1_ENV}"
fi

SWIFT_ROOT="${SWIFT_ROOT:?Set SWIFT_ROOT to your ms-swift checkout.}"
cd "${SWIFT_ROOT}"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-1024}"
export VLLM_ENABLE_CUDA_COMPATIBILITY="${VLLM_ENABLE_CUDA_COMPATIBILITY:-1}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-TORCH_SDPA}"
export PYTHONPATH="${FOODR1_ROOT}/foodr1:${FOODR1_ROOT}:${SWIFT_ROOT}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"

DETECTED_GPUS="$(python -c 'import torch; print(torch.cuda.device_count() or 1)' 2>/dev/null || echo 1)"
GPUS="${GPUS:-${MLP_WORKER_GPU:-${IDP_N_GPU:-${DETECTED_GPUS}}}}"
NNODES="${NNODES:-${MLP_WORKER_NUM:-${IDP_N_NODES:-1}}}"
NODE_RANK="${NODE_RANK:-${MLP_ROLE_INDEX:-${IDP_N_RANK:-0}}}"
MASTER_ADDR="${MASTER_ADDR:-${MLP_WORKER_0_HOST:-${IDP_MASTER_ADDR:-127.0.0.1}}}"
MASTER_PORT="${MASTER_PORT:-${MLP_WORKER_0_PORT:-29500}}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-${GPUS}}"
if [[ "${VLLM_TENSOR_PARALLEL_SIZE}" -gt 4 ]]; then
  VLLM_TENSOR_PARALLEL_SIZE=4
fi

if [[ "${NNODES}" == "1" ]]; then
  DISTRIBUTED_ARGS=(--standalone --nproc_per_node "${GPUS}")
else
  DISTRIBUTED_ARGS=(
    --nproc_per_node "${GPUS}"
    --nnodes "${NNODES}"
    --node_rank "${NODE_RANK}"
    --master_addr "${MASTER_ADDR}"
    --master_port "${MASTER_PORT}"
  )
fi

run_grpo() {
  local task_name="$1"
  local model_path="$2"
  local output_dir="$3"
  shift 3

  local reward_args=()
  while [[ "$#" -gt 0 && "$1" != "--" ]]; do
    reward_args+=("$1")
    shift
  done
  shift
  local dataset_args=("$@")

  if [[ -z "${model_path}" ]]; then
    echo "Skip ${task_name}: model path is empty." >&2
    return 0
  fi
  if [[ "${#dataset_args[@]}" -eq 0 || -z "${dataset_args[0]}" ]]; then
    echo "Skip ${task_name}: no datasets configured." >&2
    return 0
  fi
  for dataset_path in "${dataset_args[@]}"; do
    if [[ ! -f "${dataset_path}" ]]; then
      echo "Missing ${task_name} dataset: ${dataset_path}" >&2
      exit 1
    fi
  done

  local reward_weights=()
  for _ in "${reward_args[@]}"; do
    reward_weights+=(1.0)
  done

  echo "Food-R1 auxiliary GRPO: ${task_name}"
  echo "MODEL=${model_path}"
  echo "OUTPUT=${output_dir}"
  printf 'DATASET=%s\n' "${dataset_args[@]}"
  printf 'REWARD=%s\n' "${reward_args[@]}"

  MAX_PIXELS="${MAX_PIXELS:-1048576}" \
  NPROC_PER_NODE="${GPUS}" \
  torchrun "${DISTRIBUTED_ARGS[@]}" \
    swift/cli/rlhf.py \
    --rlhf_type grpo \
    --model "${model_path}" \
    --model_type "${MODEL_TYPE:-qwen3_vl}" \
    --plugin "${FOODR1_ROOT}/foodr1/rewards.py" \
    --reward_funcs "${reward_args[@]}" \
    --reward_weights "${reward_weights[@]}" \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.5}" \
    --vllm_max_model_len "${VLLM_MAX_MODEL_LEN:-8192}" \
    --vllm_tensor_parallel_size "${VLLM_TENSOR_PARALLEL_SIZE}" \
    --train_type full \
    --dataset "${dataset_args[@]}" \
    --torch_dtype bfloat16 \
    --num_train_epochs "${AUX_RL_NUM_TRAIN_EPOCHS:-3}" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate "${AUX_RL_LR:-1e-6}" \
    --weight_decay "${AUX_RL_WEIGHT_DECAY:-0.05}" \
    --gradient_accumulation_steps "${AUX_RL_GRADIENT_ACCUMULATION_STEPS:-16}" \
    --eval_steps "${AUX_RL_EVAL_STEPS:-2000}" \
    --save_steps "${AUX_RL_SAVE_STEPS:-2000}" \
    --save_total_limit "${AUX_RL_SAVE_TOTAL_LIMIT:-1}" \
    --logging_steps "${AUX_RL_LOGGING_STEPS:-5}" \
    --max_length "${AUX_RL_MAX_LENGTH:-2048}" \
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-16}" \
    --num_generations "${NUM_GENERATIONS:-8}" \
    --log_completions true \
    --epsilon "${AUX_RL_EPSILON:-0.2}" \
    --epsilon_high "${AUX_RL_EPSILON_HIGH:-0.28}" \
    --temperature "${AUX_RL_TEMPERATURE:-0.9}" \
    --top_p "${AUX_RL_TOP_P:-0.9}" \
    --max_grad_norm "${AUX_RL_MAX_GRAD_NORM:-0.5}" \
    --warmup_ratio "${AUX_RL_WARMUP_RATIO:-0.05}" \
    --dataset_num_proc "${DATASET_NUM_PROC:-16}" \
    --save_only_model true \
    --output_dir "${output_dir}" \
    --deepspeed "${AUX_DEEPSPEED:-zero2}" \
    --attn_impl "${AUX_RL_ATTN_IMPL:-flash_attn}"
}

# This script intentionally keeps auxiliary RL to ingredient and nutrition objectives.
if [[ "${RUN_VIREO172_RL:-1}" == "1" ]]; then
  IFS=':' read -r -a VIREO172_DATASET_ARRAY <<< "${VIREO172_RL_DATASETS:-}"
  run_grpo \
    "vireo172_ingredient" \
    "${VIREO172_INITIAL_MODEL:-${AUX_INITIAL_MODEL:-}}" \
    "${VIREO172_RL_OUTPUT_DIR:-outputs/grpo_vireo172_ingredient}" \
    vireo172_format vireo172_ingredient_match \
    -- "${VIREO172_DATASET_ARRAY[@]}"
fi

if [[ "${RUN_NUTRITION5K_RL:-1}" == "1" ]]; then
  IFS=':' read -r -a NUTRITION5K_DATASET_ARRAY <<< "${NUTRITION5K_RL_DATASETS:-}"
  run_grpo \
    "nutrition5k" \
    "${NUTRITION5K_INITIAL_MODEL:-${AUX_INITIAL_MODEL:-}}" \
    "${NUTRITION5K_RL_OUTPUT_DIR:-outputs/grpo_nutrition5k}" \
    nutrition5k_format nutrition5k_kcal_accuracy nutrition5k_full_nutrition_accuracy \
    -- "${NUTRITION5K_DATASET_ARRAY[@]}"
fi
