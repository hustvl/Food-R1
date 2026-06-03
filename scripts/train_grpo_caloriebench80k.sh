#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FOODR1_ROOT="${FOODR1_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

if [[ -n "${FOODR1_ENV:-}" ]]; then
  # shellcheck source=/dev/null
  source "${FOODR1_ENV}"
fi

SWIFT_ROOT="${SWIFT_ROOT:?Set SWIFT_ROOT to your ms-swift checkout.}"
INITIAL_MODEL="${INITIAL_MODEL:?Set INITIAL_MODEL to the Food-R1 SFT checkpoint.}"
RL_DATASET="${RL_DATASET:?Set RL_DATASET to the full CalorieBench-80K training GRPO dataset.}"
RL_OUTPUT_DIR="${RL_OUTPUT_DIR:-outputs/grpo_caloriebench80k_full_train}"

if [[ ! -f "${RL_DATASET}" ]]; then
  echo "Missing full CalorieBench-80K GRPO training dataset: ${RL_DATASET}" >&2
  exit 1
fi

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

MODEL_PATH="${INITIAL_MODEL}"
RESUME_ARGS=()
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  RESUME_ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
elif [[ "${AUTO_RESUME:-1}" == "1" && -d "${RL_OUTPUT_DIR}" ]]; then
  LATEST_MODEL_CHECKPOINT="$(
    find "${RL_OUTPUT_DIR}" -mindepth 2 -maxdepth 2 -type d -name 'checkpoint-*' -printf '%T@ %p\n' 2>/dev/null \
      | sort -nr | cut -d' ' -f2- | while read -r checkpoint_dir; do
          if [[ -f "${checkpoint_dir}/config.json" ]] && { [[ -f "${checkpoint_dir}/model.safetensors.index.json" ]] || find "${checkpoint_dir}" -maxdepth 1 \( -name '*.safetensors' -o -name 'pytorch_model*.bin' \) -print -quit | grep -q .; }; then
            echo "${checkpoint_dir}"
            break
          fi
        done
  )"
  if [[ -n "${LATEST_MODEL_CHECKPOINT}" ]]; then
    MODEL_PATH="${LATEST_MODEL_CHECKPOINT}"
    echo "AUTO_RESUME: using latest model checkpoint as MODEL_PATH=${MODEL_PATH}"
  fi
fi

EXTRA_ARGS=()
if [[ -n "${MAX_STEPS:-}" ]]; then
  EXTRA_ARGS+=(--max_steps "${MAX_STEPS}")
fi

echo "Food-R1 GRPO"
echo "SWIFT_ROOT=${SWIFT_ROOT}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "RL_DATASET=${RL_DATASET} (full CalorieBench-80K train)"
echo "RL_OUTPUT_DIR=${RL_OUTPUT_DIR}"

MAX_PIXELS="${MAX_PIXELS:-1048576}" \
NPROC_PER_NODE="${GPUS}" \
torchrun "${DISTRIBUTED_ARGS[@]}" \
  swift/cli/rlhf.py \
  --rlhf_type grpo \
  --model "${MODEL_PATH}" \
  --model_type "${MODEL_TYPE:-qwen3_vl}" \
  --external_plugins "${FOODR1_ROOT}/foodr1/rewards.py" \
  --reward_funcs ingredient_format ingredient_match ingredient_quantity_match total_kcal_exact \
  --reward_weights 1.0 1.0 1.0 1.0 \
  --use_vllm true \
  --vllm_mode colocate \
  --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.35}" \
  --vllm_max_model_len "${VLLM_MAX_MODEL_LEN:-8192}" \
  --vllm_tensor_parallel_size "${VLLM_TENSOR_PARALLEL_SIZE}" \
  --train_type full \
  --dataset "${RL_DATASET}" \
  --torch_dtype bfloat16 \
  --num_train_epochs "${RL_NUM_TRAIN_EPOCHS:-1}" \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --learning_rate "${RL_LR:-1e-6}" \
  --weight_decay "${RL_WEIGHT_DECAY:-0.05}" \
  --gradient_accumulation_steps "${RL_GRADIENT_ACCUMULATION_STEPS:-16}" \
  --eval_steps "${RL_EVAL_STEPS:-2000}" \
  --save_steps "${RL_SAVE_STEPS:-500}" \
  --save_total_limit "${RL_SAVE_TOTAL_LIMIT:-2}" \
  --logging_steps "${RL_LOGGING_STEPS:-5}" \
  --max_length "${RL_MAX_LENGTH:-2048}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-16}" \
  --num_generations "${NUM_GENERATIONS:-8}" \
  --log_completions true \
  --epsilon "${RL_EPSILON:-0.2}" \
  --epsilon_high "${RL_EPSILON_HIGH:-0.28}" \
  --temperature "${RL_TEMPERATURE:-0.9}" \
  --top_p "${RL_TOP_P:-0.9}" \
  --max_grad_norm "${RL_MAX_GRAD_NORM:-0.5}" \
  --warmup_ratio "${RL_WARMUP_RATIO:-0.05}" \
  --dataset_num_proc "${DATASET_NUM_PROC:-16}" \
  "${EXTRA_ARGS[@]}" \
  "${RESUME_ARGS[@]}" \
  --save_only_model true \
  --output_dir "${RL_OUTPUT_DIR}" \
  --deepspeed "${DEEPSPEED:-zero3_offload}" \
  --offload_optimizer "${OFFLOAD_OPTIMIZER:-true}" \
  --offload_model "${OFFLOAD_MODEL:-true}" \
  --attn_impl "${RL_ATTN_IMPL:-sdpa}"
