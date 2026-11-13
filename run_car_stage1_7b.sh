#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
FACTORY="$ROOT/LLaMA-Factory"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"

# This launcher is only for Qwen2.5-VL-7B-Instruct Stage 1. ZeRO-3 partitions
# model parameters, gradients, and optimizer state across the three 24 GB GPUs.
BASE_MODEL="${BASE_MODEL:-$ROOT/models/qwen/Qwen2.5-VL-7B-Instruct}"
SEED="${SEED:-42}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-3}"
MAX_STEPS="${MAX_STEPS:-}"
OUTPUT="${OUTPUT:-$ROOT/checkpoints/7b/seed_${SEED}/stage1}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/car_predpo_qwen2_5_vl_7b/gpu012/seed_${SEED}}"
DATASET_DIR="${DATASET_DIR:-$ROOT/data/prepared/stage1}"
TRAIN_DATASET="${TRAIN_DATASET:-car_predpo_stage1_full}"
DRY_RUN="${DRY_RUN:-0}"
ALLOW_EXISTING_OUTPUT="${ALLOW_EXISTING_OUTPUT:-0}"

IFS=',' read -r -a CUDA_DEVICE_ARRAY <<< "$CUDA_DEVICES"
if [[ "${#CUDA_DEVICE_ARRAY[@]}" -ne "$NPROC_PER_NODE" ]]; then
  echo "[ERROR] CUDA_DEVICES count (${#CUDA_DEVICE_ARRAY[@]}) must match NPROC_PER_NODE ($NPROC_PER_NODE)." >&2
  exit 2
fi

mkdir -p "$LOG_DIR"
cmd=(
  "$TORCHRUN_BIN" --nproc_per_node="$NPROC_PER_NODE" src/train.py
  --deepspeed examples/deepspeed/ds_z3_config.json
  --stage dpo --do_train
  --model_name_or_path "$BASE_MODEL"
  --dataset "$TRAIN_DATASET" --dataset_dir "$DATASET_DIR" --template qwen2_vl
  --finetuning_type lora
  --lora_target q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
  --lora_rank 16 --lora_alpha 32 --lora_dropout 0.05
  --output_dir "$OUTPUT" --overwrite_cache true
  --per_device_train_batch_size 1 --gradient_accumulation_steps 8
  --learning_rate 5e-6 --lr_scheduler_type cosine --warmup_ratio 0.03
  --max_grad_norm 1.0 --num_train_epochs 1.0
  --cutoff_len 2048 --image_max_pixels 262144
  --logging_steps 5
  --save_strategy steps --save_steps 100 --save_total_limit 4
  --pref_loss sigmoid --pref_beta 0.1 --bf16 true --plot_loss
  --gradient_checkpointing true --preprocessing_num_workers 2
  --eval_strategy no
  --ddp_timeout 180000 --seed "$SEED"
)
if [[ -n "$MAX_STEPS" ]]; then cmd+=(--max_steps "$MAX_STEPS"); fi

printf "CUDA_VISIBLE_DEVICES=%s\n" "$CUDA_DEVICES"
printf "CAR 7B Stage 1 (ZeRO-3): "; printf "%q " "${cmd[@]}"; printf "\n"
if [[ "$DRY_RUN" == 1 ]]; then exit 0; fi

for path in "$BASE_MODEL/config.json" "$FACTORY/src/train.py" "$FACTORY/examples/deepspeed/ds_z3_config.json" "$DATASET_DIR/dataset_info.json"; do
  [[ -e "$path" ]] || { echo "[ERROR] missing $path" >&2; exit 1; }
done
if [[ -e "$OUTPUT" && "$ALLOW_EXISTING_OUTPUT" != 1 ]]; then
  echo "[ERROR] protected output already exists: $OUTPUT" >&2
  exit 1
fi

cd "$FACTORY"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
"${cmd[@]}" 2>&1 | tee "$LOG_DIR/stage1.log"
[[ -f "$OUTPUT/adapter_config.json" && -f "$OUTPUT/trainer_state.json" ]] || {
  echo "[ERROR] incomplete Stage 1 output" >&2; exit 1;
}
compgen -G "$OUTPUT/adapter_model.*" >/dev/null || { echo "[ERROR] missing adapter weights" >&2; exit 1; }
