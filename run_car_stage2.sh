#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
FACTORY="$ROOT/LLaMA-Factory"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_MODEL="${BASE_MODEL:-${BASE_MODEL:-$ROOT/models/qwen/Qwen2.5-VL-3B-Instruct}}"
SEED="${SEED:-42}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2}"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
NPROC_PER_NODE="${NPROC_PER_NODE:-3}"
MASTER_PORT="${MASTER_PORT:-}"
USE_DEEPSPEED="${USE_DEEPSPEED:-1}"
MAX_STEPS="${MAX_STEPS:-}"
STAGE1="${STAGE1:-${STAGE1:-$ROOT/checkpoints/3b/seed_${SEED}/stage1}}"
OUTPUT="${OUTPUT:-$ROOT/checkpoints/3b/seed_${SEED}/stage2}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/car_predpo/seed_${SEED}}"
DRY_RUN="${DRY_RUN:-0}"
ALLOW_EXISTING_OUTPUT="${ALLOW_EXISTING_OUTPUT:-0}"
DATASET_DIR="${DATASET_DIR:-${DATASET_DIR:-$ROOT/data/prepared/3b/stage2}}"
TRAIN_DATASET="${TRAIN_DATASET:-car_predpo_train_scored}"
DEV_DATASET="${DEV_DATASET:-car_predpo_dev_scored}"
EXPECTED_TRAIN="${EXPECTED_TRAIN:-5131}"
EXPECTED_DEV="${EXPECTED_DEV:-570}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1.0}"
mkdir -p "$LOG_DIR"

master_port_args=()
if [[ -n "$MASTER_PORT" ]]; then master_port_args=(--master_port "$MASTER_PORT"); fi
cmd=("$TORCHRUN_BIN" --nproc_per_node="$NPROC_PER_NODE" "${master_port_args[@]}" src/train.py)
if [[ "$USE_DEEPSPEED" == 1 ]]; then cmd+=(--deepspeed examples/deepspeed/ds_z3_config.json); fi
cmd+=(
  --stage dpo --do_train --do_eval
  --model_name_or_path "$BASE_MODEL" --adapter_name_or_path "$STAGE1"
  --dataset "$TRAIN_DATASET" --eval_dataset "$DEV_DATASET"
  --dataset_dir "$DATASET_DIR" --template qwen2_vl
  --finetuning_type lora
  --lora_target q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
  --lora_rank 16 --lora_alpha 32 --lora_dropout 0.05
  --output_dir "$OUTPUT" --overwrite_cache true
  --per_device_train_batch_size 1 --gradient_accumulation_steps 8
  --per_device_eval_batch_size 1
  --learning_rate 2e-6 --lr_scheduler_type cosine --warmup_ratio 0.03
  --max_grad_norm 1.0 --num_train_epochs "$NUM_TRAIN_EPOCHS"
  --cutoff_len 2048 --image_max_pixels 262144
  --logging_steps 5 --eval_strategy steps --eval_steps 25
  --save_strategy steps --save_steps 50 --save_total_limit 4
  --pref_loss sigmoid --pref_beta 0.1 --bf16 true --plot_loss
  --gradient_checkpointing true --preprocessing_num_workers 2
  --ddp_timeout 180000 --seed "$SEED"
)
if [[ -n "$MAX_STEPS" ]]; then
  cmd+=(--max_steps "$MAX_STEPS")
fi

printf 'CUDA_VISIBLE_DEVICES=%s\n' "$CUDA_DEVICES"
printf 'CAR Stage 2: '; printf '%q ' "${cmd[@]}"; printf '\n'
PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" "$ROOT/scripts/car_predpo/validate_stage2_data.py" \
  --dataset-dir "$DATASET_DIR" \
  --train-dataset "$TRAIN_DATASET" \
  --dev-dataset "$DEV_DATASET" \
  --expected-train "$EXPECTED_TRAIN" \
  --expected-dev "$EXPECTED_DEV"
if [[ "$DRY_RUN" == 1 ]]; then
  exit 0
fi
for path in "$BASE_MODEL/config.json" "$STAGE1/adapter_config.json"; do
  [[ -e "$path" ]] || { echo "[ERROR] missing $path" >&2; exit 1; }
done
if [[ -e "$OUTPUT" && "$ALLOW_EXISTING_OUTPUT" != 1 ]]; then
  echo "[ERROR] protected output already exists: $OUTPUT" >&2
  exit 1
fi
cd "$FACTORY"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
"${cmd[@]}" 2>&1 | tee "$LOG_DIR/stage2.log"
[[ -f "$OUTPUT/adapter_config.json" && -f "$OUTPUT/trainer_state.json" ]] || {
  echo "[ERROR] incomplete Stage 2 output" >&2; exit 1;
}
compgen -G "$OUTPUT/adapter_model.*" >/dev/null || { echo "[ERROR] missing adapter weights" >&2; exit 1; }
