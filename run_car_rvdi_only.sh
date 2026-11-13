#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
FACTORY="$ROOT/LLaMA-Factory"
PYTHON_BIN="${PYTHON_BIN:-python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
MODEL_SIZE="${MODEL_SIZE:-3b}"
SEED="${SEED:-42}"
RUN_ID="${RUN_ID:?Set a unique RUN_ID}"
CUDA_DEVICES="${CUDA_DEVICES:?Set the IDs of exclusively free GPUs}"
NPROC_PER_NODE="${NPROC_PER_NODE:-3}"
MASTER_PORT="${MASTER_PORT:-29522}"
MAX_STEPS="${MAX_STEPS:-}"
DRY_RUN="${DRY_RUN:-0}"

case "$MODEL_SIZE" in
  3b)
    BASE_MODEL="${BASE_MODEL:-$ROOT/models/qwen/Qwen2.5-VL-3B-Instruct}"
    STAGE1="${STAGE1:-$ROOT/checkpoints/3b/seed_${SEED}/stage1}"
    LEARNING_RATE="${LEARNING_RATE:-2e-6}"
    ;;
  7b)
    BASE_MODEL="${BASE_MODEL:-$ROOT/models/qwen/Qwen2.5-VL-7B-Instruct}"
    STAGE1="${STAGE1:-$ROOT/checkpoints/7b/seed_${SEED}/stage1}"
    LEARNING_RATE="${LEARNING_RATE:-1e-6}"
    ;;
  *) echo "MODEL_SIZE must be 3b or 7b" >&2; exit 2 ;;
esac
case "$NPROC_PER_NODE" in
  2) GRADIENT_ACCUMULATION_STEPS=12 ;;
  3) GRADIENT_ACCUMULATION_STEPS=8 ;;
  *) echo "NPROC_PER_NODE must be 2 or 3 to preserve effective batch 24" >&2; exit 2 ;;
esac
IFS=',' read -r -a CUDA_DEVICE_ARRAY <<< "$CUDA_DEVICES"
[[ "${#CUDA_DEVICE_ARRAY[@]}" -eq "$NPROC_PER_NODE" ]] || { echo "GPU count differs from NPROC_PER_NODE" >&2; exit 2; }

DATASET_DIR="${DATASET_DIR:-${DATASET_DIR:-$ROOT/data/car_predpo_compare/$RUN_ID}}"
OUTPUT="$ROOT/checkpoints/car_predpo_compare/$RUN_ID"
LOG_DIR="$ROOT/logs/car_predpo_compare/$RUN_ID"
cmd=("$TORCHRUN_BIN" --nproc_per_node="$NPROC_PER_NODE" --master_port "$MASTER_PORT" src/train.py
  --deepspeed examples/deepspeed/ds_z3_config.json --stage dpo --do_train --do_eval
  --model_name_or_path "$BASE_MODEL" --adapter_name_or_path "$STAGE1"
  --dataset car_predpo_train_rvdi_only --eval_dataset car_predpo_dev_rvdi_only
  --dataset_dir "$DATASET_DIR" --template qwen2_vl --finetuning_type lora
  --lora_target q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
  --lora_rank 16 --lora_alpha 32 --lora_dropout 0.05
  --output_dir "$OUTPUT" --overwrite_cache true
  --per_device_train_batch_size 1 --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
  --per_device_eval_batch_size 1 --learning_rate "$LEARNING_RATE"
  --lr_scheduler_type cosine --warmup_ratio 0.03 --max_grad_norm 1.0 --num_train_epochs 1.0
  --cutoff_len 2048 --image_max_pixels 262144 --logging_steps 5 --eval_strategy steps --eval_steps 25
  --save_strategy steps --save_steps 50 --save_total_limit 4
  --pref_loss sigmoid --pref_beta 0.1
  --bf16 true --plot_loss --gradient_checkpointing true --preprocessing_num_workers 2
  --ddp_timeout 180000 --seed "$SEED")
[[ -z "$MAX_STEPS" ]] || cmd+=(--max_steps "$MAX_STEPS")

printf 'CUDA_VISIBLE_DEVICES=%s\n' "$CUDA_DEVICES"
printf 'RVDI-only %s: ' "$MODEL_SIZE"; printf '%q ' "${cmd[@]}"; printf '\n'
printf 'Output=%s Log=%s/train.log\n' "$OUTPUT" "$LOG_DIR"
[[ "$DRY_RUN" != 1 ]] || exit 0

for path in "$BASE_MODEL/config.json" "$STAGE1/adapter_config.json" "$DATASET_DIR/dataset_info.json" "$FACTORY/src/train.py"; do
  [[ -e "$path" ]] || { echo "Missing $path" >&2; exit 1; }
done
[[ ! -e "$OUTPUT" ]] || { echo "Protected output already exists: $OUTPUT" >&2; exit 1; }
PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" "$ROOT/scripts/car_predpo/validate_stage2_data.py" \
  --dataset-dir "$DATASET_DIR" --train-dataset car_predpo_train_rvdi_only \
  --dev-dataset car_predpo_dev_rvdi_only --expected-train 5131 --expected-dev 570

mkdir -p "$LOG_DIR"
cd "$FACTORY"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
"${cmd[@]}" 2>&1 | tee "$LOG_DIR/train.log"
[[ -f "$OUTPUT/adapter_config.json" && -f "$OUTPUT/trainer_state.json" ]] || { echo "Incomplete output" >&2; exit 1; }
compgen -G "$OUTPUT/adapter_model.*" >/dev/null || { echo "Missing adapter weights" >&2; exit 1; }
