#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
FACTORY="$ROOT/LLaMA-Factory"
PYTHON_BIN="${PYTHON_BIN:-python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
BASE_MODEL="${BASE_MODEL:-${BASE_MODEL:-$ROOT/models/qwen/Qwen2.5-VL-3B-Instruct}}"
SEED="${SEED:-42}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-3}"
USE_DEEPSPEED="${USE_DEEPSPEED:-1}"
MAX_STEPS="${MAX_STEPS:-}"
OUTPUT="${OUTPUT:-$ROOT/checkpoints/3b/seed_${SEED}/stage1}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/car_predpo/seed_${SEED}}"
DATASET_DIR="${DATASET_DIR:-${DATASET_DIR:-$ROOT/data/prepared/stage1}}"
TRAIN_DATASET="${TRAIN_DATASET:-car_predpo_stage1_full}"
DEV_DATASET="${DEV_DATASET:-car_predpo_dev}"
DO_EVAL="${DO_EVAL:-0}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
PURE_BF16="${PURE_BF16:-0}"
DRY_RUN="${DRY_RUN:-0}"
ALLOW_EXISTING_OUTPUT="${ALLOW_EXISTING_OUTPUT:-0}"
mkdir -p "$LOG_DIR"

cmd=("$TORCHRUN_BIN" --nproc_per_node="$NPROC_PER_NODE" src/train.py)
if [[ "$USE_DEEPSPEED" == 1 ]]; then cmd+=(--deepspeed examples/deepspeed/ds_z3_config.json); fi
cmd+=(
  --stage dpo --do_train
  --model_name_or_path "$BASE_MODEL"
  --dataset "$TRAIN_DATASET" --dataset_dir "$DATASET_DIR" --template qwen2_vl
  --finetuning_type lora
  --lora_target q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
  --lora_rank 16 --lora_alpha 32 --lora_dropout 0.05
  --output_dir "$OUTPUT" --overwrite_cache true
  --per_device_train_batch_size 1 --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
  --learning_rate 5e-6 --lr_scheduler_type cosine --warmup_ratio 0.03
  --max_grad_norm 1.0 --num_train_epochs 1.0
  --cutoff_len 2048 --image_max_pixels 262144
  --logging_steps 5
  --save_strategy steps --save_steps 100 --save_total_limit 4
  --pref_loss sigmoid --pref_beta 0.1 --bf16 true --plot_loss
  --gradient_checkpointing true --preprocessing_num_workers 2
  --ddp_timeout 180000 --seed "$SEED"
)
if [[ "$DO_EVAL" == 1 ]]; then
  cmd+=(--do_eval --eval_dataset "$DEV_DATASET" --per_device_eval_batch_size 1 --eval_strategy steps --eval_steps 25)
else
  cmd+=(--eval_strategy no)
fi
if [[ "$PURE_BF16" == 1 ]]; then cmd+=(--pure_bf16 true); fi
if [[ -n "$MAX_STEPS" ]]; then cmd+=(--max_steps "$MAX_STEPS"); fi

printf "CUDA_VISIBLE_DEVICES=%s\n" "$CUDA_DEVICES"
printf "CAR Stage 1: "; printf "%q " "${cmd[@]}"; printf "\n"
if [[ "$DRY_RUN" == 1 ]]; then exit 0; fi
for path in "$BASE_MODEL/config.json" "$FACTORY/src/train.py" "$DATASET_DIR/dataset_info.json"; do
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
