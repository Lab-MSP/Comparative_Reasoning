#!/bin/bash
# DPO training (DPO / DPO-CoT) with ms-swift, starting from the base model. Run from the repository root.
#
#   bash src/dpo_swift.sh <train_jsonl> [run_name]
#
#   run_name defaults to the file name of <train_jsonl>; outputs go to outputs/experiments/<run_name>.
#
#   DPO     : bash src/dpo_swift.sh outputs/swift/dpo.jsonl
#   DPO-CoT : bash src/dpo_swift.sh outputs/swift/dpo_cot.jsonl
#
# LoRA rank 64 / alpha 128 on all linear layers, lr 5e-5, beta 0.2, rpo_alpha 1.0 (adds the SFT
# loss on chosen responses), 10 epochs, total batch size 32.

TRAIN_DATA=${1:-outputs/swift/dpo_cot.jsonl}
RUN_NAME=${2:-$(basename "$TRAIN_DATA" .jsonl)}
OUTPUT_DIR="outputs/experiments/${RUN_NAME}"
PRETRAINED_MODEL="Qwen/Qwen2.5-Omni-3B"

# Total batch size = NPROC_PER_NODE x per_device_train_batch_size x gradient_accumulation_steps = 32.
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
swift rlhf \
    --seed 1115 \
    --rlhf_type dpo \
    --model "${PRETRAINED_MODEL}" \
    --model_type qwen2_5_omni \
    --model_name "${RUN_NAME}" \
    --train_type lora \
    --freeze_llm false \
    --freeze_vit false \
    --freeze_aligner false \
    --beta 0.2 \
    --rpo_alpha 1.0 \
    --loss_type sigmoid \
    --num_train_epochs 10 \
    --dataset "${TRAIN_DATA}" \
    --output_dir "${OUTPUT_DIR}" \
    --add_version false \
    --save_only_model false \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --logging_steps 5 \
    --torch_dtype bfloat16 \
    --gradient_checkpointing \
    --max_completion_length 2048 \
    --report_to wandb \
    --run_name "${RUN_NAME}" \
    --max_grad_norm 1.0 \
    --attn_impl flash_attn \
    --lora_rank 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --target_modules all-linear \
    --save_strategy epoch
