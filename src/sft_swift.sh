#!/bin/bash
# Supervised fine-tuning (SFT / SFT-CoT) with ms-swift. Run from the repository root.
#
#   bash src/sft_swift.sh <train_jsonl> [run_name]
#
#   run_name defaults to the file name of <train_jsonl>; outputs go to outputs/experiments/<run_name>.
#
#   SFT     : bash src/sft_swift.sh outputs/swift/sft.jsonl
#   SFT-CoT : bash src/sft_swift.sh outputs/swift/sft_cot.jsonl
#
# LoRA rank 64 / alpha 128 on all linear layers, lr 1e-4, 10 epochs, total batch size 32.

TRAIN_DATA=${1:-outputs/swift/sft.jsonl}
RUN_NAME=${2:-$(basename "$TRAIN_DATA" .jsonl)}
OUTPUT_DIR="outputs/experiments/${RUN_NAME}"
PRETRAINED_MODEL="Qwen/Qwen2.5-Omni-3B"

# Total batch size = NPROC_PER_NODE x per_device_train_batch_size x gradient_accumulation_steps = 32.
# Adjust these to your GPUs. DeepSpeed ZeRO-2 may be unnecessary depending on your GPU memory.
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
swift sft \
    --seed 1115 \
    --model ${PRETRAINED_MODEL} \
    --model_type qwen2_5_omni \
    --model_name "${RUN_NAME}" \
    --train_type lora \
    --freeze_llm false \
    --freeze_vit false \
    --freeze_aligner false \
    --dataset "${TRAIN_DATA}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs 10 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --max_grad_norm 1.0 \
    --attn_impl flash_attn \
    --save_strategy epoch \
    --gradient_checkpointing \
    --save_only_model true \
    --add_version false \
    --report_to wandb \
    --run_name "${RUN_NAME}" \
    --logging_steps 5 \
    --torch_dtype bfloat16 \
    --deepspeed zero2 \
    --lora_rank 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --target_modules all-linear
