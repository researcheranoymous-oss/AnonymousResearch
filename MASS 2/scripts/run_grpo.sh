#!/bin/bash

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/data/hf_cache
export WANDB_MODE=disabled
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
    --config_file accelerate.yaml \
    --num_processes 8 \
    --main_process_port 19346 \
    grpo_train.py \
    --learning_rate 5e-6 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --model_name_or_path /root/autodl-tmp/models/Qwen3-4B \
    --output_dir /root/autodl-tmp/checkpoints/grpo/ \
    --run_config qwen4b-2epoch \
    --num_train_epochs 2 \
    --num_iterations 2 \
    --gradient_checkpointing \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --max_prompt_length 2048 \
    --max_completion_length 16000 \
    --num_generations 8 \
    --temperature 1.2 \
    --use_vllm \
    --use_peft \
    --vllm_mode colocate \
    --logging_steps 10 \
    --save_steps 20 \
    --beta 0.0 \
    --loss_type grpo \
    --scale_rewards group \
    --wandb_project MASS
