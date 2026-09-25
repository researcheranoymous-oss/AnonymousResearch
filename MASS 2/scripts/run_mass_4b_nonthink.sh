#!/bin/bash

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/data/hf_cache
export WANDB_MODE=disabled
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

accelerate launch \
    --config_file accelerate.yaml \
    --num_processes 4 \
    --main_process_port 12949 \
    mass_train.py \
    --model_name_or_path /root/autodl-tmp/models/Qwen3-4B \
    --learning_rate 5e-6 \
    --max_grad_norm 0.1 \
    --per_device_train_batch_size 4 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 2 \
    --output_dir /root/autodl-tmp/checkpoints/ \
    --run_config mass_qwen34b_gen1024_both_nonthink_fixteacher_temp11_forwardbeta0_clip1e-6 \
    --num_train_epochs 30 \
    --max_completion_length 1024 \
    --save_steps 25 \
    --logging_steps 2 \
    --attn_implementation flash_attention_2 \
    --torch_dtype bfloat16 \
    --max_length 20000 \
    --beta 0 \
    --use_vllm \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.6 \
    --vllm_tensor_parallel_size 1 \
    --use_peft \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --temperature 1.1 \
    --top_p 0.95 \
    --top_k 20 \
    --lmbda 1 \
    --use_reference_masking True \
    --reference_mask_probability 0.35 \
    --num_masked_teacher_views 8 \
    --fixed_teacher \
    --student_thinking False \
    --teacher_thinking False \
    --jsd_token_clip 1e-6 \
    --wandb_project MASS
