#!/bin/bash
# OLMo3-7B does not use a runtime `enable_thinking` chat-template kwarg like Qwen3;
# its "Think" vs. base behavior is selected by which checkpoint you point at here.
# This script targets the *Think* checkpoint (matching the paper's OLMo3-7B-Think row).

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/data/hf_cache
export WANDB_MODE=disabled
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

accelerate launch \
    --config_file accelerate.yaml \
    --num_processes 8 \
    --main_process_port 12949 \
    mass_train.py \
    --model_name_or_path /root/autodl-tmp/models/OLMo-3-7B-Think \
    --learning_rate 5e-6 \
    --max_grad_norm 0.1 \
    --per_device_train_batch_size 2 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 2 \
    --output_dir /root/autodl-tmp/checkpoints/ \
    --run_config mass_olmo37b_gen1024_fixteacher_temp11_forwardbeta0_clip006 \
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
    --reference_mask_probability 0.3 \
    --num_masked_teacher_views 8 \
    --anchor_coefficient 0.5 \
    --weight_regularization 1e-6 \
    --shapley_regularization 0.1 \
    --ema_smoothing 0.9 \
    --mask_probability_lower_bound 0.03 \
    --mask_adaptation_rate 0.5 \
    --supports_enable_thinking False \
    --fixed_teacher \
    --jsd_token_clip 0.06 \
    --wandb_project MASS
