#!/bin/bash

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/data/hf_cache
export WANDB_MODE=disabled
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

BASE_MODEL="/root/autodl-tmp/models/Qwen3-1.7B"
MODE="${1:-mass}"

case "$MODE" in
    mass)
        EXP_DIR="/root/autodl-tmp/checkpoints/mass/mass_qwen31b_gen1024_fixteacher_temp11_forwardbeta0_clip005"
        ;;
    opsd)
        EXP_DIR="/root/autodl-tmp/checkpoints/opsd/opsd_qwen31b_gen1024_fixteacher_temp11_forwardbeta0_clip005"
        ;;
    *)
        echo "Usage: bash run_eval.sh [mass|opsd]" >&2
        exit 2
        ;;
esac

# evaluate base model performance
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python evaluate_math.py \
    --base_model "$BASE_MODEL" \
    --dataset "aime24" \
    --val_n 12 \
    --temperature 1.0 \
    --tensor_parallel_size 4
wait 

# after trained, evaluate the performance of the trained model. 
for step in 25 50 75 100; do
    NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python evaluate_math.py \
        --base_model "$BASE_MODEL" \
        --dataset "aime24" \
        --val_n 12 \
        --temperature 1.0 \
        --tensor_parallel_size 4 \
        --checkpoint_dir "$EXP_DIR/checkpoint-$step"
done
