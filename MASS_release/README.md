# MASS Deployment and Run Guide

> **Note on this release.** 
> Shapley-masking internals (Sec. 2.3: per-span coalition-utility regression and the closed-form
> masking-probability update) are withheld pending paper acceptance — see `adaptive_masking.py`.
> With the shipped default (`enable_adaptive_masking=False`), the code runs the masked-average
> teacher + anchor teacher + selective weighting (Sec. 2.1-2.2) with plain uniform per-span
> masking.

## 1. Environment Setup

```bash
conda env create -f environment.yml
source /root/miniconda3/bin/activate
conda activate mass

# Install flash-attn separately because compilation can take a while.
pip install flash-attn==2.8.3 --no-build-isolation
```

If `flash-attn` cannot be installed from source, download a prebuilt wheel that matches the server's CUDA, PyTorch, and Python versions from the [flash-attention releases page](https://github.com/Dao-AILab/flash-attention/releases), then install the wheel directly.

Create the data-disk directories used by the included scripts. The examples below use `/root/autodl-tmp`:

```bash
mkdir -p /root/autodl-tmp/models \
         /root/autodl-tmp/data/hf_cache \
         /root/autodl-tmp/datasets \
         /root/autodl-tmp/checkpoints
```

---

## 2. Download Models Through the Hugging Face Mirror

The training and evaluation scripts already configure the mirror and cache. The corresponding environment variables are:

```bash
export HF_ENDPOINT=https://hf-mirror.com          # Hugging Face mirror
export HF_HOME=/root/autodl-tmp/data/hf_cache     # Cache on the data disk
```

Download every model used by the scripts in Section 4:

```bash
hf download Qwen/Qwen3-1.7B --local-dir /root/autodl-tmp/models/Qwen3-1.7B --resume-download
hf download Qwen/Qwen3-4B --local-dir /root/autodl-tmp/models/Qwen3-4B --resume-download
hf download Qwen/Qwen3-8B --local-dir /root/autodl-tmp/models/Qwen3-8B --resume-download
hf download allenai/OLMo-3-7B-Think --local-dir /root/autodl-tmp/models/OLMo-3-7B-Think --resume-download
hf download allenai/OLMo-3-7B --local-dir /root/autodl-tmp/models/OLMo-3-7B --resume-download
```

Confirm that each downloaded directory contains `config.json`, `*.safetensors`, and `tokenizer.json`, e.g.:

```bash
ls /root/autodl-tmp/models/Qwen3-1.7B
```

---

## 3. Download Datasets

Training and evaluation use different dataset-loading paths.

### 3.1 Training dataset

`mass_train.py` loads `siyanzhao/Openthoughts_math_30k_opsd`. The loader first checks for local parquet files under:

```text
/root/autodl-tmp/datasets/Openthoughts_math_30k_opsd/
```

If no local copy is present, it downloads the dataset through the configured Hugging Face mirror and reuses the cache on subsequent runs. You can therefore start training without manually downloading the training dataset.

For a fully local setup, download it in advance:

```bash
hf download siyanzhao/Openthoughts_math_30k_opsd \
  --repo-type dataset \
  --local-dir /root/autodl-tmp/datasets/Openthoughts_math_30k_opsd \
  --max-workers 1
```

The local dataset root can be changed with `MASS_DATASET_ROOT`.

### 3.2 Evaluation datasets

`eval/evaluate_math.py` first searches for `*.parquet` files under `/root/autodl-tmp/datasets/<benchmark>/` and falls back to `load_dataset` only when a local copy is unavailable. Download the three benchmarks used in Table 1 (AIME 2024, AIME 2025, HMMT 2025) in advance:

```bash
ROOT=/root/autodl-tmp/datasets
REF="refs%2Fconvert%2Fparquet"
B="https://hf-mirror.com/datasets"

dl() {  # dl <local-name> <hub-id> <split>
  mkdir -p "$ROOT/$1"
  curl -fL -C - --retry 10 --retry-all-errors \
    "$B/$2/resolve/$REF/default/$3/0000.parquet" \
    -o "$ROOT/$1/0000.parquet" && echo "OK   $1" || echo "FAIL $1  ($2 $3)"
}

# The local names must match the values accepted by --dataset.
dl aime24 HuggingFaceH4/aime_2024   train
dl aime25 yentinglin/aime_2025      train
dl hmmt25 MathArena/hmmt_feb_2025   train

echo "=== Downloaded parquet files ==="
ls -la "$ROOT"/*/*.parquet
```

`evaluate_math.py --dataset` also accepts `math500`, `amc23`, `minerva`, and `amo-bench` for ad hoc evaluation beyond Table 1's three benchmarks; download those the same way with their hub IDs (`HuggingFaceH4/MATH-500`, `math-ai/amc23`, `math-ai/minervamath`, `meituan-longcat/AMO-Bench`, all `test` split) if you need them.

The default `run_eval.sh` / `run_eval_olmo3_7b.sh` scripts only evaluate on `aime24`; run `evaluate_math.py` directly with `--dataset aime25` or `--dataset hmmt25` for the other two.

---

## 4. Training

```bash
cd /root/MASS
pip install "huggingface-hub>=0.34.0,<1.0" -U
```

### 4.1 MASS

MASS uses eight independently masked reference views (each masked span replaced by a literal `[MASK]` token) with a target masking ratio of `0.3`, averages their teacher predictions against a full-reference anchor teacher, and applies bias-corrected selective weighting. Adaptive per-span masking-probability updates via Shapley-kernel attribution (`adaptive_masking.py`) are withheld in this release (see the note at the top of this file); masking uses the fixed target ratio instead.

Every launch script below runs MASS end to end for one model/config. All are configured for eight GPUs except the 1.7B script, which uses four:

```text
scripts/run_mass_1b.sh                 # Qwen3-1.7B
scripts/run_mass_4b.sh                 # Qwen3-4B
scripts/run_mass_4b_nonthink.sh        # Qwen3-4B, non-thinking ablation
scripts/run_mass_8b.sh                 # Qwen3-8B
scripts/run_mass_8b_nonthink.sh        # Qwen3-8B, non-thinking ablation
scripts/run_mass_olmo3_7b.sh           # OLMo-3-7B-Think
scripts/run_mass_olmo3_7b_nonthink.sh  # OLMo-3-7B (base, non-thinking)
```

```bash
bash scripts/run_mass_1b.sh 2>&1 | tee run_mass_1b.log
```

Checkpoints are saved every 25 steps under `--output_dir/mass/<run_config>/`, e.g.:

```bash
ls /root/autodl-tmp/checkpoints/mass/mass_qwen31b_gen1024_fixteacher_temp11_forwardbeta0_clip005/
# checkpoint-25  checkpoint-50  checkpoint-75  checkpoint-100 ...
```

Student rollout JSON files are saved separately under `.../generations/` inside that same directory.

OLMo3 has no `enable_thinking` chat-template kwarg (unlike Qwen3), so its scripts pass
`--supports_enable_thinking False`; thinking vs. non-thinking mode is instead selected by
which checkpoint `--model_name_or_path` points at.

### 4.2 Changing the GPU count

If a script's GPU count doesn't match your server, update both process-count settings for that script, where `N` is the number of GPUs:

```bash
sed -i 's/^num_processes: .*/num_processes: N/' accelerate.yaml
sed -i 's/--num_processes [0-9]*/--num_processes N/' scripts/run_mass_1b.sh
```

### 4.3 Background training

Use `nohup` to keep training alive after the SSH session closes:

```bash
nohup bash scripts/run_mass_1b.sh > run_mass_1b.log 2>&1 &
tail -f run_mass_1b.log
```

---

## 5. Evaluation

Confirm that the evaluation parquet files from Section 3.2 are available.

```text
eval/run_eval.sh mass           # Qwen3-1.7B
eval/run_eval_olmo3_7b.sh mass  # OLMo-3-7B-Think
```

```bash
cd /root/MASS/eval
bash run_eval.sh mass
```

Each script first evaluates the base model, then checkpoints 25, 50, 75, and 100. Each checkpoint takes approximately 30–50 minutes on four H100 GPUs. Result JSON files are written to `eval/eval_results/*.json` and include `average_at_n_pct`.

To evaluate one checkpoint or use another benchmark, invoke `evaluate_math.py` directly. For OLMo3 (or any model family without a Qwen3-style `enable_thinking` chat-template kwarg), add `--no_enable_thinking_kwarg` and set `--top_p` explicitly — the auto-tuned `--top_p` default is Qwen3-specific and is refused otherwise:

```bash
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python evaluate_math.py \
  --base_model /root/autodl-tmp/models/Qwen3-1.7B \
  --checkpoint_dir /root/autodl-tmp/checkpoints/mass/mass_qwen31b_gen1024_fixteacher_temp11_forwardbeta0_clip005/checkpoint-100 \
  --dataset aime24 \
  --val_n 12 \
  --temperature 1.0 \
  --tensor_parallel_size 4

# OLMo3 (or another non-Qwen3 family):
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python evaluate_math.py \
  --base_model /root/autodl-tmp/models/OLMo-3-7B-Think \
  --checkpoint_dir /root/autodl-tmp/checkpoints/mass/mass_olmo37b_gen1024_fixteacher_temp11_forwardbeta0_clip006/checkpoint-100 \
  --no_enable_thinking_kwarg --top_p 0.95 \
  --dataset aime24 \
  --val_n 12 \
  --temperature 1.0 \
  --tensor_parallel_size 4
```

Since adaptive masking is withheld in this release (see the note at the top of this file), expect results closer to the paper's disclosed "Weighting only" ablation numbers (Table 3) rather than the exact "Full MASS" rows in Table 1.

---

## Troubleshooting

- **Model or dataset downloads are slow or fail:** confirm that `HF_ENDPOINT=https://hf-mirror.com` is active. Use `--resume-download` with the Hugging Face CLI and `-C -` with curl to resume interrupted downloads.
- **Evaluation data cannot be found:** verify that each parquet file exists under `/root/autodl-tmp/datasets/<benchmark>/`, is not empty, and uses the same local name as `--dataset`.
- **Out of memory:** lower `--vllm_gpu_memory_utilization` or `--per_device_train_batch_size`. Increase `--gradient_accumulation_steps` if the effective batch size must remain unchanged. MASS performs multiple teacher forwards per step, but the teacher views are processed sequentially to control peak memory.
- **DeepSpeed reports a gradient-accumulation mismatch:** keep `gradient_accumulation_steps: 'auto'` in `accelerate.yaml` instead of hard-coding a number.
- **A checkpoint cannot be found:** ensure the `--run_config` value matches the directory used by `eval/run_eval.sh`.
