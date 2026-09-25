# MASS Deployment and Run Guide

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

Only Qwen3-1.7B is required for the quick start:

```bash
hf download Qwen/Qwen3-1.7B \
  --local-dir /root/autodl-tmp/models/Qwen3-1.7B

# Optional models for the 4B and 8B scripts:
# hf download Qwen/Qwen3-4B --local-dir /root/autodl-tmp/models/Qwen3-4B --resume-download
# hf download Qwen/Qwen3-8B --local-dir /root/autodl-tmp/models/Qwen3-8B --resume-download
```

Confirm that the downloaded directory contains `config.json`, `*.safetensors`, and `tokenizer.json`:

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

`eval/evaluate_math.py` first searches for `*.parquet` files under `/root/autodl-tmp/datasets/<benchmark>/` and falls back to `load_dataset` only when a local copy is unavailable. Downloading evaluation data in advance is recommended:

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
dl aime24    HuggingFaceH4/aime_2024    train
dl aime25    yentinglin/aime_2025       train
dl hmmt25    MathArena/hmmt_feb_2025    train
dl math500   HuggingFaceH4/MATH-500     test
dl amc23     math-ai/amc23              test
dl minerva   math-ai/minervamath        test
dl amo-bench meituan-longcat/AMO-Bench  test

echo "=== Downloaded parquet files ==="
ls -la "$ROOT"/*/*.parquet
```

If you only plan to run the default AIME24 evaluation, downloading `aime24` is sufficient.

---

## 4. Training on Qwen3-1.7B

The included 1.7B launch scripts are configured for four GPUs.

```bash
cd /root/MASS
pip install "huggingface-hub>=0.34.0,<1.0" -U
```

### 4.1 MASS

MASS uses eight independently masked reference views with a span-mask probability of `0.35`, averages their teacher predictions, and keeps the remaining OPSD loss unchanged.

```bash
bash scripts/run_mass_1b.sh 2>&1 | tee run_mass_1b.log
```

Checkpoints are saved every 25 steps under:

```bash
ls /root/autodl-tmp/checkpoints/mass/mass_qwen31b_gen1024_fixteacher_temp11_forwardbeta0_clip005/
# checkpoint-25  checkpoint-50  checkpoint-75  checkpoint-100 ...
```

Student rollout JSON files are saved separately under:

```text
/root/autodl-tmp/checkpoints/mass/mass_qwen31b_gen1024_fixteacher_temp11_forwardbeta0_clip005/generations/
```

### 4.2 OPSD

OPSD uses the complete reference and one teacher forward pass:

```bash
bash scripts/run_opsd_1b.sh 2>&1 | tee run_opsd_1b.log
```

Its checkpoints and generation JSON files are isolated from MASS:

```text
/root/autodl-tmp/checkpoints/opsd/opsd_qwen31b_gen1024_fixteacher_temp11_forwardbeta0_clip005/
├── checkpoint-25/
├── checkpoint-50/
├── checkpoint-75/
├── checkpoint-100/
└── generations/
```

### 4.3 Changing the GPU count

If the server does not have exactly four GPUs, update both process-count settings, where `N` is the number of GPUs:

```bash
sed -i 's/^num_processes: .*/num_processes: N/' accelerate.yaml
sed -i 's/--num_processes 4/--num_processes N/' scripts/run_mass_1b.sh
sed -i 's/--num_processes 4/--num_processes N/' scripts/run_opsd_1b.sh
```

### 4.4 Background training

Use `nohup` to keep training alive after the SSH session closes:

```bash
nohup bash scripts/run_mass_1b.sh > run_mass_1b.log 2>&1 &
tail -f run_mass_1b.log

nohup bash scripts/run_opsd_1b.sh > run_opsd_1b.log 2>&1 &
tail -f run_opsd_1b.log
```

Other MASS variants are available in:

```text
scripts/run_mass_4b.sh
scripts/run_mass_8b.sh
scripts/run_mass_4b_nonthink.sh
scripts/run_mass_8b_nonthink.sh
```

The SFT and GRPO baselines remain available as `scripts/run_sft.sh` and `scripts/run_grpo.sh`.

---

## 5. Evaluation

Confirm that the evaluation parquet files from Section 3.2 are available, then choose the training mode to evaluate.

Evaluate MASS:

```bash
cd /root/MASS/eval
bash run_eval.sh mass
```

Evaluate OPSD:

```bash
cd /root/MASS/eval
bash run_eval.sh opsd
```

`run_eval.sh` first evaluates the base model and then evaluates checkpoints 25, 50, 75, and 100. Each checkpoint takes approximately 30–50 minutes on four H100 GPUs. Result JSON files are written to `eval/eval_results/*.json` and include `average_at_n_pct`.

To evaluate one checkpoint or use another benchmark, invoke `evaluate_math.py` directly:

```bash
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python evaluate_math.py \
  --base_model /root/autodl-tmp/models/Qwen3-1.7B \
  --checkpoint_dir /root/autodl-tmp/checkpoints/mass/mass_qwen31b_gen1024_fixteacher_temp11_forwardbeta0_clip005/checkpoint-100 \
  --dataset aime24 \
  --val_n 12 \
  --temperature 1.0 \
  --tensor_parallel_size 4
```

For the original Qwen3-1.7B OPSD run on AIME24 with Avg@12, the published reference result improves from 51.5% for the base model to approximately 57.2% at step 100.

---

## Troubleshooting

- **Model or dataset downloads are slow or fail:** confirm that `HF_ENDPOINT=https://hf-mirror.com` is active. Use `--resume-download` with the Hugging Face CLI and `-C -` with curl to resume interrupted downloads.
- **Evaluation data cannot be found:** verify that each parquet file exists under `/root/autodl-tmp/datasets/<benchmark>/`, is not empty, and uses the same local name as `--dataset`.
- **Out of memory:** lower `--vllm_gpu_memory_utilization` or `--per_device_train_batch_size`. Increase `--gradient_accumulation_steps` if the effective batch size must remain unchanged. MASS performs multiple teacher forwards and is slower than OPSD, but the teacher views are processed sequentially to control peak memory.
- **DeepSpeed reports a gradient-accumulation mismatch:** keep `gradient_accumulation_steps: 'auto'` in `accelerate.yaml` instead of hard-coding a number.
- **A checkpoint cannot be found:** verify the selected mode (`mass` or `opsd`) and ensure the `--run_config` value matches the directory used by `eval/run_eval.sh`.
