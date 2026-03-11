#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
log_dir="${script_dir}/outputs"

mkdir -p "${log_dir}"
cd "${script_dir}"

echo "=== Running HF Transformers backend ==="
TORCHTITAN_LOG_FILE="${log_dir}/hf_transformers_log.txt" \
torchrun --nproc_per_node=2 minimal_fsdp.py \
  --job.config_file configs/hf_transformers_debug.toml

echo "=== Running native TorchTitan Llama 3 ==="
TORCHTITAN_LOG_FILE="${log_dir}/native_llama3_log.txt" \
torchrun --nproc_per_node=2 minimal_fsdp.py \
  --job.config_file configs/native_llama3_debug.toml

git diff --no-index --color --word-diff=color outputs/native_llama3_log.txt outputs/hf_transformers_log.txt 2>&1 | tee outputs/diff.txt
