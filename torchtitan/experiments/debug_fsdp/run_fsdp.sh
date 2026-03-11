#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
log_dir="${script_dir}/outputs"

mkdir -p "${log_dir}"
cd "${script_dir}"

echo "=== Running HF Transformers backend ==="
TORCHTITAN_LOG_FILE="${log_dir}/hf_backend_log.txt" \
torchrun --nproc_per_node=2 minimal_fsdp.py \
  --job.config_file configs/hf_backend_debug.toml

echo "=== Running TorchTitan backend ==="
TORCHTITAN_LOG_FILE="${log_dir}/torchtitan_log.txt" \
torchrun --nproc_per_node=2 minimal_fsdp.py \
  --job.config_file configs/torchtitan_debug.toml

git diff --no-index --color --word-diff=color outputs/torchtitan_log.txt outputs/hf_backend_log.txt 2>&1 | tee outputs/diff.txt
