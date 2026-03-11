#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
log_dir="${script_dir}/outputs"
seed_dump_dir="${log_dir}/shared_seed"
seed_checkpoint_path="${seed_dump_dir}/checkpoint/step-0"

mkdir -p "${log_dir}"
cd "${script_dir}"

echo "=== Creating shared seed checkpoint ==="
TORCHTITAN_LOG_FILE="${log_dir}/seed_checkpoint_log.txt" \
python minimal_fsdp.py \
  --job.config_file configs/torchtitan_debug.toml \
  --job.dump_folder "${seed_dump_dir}" \
  --checkpoint.enable \
  --checkpoint.create_seed_checkpoint

echo "=== Running HF Transformers backend ==="
TORCHTITAN_LOG_FILE="${log_dir}/hf_backend_log.txt" \
torchrun --nproc_per_node=2 minimal_fsdp.py \
  --job.config_file configs/hf_backend_debug.toml \
  --checkpoint.enable \
  --checkpoint.initial_load_path "${seed_checkpoint_path}" \
  --checkpoint.load_only

echo "=== Running TorchTitan backend ==="
TORCHTITAN_LOG_FILE="${log_dir}/torchtitan_log.txt" \
torchrun --nproc_per_node=2 minimal_fsdp.py \
  --job.config_file configs/torchtitan_debug.toml \
  --checkpoint.enable \
  --checkpoint.initial_load_path "${seed_checkpoint_path}" \
  --checkpoint.load_only

# echo "=== Running HF Transformers from_pretrained FSDP ==="
# USE_HF_FSDP=1 \
# TORCHTITAN_LOG_FILE="${log_dir}/hf_fsdp_log.txt" \
# torchrun --nproc_per_node=2 minimal_fsdp.py \
#   --job.config_file configs/hf_backend_debug.toml \
#   --checkpoint.enable \
#   --checkpoint.initial_load_path "${seed_checkpoint_path}" \
#   --checkpoint.load_only

echo "=== Diff: HF backend (torchtitan FSDP) vs Native TorchTitan ==="
git diff --no-index --color --word-diff=color \
  outputs/torchtitan_log.txt outputs/hf_backend_log.txt > outputs/diff_hf_backend_vs_torchtitan.txt 2>&1 || true
cat outputs/diff_hf_backend_vs_torchtitan.txt

# echo "=== Diff: HF from_pretrained FSDP vs Native TorchTitan ==="
# git diff --no-index --color --word-diff=color \
#   outputs/torchtitan_log.txt outputs/hf_fsdp_log.txt > outputs/diff_hf_fsdp_vs_torchtitan.txt 2>&1 || true
# cat outputs/diff_hf_fsdp_vs_torchtitan.txt
