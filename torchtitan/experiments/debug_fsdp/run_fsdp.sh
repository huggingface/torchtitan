#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
outputs_dir="${script_dir}/outputs"

dense_dir="${outputs_dir}/dense_non_tied"
dense_seed_dump_dir="${dense_dir}/seed"
dense_seed_checkpoint_path="${dense_seed_dump_dir}/checkpoint/step-0"
dense_fsdp_dir="${dense_dir}/fsdp"
dense_cpu_offload_dir="${dense_dir}/cpu_offload"
dense_mixed_precision_dir="${dense_dir}/mixed_precision"
dense_cpu_offload_mixed_precision_dir="${dense_dir}/cpu_offload_mixed_precision"

moe_dir="${outputs_dir}/moe_non_tied"
moe_seed_dump_dir="${moe_dir}/seed"
moe_seed_checkpoint_path="${moe_seed_dump_dir}/checkpoint/step-0"
moe_fsdp_dir="${moe_dir}/fsdp"
moe_cpu_offload_dir="${moe_dir}/cpu_offload"
moe_mixed_precision_dir="${moe_dir}/mixed_precision"
moe_cpu_offload_mixed_precision_dir="${moe_dir}/cpu_offload_mixed_precision"

mkdir -p \
  "${dense_seed_dump_dir}" \
  "${dense_fsdp_dir}" \
  "${dense_cpu_offload_dir}" \
  "${dense_mixed_precision_dir}" \
  "${dense_cpu_offload_mixed_precision_dir}" \
  "${moe_seed_dump_dir}" \
  "${moe_fsdp_dir}" \
  "${moe_cpu_offload_dir}" \
  "${moe_mixed_precision_dir}" \
  "${moe_cpu_offload_mixed_precision_dir}"

cd "${script_dir}"

# echo "=== Creating shared seed checkpoint: dense non_tied ==="
# TORCHTITAN_LOG_FILE="${dense_seed_dump_dir}/seed_checkpoint_log.txt" \
# python minimal_fsdp.py \
#   --job.config_file configs/torchtitan_debug.toml \
#   --job.dump_folder "${dense_seed_dump_dir}" \
#   --checkpoint.enable \
#   --checkpoint.create_seed_checkpoint

# echo "=== Dense non_tied: HF backend / FSDP ==="
# TORCHTITAN_LOG_FILE="${dense_fsdp_dir}/hf_backend_log.txt" \
# torchrun --nproc_per_node=2 minimal_fsdp.py \
#   --job.config_file configs/hf_backend_debug.toml \
#   --checkpoint.enable \
#   --checkpoint.initial_load_path "${dense_seed_checkpoint_path}" \
#   --checkpoint.load_only

# echo "=== Dense non_tied: TorchTitan / FSDP ==="
# TORCHTITAN_LOG_FILE="${dense_fsdp_dir}/torchtitan_log.txt" \
# torchrun --nproc_per_node=2 minimal_fsdp.py \
#   --job.config_file configs/torchtitan_debug.toml \
#   --checkpoint.enable \
#   --checkpoint.initial_load_path "${dense_seed_checkpoint_path}" \
#   --checkpoint.load_only

# echo "=== Dense non_tied: HF from_pretrained / FSDP ==="
# USE_HF_FSDP=1 \
# TORCHTITAN_LOG_FILE="${dense_fsdp_dir}/hf_fsdp_log.txt" \
# torchrun --nproc_per_node=2 minimal_fsdp.py \
#   --job.config_file configs/hf_backend_debug.toml \
#   --checkpoint.enable \
#   --checkpoint.initial_load_path "${dense_seed_checkpoint_path}" \
#   --checkpoint.load_only

# echo "=== Diff: dense non_tied / FSDP / HF backend vs TorchTitan ==="
# git diff --no-index --color --word-diff=color \
#   "${dense_fsdp_dir}/torchtitan_log.txt" "${dense_fsdp_dir}/hf_backend_log.txt" > "${dense_fsdp_dir}/diff_hf_backend_vs_torchtitan.txt" 2>&1 || true
# cat "${dense_fsdp_dir}/diff_hf_backend_vs_torchtitan.txt"

# echo "=== Diff: dense non_tied / FSDP / HF from_pretrained vs TorchTitan ==="
# git diff --no-index --color --word-diff=color \
#   "${dense_fsdp_dir}/torchtitan_log.txt" "${dense_fsdp_dir}/hf_fsdp_log.txt" > "${dense_fsdp_dir}/diff_hf_fsdp_vs_torchtitan.txt" 2>&1 || true
# cat "${dense_fsdp_dir}/diff_hf_fsdp_vs_torchtitan.txt"

# echo "=== Dense non_tied: HF backend / FSDP + cpu offload ==="
# TORCHTITAN_LOG_FILE="${dense_cpu_offload_dir}/hf_backend_log.txt" \
# torchrun --nproc_per_node=2 minimal_fsdp.py \
#   --job.config_file configs/hf_backend_debug.toml \
#   --training.enable_cpu_offload \
#   --checkpoint.enable \
#   --checkpoint.initial_load_path "${dense_seed_checkpoint_path}" \
#   --checkpoint.load_only

# echo "=== Dense non_tied: TorchTitan / FSDP + cpu offload ==="
# TORCHTITAN_LOG_FILE="${dense_cpu_offload_dir}/torchtitan_log.txt" \
# torchrun --nproc_per_node=2 minimal_fsdp.py \
#   --job.config_file configs/torchtitan_debug.toml \
#   --training.enable_cpu_offload \
#   --checkpoint.enable \
#   --checkpoint.initial_load_path "${dense_seed_checkpoint_path}" \
#   --checkpoint.load_only

# echo "=== Dense non_tied: HF from_pretrained / FSDP + cpu offload ==="
# USE_HF_FSDP=1 \
# TORCHTITAN_LOG_FILE="${dense_cpu_offload_dir}/hf_fsdp_log.txt" \
# torchrun --nproc_per_node=2 minimal_fsdp.py \
#   --job.config_file configs/hf_backend_debug.toml \
#   --training.enable_cpu_offload \
#   --checkpoint.enable \
#   --checkpoint.initial_load_path "${dense_seed_checkpoint_path}" \
#   --checkpoint.load_only

# echo "=== Diff: dense non_tied / FSDP + cpu offload / HF backend vs TorchTitan ==="
# git diff --no-index --color --word-diff=color \
#   "${dense_cpu_offload_dir}/torchtitan_log.txt" "${dense_cpu_offload_dir}/hf_backend_log.txt" > "${dense_cpu_offload_dir}/diff_hf_backend_vs_torchtitan.txt" 2>&1 || true
# cat "${dense_cpu_offload_dir}/diff_hf_backend_vs_torchtitan.txt"

# echo "=== Diff: dense non_tied / FSDP + cpu offload / HF from_pretrained vs TorchTitan ==="
# git diff --no-index --color --word-diff=color \
#   "${dense_cpu_offload_dir}/torchtitan_log.txt" "${dense_cpu_offload_dir}/hf_fsdp_log.txt" > "${dense_cpu_offload_dir}/diff_hf_fsdp_vs_torchtitan.txt" 2>&1 || true
# cat "${dense_cpu_offload_dir}/diff_hf_fsdp_vs_torchtitan.txt"

# echo "=== Dense non_tied: HF backend / FSDP + mixed precision ==="
# TORCHTITAN_LOG_FILE="${dense_mixed_precision_dir}/hf_backend_log.txt" \
# torchrun --nproc_per_node=2 minimal_fsdp.py \
#   --job.config_file configs/hf_backend_debug.toml \
#   --training.mixed_precision_param bfloat16 \
#   --training.mixed_precision_reduce float32 \
#   --checkpoint.enable \
#   --checkpoint.initial_load_path "${dense_seed_checkpoint_path}" \
#   --checkpoint.load_only

# echo "=== Dense non_tied: TorchTitan / FSDP + mixed precision ==="
# TORCHTITAN_LOG_FILE="${dense_mixed_precision_dir}/torchtitan_log.txt" \
# torchrun --nproc_per_node=2 minimal_fsdp.py \
#   --job.config_file configs/torchtitan_debug.toml \
#   --training.mixed_precision_param bfloat16 \
#   --training.mixed_precision_reduce float32 \
#   --checkpoint.enable \
#   --checkpoint.initial_load_path "${dense_seed_checkpoint_path}" \
#   --checkpoint.load_only

# echo "=== Dense non_tied: HF from_pretrained / FSDP + mixed precision ==="
# USE_HF_FSDP=1 \
# TORCHTITAN_LOG_FILE="${dense_mixed_precision_dir}/hf_fsdp_log.txt" \
# torchrun --nproc_per_node=2 minimal_fsdp.py \
#   --job.config_file configs/hf_backend_debug.toml \
#   --training.mixed_precision_param bfloat16 \
#   --training.mixed_precision_reduce float32 \
#   --checkpoint.enable \
#   --checkpoint.initial_load_path "${dense_seed_checkpoint_path}" \
#   --checkpoint.load_only

# echo "=== Diff: dense non_tied / FSDP + mixed precision / HF backend vs TorchTitan ==="
# git diff --no-index --color --word-diff=color \
#   "${dense_mixed_precision_dir}/torchtitan_log.txt" "${dense_mixed_precision_dir}/hf_backend_log.txt" > "${dense_mixed_precision_dir}/diff_hf_backend_vs_torchtitan.txt" 2>&1 || true
# cat "${dense_mixed_precision_dir}/diff_hf_backend_vs_torchtitan.txt"

# echo "=== Diff: dense non_tied / FSDP + mixed precision / HF from_pretrained vs TorchTitan ==="
# git diff --no-index --color --word-diff=color \
#   "${dense_mixed_precision_dir}/torchtitan_log.txt" "${dense_mixed_precision_dir}/hf_fsdp_log.txt" > "${dense_mixed_precision_dir}/diff_hf_fsdp_vs_torchtitan.txt" 2>&1 || true
# cat "${dense_mixed_precision_dir}/diff_hf_fsdp_vs_torchtitan.txt"

# echo "=== Dense non_tied: HF backend / FSDP + cpu offload + mixed precision ==="
# TORCHTITAN_LOG_FILE="${dense_cpu_offload_mixed_precision_dir}/hf_backend_log.txt" \
# torchrun --nproc_per_node=2 minimal_fsdp.py \
#   --job.config_file configs/hf_backend_debug.toml \
#   --training.enable_cpu_offload \
#   --training.mixed_precision_param bfloat16 \
#   --training.mixed_precision_reduce float32 \
#   --checkpoint.enable \
#   --checkpoint.initial_load_path "${dense_seed_checkpoint_path}" \
#   --checkpoint.load_only

# echo "=== Dense non_tied: TorchTitan / FSDP + cpu offload + mixed precision ==="
# TORCHTITAN_LOG_FILE="${dense_cpu_offload_mixed_precision_dir}/torchtitan_log.txt" \
# torchrun --nproc_per_node=2 minimal_fsdp.py \
#   --job.config_file configs/torchtitan_debug.toml \
#   --training.enable_cpu_offload \
#   --training.mixed_precision_param bfloat16 \
#   --training.mixed_precision_reduce float32 \
#   --checkpoint.enable \
#   --checkpoint.initial_load_path "${dense_seed_checkpoint_path}" \
#   --checkpoint.load_only

# echo "=== Dense non_tied: HF from_pretrained / FSDP + cpu offload + mixed precision ==="
# USE_HF_FSDP=1 \
# TORCHTITAN_LOG_FILE="${dense_cpu_offload_mixed_precision_dir}/hf_fsdp_log.txt" \
# torchrun --nproc_per_node=2 minimal_fsdp.py \
#   --job.config_file configs/hf_backend_debug.toml \
#   --training.enable_cpu_offload \
#   --training.mixed_precision_param bfloat16 \
#   --training.mixed_precision_reduce float32 \
#   --checkpoint.enable \
#   --checkpoint.initial_load_path "${dense_seed_checkpoint_path}" \
#   --checkpoint.load_only

# echo "=== Diff: dense non_tied / FSDP + cpu offload + mixed precision / HF backend vs TorchTitan ==="
# git diff --no-index --color --word-diff=color \
#   "${dense_cpu_offload_mixed_precision_dir}/torchtitan_log.txt" "${dense_cpu_offload_mixed_precision_dir}/hf_backend_log.txt" > "${dense_cpu_offload_mixed_precision_dir}/diff_hf_backend_vs_torchtitan.txt" 2>&1 || true
# cat "${dense_cpu_offload_mixed_precision_dir}/diff_hf_backend_vs_torchtitan.txt"

# echo "=== Diff: dense non_tied / FSDP + cpu offload + mixed precision / HF from_pretrained vs TorchTitan ==="
# git diff --no-index --color --word-diff=color \
#   "${dense_cpu_offload_mixed_precision_dir}/torchtitan_log.txt" "${dense_cpu_offload_mixed_precision_dir}/hf_fsdp_log.txt" > "${dense_cpu_offload_mixed_precision_dir}/diff_hf_fsdp_vs_torchtitan.txt" 2>&1 || true
# cat "${dense_cpu_offload_mixed_precision_dir}/diff_hf_fsdp_vs_torchtitan.txt"

echo "=== Creating shared seed checkpoint: moe non_tied ==="
TORCHTITAN_LOG_FILE="${moe_seed_dump_dir}/seed_checkpoint_log.txt" \
python minimal_fsdp.py \
  --job.config_file configs/torchtitan_debug.toml \
  --model.flavor debugmodel_moe \
  --job.dump_folder "${moe_seed_dump_dir}" \
  --checkpoint.enable \
  --checkpoint.create_seed_checkpoint

echo "=== MoE non_tied: TorchTitan / FSDP ==="
TORCHTITAN_LOG_FILE="${moe_fsdp_dir}/torchtitan_log.txt" \
torchrun --nproc_per_node=2 minimal_fsdp.py \
  --job.config_file configs/torchtitan_debug.toml \
  --model.flavor debugmodel_moe \
  --checkpoint.enable \
  --checkpoint.initial_load_path "${moe_seed_checkpoint_path}" \
  --checkpoint.load_only

echo "=== MoE non_tied: HF from_pretrained / FSDP ==="
USE_HF_FSDP=1 \
TORCHTITAN_LOG_FILE="${moe_fsdp_dir}/hf_fsdp_log.txt" \
torchrun --nproc_per_node=2 minimal_fsdp.py \
  --job.config_file configs/hf_qwen3_moe_fsdp_debug.toml \
  --checkpoint.enable \
  --checkpoint.initial_load_path "${moe_seed_checkpoint_path}" \
  --checkpoint.load_only

echo "=== Diff: MoE non_tied / FSDP / HF from_pretrained vs TorchTitan ==="
git diff --no-index --color --word-diff=color \
  "${moe_fsdp_dir}/torchtitan_log.txt" "${moe_fsdp_dir}/hf_fsdp_log.txt" > "${moe_fsdp_dir}/diff_hf_fsdp_vs_torchtitan.txt" 2>&1 || true
cat "${moe_fsdp_dir}/diff_hf_fsdp_vs_torchtitan.txt"

echo "=== MoE non_tied: TorchTitan / FSDP + cpu offload ==="
TORCHTITAN_LOG_FILE="${moe_cpu_offload_dir}/torchtitan_log.txt" \
torchrun --nproc_per_node=2 minimal_fsdp.py \
  --job.config_file configs/torchtitan_debug.toml \
  --model.flavor debugmodel_moe \
  --training.enable_cpu_offload \
  --checkpoint.enable \
  --checkpoint.initial_load_path "${moe_seed_checkpoint_path}" \
  --checkpoint.load_only

echo "=== MoE non_tied: HF from_pretrained / FSDP + cpu offload ==="
USE_HF_FSDP=1 \
TORCHTITAN_LOG_FILE="${moe_cpu_offload_dir}/hf_fsdp_log.txt" \
torchrun --nproc_per_node=2 minimal_fsdp.py \
  --job.config_file configs/hf_qwen3_moe_fsdp_debug.toml \
  --training.enable_cpu_offload \
  --checkpoint.enable \
  --checkpoint.initial_load_path "${moe_seed_checkpoint_path}" \
  --checkpoint.load_only

echo "=== Diff: MoE non_tied / FSDP + cpu offload / HF from_pretrained vs TorchTitan ==="
git diff --no-index --color --word-diff=color \
  "${moe_cpu_offload_dir}/torchtitan_log.txt" "${moe_cpu_offload_dir}/hf_fsdp_log.txt" > "${moe_cpu_offload_dir}/diff_hf_fsdp_vs_torchtitan.txt" 2>&1 || true
cat "${moe_cpu_offload_dir}/diff_hf_fsdp_vs_torchtitan.txt"

echo "=== MoE non_tied: TorchTitan / FSDP + mixed precision ==="
TORCHTITAN_LOG_FILE="${moe_mixed_precision_dir}/torchtitan_log.txt" \
torchrun --nproc_per_node=2 minimal_fsdp.py \
  --job.config_file configs/torchtitan_debug.toml \
  --model.flavor debugmodel_moe \
  --training.mixed_precision_param bfloat16 \
  --training.mixed_precision_reduce float32 \
  --checkpoint.enable \
  --checkpoint.initial_load_path "${moe_seed_checkpoint_path}" \
  --checkpoint.load_only

echo "=== MoE non_tied: HF from_pretrained / FSDP + mixed precision ==="
USE_HF_FSDP=1 \
TORCHTITAN_LOG_FILE="${moe_mixed_precision_dir}/hf_fsdp_log.txt" \
torchrun --nproc_per_node=2 minimal_fsdp.py \
  --job.config_file configs/hf_qwen3_moe_fsdp_debug.toml \
  --training.mixed_precision_param bfloat16 \
  --training.mixed_precision_reduce float32 \
  --checkpoint.enable \
  --checkpoint.initial_load_path "${moe_seed_checkpoint_path}" \
  --checkpoint.load_only

echo "=== Diff: MoE non_tied / FSDP + mixed precision / HF from_pretrained vs TorchTitan ==="
git diff --no-index --color --word-diff=color \
  "${moe_mixed_precision_dir}/torchtitan_log.txt" "${moe_mixed_precision_dir}/hf_fsdp_log.txt" > "${moe_mixed_precision_dir}/diff_hf_fsdp_vs_torchtitan.txt" 2>&1 || true
cat "${moe_mixed_precision_dir}/diff_hf_fsdp_vs_torchtitan.txt"

echo "=== MoE non_tied: TorchTitan / FSDP + cpu offload + mixed precision ==="
TORCHTITAN_LOG_FILE="${moe_cpu_offload_mixed_precision_dir}/torchtitan_log.txt" \
torchrun --nproc_per_node=2 minimal_fsdp.py \
  --job.config_file configs/torchtitan_debug.toml \
  --model.flavor debugmodel_moe \
  --training.enable_cpu_offload \
  --training.mixed_precision_param bfloat16 \
  --training.mixed_precision_reduce float32 \
  --checkpoint.enable \
  --checkpoint.initial_load_path "${moe_seed_checkpoint_path}" \
  --checkpoint.load_only

echo "=== MoE non_tied: HF from_pretrained / FSDP + cpu offload + mixed precision ==="
USE_HF_FSDP=1 \
TORCHTITAN_LOG_FILE="${moe_cpu_offload_mixed_precision_dir}/hf_fsdp_log.txt" \
torchrun --nproc_per_node=2 minimal_fsdp.py \
  --job.config_file configs/hf_qwen3_moe_fsdp_debug.toml \
  --training.enable_cpu_offload \
  --training.mixed_precision_param bfloat16 \
  --training.mixed_precision_reduce float32 \
  --checkpoint.enable \
  --checkpoint.initial_load_path "${moe_seed_checkpoint_path}" \
  --checkpoint.load_only

echo "=== Diff: MoE non_tied / FSDP + cpu offload + mixed precision / HF from_pretrained vs TorchTitan ==="
git diff --no-index --color --word-diff=color \
  "${moe_cpu_offload_mixed_precision_dir}/torchtitan_log.txt" "${moe_cpu_offload_mixed_precision_dir}/hf_fsdp_log.txt" > "${moe_cpu_offload_mixed_precision_dir}/diff_hf_fsdp_vs_torchtitan.txt" 2>&1 || true
cat "${moe_cpu_offload_mixed_precision_dir}/diff_hf_fsdp_vs_torchtitan.txt"
