#!/bin/bash
#SBATCH --job-name=llama3_8b_fsdp_mixed_precision_torchtitan
#SBATCH --output=outputs/llama3_8b/fsdp_mixed_precision/torchtitan/slurm_%j.out
#SBATCH --error=outputs/llama3_8b/fsdp_mixed_precision/torchtitan/slurm_%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --qos=high
#SBATCH --cpus-per-task=12
#SBATCH --partition=hopper-prod
#SBATCH --time=6:00:00

# Misc initializations.
echo "========================"
echo "START TIME: $(date)"
source /etc/profile.d/modules.sh
source /fsx/ferdinandmom/ferdinand-hf/huggingface/torchtitan/env_torchtitan_official/bin/activate
echo python3 version = $(python3 --version)
echo "==========="

# Slurm stuff
export HOSTNAMES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((1024 + RANDOM % 64511))

export TMPDIR=/scratch
export TORCH_HOME="/fsx/ferdinandmom/cache/torch"
export HF_HOME="/fsx/ferdinandmom/cache/huggingface"
export HF_DATASETS_CACHE="/fsx/ferdinandmom/cache/huggingface/datasets"
export TRANSFORMERS_CACHE="/fsx/ferdinandmom/cache/huggingface/transformers"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export CUDA_DEVICE_MAX_CONNECTIONS="1"
export UV_CACHE_DIR="/fsx/ferdinandmom/.cache/uv"

# EFA settings
export FI_PROVIDER=efa
export FI_EFA_FORK_SAFE=1
export FI_EFA_ENABLE_SHM_TRANSFER=1
export NCCL_PROTO=simple
export NCCL_SOCKET_IFNAME=enp

module load cuda/12.4

SCRIPT_DIR="/fsx/ferdinandmom/ferdinand-hf/huggingface/torchtitan/torchtitan/experiments/debug_fsdp"

echo "Running job: llama3_8b_fsdp_mixed_precision_torchtitan"
echo "Config file: /fsx/ferdinandmom/ferdinand-hf/huggingface/torchtitan/torchtitan/experiments/debug_fsdp/configs/torchtitan_llama3_8b.toml"

# Function to update status based on squeue output
update_status() {
    job_id=$1
    status_file=$2
    while true; do
        job_status=$(squeue --job $job_id --noheader --format=%T)
        echo "Job status: $job_status"
        if [ -z "$job_status" ]; then
            break
        elif [ "$job_status" = "RUNNING" ]; then
            printf "running" > $status_file
            break
        fi
        sleep 10
    done
}

# Update status in the background
update_status $SLURM_JOB_ID outputs/llama3_8b/fsdp_mixed_precision/torchtitan/status.txt &

cd "${SCRIPT_DIR}"



CMD="torchrun \
    --nproc_per_node 8 \
    --nnodes 1 \
    --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --role rank \
    --local_ranks_filter 0 \
    --tee 3 \
    minimal_fsdp.py \
    --job.config_file /fsx/ferdinandmom/ferdinand-hf/huggingface/torchtitan/torchtitan/experiments/debug_fsdp/configs/torchtitan_llama3_8b.toml \
    --checkpoint.enable \
    --checkpoint.initial_load_path outputs/llama3_8b/seed_checkpoint/checkpoint/step-0 \
    --checkpoint.load_only \
    --training.mixed_precision_param bfloat16 \
    --training.mixed_precision_reduce float32 \
    --model.flavor 8B"

echo "Running command: srun -u $CMD"
srun -u $CMD


exit_status=$?

# Update status based on exit
if [ $exit_status -eq 0 ]; then
    printf "completed" > outputs/llama3_8b/fsdp_mixed_precision/torchtitan/status.txt
else
    printf "fail" > outputs/llama3_8b/fsdp_mixed_precision/torchtitan/status.txt
fi