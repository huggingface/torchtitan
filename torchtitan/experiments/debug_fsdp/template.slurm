#!/bin/bash
#SBATCH --job-name={{ name }}
#SBATCH --output={{ root_path }}/slurm_%j.out
#SBATCH --error={{ root_path }}/slurm_%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:{{ n_proc_per_node }}
#SBATCH --ntasks-per-node=1
#SBATCH --qos={{ qos }}
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

echo "Running job: {{ name }}"
echo "Config file: {{ config_path }}"

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
update_status $SLURM_JOB_ID {{ root_path }}/status.txt &

cd "${SCRIPT_DIR}"
{% if is_seed %}

# Seed checkpoint creation (single process, no torchrun)
CMD="python minimal_fsdp.py \
    --job.config_file {{ config_path }} \
    --model.flavor {{ model_flavor }} \
    --job.dump_folder {{ root_path }} \
    --checkpoint.enable \
    --checkpoint.create_seed_checkpoint"

echo "Running command: $CMD"
$CMD
{% else %}
{% if use_hf_fsdp %}
export USE_HF_FSDP=1
echo "USE_HF_FSDP=1"
{% endif %}

CMD="torchrun \
    --nproc_per_node {{ n_proc_per_node }} \
    --nnodes 1 \
    --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --role rank \
    --local_ranks_filter 0 \
    --tee 3 \
    minimal_fsdp.py \
    --job.config_file {{ config_path }} \
    --checkpoint.enable \
    --checkpoint.initial_load_path {{ seed_checkpoint_path }} \
    --checkpoint.load_only{% if enable_cpu_offload %} \
    --training.enable_cpu_offload{% endif %}{% if mixed_precision_param %} \
    --training.mixed_precision_param {{ mixed_precision_param }}{% endif %}{% if mixed_precision_reduce %} \
    --training.mixed_precision_reduce {{ mixed_precision_reduce }}{% endif %}{% if model_flavor %} \
    --model.flavor {{ model_flavor }}{% endif %}"

echo "Running command: srun -u $CMD"
srun -u $CMD
{% endif %}

exit_status=$?

# Update status based on exit
if [ $exit_status -eq 0 ]; then
    printf "completed" > {{ root_path }}/status.txt
else
    printf "fail" > {{ root_path }}/status.txt
fi
