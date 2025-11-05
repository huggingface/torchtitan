# aa.sh
torchrun \
  --nproc_per_node 1 \
  --nnodes 1 \
  --rdzv_endpoint localhost:0 \
  --rdzv_backend c10d \
  --max_restarts 0 \
  --role rank \
  --local_ranks_filter 0 \
  --tee 3 \
  -m torchtitan.train \
  --checkpoint.enable \
  --checkpoint.initial_load_path debug_local_results/meta-llama/Llama-3.2-1B/debugmodel/seed_checkpoint/checkpoint/step-0 \
  --training.seed 42 \
  --training.deterministic \
  --training.steps 2 \
  --job.custom_config_module=torchtitan.experiments.transformers_backend.job_config \
  --job.config_file debug_local_results/meta-llama/Llama-3.2-1B/debugmodel/fsdp1_tp1_cp1_pp1/config.toml \
  2>&1 | tee log_baseline.txt

# Rank 0
CUDA_VISIBLE_DEVICES=0 torchrun \
  --nproc_per_node 1 \
  --nnodes 2 \
  --node_rank 0 \
  --rdzv_endpoint localhost:29500 \
  --rdzv_backend c10d \
  --max_restarts 0 \
  --role rank \
  --tee 3 \
  -m torchtitan.train \
  --checkpoint.enable \
  --checkpoint.initial_load_path debug_local_results/meta-llama/Llama-3.2-1B/debugmodel/seed_checkpoint/checkpoint/step-0 \
  --training.seed 42 \
  --training.deterministic \
  --training.steps 2 \
  --job.custom_config_module=torchtitan.experiments.transformers_backend.job_config \
  --job.config_file debug_local_results/meta-llama/Llama-3.2-1B/debugmodel/fsdp1_tp1_cp1_pp2/config.toml \
  2>&1 | tee log_pp2_rank0.txt &


# rank 1
CUDA_VISIBLE_DEVICES=1 torchrun \
  --nproc_per_node 1 \
  --nnodes 2 \
  --node_rank 1 \
  --rdzv_endpoint localhost:29500 \
  --rdzv_backend c10d \
  --max_restarts 0 \
  --role rank \
  --tee 3 \
  -m torchtitan.train \
  --checkpoint.enable \
  --checkpoint.initial_load_path debug_local_results/meta-llama/Llama-3.2-1B/debugmodel/seed_checkpoint/checkpoint/step-0 \
  --training.seed 42 \
  --training.steps 2 \
  --training.deterministic \
  --job.custom_config_module=torchtitan.experiments.transformers_backend.job_config \
  --job.config_file debug_local_results/meta-llama/Llama-3.2-1B/debugmodel/fsdp1_tp1_cp1_pp2/config.toml \
  2>&1 | tee log_pp2_rank1.txt &