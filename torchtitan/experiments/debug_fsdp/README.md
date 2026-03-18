# Reproducibility Report — Llama3 8B FSDP Debug

## Experiment

Comparing **TorchTitan native** vs **HF `from_pretrained` + FSDP** (`fsdp_plan="auto"`) across 4 FSDP scenarios, both starting from the same seed checkpoint.

### Scenarios

| Scenario | FSDP | CPU Offload | Mixed Precision |
|----------|------|-------------|-----------------|
| `fsdp` | Yes | No | No (fp32/fp32) |
| `fsdp_mixed_precision` | Yes | No | Yes (bf16 param / fp32 reduce) |
| `fsdp_cpu_offload` | Yes | Yes | No (fp32/fp32) |
| `fsdp_cpu_offload_mixed_precision` | Yes | Yes | Yes (bf16 param / fp32 reduce) |

### Backends

- **TorchTitan**: Native `llama3` model with `apply_fsdp` from `torchtitan.models.llama3.infra.parallelize`
- **HF from_pretrained**: `AutoModelForCausalLM.from_pretrained` with `fsdp_plan={"mode": "auto", ...}` via `transformers-v5-fsdp`

---

## Environment

### Hardware
- **GPU**: 8× NVIDIA H100 80GB HBM3 (79.4 GiB each)
- **Nodes**: 1
- **Interconnect**: EFA (AWS)
- **Partition**: hopper-prod

### Software
| Component | Version |
|-----------|---------|
| OS | Ubuntu 20.04.6 LTS |
| Kernel | 5.15.0-1048-aws |
| NVIDIA Driver | 575.57.08 |
| CUDA (module) | 12.4 |
| Python | 3.12.9 |
| PyTorch | 2.11.0.dev20251214+cu126 |
| CUDA (torch) | 12.6 |
| cuDNN | 9.10.02 |
| NCCL | 2.28.9 |
| NumPy | 1.26.4 |
| transformers (installed) | 5.3.0.dev0 |
| datasets | 4.4.1 |
| safetensors | 0.6.2 |
| tokenizers | 0.22.1 |
| Jinja2 | 3.1.6 |
| matplotlib | 3.10.7 |

### transformers-v5-fsdp (HF FSDP backend)
- **Version**: 5.3.0.dev0
- **Commit**: `c209232aefedc116cfc5c026631b572e63f20f6c`
- **Branch**: `fsdp-vs-ddp`

### torchtitan
- **Commit**: `5bcd17e0bf02419bfa569ae292920a308d810e62`
- **Branch**: `sanity-check-fsdp`
- **Note**: Working tree has uncommitted changes to `minimal_fsdp.py` (added llama3 `apply_fsdp` dispatch)

---

## Model Architecture — Llama3 8B

| Parameter | Value |
|-----------|-------|
| Architecture | LlamaForCausalLM |
| hidden_size | 4096 |
| num_hidden_layers | 32 |
| num_attention_heads | 32 |
| num_key_value_heads | 8 |
| intermediate_size | 14336 |
| head_dim | 128 |
| vocab_size | 128256 |
| rms_norm_eps | 1e-5 |
| rope_theta | 500000.0 |
| tie_word_embeddings | false |
| max_position_embeddings | 131072 |

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 8e-4 |
| Epsilon | 1e-8 |
| LR warmup steps | 100 |
| Local batch size | 1 |
| Sequence length | 2048 |
| Grad clip (max_norm) | 1.0 |
| Training steps | 3000 |
| Dataset | c4_test (small test split, re-looped) |
| Seed | 2026 |
| Deterministic | true |
| Activation checkpoint | none |
| torch.compile | disabled |

### Parallelism
| Parameter | Value |
|-----------|-------|
| data_parallel_replicate_degree | 1 |
| data_parallel_shard_degree | -1 (auto = 8) |
| tensor_parallel_degree | 1 |
| pipeline_parallel_degree | 1 |
| context_parallel_degree | 1 |

---

## Seed Checkpoint

Both backends load the same seed checkpoint to ensure identical initial weights.

1. TorchTitan builds the model on CPU, calls `init_weights()`, exports to HF-format state dict via `Llama3StateDictAdapter.to_hf()`
2. Saved as `seed_model.pt` in `seed_checkpoint/checkpoint/step-0/`
3. TorchTitan backend loads via `Llama3StateDictAdapter.from_hf()`
4. HF backend loads via `from_pretrained()` after converting seed to safetensors format

---

## Environment Variables (SLURM)

```bash
CUBLAS_WORKSPACE_CONFIG=":4096:8"    # Deterministic cuBLAS
CUDA_DEVICE_MAX_CONNECTIONS="1"
FI_PROVIDER=efa
FI_EFA_FORK_SAFE=1
FI_EFA_ENABLE_SHM_TRANSFER=1
NCCL_PROTO=simple
NCCL_SOCKET_IFNAME=enp
```

---

## How to Reproduce

```bash
cd torchtitan/experiments/debug_fsdp

# 1. Create configs (all 4 FSDP scenarios, both backends)
python runner.py create_configs --out_dir ./outputs/llama3_8b --model llama3_8b --qos high

# 2. Submit seed checkpoint
python runner.py submit_jobs --inp_dir ./outputs/llama3_8b --qos high --only seed_checkpoint

# 3. Wait for seed, then submit training jobs
python runner.py check_status --inp_dir ./outputs/llama3_8b
python runner.py submit_jobs --inp_dir ./outputs/llama3_8b --qos high

# 4. Monitor
python runner.py check_status --inp_dir ./outputs/llama3_8b

# 5. Generate report + plots
python runner.py report --inp_dir ./outputs/llama3_8b
python plot.py --inp_dir ./outputs/llama3_8b
```

---

## Files

| File | Purpose |
|------|---------|
| `configs/torchtitan_llama3_8b.toml` | TorchTitan training config |
| `configs/hf_llama3_8b_fsdp.toml` | HF from_pretrained training config |
| `configs/hf_llama3_8b/config.json` | HF model architecture config |
| `minimal_fsdp.py` | Training script (shared by all backends) |
| `runner.py` | SLURM job management (create/submit/status/report) |
| `template.slurm` | SLURM job template |
| `plot.py` | Metrics visualization |
