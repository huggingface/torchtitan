#!/usr/bin/env python3
"""
Minimal TorchTitan FSDP debug script with synthetic data.
Supports both native TorchTitan models (e.g. llama3) and the
HF Transformers modeling backend.

Usage (with a TOML config):
    torchrun --nproc_per_node=2 minimal_fsdp.py \
        --job.config_file path/to/config.toml

Any TOML field can be overridden from the CLI, e.g.:
    torchrun --nproc_per_node=2 minimal_fsdp.py \
        --job.config_file path/to/config.toml \
        --training.steps 20
"""

from __future__ import annotations

import copy
import os
import time
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.elastic.multiprocessing.errors import record

from torchtitan.components.metrics import DeviceMemoryMonitor
from torchtitan.config import ConfigManager, JobConfig, TORCH_DTYPE_MAP
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.protocols.train_spec import BaseModelArgs, get_train_spec, TrainSpec
from torchtitan.tools import utils as titan_utils
from torchtitan.tools.logging import init_logger, logger
from torchtitan.tools.utils import device_module, device_type, set_default_dtype


@dataclass
class PerfStats:
    color: titan_utils.Color | titan_utils.NoColor
    device_memory_monitor: DeviceMemoryMonitor
    num_flops_per_token: int
    gpu_peak_flops: int
    ntokens_since_last_log: int = 0
    time_last_log: float = 0.0
    post_warmup_tps: list[float] = None
    post_warmup_tflops: list[float] = None
    post_warmup_mfu: list[float] = None

    def __post_init__(self) -> None:
        if self.post_warmup_tps is None:
            self.post_warmup_tps = []
        if self.post_warmup_tflops is None:
            self.post_warmup_tflops = []
        if self.post_warmup_mfu is None:
            self.post_warmup_mfu = []


def rank_zero_log(message: str) -> None:
    if not dist.is_initialized() or dist.get_rank() == 0:
        logger.info(message)


def build_model(
    job_config: JobConfig,
    train_spec: TrainSpec,
) -> tuple[torch.nn.Module, BaseModelArgs]:
    flavor = job_config.model.flavor
    model_args = copy.deepcopy(train_spec.model_args[flavor])
    model_args.update_from_config(job_config)

    rank_zero_log(
        f"Building {job_config.model.name} {flavor} with {model_args}"
    )

    with (
        torch.device("meta"),
        set_default_dtype(TORCH_DTYPE_MAP[job_config.training.dtype]),
    ):
        model = train_spec.model_cls(model_args)

    return model, model_args


def init_distributed(job_config: JobConfig) -> tuple[torch.device, ParallelDims]:
    required_env = ("RANK", "LOCAL_RANK", "WORLD_SIZE")
    missing_env = [name for name in required_env if name not in os.environ]
    if missing_env:
        missing = ", ".join(missing_env)
        raise RuntimeError(
            f"Missing distributed env vars: {missing}. Launch with torchrun."
        )

    if device_type != "cuda":
        raise RuntimeError(
            f"This debug script expects CUDA, but torchtitan device_type resolved to {device_type!r}."
        )

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size < 2:
        raise RuntimeError(
            "FSDP-only debug run requires WORLD_SIZE >= 2. "
            "Use torchrun --nproc_per_node=2 (or more)."
        )

    device = torch.device(f"cuda:{local_rank}")
    device_module.set_device(device)

    dist_utils.init_distributed(job_config.comm, enable_cpu_backend=False)

    parallel_dims = ParallelDims(
        dp_replicate=job_config.parallelism.data_parallel_replicate_degree,
        dp_shard=job_config.parallelism.data_parallel_shard_degree,
        cp=job_config.parallelism.context_parallel_degree,
        tp=job_config.parallelism.tensor_parallel_degree,
        pp=job_config.parallelism.pipeline_parallel_degree,
        ep=job_config.parallelism.expert_parallel_degree,
        etp=job_config.parallelism.expert_tensor_parallel_degree,
        world_size=world_size,
    )

    dist_utils.set_determinism(
        parallel_dims.world_mesh,
        device,
        job_config.debug,
        distinct_seed_mesh_dims=[],
    )

    rank_zero_log(
        f"Initialized distributed run rank={dist.get_rank()} world_size={world_size} "
        f"local_rank={local_rank} mesh={parallel_dims.world_mesh.mesh.tolist()}"
    )
    return device, parallel_dims


def parallelize_and_materialize(
    model: torch.nn.Module,
    train_spec: TrainSpec,
    job_config: JobConfig,
    parallel_dims: ParallelDims,
    device: torch.device,
) -> torch.nn.Module:
    model = train_spec.parallelize_fn(model, parallel_dims, job_config)

    model.to_empty(device=device)
    with torch.no_grad():
        model.init_weights()
    model.train()
    return model


def make_synthetic_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
) -> torch.Tensor:
    if vocab_size <= 0:
        raise ValueError(f"Invalid vocab size: {vocab_size}")
    return torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, seq_len),
        device=device,
    )


def cross_entropy_lm_loss(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = tokens[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )


def get_global_mean(value: torch.Tensor, parallel_dims: ParallelDims) -> float:
    if parallel_dims.dp_cp_enabled:
        return dist_utils.dist_mean(value.detach(), parallel_dims.world_mesh["dp_cp"])
    return value.detach().item()


def init_perf_stats(
    device: torch.device,
    num_flops_per_token: int,
    job_config: JobConfig,
) -> PerfStats:
    color = (
        titan_utils.NoColor()
        if job_config.metrics.disable_color_printing
        else titan_utils.Color()
    )
    device_memory_monitor = DeviceMemoryMonitor(str(device))
    gpu_peak_flops = titan_utils.get_peak_flops(device_memory_monitor.device_name)
    model_mem_stats = device_memory_monitor.get_peak_stats()

    rank_zero_log(f"Peak FLOPS used for computing MFU: {gpu_peak_flops:.3e}")
    rank_zero_log(
        f"{device.type.upper()} memory usage for model: "
        f"{model_mem_stats.max_reserved_gib:.2f}GiB"
        f"({model_mem_stats.max_reserved_pct:.2f}%)"
    )
    device_memory_monitor.reset_peak_stats()

    return PerfStats(
        color=color,
        device_memory_monitor=device_memory_monitor,
        num_flops_per_token=num_flops_per_token,
        gpu_peak_flops=gpu_peak_flops,
    )


def log_train_step(
    step: int,
    loss: float,
    grad_norm: float,
    perf_stats: PerfStats,
    parallel_dims: ParallelDims,
    warmup_steps: int,
) -> None:
    time_delta = time.perf_counter() - perf_stats.time_last_log
    tps = perf_stats.ntokens_since_last_log / (
        time_delta * parallel_dims.non_data_parallel_size
    )
    tflops = perf_stats.num_flops_per_token * tps / 1e12
    mfu = 100 * perf_stats.num_flops_per_token * tps / perf_stats.gpu_peak_flops
    device_mem_stats = perf_stats.device_memory_monitor.get_peak_stats()
    color = perf_stats.color

    rank_zero_log(
        f"{color.red}step: {step:2}  "
        f"{color.green}loss: {loss:7.4f}  "
        f"{color.orange}grad_norm: {grad_norm:7.4f}  "
        f"{color.turquoise}memory: {device_mem_stats.max_reserved_gib:5.2f}GiB"
        f"({device_mem_stats.max_reserved_pct:.2f}%)  "
        f"{color.blue}tps: {round(tps):,}  "
        f"{color.cyan}tflops: {tflops:,.2f}  "
        f"{color.magenta}mfu: {mfu:.2f}%{color.reset}"
    )

    if step > warmup_steps:
        perf_stats.post_warmup_tps.append(tps)
        perf_stats.post_warmup_tflops.append(tflops)
        perf_stats.post_warmup_mfu.append(mfu)

    perf_stats.ntokens_since_last_log = 0
    perf_stats.time_last_log = time.perf_counter()
    perf_stats.device_memory_monitor.reset_peak_stats()


def log_post_warmup_summary(perf_stats: PerfStats, warmup_steps: int) -> None:
    if not perf_stats.post_warmup_tps:
        return

    avg_tps = sum(perf_stats.post_warmup_tps) / len(perf_stats.post_warmup_tps)
    avg_tflops = sum(perf_stats.post_warmup_tflops) / len(
        perf_stats.post_warmup_tflops
    )
    avg_mfu = sum(perf_stats.post_warmup_mfu) / len(perf_stats.post_warmup_mfu)
    color = perf_stats.color
    rank_zero_log(
        f"{color.yellow}post-warmup ({warmup_steps} steps) avg  "
        f"{color.blue}tps: {round(avg_tps):,}  "
        f"{color.cyan}tflops: {avg_tflops:,.2f}  "
        f"{color.magenta}mfu: {avg_mfu:.2f}%{color.reset}"
    )


@record
def main() -> None:
    init_logger()
    config_manager = ConfigManager()
    job_config = config_manager.parse_args()

    train_spec = get_train_spec(job_config.model.name)

    device, parallel_dims = init_distributed(job_config)
    try:
        model, model_args = build_model(job_config, train_spec)
        seq_len = job_config.training.seq_len
        _, num_flops_per_token = model_args.get_nparams_and_flops(model, seq_len)
        model = parallelize_and_materialize(
            model, train_spec, job_config, parallel_dims, device
        )
        perf_stats = init_perf_stats(device, num_flops_per_token, job_config)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=job_config.optimizer.lr,
            weight_decay=job_config.optimizer.weight_decay,
        )

        world_size = dist.get_world_size()
        batch_size = job_config.training.local_batch_size
        steps = job_config.training.steps
        max_norm = job_config.training.max_norm
        warmup_steps = job_config.lr_scheduler.warmup_steps
        log_freq = job_config.metrics.log_freq
        vocab_size = model_args.vocab_size
        tokens_per_step = batch_size * seq_len * world_size
        rank_zero_log(
            "Trainer is initialized with "
            f"local batch size {batch_size}, "
            f"global batch size {batch_size * world_size}, "
            "gradient accumulation steps 1, "
            f"sequence length {seq_len}, "
            f"total steps {steps} (warmup {warmup_steps})"
        )
        rank_zero_log("Training starts at step 1")
        perf_stats.time_last_log = time.perf_counter()

        for step in range(1, steps + 1):
            optimizer.zero_grad(set_to_none=True)

            tokens = make_synthetic_batch(
                batch_size=batch_size,
                seq_len=seq_len,
                vocab_size=vocab_size,
                device=device,
            )

            logits = model(tokens)
            loss = cross_entropy_lm_loss(logits, tokens)
            loss.backward()

            grad_norm = dist_utils.clip_grad_norm_(
                list(model.parameters()),
                max_norm=max_norm,
                foreach=True,
            )
            optimizer.step()

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            perf_stats.ntokens_since_last_log += tokens_per_step

            if step % log_freq == 0 or step == 1 or step == steps:
                avg_loss = get_global_mean(loss, parallel_dims)
                log_train_step(
                    step=step,
                    loss=avg_loss,
                    grad_norm=grad_norm.item(),
                    perf_stats=perf_stats,
                    parallel_dims=parallel_dims,
                    warmup_steps=warmup_steps,
                )

            if step == 1:
                dist_utils.set_pg_timeouts(
                    timeout=timedelta(
                        seconds=job_config.comm.train_timeout_seconds
                    ),
                    world_mesh=parallel_dims.world_mesh,
                )

        dist.barrier()
        log_post_warmup_summary(perf_stats, warmup_steps)
        rank_zero_log("Training completed")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
