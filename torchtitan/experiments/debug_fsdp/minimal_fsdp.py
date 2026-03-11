#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import os
import time
from dataclasses import dataclass
from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.elastic.multiprocessing.errors import record

from torchtitan.components.metrics import DeviceMemoryMonitor
from torchtitan.config import Comm, Debug, TORCH_DTYPE_MAP
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.experiments.transformers_modeling_backend.infra.parallelize import (
    apply_fsdp,
)
from torchtitan.experiments.transformers_modeling_backend.model.args import (
    HFTransformerModelArgs,
    TitanDenseModelArgs,
)
from torchtitan.experiments.transformers_modeling_backend.model.model import (
    HFTransformerModel,
)
from torchtitan.tools import utils as titan_utils
from torchtitan.tools.logging import init_logger, logger
from torchtitan.tools.utils import device_module, device_type, set_default_dtype


FLAVORS = {
    "debugmodel": HFTransformerModelArgs(
        titan_dense_args=TitanDenseModelArgs(
            dim=256,
            n_layers=2,
            n_heads=16,
            n_kv_heads=16,
            vocab_size=2048,
        ),
    ),
    "full": HFTransformerModelArgs(
        titan_dense_args=TitanDenseModelArgs(),
    ),
}


@dataclass
class RuntimeConfig:
    hf_transformers: SimpleNamespace
    training: SimpleNamespace
    debug: SimpleNamespace


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Minimal TorchTitan FSDP debug script using the Hugging Face "
            "transformers modeling backend with a Llama 3 architecture."
        )
    )
    parser.add_argument(
        "--hf-model",
        default="meta-llama/Llama-3.2-1B",
        help="HF model id or local config path used to pick the architecture/config.",
    )
    parser.add_argument(
        "--flavor",
        default="debugmodel",
        choices=sorted(FLAVORS.keys()),
        help="TorchTitan backend flavor. debugmodel keeps the network tiny.",
    )
    parser.add_argument("--steps", type=int, default=10, help="Number of train steps.")
    parser.add_argument(
        "--batch-size", type=int, default=2, help="Per-rank batch size for synthetic data."
    )
    parser.add_argument(
        "--seq-len", type=int, default=512, help="Sequence length for synthetic tokens."
    )
    parser.add_argument("--lr", type=float, default=8e-4, help="AdamW learning rate.")
    parser.add_argument(
        "--max-norm", type=float, default=1.0, help="Gradient clipping max norm."
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.1, help="AdamW weight decay."
    )
    parser.add_argument(
        "--init-dtype",
        default="float32",
        choices=sorted(TORCH_DTYPE_MAP.keys()),
        help="Default dtype used while materializing the model.",
    )
    parser.add_argument(
        "--param-dtype",
        default="float32",
        choices=sorted(TORCH_DTYPE_MAP.keys()),
        help="FSDP mixed-precision parameter dtype.",
    )
    parser.add_argument(
        "--reduce-dtype",
        default="float32",
        choices=sorted(TORCH_DTYPE_MAP.keys()),
        help="FSDP reduction dtype.",
    )
    parser.add_argument(
        "--reshard-after-forward",
        default="default",
        choices=("default", "always", "never"),
        help="Pass-through to TorchTitan's FSDP helper.",
    )
    parser.add_argument("--seed", type=int, default=2026, help="Base RNG seed.")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic algorithms where possible.",
    )
    parser.add_argument(
        "--log-every", type=int, default=1, help="Rank-0 log frequency in steps."
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=2,
        help="Number of initial steps excluded from the final throughput summary.",
    )
    parser.add_argument(
        "--disable-color-printing",
        action="store_true",
        help="Disable ANSI colors in rank-0 metric logging.",
    )
    parser.add_argument(
        "--train-timeout-seconds",
        type=int,
        default=100,
        help="Timeout applied to all process groups after the first train step.",
    )
    return parser.parse_args()


def rank_zero_log(message: str) -> None:
    if not dist.is_initialized() or dist.get_rank() == 0:
        logger.info(message)


def build_runtime_config(args: argparse.Namespace) -> RuntimeConfig:
    return RuntimeConfig(
        hf_transformers=SimpleNamespace(model=args.hf_model),
        training=SimpleNamespace(seq_len=args.seq_len),
        debug=SimpleNamespace(deterministic=args.deterministic),
    )


def build_model(args: argparse.Namespace) -> tuple[HFTransformerModel, HFTransformerModelArgs]:
    model_args = copy.deepcopy(FLAVORS[args.flavor])
    model_args.update_from_config(build_runtime_config(args))

    rank_zero_log(
        "Building HF backend model "
        f"hf_model={args.hf_model} flavor={args.flavor} "
        f"layers={model_args.n_layers} dim={model_args.dim} "
        f"heads={model_args.n_heads} vocab={model_args.vocab_size} "
        f"seq_len={model_args.max_seq_len}"
    )

    with (
        torch.device("meta"),
        set_default_dtype(TORCH_DTYPE_MAP[args.init_dtype]),
    ):
        model = HFTransformerModel(model_args)

    return model, model_args


def init_distributed(args: argparse.Namespace) -> tuple[torch.device, ParallelDims]:
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

    comm_config = Comm(trace_buf_size=0)
    dist_utils.init_distributed(comm_config, enable_cpu_backend=False)

    parallel_dims = ParallelDims(
        dp_replicate=1,
        # ParallelDims only accepts -1 as the "use remaining ranks" sentinel.
        dp_shard=-1,
        cp=1,
        tp=1,
        pp=1,
        ep=1,
        etp=1,
        world_size=world_size,
    )

    debug_config = Debug(
        seed=args.seed,
        deterministic=args.deterministic,
        deterministic_warn_only=args.deterministic,
    )
    dist_utils.set_determinism(
        parallel_dims.world_mesh,
        device,
        debug_config,
        distinct_seed_mesh_dims=[],
    )

    rank_zero_log(
        f"Initialized distributed run rank={dist.get_rank()} world_size={world_size} "
        f"local_rank={local_rank} mesh={parallel_dims.world_mesh.mesh.tolist()}"
    )
    return device, parallel_dims


def assert_fsdp_only(parallel_dims: ParallelDims) -> None:
    if not parallel_dims.dp_shard_enabled:
        raise RuntimeError("This debug script requires FSDP sharding to be enabled.")
    if parallel_dims.dp_replicate_enabled:
        raise RuntimeError("This debug script is FSDP-only and does not support DDP/HSDP.")
    if (
        parallel_dims.cp_enabled
        or parallel_dims.tp_enabled
        or parallel_dims.pp_enabled
        or parallel_dims.ep_enabled
        or parallel_dims.etp_enabled
    ):
        raise RuntimeError(
            "This debug script is FSDP-only and requires cp=tp=pp=ep=etp=1."
        )


def materialize_and_wrap_model(
    model: HFTransformerModel,
    args: argparse.Namespace,
    parallel_dims: ParallelDims,
    device: torch.device,
) -> HFTransformerModel:
    assert_fsdp_only(parallel_dims)
    apply_fsdp(
        model,
        parallel_dims.world_mesh["dp_shard_cp"],
        param_dtype=TORCH_DTYPE_MAP[args.param_dtype],
        reduce_dtype=TORCH_DTYPE_MAP[args.reduce_dtype],
        pp_enabled=False,
        cpu_offload=False,
        reshard_after_forward_policy=args.reshard_after_forward,
    )

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
    args: argparse.Namespace,
) -> PerfStats:
    color = (
        titan_utils.NoColor() if args.disable_color_printing else titan_utils.Color()
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
    args = parse_args()

    device, parallel_dims = init_distributed(args)
    try:
        model, model_args = build_model(args)
        _, num_flops_per_token = model_args.get_nparams_and_flops(model, args.seq_len)
        model = materialize_and_wrap_model(model, args, parallel_dims, device)
        perf_stats = init_perf_stats(device, num_flops_per_token, args)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        world_size = dist.get_world_size()
        vocab_size = model.model.config.vocab_size
        tokens_per_step = args.batch_size * args.seq_len * world_size
        rank_zero_log(
            "Trainer is initialized with "
            f"local batch size {args.batch_size}, "
            f"global batch size {args.batch_size * world_size}, "
            "gradient accumulation steps 1, "
            f"sequence length {args.seq_len}, "
            f"total steps {args.steps} (warmup {args.warmup_steps})"
        )
        rank_zero_log("Training starts at step 1")
        perf_stats.time_last_log = time.perf_counter()

        for step in range(1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)

            tokens = make_synthetic_batch(
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                vocab_size=vocab_size,
                device=device,
            )

            logits = model(tokens)
            loss = cross_entropy_lm_loss(logits, tokens)
            loss.backward()

            grad_norm = dist_utils.clip_grad_norm_(
                list(model.parameters()),
                max_norm=args.max_norm,
                foreach=True,
            )
            optimizer.step()

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            perf_stats.ntokens_since_last_log += tokens_per_step

            if step % args.log_every == 0 or step == 1 or step == args.steps:
                avg_loss = get_global_mean(loss, parallel_dims)
                log_train_step(
                    step=step,
                    loss=avg_loss,
                    grad_norm=grad_norm.item(),
                    perf_stats=perf_stats,
                    parallel_dims=parallel_dims,
                    warmup_steps=args.warmup_steps,
                )

            if step == 1:
                dist_utils.set_pg_timeouts(
                    timeout=timedelta(seconds=args.train_timeout_seconds),
                    world_mesh=parallel_dims.world_mesh,
                )

        dist.barrier()
        log_post_warmup_summary(perf_stats, args.warmup_steps)
        rank_zero_log("Training completed")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
