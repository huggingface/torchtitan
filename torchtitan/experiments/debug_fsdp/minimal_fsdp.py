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
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import config as datasets_config
from torch.distributed.elastic.multiprocessing.errors import record

from torchtitan.components.dataloader import DataloaderExhaustedError
from torchtitan.components.metrics import DeviceMemoryMonitor
from torchtitan.config import ConfigManager, JobConfig, TORCH_DTYPE_MAP
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.models.llama3.model.state_dict_adapter import Llama3StateDictAdapter
from torchtitan.models.qwen3.model.state_dict_adapter import Qwen3StateDictAdapter
from torchtitan.protocols.train_spec import BaseModelArgs, get_train_spec, TrainSpec
from torchtitan.tools import utils as titan_utils
from torchtitan.tools.logging import init_logger, logger
from torchtitan.tools.utils import device_module, device_type, set_default_dtype

USE_HF_FSDP = os.environ.get("USE_HF_FSDP", "0") == "1"


SEED_CHECKPOINT_FILENAME = "seed_model.pt"
SEED_CHECKPOINT_METADATA_FILENAME = "seed_model.json"
HF_ALLOWED_MISSING_KEY_SUFFIXES = ("rotary_emb.inv_freq",)
HF_TIED_WEIGHT_ALLOWED_MISSING_KEY_SUFFIXES = ("lm_head.weight",)


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
    use_meta_init: bool = True,
) -> tuple[torch.nn.Module, BaseModelArgs]:
    flavor = job_config.model.flavor
    model_args = copy.deepcopy(train_spec.model_args[flavor])
    model_args.update_from_config(job_config)

    rank_zero_log(
        f"Building {job_config.model.name} {flavor} with {model_args}"
    )

    with set_default_dtype(TORCH_DTYPE_MAP[job_config.training.dtype]):
        if use_meta_init:
            with torch.device("meta"):
                model = train_spec.model_cls(model_args)
        else:
            model = train_spec.model_cls(model_args)

    return model, model_args


def _prepare_hf_fsdp_model_dir(job_config: JobConfig) -> Path:
    """Prepare a HF-format model directory with seed checkpoint weights.

    Copies config.json from the HF model config path and converts the seed
    checkpoint (.pt with HF-compatible keys) into safetensors so that
    ``from_pretrained`` can load everything in one shot.
    """
    import shutil

    from safetensors.torch import save_file

    hf_config_dir = Path(job_config.hf_transformers.model)
    seed_checkpoint_path = job_config.checkpoint.initial_load_path

    # Build the output directory next to the seed checkpoint
    out_dir = Path(job_config.job.dump_folder) / "hf_fsdp_pretrained"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy config.json
    shutil.copy2(hf_config_dir / "config.json", out_dir / "config.json")

    if seed_checkpoint_path:
        seed_state_dict = load_seed_checkpoint(seed_checkpoint_path)
        save_file(seed_state_dict, out_dir / "model.safetensors")
        rank_zero_log(
            f"Prepared HF model dir at {out_dir} with seed weights from {seed_checkpoint_path}"
        )
    else:
        rank_zero_log(
            f"Prepared HF model dir at {out_dir} (random init, no seed checkpoint)"
        )

    return out_dir


def build_hf_fsdp_model(
    job_config: JobConfig,
    device: torch.device,
    world_size: int,
) -> tuple[torch.nn.Module, int]:
    """Build model using HF Transformers from_pretrained with fsdp_plan='auto'.

    This bypasses torchtitan's parallelization and uses the FSDP2 integration
    built into transformers-v5.
    """
    # Add transformers-v5-fsdp to the path so we use the right version
    v5_fsdp_path = str(Path(__file__).resolve().parents[3] / "transformers-v5-fsdp" / "src")
    if v5_fsdp_path not in sys.path:
        sys.path.insert(0, v5_fsdp_path)

    from transformers import AutoModelForCausalLM

    # Only rank 0 prepares the directory to avoid races
    if dist.get_rank() == 0:
        model_dir = _prepare_hf_fsdp_model_dir(job_config)
    dist.barrier()
    # All ranks resolve the same path
    model_dir = Path(job_config.job.dump_folder) / "hf_fsdp_pretrained"

    train_dtype = TORCH_DTYPE_MAP[job_config.training.dtype]

    fsdp_mesh = torch.distributed.init_device_mesh(
        device.type, (world_size,), mesh_dim_names=("dp_shard",)
    )

    rank_zero_log(
        f"Building HF FSDP model from {model_dir} with fsdp_plan='auto' "
        f"(cpu_offload=False, mixed_precision=False)"
    )

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        fsdp_plan={"mode": "auto", "cpu_offload": False, "mixed_precision": False},
        device_mesh=fsdp_mesh,
        torch_dtype=train_dtype,
    )

    # Count parameters and estimate FLOPs
    num_params = sum(p.numel() for p in model.parameters())
    seq_len = job_config.training.seq_len
    # Rough FLOPs estimate: 6 * num_params (forward + backward) per token
    num_flops_per_token = 6 * num_params

    rank_zero_log(
        f"HF FSDP model loaded: {num_params:,} params, "
        f"estimated {num_flops_per_token:,} flops/token"
    )

    model.train()
    return model, num_flops_per_token


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

    dist_utils.init_distributed(
        job_config.comm,
        enable_cpu_backend=False,
        base_folder=job_config.job.dump_folder,
    )

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


def move_and_parallelize_loaded_model(
    model: torch.nn.Module,
    train_spec: TrainSpec,
    job_config: JobConfig,
    parallel_dims: ParallelDims,
    device: torch.device,
) -> torch.nn.Module:
    model.to(device)
    model = train_spec.parallelize_fn(model, parallel_dims, job_config)
    model.train()
    return model


def build_dataloader(
    job_config: JobConfig,
    train_spec: TrainSpec,
    parallel_dims: ParallelDims,
):
    if "HF_DATASETS_CACHE" not in os.environ:
        datasets_cache = Path(job_config.job.dump_folder) / "hf_datasets_cache"
        datasets_cache.mkdir(parents=True, exist_ok=True)
        os.environ["HF_DATASETS_CACHE"] = str(datasets_cache)
        datasets_config.HF_DATASETS_CACHE = str(datasets_cache)

    tokenizer = (
        train_spec.build_tokenizer_fn(job_config)
        if train_spec.build_tokenizer_fn is not None
        else None
    )
    if tokenizer is None:
        raise RuntimeError(
            f"{job_config.model.name!r} does not provide a tokenizer for dataset-backed debug runs."
        )

    if parallel_dims.dp_enabled:
        dp_mesh = parallel_dims.world_mesh["dp"]
        dp_world_size = dp_mesh.size()
        dp_rank = dp_mesh.get_local_rank()
    else:
        dp_world_size = 1
        dp_rank = 0

    dataloader = train_spec.build_dataloader_fn(
        dp_world_size=dp_world_size,
        dp_rank=dp_rank,
        tokenizer=tokenizer,
        job_config=job_config,
    )
    return dataloader


def get_next_batch(
    data_iterator,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    try:
        input_dict, labels = next(data_iterator)
    except StopIteration as ex:
        raise DataloaderExhaustedError() from ex

    for key, value in input_dict.items():
        if isinstance(value, torch.Tensor):
            input_dict[key] = value.to(device)
    labels = labels.to(device)
    return input_dict, labels


def cross_entropy_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
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


def get_seed_checkpoint_dir(
    job_config: JobConfig,
    path_override: str | None = None,
) -> Path:
    if path_override:
        return Path(path_override)
    return (
        Path(job_config.job.dump_folder)
        / job_config.checkpoint.folder
        / "step-0"
    )


def get_seed_checkpoint_file(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / SEED_CHECKPOINT_FILENAME


def get_seed_checkpoint_metadata_file(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / SEED_CHECKPOINT_METADATA_FILENAME


def clone_state_dict_to_cpu(
    state_dict: dict[str, torch.Tensor],
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().to(dtype).clone()
        for key, value in state_dict.items()
    }


def build_native_seed_adapter(
    model_args: BaseModelArgs,
    job_config: JobConfig,
) -> Llama3StateDictAdapter | Qwen3StateDictAdapter:
    if job_config.model.name == "llama3":
        return Llama3StateDictAdapter(model_args, job_config.model.hf_assets_path)
    if job_config.model.name == "qwen3":
        return Qwen3StateDictAdapter(model_args, job_config.model.hf_assets_path)
    raise NotImplementedError(
        f"Shared seed checkpoint adapter is not implemented for {job_config.model.name!r}."
    )


def export_seed_state_dict(
    model: torch.nn.Module,
    model_args: BaseModelArgs,
    job_config: JobConfig,
) -> dict[str, torch.Tensor]:
    export_dtype = TORCH_DTYPE_MAP[job_config.checkpoint.export_dtype]
    state_dict = clone_state_dict_to_cpu(model.state_dict(), export_dtype)

    if job_config.model.name in {"llama3", "qwen3"}:
        adapter = build_native_seed_adapter(model_args, job_config)
        return adapter.to_hf(state_dict)

    if job_config.model.name == "transformers_modeling_backend":
        normalized_state_dict = {}
        for key, value in state_dict.items():
            normalized_key = key.removeprefix("model.")
            if (
                normalized_key == "lm_head.weight"
                and getattr(model_args, "tie_word_embeddings", False)
            ):
                continue
            normalized_state_dict[normalized_key] = value
        return normalized_state_dict

    raise NotImplementedError(
        f"Shared seed checkpoint export is not implemented for {job_config.model.name!r}."
    )


def save_seed_checkpoint(
    model: torch.nn.Module,
    model_args: BaseModelArgs,
    job_config: JobConfig,
) -> Path:
    checkpoint_dir = get_seed_checkpoint_dir(job_config)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        export_seed_state_dict(model, model_args, job_config),
        get_seed_checkpoint_file(checkpoint_dir),
    )

    metadata = {
        "format": "hf_compatible_state_dict",
        "model_name": job_config.model.name,
        "flavor": job_config.model.flavor,
    }
    get_seed_checkpoint_metadata_file(checkpoint_dir).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return checkpoint_dir


def load_seed_checkpoint(path: str) -> dict[str, torch.Tensor]:
    checkpoint_dir = Path(path)
    checkpoint_file = get_seed_checkpoint_file(checkpoint_dir)
    if not checkpoint_file.is_file():
        raise FileNotFoundError(
            f"Shared seed checkpoint file not found at {checkpoint_file}."
        )
    return torch.load(checkpoint_file, map_location="cpu", weights_only=True)


def validate_loaded_state_dict(
    incompatible_keys: torch.nn.modules.module._IncompatibleKeys,
    allowed_missing_suffixes: tuple[str, ...] = (),
) -> None:
    missing_keys = [
        key
        for key in incompatible_keys.missing_keys
        if not any(key.endswith(suffix) for suffix in allowed_missing_suffixes)
    ]
    unexpected_keys = list(incompatible_keys.unexpected_keys)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "Seed checkpoint load mismatch: "
            f"missing_keys={missing_keys}, unexpected_keys={unexpected_keys}"
        )


def load_seed_checkpoint_into_model(
    model: torch.nn.Module,
    model_args: BaseModelArgs,
    job_config: JobConfig,
) -> None:
    if not job_config.checkpoint.initial_load_path:
        return

    shared_seed_state_dict = load_seed_checkpoint(
        job_config.checkpoint.initial_load_path
    )

    if job_config.model.name in {"llama3", "qwen3"}:
        adapter = build_native_seed_adapter(model_args, job_config)
        incompatible_keys = model.load_state_dict(
            adapter.from_hf(shared_seed_state_dict),
            strict=False,
        )
        validate_loaded_state_dict(incompatible_keys)
    elif job_config.model.name == "transformers_modeling_backend":
        wrapped_state_dict = {
            f"model.{key}": value for key, value in shared_seed_state_dict.items()
        }
        incompatible_keys = model.load_state_dict(
            wrapped_state_dict,
            strict=False,
        )
        allowed_missing_suffixes = HF_ALLOWED_MISSING_KEY_SUFFIXES
        if getattr(model_args, "tie_word_embeddings", False):
            allowed_missing_suffixes += HF_TIED_WEIGHT_ALLOWED_MISSING_KEY_SUFFIXES
        validate_loaded_state_dict(
            incompatible_keys,
            allowed_missing_suffixes=allowed_missing_suffixes,
        )
    else:
        raise NotImplementedError(
            f"Shared seed checkpoint load is not implemented for {job_config.model.name!r}."
        )

    rank_zero_log(
        f"Loaded shared seed checkpoint from {job_config.checkpoint.initial_load_path}"
    )


def initialize_full_model(
    model: torch.nn.Module,
    job_config: JobConfig,
) -> None:
    init_weights = getattr(model, "init_weights", None)
    if init_weights is None or job_config.model.name == "transformers_modeling_backend":
        return
    with torch.no_grad():
        init_weights()


def create_seed_checkpoint(
    job_config: JobConfig,
    train_spec: TrainSpec,
) -> None:
    dist_utils.set_determinism(
        world_mesh=None,
        device=torch.device("cpu"),
        debug_config=job_config.debug,
        distinct_seed_mesh_dims=[],
    )

    model, model_args = build_model(
        job_config,
        train_spec,
        use_meta_init=False,
    )
    initialize_full_model(model, job_config)
    checkpoint_dir = save_seed_checkpoint(model, model_args, job_config)
    logger.info("Saved shared seed checkpoint to %s", checkpoint_dir)


@record
def main() -> None:
    init_logger()
    config_manager = ConfigManager()
    job_config = config_manager.parse_args()

    train_spec = get_train_spec(job_config.model.name)

    if job_config.checkpoint.enable and job_config.checkpoint.create_seed_checkpoint:
        create_seed_checkpoint(job_config, train_spec)
        return

    device, parallel_dims = init_distributed(job_config)
    try:
        world_size = dist.get_world_size()
        seq_len = job_config.training.seq_len

        if USE_HF_FSDP:
            rank_zero_log("Using HF Transformers from_pretrained with fsdp_plan='auto'")
            model, num_flops_per_token = build_hf_fsdp_model(
                job_config, device, world_size
            )
        else:
            load_shared_seed = bool(
                job_config.checkpoint.enable
                and job_config.checkpoint.initial_load_path
            )
            model, model_args = build_model(
                job_config,
                train_spec,
                use_meta_init=not load_shared_seed,
            )
            _, num_flops_per_token = model_args.get_nparams_and_flops(model, seq_len)
            if load_shared_seed:
                load_seed_checkpoint_into_model(model, model_args, job_config)
                model = move_and_parallelize_loaded_model(
                    model,
                    train_spec,
                    job_config,
                    parallel_dims,
                    device,
                )
            else:
                model = parallelize_and_materialize(
                    model, train_spec, job_config, parallel_dims, device
                )

        perf_stats = init_perf_stats(device, num_flops_per_token, job_config)
        dataloader = build_dataloader(job_config, train_spec, parallel_dims)
        data_iterator = iter(dataloader)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=job_config.optimizer.lr,
            weight_decay=job_config.optimizer.weight_decay,
        )

        batch_size = job_config.training.local_batch_size
        steps = job_config.training.steps
        max_norm = job_config.training.max_norm
        warmup_steps = job_config.lr_scheduler.warmup_steps
        log_freq = job_config.metrics.log_freq
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

            input_dict, labels = get_next_batch(data_iterator, device)

            if USE_HF_FSDP:
                inputs = input_dict["input"]
                outputs = model(input_ids=inputs)
                logits = outputs.logits
            else:
                inputs = input_dict["input"]
                logits = model(inputs)

            loss = cross_entropy_lm_loss(logits, labels)
            loss.backward()

            grad_norm = dist_utils.clip_grad_norm_(
                list(model.parameters()),
                max_norm=max_norm,
                foreach=True,
            )
            optimizer.step()

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            perf_stats.ntokens_since_last_log += labels.numel()

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
