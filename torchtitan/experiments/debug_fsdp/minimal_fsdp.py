#!/usr/bin/env python3
"""
Minimal FSDP debug script comparing three backends on the same seed weights:

  1. TorchTitan native (qwen3)
  2. HF Transformers modeling backend (torchtitan FSDP)
  3. HF Transformers from_pretrained (standalone FSDP, no torchtitan wrapping)

Usage:
    torchrun --nproc_per_node=2 minimal_fsdp.py \
        --job.config_file path/to/config.toml

Set USE_HF_FSDP=1 to use backend 3 instead of the default torchtitan path.
"""

from __future__ import annotations

import copy
import json
import os
import sys
import time
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
from torchtitan.models.llama4.infra.parallelize import apply_fsdp
from torchtitan.models.qwen3.model.state_dict_adapter import Qwen3StateDictAdapter
from torchtitan.protocols.train_spec import get_train_spec, TrainSpec
from torchtitan.tools import utils as titan_utils
from torchtitan.tools.logging import init_logger, logger
from torchtitan.tools.utils import device_module, device_type, set_default_dtype

USE_HF_FSDP = os.environ.get("USE_HF_FSDP", "0") == "1"

SEED_CHECKPOINT_FILENAME = "seed_model.pt"
SEED_CHECKPOINT_METADATA_FILENAME = "seed_model.json"
HF_ALLOWED_MISSING_KEY_SUFFIXES = ("rotary_emb.inv_freq",)
HF_TIED_WEIGHT_ALLOWED_MISSING_KEY_SUFFIXES = ("lm_head.weight",)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rank_zero_log(message: str) -> None:
    if not dist.is_initialized() or dist.get_rank() == 0:
        logger.info(message)



# ---------------------------------------------------------------------------
# Distributed init
# ---------------------------------------------------------------------------

def init_distributed(job_config: JobConfig) -> tuple[torch.device, ParallelDims]:
    required_env = ("RANK", "LOCAL_RANK", "WORLD_SIZE")
    missing_env = [v for v in required_env if v not in os.environ]
    if missing_env:
        raise RuntimeError(
            f"Missing distributed env vars: {', '.join(missing_env)}. Launch with torchrun."
        )
    if device_type != "cuda":
        raise RuntimeError(f"Expected CUDA, got {device_type!r}.")

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size < 2:
        raise RuntimeError("FSDP requires WORLD_SIZE >= 2.")

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
        parallel_dims.world_mesh, device, job_config.debug,
        distinct_seed_mesh_dims=[],
    )

    rank_zero_log(
        f"Initialized distributed run rank={dist.get_rank()} world_size={world_size} "
        f"local_rank={local_rank} mesh={parallel_dims.world_mesh.mesh.tolist()}"
    )
    return device, parallel_dims


# ---------------------------------------------------------------------------
# Seed checkpoint (shared by all backends)
# ---------------------------------------------------------------------------

def load_seed_checkpoint(path: str) -> dict[str, torch.Tensor]:
    """Load a seed .pt file from disk. Used by both torchtitan and HF FSDP paths."""
    f = Path(path) / SEED_CHECKPOINT_FILENAME
    if not f.is_file():
        raise FileNotFoundError(f"Seed checkpoint not found at {f}.")
    return torch.load(f, map_location="cpu", weights_only=True)


def load_seed_into_torchtitan_model(model, model_args, job_config):
    """Load seed checkpoint into a torchtitan-wrapped model (backends 1 & 2)."""
    if not job_config.checkpoint.initial_load_path:
        return
    sd = load_seed_checkpoint(job_config.checkpoint.initial_load_path)

    if job_config.model.name in {"llama3", "qwen3"}:
        adapters = {"llama3": Llama3StateDictAdapter, "qwen3": Qwen3StateDictAdapter}
        adapter = adapters[job_config.model.name](model_args, job_config.model.hf_assets_path)
        ik = model.load_state_dict(adapter.from_hf(sd), strict=False)
        allowed = ()
    elif job_config.model.name == "transformers_modeling_backend":
        ik = model.load_state_dict({f"model.{k}": v for k, v in sd.items()}, strict=False)
        allowed = HF_ALLOWED_MISSING_KEY_SUFFIXES
        if getattr(model_args, "tie_word_embeddings", False):
            allowed += HF_TIED_WEIGHT_ALLOWED_MISSING_KEY_SUFFIXES
    else:
        raise NotImplementedError(f"No seed loader for {job_config.model.name!r}")

    missing = [k for k in ik.missing_keys if not any(k.endswith(s) for s in allowed)]
    if missing or ik.unexpected_keys:
        raise RuntimeError(
            f"Seed checkpoint mismatch: missing={missing}, unexpected={list(ik.unexpected_keys)}"
        )
    rank_zero_log(f"Loaded seed checkpoint from {job_config.checkpoint.initial_load_path}")


def create_seed_checkpoint(job_config: JobConfig, train_spec: TrainSpec) -> None:
    """Build a model on CPU, init weights, and save as HF-compatible state dict."""
    dist_utils.set_determinism(
        world_mesh=None, device=torch.device("cpu"),
        debug_config=job_config.debug, distinct_seed_mesh_dims=[],
    )
    model, model_args = build_torchtitan_model(job_config, train_spec, use_meta_init=False)
    init_weights = getattr(model, "init_weights", None)
    if init_weights is not None and job_config.model.name != "transformers_modeling_backend":
        with torch.no_grad():
            init_weights()

    # Export to HF-compatible state dict
    export_dtype = TORCH_DTYPE_MAP[job_config.checkpoint.export_dtype]
    raw_sd = {k: v.detach().cpu().to(export_dtype).clone() for k, v in model.state_dict().items()}

    if job_config.model.name in {"llama3", "qwen3"}:
        adapters = {"llama3": Llama3StateDictAdapter, "qwen3": Qwen3StateDictAdapter}
        adapter = adapters[job_config.model.name](model_args, job_config.model.hf_assets_path)
        sd = adapter.to_hf(raw_sd)
    elif job_config.model.name == "transformers_modeling_backend":
        sd = {}
        for key, value in raw_sd.items():
            nkey = key.removeprefix("model.")
            if nkey == "lm_head.weight" and getattr(model_args, "tie_word_embeddings", False):
                continue
            sd[nkey] = value
    else:
        raise NotImplementedError(f"No seed export for {job_config.model.name!r}")

    # Save
    checkpoint_dir = Path(job_config.job.dump_folder) / job_config.checkpoint.folder / "step-0"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(sd, checkpoint_dir / SEED_CHECKPOINT_FILENAME)
    (checkpoint_dir / SEED_CHECKPOINT_METADATA_FILENAME).write_text(
        json.dumps({
            "format": "hf_compatible_state_dict",
            "model_name": job_config.model.name,
            "flavor": job_config.model.flavor,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    logger.info("Saved seed checkpoint to %s", checkpoint_dir)


# ---------------------------------------------------------------------------
# Backend 1 & 2: TorchTitan model building (native qwen3 / HF backend)
# ---------------------------------------------------------------------------

def build_torchtitan_model(job_config, train_spec, use_meta_init=True):
    flavor = job_config.model.flavor
    model_args = copy.deepcopy(train_spec.model_args[flavor])
    model_args.update_from_config(job_config)
    rank_zero_log(f"Building {job_config.model.name} {flavor} with {model_args}")

    with set_default_dtype(TORCH_DTYPE_MAP[job_config.training.dtype]):
        if use_meta_init:
            with torch.device("meta"):
                model = train_spec.model_cls(model_args)
        else:
            model = train_spec.model_cls(model_args)
    return model, model_args


def build_and_parallelize_torchtitan(job_config, train_spec, parallel_dims, device):
    """Build, load seed, parallelize, and materialize a torchtitan model."""
    load_shared_seed = bool(
        job_config.checkpoint.enable and job_config.checkpoint.initial_load_path
    )
    model, model_args = build_torchtitan_model(
        job_config, train_spec, use_meta_init=not load_shared_seed
    )
    seq_len = job_config.training.seq_len
    _, num_flops_per_token = model_args.get_nparams_and_flops(model, seq_len)

    if load_shared_seed:
        load_seed_into_torchtitan_model(model, model_args, job_config)
        model.to(device)
    else:
        model.to_empty(device=device)
        with torch.no_grad():
            model.init_weights()

    dp_mesh_dim_names = ("dp_shard",)
    #TODO(3ou): add mixed precision support and cpu_offload support
    apply_fsdp(
        model,
        parallel_dims.world_mesh[tuple(dp_mesh_dim_names)],
        param_dtype=None,
        reduce_dtype=None,
        pp_enabled=parallel_dims.pp_enabled,
        # cpu_offload=job_config.training.enable_cpu_offload,
        cpu_offload=False,
        reshard_after_forward_policy=job_config.parallelism.fsdp_reshard_after_forward,
    )

    model.train()
    return model, num_flops_per_token


# ---------------------------------------------------------------------------
# Backend 3: HF from_pretrained + FSDP via fsdp_plan
# ---------------------------------------------------------------------------

def _prepare_hf_pretrained_dir(job_config: JobConfig) -> Path:
    """Create a HF-format directory (config.json + model.safetensors) from seed."""
    import shutil
    from safetensors.torch import save_file

    hf_config_dir = Path(job_config.hf_transformers.model)
    out_dir = Path(job_config.job.dump_folder) / "hf_fsdp_pretrained"
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(hf_config_dir / "config.json", out_dir / "config.json")

    seed_path = job_config.checkpoint.initial_load_path
    if seed_path:
        sd = load_seed_checkpoint(seed_path)
        save_file(sd, out_dir / "model.safetensors")
        rank_zero_log(f"Prepared HF model dir at {out_dir} with seed from {seed_path}")
    else:
        rank_zero_log(f"Prepared HF model dir at {out_dir} (random init)")

    return out_dir


def build_hf_fsdp_model(job_config, world_size):
    """Build model via HF from_pretrained with fsdp_plan=auto."""
    v5_path = str(Path(__file__).resolve().parents[3] / "transformers-v5-fsdp" / "src")
    if v5_path not in sys.path:
        sys.path.insert(0, v5_path)
    from transformers import AutoModelForCausalLM

    # Rank 0 prepares the directory, others wait
    if dist.get_rank() == 0:
        _prepare_hf_pretrained_dir(job_config)
    dist.barrier()
    model_dir = Path(job_config.job.dump_folder) / "hf_fsdp_pretrained"

    rank_zero_log(f"Building HF model from {model_dir}")

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=torch.float32,
        fsdp_plan={"mode": "auto", "cpu_offload": False, "mixed_precision": False},
    )

    # Param count (local shard × world_size for full count)
    local_params = sum(p.numel() for p in model.parameters())
    num_params = local_params * world_size
    num_flops_per_token = 6 * num_params
    rank_zero_log(
        f"HF model: {num_params:,} params ({local_params:,} local/rank), "
        f"{num_flops_per_token:,} flops/token"
    )

    model.train()
    return model, num_flops_per_token


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def build_dataloader(job_config, train_spec, parallel_dims):
    if "HF_DATASETS_CACHE" not in os.environ:
        cache = Path(job_config.job.dump_folder) / "hf_datasets_cache"
        cache.mkdir(parents=True, exist_ok=True)
        os.environ["HF_DATASETS_CACHE"] = str(cache)
        datasets_config.HF_DATASETS_CACHE = str(cache)

    tokenizer = (
        train_spec.build_tokenizer_fn(job_config)
        if train_spec.build_tokenizer_fn is not None else None
    )
    if tokenizer is None:
        raise RuntimeError(f"{job_config.model.name!r} has no tokenizer.")

    if parallel_dims.dp_enabled:
        dp_mesh = parallel_dims.world_mesh["dp"]
        dp_world_size, dp_rank = dp_mesh.size(), dp_mesh.get_local_rank()
    else:
        dp_world_size, dp_rank = 1, 0

    return train_spec.build_dataloader_fn(
        dp_world_size=dp_world_size, dp_rank=dp_rank,
        tokenizer=tokenizer, job_config=job_config,
    )


def get_next_batch(data_iterator, device):
    try:
        input_dict, labels = next(data_iterator)
    except StopIteration as ex:
        raise DataloaderExhaustedError() from ex
    for key, value in input_dict.items():
        if isinstance(value, torch.Tensor):
            input_dict[key] = value.to(device)
    return input_dict, labels.to(device)


# ---------------------------------------------------------------------------
# Training loop (shared by all backends)
# ---------------------------------------------------------------------------

def train_loop(model, job_config, train_spec, parallel_dims, device,
               num_flops_per_token, is_hf_fsdp):
    color = (titan_utils.NoColor() if job_config.metrics.disable_color_printing
             else titan_utils.Color())
    mem_mon = DeviceMemoryMonitor(str(device))
    gpu_peak_flops = titan_utils.get_peak_flops(mem_mon.device_name)
    mem = mem_mon.get_peak_stats()
    rank_zero_log(f"Peak FLOPS used for computing MFU: {gpu_peak_flops:.3e}")
    rank_zero_log(
        f"{device.type.upper()} memory usage for model: "
        f"{mem.max_reserved_gib:.2f}GiB({mem.max_reserved_pct:.2f}%)"
    )
    mem_mon.reset_peak_stats()

    dataloader = build_dataloader(job_config, train_spec, parallel_dims)
    data_iterator = iter(dataloader)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=job_config.optimizer.lr,
        weight_decay=job_config.optimizer.weight_decay,
    )

    world_size = dist.get_world_size()
    batch_size = job_config.training.local_batch_size
    seq_len = job_config.training.seq_len
    steps = job_config.training.steps
    max_norm = job_config.training.max_norm
    warmup_steps = job_config.lr_scheduler.warmup_steps
    log_freq = job_config.metrics.log_freq

    rank_zero_log(
        f"Trainer: local_bs={batch_size}, global_bs={batch_size * world_size}, "
        f"seq_len={seq_len}, steps={steps} (warmup {warmup_steps})"
    )
    rank_zero_log("Training starts at step 1")

    ntokens_since_last_log = 0
    time_last_log = time.perf_counter()
    post_warmup_tps, post_warmup_tflops, post_warmup_mfu = [], [], []

    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)

        input_dict, labels = get_next_batch(data_iterator, device)
        inputs = input_dict["input"]

        if is_hf_fsdp:
            logits = model(input_ids=inputs).logits
        else:
            logits = model(inputs)

        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss.backward()

        grad_norm = dist_utils.clip_grad_norm_(
            list(model.parameters()), max_norm=max_norm, foreach=True,
        )
        optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        ntokens_since_last_log += labels.numel()

        if step % log_freq == 0 or step == 1 or step == steps:
            if parallel_dims.dp_cp_enabled:
                avg_loss = dist_utils.dist_mean(loss.detach(), parallel_dims.world_mesh["dp_cp"])
            else:
                avg_loss = loss.detach().item()

            dt = time.perf_counter() - time_last_log
            tps = ntokens_since_last_log / (dt * parallel_dims.non_data_parallel_size)
            tflops = num_flops_per_token * tps / 1e12
            mfu = 100 * num_flops_per_token * tps / gpu_peak_flops
            mem = mem_mon.get_peak_stats()
            c = color
            rank_zero_log(
                f"{c.red}step: {step:2}  "
                f"{c.green}loss: {avg_loss:7.4f}  "
                f"{c.orange}grad_norm: {grad_norm.item():7.4f}  "
                f"{c.turquoise}memory: {mem.max_reserved_gib:5.2f}GiB({mem.max_reserved_pct:.2f}%)  "
                f"{c.blue}tps: {round(tps):,}  "
                f"{c.cyan}tflops: {tflops:,.2f}  "
                f"{c.magenta}mfu: {mfu:.2f}%{c.reset}"
            )
            if step > warmup_steps:
                post_warmup_tps.append(tps)
                post_warmup_tflops.append(tflops)
                post_warmup_mfu.append(mfu)
            ntokens_since_last_log = 0
            time_last_log = time.perf_counter()
            mem_mon.reset_peak_stats()

        if step == 1:
            dist_utils.set_pg_timeouts(
                timeout=timedelta(seconds=job_config.comm.train_timeout_seconds),
                world_mesh=parallel_dims.world_mesh,
            )

    dist.barrier()
    if post_warmup_tps:
        n = len(post_warmup_tps)
        c = color
        rank_zero_log(
            f"{c.yellow}post-warmup ({warmup_steps} steps) avg  "
            f"{c.blue}tps: {round(sum(post_warmup_tps) / n):,}  "
            f"{c.cyan}tflops: {sum(post_warmup_tflops) / n:,.2f}  "
            f"{c.magenta}mfu: {sum(post_warmup_mfu) / n:.2f}%{c.reset}"
        )
    rank_zero_log("Training completed")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@record
def main() -> None:
    init_logger()
    job_config = ConfigManager().parse_args()
    train_spec = get_train_spec(job_config.model.name)

    # Seed checkpoint creation (single-process, no distributed)
    if job_config.checkpoint.enable and job_config.checkpoint.create_seed_checkpoint:
        create_seed_checkpoint(job_config, train_spec)
        return

    device, parallel_dims = init_distributed(job_config)
    try:
        if USE_HF_FSDP:
            rank_zero_log("Backend: HF from_pretrained + manual FSDP")
            model, num_flops_per_token = build_hf_fsdp_model(
                job_config, dist.get_world_size()
            )
        else:
            rank_zero_log(f"Backend: torchtitan ({job_config.model.name})")
            model, num_flops_per_token = build_and_parallelize_torchtitan(
                job_config, train_spec, parallel_dims, device
            )

        train_loop(
            model, job_config, train_spec, parallel_dims, device,
            num_flops_per_token, is_hf_fsdp=USE_HF_FSDP,
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
